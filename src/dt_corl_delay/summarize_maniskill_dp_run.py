"""Convert official DP TensorBoard logs into a checksum-bound JSON summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .provenance import package_source_provenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ema_states_equal(left_path: Path, right_path: Path) -> bool:
    left = torch.load(left_path, map_location="cpu", weights_only=False)["ema_agent"]
    right = torch.load(right_path, map_location="cpu", weights_only=False)["ema_agent"]
    if set(left) != set(right):
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def summarize(run_dir: Path, output: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event file in {run_dir}")
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = sorted(accumulator.Tags().get("scalars", []))
    scalar_series: dict[str, list[dict[str, float | int]]] = {}
    scalar_wall_times: list[float] = []
    for tag in scalar_tags:
        events = accumulator.Scalars(tag)
        scalar_wall_times.extend(float(event.wall_time) for event in events)
        scalar_series[tag] = [
            {
                "step": int(event.step),
                "value": float(event.value),
                "wall_time_unix": float(event.wall_time),
            }
            for event in events
        ]
    best: dict[str, dict[str, float | int]] = {}
    for metric in ("eval/success_once", "eval/success_at_end", "eval/return"):
        values = scalar_series.get(metric, [])
        if values:
            selected = max(values, key=lambda row: (float(row["value"]), -int(row["step"])))
            best[metric] = selected
    checkpoints = []
    for path in sorted((run_dir / "checkpoints").glob("*.pt")):
        checkpoints.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    clean_selection: dict[str, Any] = {
        "metric": "eval/success_once",
        "tie_break": "earliest_step",
        "selected_event": best.get("eval/success_once"),
    }
    selected_event = clean_selection["selected_event"]
    alias_path = run_dir / "checkpoints" / "best_eval_success_once.pt"
    if selected_event is not None and alias_path.is_file():
        selected_step = int(selected_event["step"])
        step_path = run_dir / "checkpoints" / f"{selected_step}.pt"
        clean_selection["checkpoint_alias"] = {
            "name": alias_path.name,
            "bytes": alias_path.stat().st_size,
            "sha256": _sha256(alias_path),
        }
        if step_path.is_file():
            if not _ema_states_equal(alias_path, step_path):
                raise ValueError(
                    "best_eval_success_once.pt does not match the clean-selected step "
                    f"checkpoint: {selected_step}"
                )
            clean_selection["selected_step_checkpoint"] = {
                "name": step_path.name,
                "bytes": step_path.stat().st_size,
                "sha256": _sha256(step_path),
                "ema_state_exact_match": True,
            }
        else:
            clean_selection["selected_step_checkpoint"] = {
                "name": step_path.name,
                "available": False,
                "validation": "upstream_best_alias_with_frozen_strict_improvement_rule",
            }
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "official_ManiSkill_v3.0.1_state_Diffusion_Policy",
        "summary_source_provenance": package_source_provenance(),
        "run_dir": str(run_dir),
        "run_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "event_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in event_files
        ],
        "event_wall_time_span_seconds": (
            max(scalar_wall_times) - min(scalar_wall_times)
            if scalar_wall_times
            else 0.0
        ),
        "scalar_series": scalar_series,
        "best_metrics": best,
        "clean_checkpoint_selection": clean_selection,
        "checkpoints": checkpoints,
    }
    output = output.resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
