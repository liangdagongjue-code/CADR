"""Write a checksum-bound manifest for an official ManiSkill DP run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from .provenance import package_source_provenance


EXPECTED_MANISKILL_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(args: argparse.Namespace) -> dict[str, object]:
    dataset = args.dataset.resolve()
    metadata = dataset.with_suffix(".json")
    source_root = args.source_root.resolve()
    dependency_lock = args.dependency_lock.resolve()
    for required in (dataset, metadata, source_root / ".git", dependency_lock):
        if not required.exists():
            raise FileNotFoundError(required)
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_MANISKILL_COMMIT:
        raise ValueError(f"unexpected ManiSkill source commit: {commit}")
    replay_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    recorded_control = replay_metadata["env_info"]["env_kwargs"].get(
        "control_mode",
        replay_metadata["episodes"][0].get("control_mode"),
    )
    if recorded_control != args.control_mode:
        raise ValueError(
            f"dataset control mode {recorded_control!r} != {args.control_mode!r}"
        )
    official_files = [
        "examples/baselines/diffusion_policy/train.py",
        "examples/baselines/diffusion_policy/baselines.sh",
        "examples/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py",
        "examples/baselines/diffusion_policy/diffusion_policy/evaluate.py",
        "scripts/data_generation/replay_for_il_baselines.sh",
    ]
    result: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "official_maniskill_v3.0.1_state_diffusion_policy",
        "environment_id": args.environment_id,
        "control_mode": args.control_mode,
        "sim_backend": "physx_cpu",
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
        "num_demos": args.num_demos,
        "total_iters": args.total_iters,
        "local_source_provenance": package_source_provenance(),
        "dataset": {
            "path": str(dataset),
            "sha256": _sha256(dataset),
            "metadata_path": str(metadata),
            "metadata_sha256": _sha256(metadata),
            "episodes_in_metadata": len(replay_metadata["episodes"]),
        },
        "upstream": {
            "repository": "https://github.com/haosulab/ManiSkill.git",
            "tag": "v3.0.1",
            "commit": commit,
            "file_sha256": {
                relative: _sha256(source_root / relative) for relative in official_files
            },
        },
        "dependency_lock": {
            "path": str(dependency_lock),
            "sha256": _sha256(dependency_lock),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--control-mode", required=True)
    parser.add_argument("--max-episode-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-demos", type=int, required=True)
    parser.add_argument("--total-iters", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(write_manifest(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
