"""Convert source-bound Windows WiFi RTT measurements into control traces."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

from .network_traces import _delay_steps, _sha256


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000.0


def convert(
    raw_directory: Path,
    output_file: Path,
    control_period_ms: float,
    one_way_factor: float,
    stress_multipliers: tuple[float, ...],
    maximum_delay: int,
    segment_break_ms: float,
) -> dict[str, Any]:
    metadata_path = raw_directory / "metadata.json"
    source_path = raw_directory / "rtt.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if not metadata.get("route_verified_as_wlan", False):
        raise ValueError("trace was not verified as using the WLAN interface")
    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            success = str(raw["success"]).lower() == "true"
            rtt = float(raw["rtt_ms"]) if success and raw["rtt_ms"] else None
            if rtt is not None and (not math.isfinite(rtt) or rtt < 0.0):
                success, rtt = False, None
            rows.append({
                "sequence": int(raw["sequence"]),
                "timestamp_ms": _timestamp(raw["timestamp_utc"]),
                "success": success,
                "rtt_ms": rtt,
                "status": raw["status"],
            })
    if len(rows) < 2:
        raise ValueError("WiFi trace requires at least two samples")
    rows.sort(key=lambda row: row["timestamp_ms"])
    columns = [f"delay_steps_x{multiplier:g}" for multiplier in stress_multipliers]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "segment_id", "sequence", "source_timestamp_ms", "rtt_ms", "success",
        "status", "duration_steps", "is_inferred_outage", *columns,
    ]
    histograms = {column: Counter() for column in columns}
    segment_id = total_steps = failed_steps = 0
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            if index + 1 < len(rows):
                gap = rows[index + 1]["timestamp_ms"] - row["timestamp_ms"]
                if gap <= 0.0:
                    continue
                duration = 1 if gap > segment_break_ms else max(1, int(round(gap / control_period_ms)))
                next_segment = gap > segment_break_ms
            else:
                duration, next_segment = 1, False
            output = {
                "segment_id": segment_id,
                "sequence": row["sequence"],
                "source_timestamp_ms": f"{row['timestamp_ms']:.3f}",
                "rtt_ms": "" if row["rtt_ms"] is None else f"{row['rtt_ms']:.6g}",
                "success": int(row["success"]),
                "status": row["status"],
                "duration_steps": duration,
                "is_inferred_outage": int(not row["success"]),
            }
            for multiplier, column in zip(stress_multipliers, columns):
                delay = maximum_delay if not row["success"] else _delay_steps(
                    row["rtt_ms"], control_period_ms, one_way_factor * multiplier, maximum_delay
                )
                output[column] = delay
                histograms[column][delay] += duration
            writer.writerow(output)
            total_steps += duration
            if not row["success"]:
                failed_steps += duration
            if next_segment:
                segment_id += 1
    valid_rtts = [row["rtt_ms"] for row in rows if row["rtt_ms"] is not None]
    manifest = {
        "schema_version": 1,
        "raw_directory": str(raw_directory.resolve()),
        "source_sha256": _sha256(source_path),
        "metadata_sha256": _sha256(metadata_path),
        "derived_file": str(output_file.resolve()),
        "derived_sha256": _sha256(output_file),
        "metadata": metadata,
        "samples": len(rows),
        "successful_samples": len(valid_rtts),
        "failed_samples": len(rows) - len(valid_rtts),
        "control_steps": total_steps,
        "failed_control_steps": failed_steps,
        "segments": segment_id + 1,
        "control_period_ms": control_period_ms,
        "one_way_factor": one_way_factor,
        "stress_multipliers": stress_multipliers,
        "maximum_delay": maximum_delay,
        "rtt_ms": {
            "min": min(valid_rtts) if valid_rtts else None,
            "mean": sum(valid_rtts) / len(valid_rtts) if valid_rtts else None,
            "max": max(valid_rtts) if valid_rtts else None,
        },
        "delay_histograms": {
            column: {str(delay): count for delay, count in sorted(histogram.items())}
            for column, histogram in histograms.items()
        },
    }
    manifest_path = output_file.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--control-period-ms", type=float, default=50.0)
    parser.add_argument("--one-way-factor", type=float, default=0.5)
    parser.add_argument("--stress-multipliers", default="1,2,4")
    parser.add_argument("--maximum-delay", type=int, default=32)
    parser.add_argument("--segment-break-ms", type=float, default=5000.0)
    args = parser.parse_args()
    result = convert(
        args.raw_directory.resolve(), args.output_file.resolve(), args.control_period_ms,
        args.one_way_factor,
        tuple(float(value) for value in args.stress_multipliers.split(",")),
        args.maximum_delay, args.segment_break_ms,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
