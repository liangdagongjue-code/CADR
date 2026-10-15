"""Audit real RTT datasets and create loss-aware control-delay traces."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RTT_COLUMN = "Time (ms)"
PING_TIMESTAMP_COLUMN = "Timestamp"
NR_FREQUENCY_COLUMN = "5G KPI PCell RF Frequency [MHz]"
NR_PCI_COLUMN = "5G KPI PCell RF Serving PCI"
LTE_EARFCN_COLUMN = "LTE KPI PCell Serving EARFCN(DL)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _weighted_percentile(values: Iterable[float], weights: Iterable[int], percentile: float) -> float:
    value_array = np.asarray(list(values), dtype=np.float64)
    weight_array = np.asarray(list(weights), dtype=np.int64)
    if value_array.size == 0 or weight_array.sum() <= 0:
        raise ValueError("weighted percentile requires non-empty positive weights")
    order = np.argsort(value_array)
    sorted_values = value_array[order]
    cumulative = np.cumsum(weight_array[order])
    target = percentile / 100.0 * cumulative[-1]
    index = min(np.searchsorted(cumulative, target, side="left"), len(order) - 1)
    return float(sorted_values[index])


def _delay_steps(rtt_ms: float, control_period_ms: float, factor: float, maximum: int) -> int:
    # delay=0 means arrival by the immediately following decision; delay=k
    # means k additional decisions are missed.
    intervals = math.ceil(max(0.0, rtt_ms * factor) / control_period_ms - 1e-12)
    return min(maximum, max(0, intervals - 1))


def _read_valid_rtt_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counters = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {PING_TIMESTAMP_COLUMN, RTT_COLUMN}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks required columns {sorted(required)}")
        for raw in reader:
            counters["source_rows"] += 1
            timestamp = _finite_float(raw.get(PING_TIMESTAMP_COLUMN))
            rtt = _finite_float(raw.get(RTT_COLUMN))
            if timestamp is None or rtt is None or rtt < 0.0:
                counters["rows_without_valid_rtt"] += 1
                continue
            rows.append(
                {
                    "timestamp_ms": timestamp,
                    "rtt_ms": rtt,
                    "nr_frequency_mhz": raw.get(NR_FREQUENCY_COLUMN, ""),
                    "nr_pci": raw.get(NR_PCI_COLUMN, ""),
                    "lte_earfcn": raw.get(LTE_EARFCN_COLUMN, ""),
                }
            )
            counters["valid_rtt_rows"] += 1
            if str(raw.get(NR_FREQUENCY_COLUMN, "")).strip():
                counters["rows_with_nr_frequency"] += 1
            if str(raw.get(LTE_EARFCN_COLUMN, "")).strip():
                counters["rows_with_lte_earfcn"] += 1
    rows.sort(key=lambda row: row["timestamp_ms"])
    deduplicated: list[dict[str, Any]] = []
    for row in rows:
        if deduplicated and row["timestamp_ms"] == deduplicated[-1]["timestamp_ms"]:
            deduplicated[-1] = row
            counters["duplicate_timestamps"] += 1
        else:
            deduplicated.append(row)
    return deduplicated, dict(counters)


def convert_file(
    source: Path,
    destination: Path,
    control_period_ms: float,
    one_way_factor: float,
    stress_multipliers: tuple[float, ...],
    maximum_delay: int,
    outage_threshold_ms: float,
    segment_break_ms: float,
) -> dict[str, Any]:
    rows, source_counts = _read_valid_rtt_rows(source)
    if len(rows) < 2:
        raise ValueError(f"{source} has fewer than two valid RTT rows")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stress_columns = [f"delay_steps_x{multiplier:g}" for multiplier in stress_multipliers]
    fieldnames = [
        "segment_id", "source_timestamp_ms", "rtt_ms", "nr_frequency_mhz",
        "nr_pci", "lte_earfcn", "duration_steps", "is_inferred_outage",
        *stress_columns,
    ]
    delay_histograms = {column: Counter() for column in stress_columns}
    weighted_rtts: list[float] = []
    weights: list[int] = []
    output_rows = total_control_steps = inferred_outage_steps = segment_id = 0
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            if index + 1 < len(rows):
                gap_ms = rows[index + 1]["timestamp_ms"] - row["timestamp_ms"]
                if gap_ms <= 0.0:
                    continue
                if gap_ms > segment_break_ms:
                    duration_steps, next_segment = 1, True
                else:
                    duration_steps = max(
                        1, int(round(min(gap_ms, outage_threshold_ms) / control_period_ms))
                    )
                    next_segment = False
            else:
                duration_steps, gap_ms, next_segment = 1, control_period_ms, False
            output = {
                "segment_id": segment_id,
                "source_timestamp_ms": f"{row['timestamp_ms']:.3f}",
                "rtt_ms": f"{row['rtt_ms']:.6g}",
                "nr_frequency_mhz": row["nr_frequency_mhz"],
                "nr_pci": row["nr_pci"],
                "lte_earfcn": row["lte_earfcn"],
                "duration_steps": duration_steps,
                "is_inferred_outage": 0,
            }
            for multiplier, column in zip(stress_multipliers, stress_columns):
                delay = _delay_steps(
                    row["rtt_ms"], control_period_ms, one_way_factor * multiplier, maximum_delay
                )
                output[column] = delay
                delay_histograms[column][delay] += duration_steps
            writer.writerow(output)
            output_rows += 1
            total_control_steps += duration_steps
            weighted_rtts.append(row["rtt_ms"])
            weights.append(duration_steps)
            if outage_threshold_ms < gap_ms <= segment_break_ms:
                outage_steps = max(
                    1, int(round((gap_ms - outage_threshold_ms) / control_period_ms))
                )
                outage_output = dict(output)
                outage_output.update(
                    source_timestamp_ms=f"{row['timestamp_ms'] + outage_threshold_ms:.3f}",
                    rtt_ms="", duration_steps=outage_steps, is_inferred_outage=1,
                )
                for column in stress_columns:
                    outage_output[column] = maximum_delay
                    delay_histograms[column][maximum_delay] += outage_steps
                writer.writerow(outage_output)
                output_rows += 1
                total_control_steps += outage_steps
                inferred_outage_steps += outage_steps
            if next_segment:
                segment_id += 1
    return {
        "source": str(source),
        "source_sha256": _sha256(source),
        "derived": str(destination),
        "derived_sha256": _sha256(destination),
        **source_counts,
        "derived_rows": output_rows,
        "segments": segment_id + 1,
        "control_steps": total_control_steps,
        "inferred_outage_steps": inferred_outage_steps,
        "rtt_ms": {
            "min": min(row["rtt_ms"] for row in rows),
            "p50_time_weighted": _weighted_percentile(weighted_rtts, weights, 50),
            "p95_time_weighted": _weighted_percentile(weighted_rtts, weights, 95),
            "p99_time_weighted": _weighted_percentile(weighted_rtts, weights, 99),
            "max": max(row["rtt_ms"] for row in rows),
        },
        "delay_histograms": {
            column: {str(delay): count for delay, count in sorted(histogram.items())}
            for column, histogram in delay_histograms.items()
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root, output_root = args.source_root.resolve(), args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    multipliers = tuple(float(value) for value in args.stress_multipliers.split(","))
    files = sorted(source_root.rglob("*_rtt.csv"))
    if not files:
        raise FileNotFoundError(f"no *_rtt.csv files under {source_root}")
    audits: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for source in files:
        destination = output_root / source.relative_to(source_root)
        try:
            audits.append(convert_file(
                source, destination, args.control_period_ms, args.one_way_factor,
                multipliers, args.maximum_delay, args.outage_threshold_ms,
                args.segment_break_ms,
            ))
        except Exception as error:
            failures.append({"source": str(source), "error": repr(error)})
    manifest = {
        "schema_version": 1,
        "source_root": str(source_root), "output_root": str(output_root),
        "control_period_ms": args.control_period_ms, "one_way_factor": args.one_way_factor,
        "stress_multipliers": multipliers, "maximum_delay": args.maximum_delay,
        "outage_threshold_ms": args.outage_threshold_ms,
        "segment_break_ms": args.segment_break_ms,
        "source_files_found": len(files), "files_converted": len(audits),
        "failures": failures, "files": audits,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control-period-ms", type=float, default=50.0)
    parser.add_argument("--one-way-factor", type=float, default=0.5)
    parser.add_argument("--stress-multipliers", default="1,2,4")
    parser.add_argument("--maximum-delay", type=int, default=32)
    parser.add_argument("--outage-threshold-ms", type=float, default=1000.0)
    parser.add_argument("--segment-break-ms", type=float, default=5000.0)
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps({
        "source_files_found": manifest["source_files_found"],
        "files_converted": manifest["files_converted"],
        "failures": manifest["failures"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
