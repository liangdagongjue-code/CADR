"""Create or verify the immutable source/protocol manifest for formal_v1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PINNED_RESEARCH_FILES = (
    "research/DATA_SOURCES.md",
    "research/EXPERIMENT_PROTOCOL.md",
    "research/LITERATURE_POSITIONING.md",
    "research/METHOD_CADR_DP.md",
    "research/STATISTICAL_ANALYSIS_PLAN.md",
    "research/upstreams.json",
)

PINNED_UPSTREAM_FILES = {
    "DT-CORL": (
        "utils/network.py",
    ),
    "ManiSkill": (
        "examples/baselines/diffusion_policy/train.py",
        "examples/baselines/diffusion_policy/baselines.sh",
        "examples/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py",
        "examples/baselines/diffusion_policy/diffusion_policy/evaluate.py",
        "scripts/data_generation/replay_for_il_baselines.sh",
    ),
}
EXPECTED_UPSTREAM_COMMITS = {
    "DT-CORL": "7e44adf934b8aef91ca6fb9e057a1935c6a167c9",
    "ManiSkill": "a4a4f9272ad64b1564035874b605ceb687b63ed8",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_formal_files(project_root: Path) -> dict[str, str]:
    candidates: set[Path] = set()
    candidates.update((project_root / "src" / "dt_corl_delay").rglob("*.py"))
    candidates.update((project_root / "tests").rglob("*.py"))
    candidates.update((project_root / "scripts" / "remote").glob("*.sh"))
    candidates.update(
        path
        for path in (project_root / "configs").glob("*formal*.json")
    )
    candidates.update(
        (project_root / "configs").glob("generated_ablation_v1_*.json")
    )
    formal_configs = [
        path
        for path in candidates
        if path.parent == project_root / "configs" and path.suffix == ".json"
    ]
    for config_path in formal_configs:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for scenario in config.get("scenarios", []):
            if scenario.get("type") == "trace" and "path" in scenario:
                candidates.add(
                    project_root / "data" / "processed" / str(scenario["path"])
                )
                trace_manifest = (
                    project_root / "data" / "processed" / str(scenario["path"])
                ).with_suffix(".manifest.json")
                if trace_manifest.is_file():
                    candidates.add(trace_manifest)
    candidates.update((project_root / "data" / "manifests").rglob("*.json"))
    candidates.update((project_root / "datasets" / "minari").rglob("download_manifest.json"))
    candidates.update((project_root / "datasets" / "minari").rglob("*metadata.json"))
    candidates.update(
        path
        for path in (project_root / "audits").glob("*.json")
        if path.name.startswith(("mujoco_", "D4RL_", "ManiSkill_"))
    )
    for relative in (
        "pyproject.toml",
        "requirements-research.in",
        "requirements-maniskill-dp.in",
        "envs/research.freeze.txt",
        "envs/maniskill-dp.freeze.txt",
        *PINNED_RESEARCH_FILES,
    ):
        candidates.add(project_root / relative)
    upstreams_path = project_root / "research" / "upstreams.json"
    upstreams = json.loads(upstreams_path.read_text(encoding="utf-8"))["upstreams"]
    for upstream in upstreams:
        name = str(upstream["name"])
        relative_root = upstream.get("remote_path", upstream.get("local_path"))
        if not relative_root or name not in PINNED_UPSTREAM_FILES:
            raise ValueError(f"unsupported frozen upstream declaration: {upstream}")
        source_root = project_root / str(relative_root)
        expected_commit = str(upstream["commit"])
        if expected_commit != EXPECTED_UPSTREAM_COMMITS[name]:
            raise ValueError(
                f"unpinned {name} declaration: {expected_commit}"
            )
        if (source_root / ".git").exists():
            actual_commit = subprocess.run(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if actual_commit != expected_commit:
                raise ValueError(
                    f"unexpected {name} commit: {actual_commit} != {expected_commit}"
                )
            dirty = subprocess.run(
                [
                    "git", "-C", str(source_root), "status", "--porcelain", "--",
                    *PINNED_UPSTREAM_FILES[name],
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if dirty:
                raise ValueError(f"pinned {name} source files are dirty: {dirty}")
        for relative in PINNED_UPSTREAM_FILES[name]:
            candidates.add(source_root / relative)
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return {
        path.relative_to(project_root).as_posix(): _sha256(path)
        for path in sorted(candidates)
    }


def build_manifest(project_root: Path, series: str) -> dict[str, Any]:
    hashes = collect_formal_files(project_root.resolve())
    combined = hashlib.sha256()
    for relative_path, digest in hashes.items():
        combined.update(relative_path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {
        "series": series,
        "scheme": "sha256(relative_path_NUL_file_sha256_LF)",
        "file_count": len(hashes),
        "tree_sha256": combined.hexdigest(),
        "file_sha256": hashes,
    }


def verify_or_create(
    project_root: Path,
    manifest_path: Path,
    series: str,
) -> dict[str, Any]:
    current = build_manifest(project_root, series)
    manifest_path = manifest_path.resolve()
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {
            key: recorded[key]
            for key in ("series", "scheme", "file_count", "tree_sha256", "file_sha256")
        }
        if comparable != current:
            before = recorded.get("file_sha256", {})
            after = current["file_sha256"]
            changed = sorted(
                path
                for path in set(before) | set(after)
                if before.get(path) != after.get(path)
            )
            raise ValueError(
                "formal source/protocol freeze mismatch: " + ", ".join(changed)
            )
        return recorded
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **current,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--series", default="formal_v1")
    args = parser.parse_args()
    result = verify_or_create(
        args.project_root,
        args.manifest,
        args.series,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "series": result["series"],
                "file_count": result["file_count"],
                "tree_sha256": result["tree_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
