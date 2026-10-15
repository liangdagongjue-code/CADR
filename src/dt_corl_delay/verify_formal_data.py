"""Reject formal runs whose exact offline dataset differs from its audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} checksum mismatch: {actual} != {expected}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def verify(project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_id = str(config["dataset_id"])
    if str(config.get("dataset_type", "minari")) == "maniskill":
        dataset_path = Path(str(config["dataset_path"]))
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path
        metadata_path = dataset_path.with_suffix(".json")
        environment_id = str(config["environment"]["env_id"])
        audit_path = (
            project_root
            / "audits"
            / f"ManiSkill_{environment_id}_official_dp_v3_0_1.json"
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audited = audit["metadata"]
        if str(audited["dataset_id"]) != dataset_id:
            raise ValueError("ManiSkill audit dataset id does not match configuration")
        return {
            "dataset_id": dataset_id,
            "dataset": _verified(
                dataset_path, str(audited["dataset_sha256"]), "ManiSkill dataset"
            ),
            "metadata": _verified(
                metadata_path,
                str(audited["metadata_sha256"]),
                "ManiSkill metadata",
            ),
            "audit": str(audit_path),
        }

    dataset_root = project_root / "datasets" / "minari" / dataset_id
    manifest_path = dataset_root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest["dataset_id"]) != dataset_id:
        raise ValueError("Minari download manifest dataset id does not match configuration")
    expected = str(manifest["main_data_sha256"])
    if manifest.get("expected_sha256") not in (None, expected):
        raise ValueError("Minari known and downloaded checksums disagree")
    metadata_path = dataset_root / "data" / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    return {
        "dataset_id": dataset_id,
        "dataset": _verified(
            dataset_root / "data" / "main_data.hdf5", expected, "Minari dataset"
        ),
        "metadata_path": str(metadata_path),
        "manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.project_root, args.config.resolve()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
