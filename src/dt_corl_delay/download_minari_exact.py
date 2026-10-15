"""Download a known Minari dataset without enumerating the entire HF Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


KNOWN_DATA_SHA256 = {
    "mujoco/halfcheetah/medium-v0": "db46045497395bc621f2b409089e495f0e51a1182e6257f2f319f7d6654f66c5",
    "D4RL/door/expert-v2": "acde1fbf64d02bf8b9778ad7ad8b72fbc5e1f314c514abd2e155b24d33f9a25f",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(dataset_id: str, datasets_root: Path) -> dict[str, object]:
    parts = dataset_id.split("/")
    if len(parts) < 3 or any(not part for part in parts):
        raise ValueError(f"expected namespace/environment/dataset, got {dataset_id}")
    namespace = parts[0]
    relative_dataset = "/".join(parts[1:])
    repository_id = f"farama-minari/{namespace}"
    namespace_root = datasets_root / namespace
    required_files = [
        f"{relative_dataset}/data/metadata.json",
        f"{relative_dataset}/data/main_data.hdf5",
    ]
    optional_files = [
        "namespace_metadata.json",
        f"{parts[1]}/namespace_metadata.json",
    ]
    downloaded: list[str] = []
    for filename in required_files:
        local_path = hf_hub_download(
            repo_id=repository_id,
            repo_type="dataset",
            filename=filename,
            local_dir=namespace_root,
        )
        downloaded.append(str(Path(local_path).resolve()))
    for filename in optional_files:
        try:
            local_path = hf_hub_download(
                repo_id=repository_id,
                repo_type="dataset",
                filename=filename,
                local_dir=namespace_root,
            )
            downloaded.append(str(Path(local_path).resolve()))
        except EntryNotFoundError:
            pass

    data_path = namespace_root / relative_dataset / "data" / "main_data.hdf5"
    actual_sha256 = _sha256(data_path)
    expected_sha256 = KNOWN_DATA_SHA256.get(dataset_id)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {dataset_id}: expected {expected_sha256}, got {actual_sha256}"
        )
    manifest = {
        "dataset_id": dataset_id,
        "repository_id": repository_id,
        "main_data_path": str(data_path.resolve()),
        "main_data_bytes": data_path.stat().st_size,
        "main_data_sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "downloaded_files": downloaded,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    }
    manifest_path = namespace_root / relative_dataset / "download_manifest.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    args = parser.parse_args()
    report = download(args.dataset, args.datasets_root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
