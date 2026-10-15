"""Stable provenance for the local CADR implementation loaded by a run."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of one immutable run input."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_input_provenance(path: Path) -> dict[str, Any]:
    """Describe a file input strongly enough to reject unsafe partial resumes."""

    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def package_source_provenance() -> dict[str, Any]:
    """Hash every Python source file in the installed/local package tree."""

    package_root = Path(__file__).resolve().parent
    hashes = {
        path.relative_to(package_root).as_posix(): file_sha256(path)
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    combined = hashlib.sha256()
    for relative_path, digest in hashes.items():
        combined.update(relative_path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {
        "scheme": "sha256(relative_path_NUL_file_sha256_LF)",
        "package": "dt_corl_delay",
        "file_count": len(hashes),
        "tree_sha256": combined.hexdigest(),
        "file_sha256": hashes,
    }


def verify_package_source_unchanged(expected: dict[str, Any]) -> None:
    """Fail a long run if package bytes changed after its start snapshot."""

    actual = package_source_provenance()
    if actual != expected:
        raise RuntimeError(
            "dt_corl_delay source changed while the run was active: "
            f"started={expected.get('tree_sha256')} "
            f"current={actual.get('tree_sha256')}"
        )
