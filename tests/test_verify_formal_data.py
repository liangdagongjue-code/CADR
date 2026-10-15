from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dt_corl_delay.verify_formal_data import verify


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class FormalDataVerificationTest(unittest.TestCase):
    def test_minari_manifest_binds_dataset_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_id = "mujoco/example/medium-v0"
            dataset_root = root / "datasets" / "minari" / dataset_id
            (dataset_root / "data").mkdir(parents=True)
            payload = b"exact-minari-data"
            (dataset_root / "data" / "main_data.hdf5").write_bytes(payload)
            (dataset_root / "data" / "metadata.json").write_text("{}")
            checksum = _sha256(payload)
            (dataset_root / "download_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_id": dataset_id,
                        "main_data_sha256": checksum,
                        "expected_sha256": checksum,
                    }
                )
            )
            config = root / "config.json"
            config.write_text(json.dumps({"dataset_id": dataset_id}))
            self.assertEqual(verify(root, config)["dataset"]["sha256"], checksum)
            (dataset_root / "data" / "main_data.hdf5").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify(root, config)

    def test_maniskill_audit_binds_dataset_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            relative = Path("data/maniskill/processed/PickCube-v1/example.h5")
            dataset = root / relative
            dataset.parent.mkdir(parents=True)
            dataset_bytes = b"trajectory"
            metadata_bytes = b"metadata"
            dataset.write_bytes(dataset_bytes)
            dataset.with_suffix(".json").write_bytes(metadata_bytes)
            (root / "audits").mkdir()
            dataset_id = "maniskill/PickCube-v1/example"
            (root / "audits" / "ManiSkill_PickCube-v1_official_dp_v3_0_1.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "dataset_id": dataset_id,
                            "dataset_sha256": _sha256(dataset_bytes),
                            "metadata_sha256": _sha256(metadata_bytes),
                        }
                    }
                )
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "dataset_type": "maniskill",
                        "dataset_id": dataset_id,
                        "dataset_path": relative.as_posix(),
                        "environment": {"env_id": "PickCube-v1"},
                    }
                )
            )
            result = verify(root, config)
            self.assertEqual(result["metadata"]["sha256"], _sha256(metadata_bytes))
            dataset.with_suffix(".json").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "metadata checksum mismatch"):
                verify(root, config)


if __name__ == "__main__":
    unittest.main()
