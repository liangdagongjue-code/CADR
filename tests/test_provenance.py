from __future__ import annotations

import re
import unittest

from dt_corl_delay.provenance import (
    package_source_provenance,
    verify_package_source_unchanged,
)


class PackageSourceProvenanceTest(unittest.TestCase):
    def test_manifest_is_deterministic_and_contains_core_modules(self) -> None:
        first = package_source_provenance()
        second = package_source_provenance()
        self.assertEqual(first, second)
        self.assertEqual(first["package"], "dt_corl_delay")
        self.assertGreater(first["file_count"], 10)
        self.assertIn("delayed_eval.py", first["file_sha256"])
        self.assertIn("maniskill_diffusion.py", first["file_sha256"])
        self.assertRegex(first["tree_sha256"], re.compile(r"^[0-9a-f]{64}$"))

    def test_source_guard_accepts_snapshot_and_rejects_wrong_tree(self) -> None:
        snapshot = package_source_provenance()
        verify_package_source_unchanged(snapshot)
        wrong = dict(snapshot)
        wrong["tree_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            verify_package_source_unchanged(wrong)


if __name__ == "__main__":
    unittest.main()
