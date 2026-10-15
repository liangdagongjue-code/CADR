from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from dt_corl_delay.formal_freeze import verify_or_create


class FormalFreezeTest(unittest.TestCase):
    def test_create_verify_and_detect_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "src" / "dt_corl_delay").mkdir(parents=True)
            (root / "scripts" / "remote").mkdir(parents=True)
            (root / "configs").mkdir()
            (root / "research").mkdir()
            (root / "envs").mkdir()
            (root / "src" / "dt_corl_delay" / "core.py").write_text(
                "VALUE = 1\n", encoding="utf-8"
            )
            (root / "scripts" / "remote" / "run_formal.sh").write_text(
                "#!/bin/sh\n", encoding="utf-8"
            )
            (root / "configs" / "task_formal.json").write_text(
                "{}\n", encoding="utf-8"
            )
            for relative in (
                "pyproject.toml",
                "requirements-research.in",
                "requirements-maniskill-dp.in",
                "envs/research.freeze.txt",
                "envs/maniskill-dp.freeze.txt",
            ):
                (root / relative).write_text("\n", encoding="utf-8")
            (root / "research" / "upstreams.json").write_text(
                json.dumps({"upstreams": []}) + "\n", encoding="utf-8"
            )
            for relative in (
                "DATA_SOURCES.md",
                "EXPERIMENT_PROTOCOL.md",
                "LITERATURE_POSITIONING.md",
                "METHOD_CADR_DP.md",
                "STATISTICAL_ANALYSIS_PLAN.md",
            ):
                (root / "research" / relative).write_text("\n", encoding="utf-8")

            manifest_path = root / "freeze.json"
            created = verify_or_create(root, manifest_path, "formal_v1")
            verified = verify_or_create(root, manifest_path, "formal_v1")
            self.assertEqual(created, verified)
            (root / "src" / "dt_corl_delay" / "core.py").write_text(
                "VALUE = 2\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "core.py"):
                verify_or_create(root, manifest_path, "formal_v1")


if __name__ == "__main__":
    unittest.main()
