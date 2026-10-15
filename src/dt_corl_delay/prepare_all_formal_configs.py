"""Materialize every deterministic formal_v1 run configuration before freezing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .prepare_cadr_ablation_config import materialize as materialize_ablation
from .prepare_locomotion_task_config import materialize as materialize_minari
from .prepare_maniskill_trace_config import materialize as materialize_trace
from .prepare_seed_config import materialize as materialize_seed


FORMAL_SEEDS = (101, 102, 103, 104, 105)
MANISKILL_TASKS = {
    "pickcube": "maniskill_pickcube_official_dp_formal.json",
    "stackcube": "maniskill_stackcube_official_dp_formal.json",
    "peginsertion": "maniskill_peginsertion_official_dp_formal.json",
}
LOCOMOTION_TASKS = {
    "halfcheetah": "mujoco/halfcheetah/medium-v0",
    "hopper": "mujoco/hopper/medium-v0",
    "walker2d": "mujoco/walker2d/medium-v0",
}
ADROIT_TASKS = {
    task: f"D4RL/{task}/expert-v2"
    for task in ("door", "pen", "hammer", "relocate")
}
ABLATION_VARIANTS = (
    "ack_map",
    "no_content",
    "fixed_prior",
    "single_model",
    "replan_every_step",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_all(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_dir = project_root / "configs"
    generated: set[Path] = set()
    trace_template = config_dir / "maniskill_real_network_50ms_formal.json"

    for slug, template_name in MANISKILL_TASKS.items():
        template = config_dir / template_name
        for seed in FORMAL_SEEDS:
            synthetic = config_dir / f"generated_formal_v1_{slug}_synthetic_seed{seed}.json"
            trace = config_dir / f"generated_formal_v1_{slug}_real_network_seed{seed}.json"
            materialize_seed(template, synthetic, seed)
            materialize_trace(template, trace_template, trace, seed)
            generated.update((synthetic, trace))

    pick_template = config_dir / MANISKILL_TASKS["pickcube"]
    for seed in FORMAL_SEEDS:
        synthetic = config_dir / f"generated_formal_v1_pickcube_synthetic_seed{seed}.json"
        materialize_seed(pick_template, synthetic, seed)
        for variant in ABLATION_VARIANTS:
            output = config_dir / f"generated_ablation_v1_pickcube_{variant}_seed{seed}.json"
            materialize_ablation(synthetic, output, variant)
            generated.add(output)

    locomotion_templates = {
        "iql": "iql_halfcheetah_formal500k.json",
        "cql": "cql_halfcheetah_formal500k.json",
        "cadr": "cadr_halfcheetah_formal50k.json",
        "dtcorl": "dtcorl_halfcheetah_formal50k.json",
        "real_network": "eval_halfcheetah_real_network_formal.json",
    }
    for slug, dataset_id in LOCOMOTION_TASKS.items():
        for seed in FORMAL_SEEDS:
            for role, template_name in locomotion_templates.items():
                output = config_dir / f"generated_formal_v1_{slug}_{role}_seed{seed}.json"
                materialize_minari(
                    config_dir / template_name,
                    output,
                    dataset_id,
                    seed,
                )
                generated.add(output)

    adroit_templates = {
        "iql": "iql_halfcheetah_formal500k.json",
        "cadr": "cadr_halfcheetah_formal50k.json",
        "dtcorl": "dtcorl_halfcheetah_formal50k.json",
        "real_network": "eval_halfcheetah_real_network_formal.json",
    }
    for task, dataset_id in ADROIT_TASKS.items():
        for seed in FORMAL_SEEDS:
            for role, template_name in adroit_templates.items():
                output = config_dir / f"generated_formal_v1_adroit_{task}_{role}_seed{seed}.json"
                materialize_minari(
                    config_dir / template_name,
                    output,
                    dataset_id,
                    seed,
                )
                generated.add(output)

    if len(generated) != 210:
        raise AssertionError(f"expected 210 generated formal configs, found {len(generated)}")
    hashes = {
        path.relative_to(project_root).as_posix(): _sha256(path)
        for path in sorted(generated)
    }
    return {"config_count": len(hashes), "file_sha256": hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_all(args.project_root)
    print(json.dumps({"config_count": result["config_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
