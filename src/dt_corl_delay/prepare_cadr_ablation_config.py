"""Materialize immutable CADR ablation configurations from a frozen template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUPPORTED_VARIANTS = {
    "ack_map",
    "no_content",
    "fixed_prior",
    "single_model",
    "replan_every_step",
}


def materialize(
    template: Path,
    output: Path,
    variant: str,
) -> dict[str, Any]:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported CADR ablation: {variant}")
    config = json.loads(template.read_text(encoding="utf-8"))
    config["ablation_variant"] = variant
    base_series = str(config.get("experiment_series", "development"))
    config["experiment_series"] = f"ablation_v1_{base_series}_{variant}"
    # Ablation runs isolate one deployable CADR component at a time; the
    # privileged upper bound is evaluated only in the frozen main series.
    config["include_oracle_upper_bound"] = False
    if "act_horizon" in config:
        config["required_methods"] = [
            "official_maniskill_dp_direct_delayed_observation",
            f"cadr_dp_ablation_{variant}_plus_official_maniskill_dp",
        ]
    if variant == "ack_map":
        config["age_decision_rule"] = "map"
        config["use_content_conditioning"] = False
    elif variant == "no_content":
        config["age_decision_rule"] = "posterior_mean"
        config["use_content_conditioning"] = False
    elif variant == "fixed_prior":
        config["posterior_updates"] = False
    elif variant == "single_model":
        config["ensemble_size"] = 1
    elif variant == "replan_every_step":
        if "act_horizon" not in config:
            raise ValueError("replan_every_step requires a chunked-policy template")
        config["cadr_act_horizon"] = 1

    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise ValueError(
            f"generated ablation config already exists with different content: {output}"
        )
    output.write_text(serialized, encoding="utf-8", newline="\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(SUPPORTED_VARIANTS), required=True)
    args = parser.parse_args()
    result = materialize(
        args.template.resolve(),
        args.output.resolve(),
        args.variant,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "variant": result["ablation_variant"],
                "experiment_series": result["experiment_series"],
            }
        )
    )


if __name__ == "__main__":
    main()
