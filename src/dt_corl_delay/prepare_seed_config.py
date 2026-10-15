"""Materialize an immutable per-seed JSON config from a tracked template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def materialize(template: Path, output: Path, seed: int) -> dict[str, object]:
    config = json.loads(template.read_text(encoding="utf-8"))
    config["seed"] = seed
    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise ValueError(f"generated config already exists with different content: {output}")
    output.write_text(serialized, encoding="utf-8", newline="\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    result = materialize(args.template.resolve(), args.output.resolve(), args.seed)
    print(json.dumps({"output": str(args.output.resolve()), "seed": result["seed"]}))


if __name__ == "__main__":
    main()
