"""Delay processes used only by the environment-side channel.

Policies must never receive a DelayProcess instance. Keeping delay generation
behind the channel boundary makes accidental privileged-input leakage easier to
detect in tests and code review.
"""

from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Any, Mapping, Protocol, Sequence


class DelayProcess(Protocol):
    def reset(self) -> None:
        """Reset episode-local state."""

    def sample(self, step: int, rng: random.Random) -> int:
        """Sample a non-negative integer observation delay."""


def _validate_distribution(delays: Sequence[int], probabilities: Sequence[float]) -> None:
    if not delays or len(delays) != len(probabilities):
        raise ValueError("delays and probabilities must be non-empty and equally sized")
    if any(delay < 0 for delay in delays):
        raise ValueError("delays must be non-negative")
    if any(probability < 0 for probability in probabilities):
        raise ValueError("probabilities must be non-negative")
    if abs(sum(probabilities) - 1.0) > 1e-8:
        raise ValueError("probabilities must sum to one")


@dataclass
class FixedDelay:
    delay: int

    def __post_init__(self) -> None:
        if self.delay < 0:
            raise ValueError("delay must be non-negative")

    def reset(self) -> None:
        return None

    def sample(self, step: int, rng: random.Random) -> int:
        del step, rng
        return self.delay


@dataclass
class CategoricalDelay:
    delays: tuple[int, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_distribution(self.delays, self.probabilities)

    def reset(self) -> None:
        return None

    def sample(self, step: int, rng: random.Random) -> int:
        del step
        return rng.choices(self.delays, weights=self.probabilities, k=1)[0]


class MarkovSwitchingDelay:
    def __init__(
        self,
        regimes: Sequence[CategoricalDelay],
        transition: Sequence[Sequence[float]],
        initial_regime: int = 0,
    ) -> None:
        if not regimes:
            raise ValueError("at least one regime is required")
        if len(transition) != len(regimes):
            raise ValueError("transition matrix must have one row per regime")
        for row in transition:
            _validate_distribution(tuple(range(len(regimes))), tuple(row))
        if not 0 <= initial_regime < len(regimes):
            raise ValueError("initial_regime is out of range")
        self.regimes = tuple(regimes)
        self.transition = tuple(tuple(row) for row in transition)
        self.initial_regime = initial_regime
        self.regime = initial_regime

    def reset(self) -> None:
        self.regime = self.initial_regime

    def sample(self, step: int, rng: random.Random) -> int:
        if step > 0:
            states = tuple(range(len(self.regimes)))
            self.regime = rng.choices(states, weights=self.transition[self.regime], k=1)[0]
        return self.regimes[self.regime].sample(step, rng)


@dataclass
class ChangepointDelay:
    before: CategoricalDelay
    after: CategoricalDelay
    change_step: int

    def __post_init__(self) -> None:
        if self.change_step < 0:
            raise ValueError("change_step must be non-negative")

    def reset(self) -> None:
        return None

    def sample(self, step: int, rng: random.Random) -> int:
        process = self.before if step < self.change_step else self.after
        return process.sample(step, rng)


@dataclass
class DriftingCategoricalDelay:
    """Linearly drift between two delay laws over a fixed horizon."""

    delays: tuple[int, ...]
    start_probabilities: tuple[float, ...]
    end_probabilities: tuple[float, ...]
    horizon_steps: int

    def __post_init__(self) -> None:
        _validate_distribution(self.delays, self.start_probabilities)
        _validate_distribution(self.delays, self.end_probabilities)
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")

    def reset(self) -> None:
        return None

    def sample(self, step: int, rng: random.Random) -> int:
        fraction = min(1.0, max(0.0, float(step) / self.horizon_steps))
        probabilities = tuple(
            (1.0 - fraction) * start + fraction * end
            for start, end in zip(self.start_probabilities, self.end_probabilities)
        )
        return rng.choices(self.delays, weights=probabilities, k=1)[0]


class TraceDelay:
    """Run-length encoded real delay trace with randomized episode starts."""

    def __init__(
        self,
        delays: Sequence[int],
        durations: Sequence[int] | None = None,
        random_start: bool = True,
        cycle: bool = True,
    ) -> None:
        if not delays:
            raise ValueError("trace must contain at least one delay")
        if any(int(delay) < 0 for delay in delays):
            raise ValueError("trace delays must be non-negative")
        if durations is None:
            durations = [1] * len(delays)
        if len(durations) != len(delays) or any(int(duration) <= 0 for duration in durations):
            raise ValueError("trace durations must be positive and match delays")
        self.delays = tuple(int(delay) for delay in delays)
        self.durations = tuple(int(duration) for duration in durations)
        cumulative: list[int] = []
        total = 0
        for duration in self.durations:
            total += duration
            cumulative.append(total)
        self.cumulative = tuple(cumulative)
        self.total_steps = total
        self.random_start = random_start
        self.cycle = cycle
        self._offset: int | None = None

    def reset(self) -> None:
        self._offset = None

    def sample(self, step: int, rng: random.Random) -> int:
        if self._offset is None:
            self._offset = rng.randrange(self.total_steps) if self.random_start else 0
        position = self._offset + max(0, step - 1)
        position = position % self.total_steps if self.cycle else min(position, self.total_steps - 1)
        index = bisect.bisect_right(self.cumulative, position)
        return self.delays[index]


def _categorical_from_config(config: Mapping[str, Any]) -> CategoricalDelay:
    return CategoricalDelay(
        delays=tuple(int(value) for value in config["delays"]),
        probabilities=tuple(float(value) for value in config["probabilities"]),
    )


def _trace_from_config(config: Mapping[str, Any]) -> TraceDelay:
    if "delays" in config:
        delays = [int(value) for value in config["delays"]]
        durations = [int(value) for value in config.get("durations", [1] * len(delays))]
    else:
        path = Path(str(config["path"]))
        if not path.is_absolute():
            path = Path(os.environ.get("DT_CORL_DATA_ROOT", ".")) / path
        delays: list[int] = []
        durations: list[int] = []
        delay_column = str(config.get("delay_column", "delay_steps_x1"))
        duration_column = str(config.get("duration_column", "duration_steps"))
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or delay_column not in reader.fieldnames:
                raise ValueError(f"trace {path} lacks delay column {delay_column}")
            for row in reader:
                delays.append(int(row[delay_column]))
                durations.append(int(row.get(duration_column, "1")))
    return TraceDelay(
        delays,
        durations,
        random_start=bool(config.get("random_start", True)),
        cycle=bool(config.get("cycle", True)),
    )


def build_delay_process(config: Mapping[str, Any]) -> DelayProcess:
    """Build a validated delay process from a JSON-compatible mapping."""

    kind = config["type"]
    if kind == "fixed":
        return FixedDelay(int(config["delay"]))
    if kind == "categorical":
        return _categorical_from_config(config)
    if kind == "markov":
        regimes = [_categorical_from_config(item) for item in config["regimes"]]
        return MarkovSwitchingDelay(regimes, config["transition"])
    if kind == "changepoint":
        return ChangepointDelay(
            before=_categorical_from_config(config["before"]),
            after=_categorical_from_config(config["after"]),
            change_step=int(config["change_step"]),
        )
    if kind == "drifting":
        return DriftingCategoricalDelay(
            delays=tuple(int(value) for value in config["delays"]),
            start_probabilities=tuple(float(value) for value in config["start_probabilities"]),
            end_probabilities=tuple(float(value) for value in config["end_probabilities"]),
            horizon_steps=int(config["horizon_steps"]),
        )
    if kind == "trace":
        return _trace_from_config(config)
    raise ValueError(f"unsupported delay process type: {kind}")
