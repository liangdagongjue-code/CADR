# CADR

**Censored-ACK Delay Reconstruction** adapts frozen offline policies to unknown,
non-stationary observation delays. It combines observation-age inference from
received and outstanding acknowledgements with state reconstruction along
actions that were actually executed. Policy and dynamics parameters stay fixed
during delayed deployment.

This repository accompanies *Censored-ACK State Reconstruction for Frozen
Offline Policies under Hidden Non-Stationary Observation Delays*. The manuscript
is in preparation; this repository does not imply acceptance or publication.

## What is included

- CADR age estimation, delayed channels, dynamics ensembles, and policy interfaces.
- IQL/CQL training and evaluation, an adapter for the pinned upstream DT-CORL
  architecture, and chunk-consistent CADR-DP evaluation.
- Experiment configurations, core unit tests, and captured environment package lists.
- A small linear-control smoke example that does not require a GPU or simulators.

The manuscript studies ten tasks: three MuJoCo locomotion tasks, four Adroit
tasks, and three ManiSkill tasks. The IQL and Diffusion Policy interfaces are
the CADR evaluations; CQL is a direct-policy baseline in locomotion. The
DT-CORL comparison is an architecture-matched adapter, not a reproduction of
its original end-to-end training procedure.

## Quick start: lightweight smoke example

Use Python 3.10 or newer in an isolated environment. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m dt_corl_delay.experiment --config configs/smoke.json --output outputs/smoke
```

This writes a toy checkpoint, metrics, and SVG trajectories. It checks the
clean-training/delayed-deployment pipeline; it is not a reproduction of the
MuJoCo, Adroit, or ManiSkill paper results.

For the dependency-light tests:

```bash
python -m pip install numpy pytest
python -m pytest tests/test_delay_channel.py tests/test_smoke.py tests/test_trace_delay.py -q
```

## Reproducing the research experiments

See [the reproduction guide](docs/REPRODUCING.md) for dependencies, dataset IDs,
training/evaluation entry points, and the scope of this code-only release.
Keep the research and ManiSkill environments separate: their Gymnasium versions
are intentionally different. Captured package lists are environment records,
not a promise that every package can be installed on every operating system.

### Main modules

| Component | Module under `src/dt_corl_delay/` |
|---|---|
| Delayed observations and ACK records | `channel.py` |
| Censored-ACK age inference | `delay_belief.py` |
| Dynamics and frozen-policy interface | `dynamics.py`, `controllers.py` |
| Offline IQL/CQL learning | `train_offline.py`, `iql.py`, `cql.py` |
| CADR dynamics learning | `train_cadr.py` |
| IQL/CQL delayed evaluation | `evaluate_checkpoints.py` |
| Chunk-consistent diffusion evaluation | `evaluate_maniskill_cadr_diffusion.py` |
| Upstream DT-CORL architecture adapter | `official_dtcorl.py`, `train_dtcorl_adapter.py` |
| Aggregate analysis | `aggregate_formal_results.py`, `build_paper_artifacts.py` |

## Release scope

This is a **code-only release**. Datasets, checkpoints, raw network captures,
per-run results, internal archive-audit infrastructure, server synchronization
scripts, and manuscript drafts are not distributed here. Trace-based configs
require the corresponding processed trace files; those files are not bundled.
The legacy `formal_freeze` and archival tools expect additional research assets
and are not a self-contained public reproduction gate.

Upstream source code is not vendored. Obtain it from the original projects and
follow their licenses; see [third-party sources](THIRD_PARTY.md). Code availability
is distinct from a grant of reuse rights: **no repository-wide license has yet
been selected by the authors**.

## Citation

The manuscript has no assigned DOI or acceptance status. A formal citation will
be added when an author-approved public manuscript identifier is available.

## Issues

Please report reproducible code issues through this repository's issue tracker.
Do not attach credentials, private network metadata, or restricted datasets.
