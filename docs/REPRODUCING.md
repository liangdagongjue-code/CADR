# Reproducing CADR experiments

## 1. Environments and data

Use separate Python environments for the Minari and ManiSkill tracks.

- Minari track: `requirements-research.in`; the recorded environment uses
  PyTorch 2.8.0 with CUDA 12.8, Minari 0.5.3, and MuJoCo 3.2.3. Choose a
  compatible PyTorch build for your hardware before installing the other pins.
- ManiSkill track: `requirements-maniskill.in` and
  `requirements-maniskill-dp.in`. In particular, the upstream Diffusion Policy
  evaluator requires Gymnasium 0.29.1 rather than Gymnasium 1.x.
- `requirements-locks/remote/` contains full package inventories from the
  experiment environments, including unrelated environment packages.
- `requirements-mujoco-lock.txt` records a separate local visualization
  environment; do not substitute it for the formal-training environment.

Data must be obtained separately, with the original sources' access and license
conditions respected. Set `MINARI_DATASETS_PATH` to your dataset directory.

| Task group | Data / environment boundary |
|---|---|
| Locomotion | Minari `mujoco/{halfcheetah,hopper,walker2d}/medium-v0`; recovered v5 environments |
| Adroit | Minari `D4RL/{door,pen,hammer,relocate}` expert datasets; exact IDs in task configs; Gymnasium-Robotics 1.2.0 |
| ManiSkill | Official v3.0.1 state demonstrations for PickCube, StackCube, PegInsertionSide; 100 trajectories per task |

ManiSkill demonstrations must be replayed into the official state-baseline
format: `pd_ee_delta_pos` for PickCube/StackCube and `pd_ee_delta_pose` for
PegInsertionSide, with `physx_cpu`. Do not silently replace these with a different
control mode or newer environment version.

## 2. Example: HalfCheetah IQL and CADR

The following entry points are read from the released code. Full training is
compute intensive; this release check does not rerun the full training matrix.
Run from the repository root after installing the research dependencies:

```bash
python -m dt_corl_delay.download_minari_exact --dataset mujoco/halfcheetah/medium-v0 --datasets-root datasets/minari
# Point MINARI_DATASETS_PATH at the same datasets/minari directory.
python -m dt_corl_delay.prepare_seed_config --template configs/iql_halfcheetah_formal500k.json --output outputs/iql_seed101.json --seed 101
python -m dt_corl_delay.prepare_seed_config --template configs/cadr_halfcheetah_formal50k.json --output outputs/cadr_seed101.json --seed 101
python -m dt_corl_delay.train_offline --config outputs/iql_seed101.json --output outputs/iql_seed101
python -m dt_corl_delay.train_cadr --config outputs/cadr_seed101.json --base-checkpoint outputs/iql_seed101/checkpoint.pt --output outputs/cadr_seed101
python -m dt_corl_delay.evaluate_checkpoints --config outputs/cadr_seed101.json --base-checkpoint outputs/iql_seed101/checkpoint.pt --dynamics-checkpoint outputs/cadr_seed101/dynamics_checkpoint.pt --output outputs/eval_seed101
```

`prepare_seed_config` creates the output parent directory. Repeat with training
seeds 101--105 and the task-specific configs for the intended comparison.
`generated_formal_v1_*` and `generated_ablation_v1_*` preserve the recorded task,
seed, and ablation settings. `prepare_all_formal_configs` can regenerate the
formal training/evaluation config matrix in a separate workspace.

## 3. Diffusion Policy and DT-CORL interfaces

Obtain the upstream sources at the pinned commits in [THIRD_PARTY.md](../THIRD_PARTY.md).
For DT-CORL, place the source checkout at `external/DT-CORL`; the architecture
loader checks the pinned file hash. For ManiSkill, pass the checkout path to
the evaluator's `--source-root` option.

The CADR-DP evaluator accepts `--config`, `--policy-checkpoint`,
`--dynamics-checkpoint`, `--source-root`, and `--output`:

```bash
python -m dt_corl_delay.evaluate_maniskill_cadr_diffusion --help
python -m dt_corl_delay.train_cadr --help
python -m dt_corl_delay.train_dtcorl_adapter --help
```

Policy checkpoints, trained dynamics, demonstrations, and upstream code are
not included. Train or obtain them separately with the matching configuration;
do not interpret this repository as a checkpoint distribution.

## 4. Delays, comparisons, and metrics

The configs include fixed, IID, Markov-switching, changepoint, gradual-drift,
and support-shift settings. Delayed deployment keeps policy and dynamics
parameters fixed. Methods use matched seeds, initial conditions, and delay
traces. True-age access is reserved for explicitly labeled comparators; such
comparators are not guaranteed mathematical performance upper bounds.

Locomotion reports return; manipulation reports success measures. The paper
uses paired differences with hierarchical bootstrap intervals. Primary formal
experiments use five training seeds. Consult the manuscript/supplement for the
complete endpoint, aggregation, and comparison definitions.

Trace configs additionally require processed input files. The experiment
convention maps half RTT to discrete observation delay using a 50 ms reference
period, with explicit x2/x4 stress scaling. Raw traces and identifying network
metadata are not in this repository. A trace config alone is not sufficient to
recreate its original measured-network result.

## 5. Tests and release boundaries

The public `tests/` directory includes core algorithm/configuration-independent
unit tests. Internal archive-audit tests and tests of machine-specific remote
launchers are excluded together with that infrastructure. Optional PyTorch or
simulator tests require their matching dependencies; skip counts must not be
reported as passing full simulator validation.

The local release check runs the available tests and lightweight smoke pipeline.
The initial Windows check could not collect `test_maniskill_diffusion.py`
because local application policy blocked an h5py DLL. That module is retained
in the repository and must be tested in a functioning research environment;
this release does not claim a fully passing simulator/dependency test suite.
It does not repeat the ten-task/five-seed experiments, distribute raw training
data, or assert that the private archive/audit gate can be run from this public
code-only checkout.
