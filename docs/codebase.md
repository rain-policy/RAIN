# Codebase guide

## Entry points

| File | Responsibility |
|---|---|
| `run.py` | Model checks, task selection, route-path resolution and evaluation |
| `train.py` | Preview or execute the nine training/calibration stages |
| `tools/download_models.py` | Download, verify and install model bundles |
| `tools/install_simulator.py` | Verify and install the reference ManiSkill source |

Start with these entry points for normal use. Low-level scripts expose their
own `--help` for development and experiments.

## Method components

| Component | Implementation |
|---|---|
| ACT proposal and uncertainty | `src/policies/maniskill_uncertain_act.py` |
| Temporal action aggregation | `src/policies/maniskill_uact_current_action_flow.py` |
| Joint ACT/FM refinement | `src/policies/maniskill_uact_joint_refiner.py` |
| Flow source construction | `src/policies/maniskill_uact_mars_action_flow.py` |
| Error ensemble and refinement selection | `src/policies/maniskill_uact_fm_learned_route.py` |
| CUDA Graph and proposal runtime | `src/runtime/maniskill_fair_inference.py` |
| Demonstration loading and fixed splits | `src/data/maniskill_uncertain_act_dataset.py` |
| Environment rollout and result writing | `scripts/evaluate_rain.py` |

The `maniskill_uact_*` identifiers are retained so the released checkpoints and
training stages use their original module/configuration conventions. This
release's user-facing method is RAIN.

## Configuration and data flow

1. `manifests/tasks.json` selects the task environment, control mode, seeds and
   the four model files.
2. `run.py` checks all four hashes and writes a locally resolved route JSON
   into the new output directory.
3. `scripts/evaluate_rain.py` loads the proposal, refiner and error ensemble,
   chooses a refinement count and executes the H8 action block.
4. The evaluator writes task-level `results.json`; `run.py` adds `summary.json`.

For training, `train.py` builds the stage commands. The implementations live
in `scripts/`; the [training guide](training.md) lists their dependencies.

## Release manifests

| Manifest | Contents |
|---|---|
| `tasks.json` | Runtime task settings and release/source model hashes |
| `release.json` | Model archive location, version, size and SHA-256 |
| `simulator.json` | Reference simulator archive, revision and SHA-256 |
| `datasets.json` | Processed demonstration identities and trajectory split IDs |

`hashes` identify the portable released model files; `source_hashes` identify
their original experiment artifacts. These may differ after path metadata
sanitization. Model and data manifests are versioned release contracts.
