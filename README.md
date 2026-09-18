# RAIN

RAIN for state-observation manipulation in ManiSkill. This repository contains
the RAIN method, pretrained models for eight tasks, and its training and
evaluation tools. It is currently a private release.

## Install

Use Linux, Python 3.10 and an NVIDIA GPU. The reference environment uses
PyTorch 2.6.0 with CUDA 12.4. SAPIEN also requires a working Vulkan loader and
ICD, including for state-only simulation. See [installation](docs/installation.md).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Download models

Authenticate GitHub CLI with an account that can read this private repository:

```bash
gh auth login
python tools/download_models.py
python run.py --task all --verify-only
```

Alternatively download `rain-models-v0.1.0.tar.gz` from this repository's Release
and pass `--archive /path/to/rain-models-v0.1.0.tar.gz` to the download script.
The installer verifies the archive and every model before installation.

## Run RAIN

From the repository root, run one episode first:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --task pickcube --episodes 1 --output outputs/pickcube-smoke
```

Run all eight tasks, or use the original 100-episode protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --task all --episodes 1 --output outputs/all-smoke
CUDA_VISIBLE_DEVICES=0 python run.py --task all --episodes 100 --output outputs/evaluation-100
```

Choose a new output directory for each invocation. `--dry-run` prints commands
without starting simulation. `--video` also records episode videos when rendering
is available. `--backend eager` is a diagnostic alternative to the default CUDA
Graph execution; it is not the reference latency setting.

| Task argument | Environment |
|---|---|
| `pickcube` | PickCube-v1 |
| `pushcube` | PushCube-v1 |
| `stackcube` | StackCube-v1 |
| `peginsertionside` | PegInsertionSide-v1 |
| `plugcharger` | PlugCharger-v1 |
| `pullcubetool` | PullCubeTool-v1 |
| `pusht` | PushT-v1 |
| `stackpyramid` | StackPyramid-v1 |

The released configuration executes eight actions per decision and dynamically
selects zero to three FM refinements. The proposal, uncertainty, refinement and
routing components are all part of RAIN.

The original archived evaluation achieved 533 successes in 800 episodes
(66.625%). This is a reference result, not a promise for every software/hardware
setup. See [validation](docs/validation.md) for the checks actually performed on
this release and [training](docs/training.md) for retraining RAIN.

## Files

- `src/`: RAIN model, data and runtime components.
- `scripts/`: RAIN training, calibration and rollout implementations.
- `run.py`, `train.py`: user-facing entry points.
- `manifests/`: task settings, model checksums and source identities.
- `models/`: downloaded models, excluded from Git.

Historical internal checkpoint identifiers are preserved for compatibility.
See [third-party notices](THIRD_PARTY_NOTICES.md) and [license status](LICENSE).
