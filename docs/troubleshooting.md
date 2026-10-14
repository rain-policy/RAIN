# Troubleshooting

## Model access or download fails

The release is public and requires no login. Check that your network can reach
GitHub and its release download service. You can also download the archive from
the [release page](https://github.com/rain-policy/RAIN/releases/tag/v0.1.0) and
install it using `--archive`; see [models](models.md).

## Missing model or checksum mismatch

Run `python run.py --task all --verify-only`. Download the matching release
again if a file is incomplete. The installer preserves a different existing
model; select a clean checkout or move your own model to a separate directory
before installing the frozen release.

## `ErrorIncompatibleDriver` from Vulkan

SAPIEN needs a Vulkan loader and a usable ICD even for state-observation tasks.
Check the system's Vulkan installation and select its configured device. See
[installation](installation.md) for loader variables and the software-renderer
example. An eager policy backend does not fix a missing Vulkan driver.

## Missing simulation assets

Use ManiSkill's asset downloader for the asset ID reported by environment
creation, or set `MS_ASSET_DIR` to an existing matching asset directory. Asset
availability and licenses are managed by the upstream simulator.

## CUDA Graph execution fails

First try a single episode with `--backend eager` to isolate graph capture from
model/environment loading. Include the GPU, PyTorch/CUDA versions, exact task
and traceback in an issue. Eager timing is separate from the reference CUDA
Graph setting.

## Output directory already exists

Choose a new `--output` path. Evaluation preserves prior run directories.
Training paths and completed stage artifacts should also use a distinct run
namespace.

## Results differ from the task table

Check the task's model hashes, control mode, seeds, episode limit and runtime
settings. A single episode is an integration check; the reference success
rates are from 100 episodes per task. Use the environment versions listed in
[installation](installation.md).

## Prepare an issue report

Include the commit, command, task, environment versions, model release, renderer
and relevant error/output fields. The [bug-report form](https://github.com/rain-policy/RAIN/issues/new?template=bug_report.yml)
provides these fields. Remove credentials and unrelated personal paths from logs.
