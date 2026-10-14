# Contributing to RAIN

This repository maintains the RAIN method for eight ManiSkill tasks. For the
research overview, see the [project page](https://rain-policy.github.io/).

## Development checks

Use Python 3.10. The CPU regression suite only needs PyTorch, NumPy and the
development requirements:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6 -r requirements-dev.txt
python -m compileall -q src scripts run.py train.py tools
python -m pytest tests -q
```

The CI suite does not need downloaded models, demonstrations or a GPU. GPU
integration evaluation is a separate check described in [validation](docs/validation.md).

## Changes and pull requests

- Keep public commands and documentation synchronized.
- Preserve checkpoint module names, data splits and release model checksums.
- Add targeted regression coverage when changing routing, action aggregation
  or runtime behavior.
- Summarize the behavior change and the checks performed in the pull request.
- Keep datasets, checkpoints, generated outputs and private experiment logs
  outside Git.

## Questions and bug reports

Use an issue for an installation failure, unexpected runtime behavior or a
question about the method. Include a minimal command and task-specific result
fields when possible. The current repository access and license are described
in [LICENSE](LICENSE).
