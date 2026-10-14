# RAIN documentation

RAIN's ManiSkill release supports pretrained evaluation and the method's own
training and calibration workflow. The project overview and videos are on the
[project page](https://rain-policy.github.io/).

## Run a pretrained policy

1. [Install the reference environment](installation.md).
2. [Download and verify the models](models.md).
3. [Run an evaluation episode](evaluation.md).
4. Use [troubleshooting](troubleshooting.md) if environment creation or model
   loading fails.

## Train a policy

1. [Prepare the demonstration HDF5 file](data.md).
2. [Run the training and calibration stages](training.md).
3. Evaluate the exported models using the command in the training guide.

## Develop and verify

- [Codebase guide](codebase.md): public entry points and implementation map.
- [Validation record](validation.md): completed checks and integration results.
- [Contributing](../CONTRIBUTING.md): development setup and contribution workflow.
- [Changelog](../CHANGELOG.md): release history.

The official paper citation and arXiv link will appear in the
[repository README](../README.md) when the public preprint is available.
