# Pretrained models

The [v0.1.0 release](https://github.com/rain-policy/RAIN/releases/tag/v0.1.0)
contains `rain-models-v0.1.0.tar.gz`, a 566,963,392-byte archive for eight tasks.
The repository and its release assets currently require private-repository access.

## Download and install

From the repository root:

```bash
gh auth login
python tools/download_models.py --task pickcube
python run.py --task pickcube --verify-only
```

Omit `--task` to install all eight tasks:

```bash
python tools/download_models.py
python run.py --task all --verify-only
```

Selecting a task installs only its four files; the downloader still fetches
and verifies the complete release archive. The installer checks the archive
SHA-256, the exact member list and every individual model hash. An existing
model with different bytes is preserved and reported as an error.

## Install an existing archive

```bash
python tools/download_models.py --archive /path/to/rain-models-v0.1.0.tar.gz
python run.py --task all --verify-only
```

This route only needs Python's standard library. Pretrained evaluation does
not require demonstration datasets.

## Bundle layout

```text
models/<task>/
├── uact.pt       # ACT proposal, uncertainty and normalization metadata
├── flow.pt       # Flow-matching refiner
├── heads.pt      # Error-prediction ensemble
└── route.json    # Frozen refinement-selection coefficients
```

Use the four files from the same task bundle together. `run.py` resolves the
linked model paths locally. The model archive and individual checksums are
defined in [release.json](../manifests/release.json) and
[tasks.json](../manifests/tasks.json).

The exported tensors and numeric route settings match the original models;
private path metadata was removed. See [validation](validation.md) for the
export checks.

## License

The eight-task v0.1.0 RAIN model weights, including the proposal, refiner and
error-ensemble checkpoints and associated route configuration, are released
under the [MIT License](../LICENSE), as selected by the authors on 2026-10-14.
The release remains private until public distribution is authorized.

Simulator source, simulator assets and demonstration datasets retain their
upstream terms; see [third-party notices](../THIRD_PARTY_NOTICES.md).
