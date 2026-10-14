# Installation and platform requirements

## Install the reference environment

Use Linux, Python 3.10 and an NVIDIA GPU. Clone the public repository:

```bash
git clone https://github.com/rain-policy/RAIN.git
cd RAIN
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python tools/install_simulator.py
```

The simulator installer downloads the exact retained ManiSkill 3.0.1 source
snapshot and verifies its archive hash and layout before installation. The
recorded reference revision is `fb3328d3b4e29fad8753e2c26a713320be786fd8`; its
old upstream Git URL is currently unavailable. The source archive avoids
substituting a newer simulator for the version used by the released models.
Its size and checksum are in [simulator.json](../manifests/simulator.json).

To use an already downloaded simulator archive:

```bash
python tools/install_simulator.py --archive /path/to/rain-maniskill-source-fb3328d.tar.gz
```

Add `--verify-only` to check the source archive without installing it.

The release was checked with Linux, Python 3.10.20, PyTorch 2.6.0+cu124,
NumPy 2.2.6, Gymnasium 1.3.0, SAPIEN 3.0.3 and ManiSkill 3.0.1 at the pinned
commit. Training utilities additionally use HDF5, SciPy, diffusers and TensorBoard.
Dependency versions are pinned in `requirements.txt`; PyTorch's CUDA wheel
and the reference simulator are installed explicitly.

## Check the environment

```bash
python -c "import torch, mani_skill, sapien; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python run.py --help
```

Then [install the models](models.md) and start a one-episode run. The recorded
GPU integration checks used an isolated source checkout with inherited
dependencies; a complete empty-machine CUDA installation has not yet been
validated. See [validation](validation.md) for the exact checks.

## Vulkan on headless servers

CUDA availability alone is not sufficient for SAPIEN. A headless container must
expose the Vulkan loader and the correct driver ICD. An `ErrorIncompatibleDriver`
from `vk::createInstanceUnique` indicates an environment setup problem.

For a system with a configured Vulkan ICD, `VK_ICD_FILENAMES` (and, when needed,
`VK_DRIVER_FILES`) can select its JSON file. Use the ICD and libraries installed
for your system, not paths copied from another server. Mesa Lavapipe is an
alternative for software rendering while keeping RAIN inference on CUDA. Select
the Vulkan device exposed by that installation with `--render-backend`; for
example, Lavapipe on the validation server advertises `pci:0000:00:00.0`:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --task pickcube --episodes 1 --render-backend pci:0000:00:00.0 --output outputs/software-renderer-smoke
```

That command assumes its Vulkan loader, ICD and libraries are already configured.
The evaluator includes a small compatibility fix for PCI device string parsing in
the pinned ManiSkill version; task code and physics are unchanged. Do not select
`none` as a universal workaround: some tasks still construct rendering resources.
Rendering performance from software Vulkan is not a GPU latency benchmark.

Required robot/scene assets must be available to ManiSkill. It will report missing
assets at environment creation. Use its asset downloader for the asset IDs it
requests; do not replace missing assets with a different robot or scene. You can
set `MS_ASSET_DIR` to an existing ManiSkill asset directory.

Windows and CPU-only end-to-end execution have not been validated. Hash checking,
archive installation and command preview use only the Python standard library.
