# Third-party dependencies

RAIN uses ManiSkill for simulation, SAPIEN for physics/rendering, PyTorch for
models, and diffusers for training utilities. These are installed as dependencies;
their upstream license terms continue to apply.

- ManiSkill: https://github.com/mani-skill/ManiSkill. The retained reference
  source archive is indexed in `manifests/simulator.json`; its recorded
  revision is `fb3328d3b4e29fad8753e2c26a713320be786fd8`. The archive retains
  ManiSkill's Apache-2.0 `LICENSE`, `LICENSE-3RD-PARTY` and asset-specific
  notices. Asset terms remain separate from the code license.
- SAPIEN: https://github.com/haosulab/SAPIEN.
- PyTorch: https://github.com/pytorch/pytorch.
- diffusers: https://github.com/huggingface/diffusers.

No upstream baseline implementation or full ManiSkill source tree is vendored
in Git. The reference simulator is supplied as a separate, verified Release
attachment, including its packaged simulation assets. The MIT license in
`LICENSE` covers original RAIN code, documentation and the eight-task v0.1.0
model weights. It does not replace upstream software, simulator asset or
demonstration dataset terms. The separately distributed manuscript is not
covered by the code and model MIT license.
