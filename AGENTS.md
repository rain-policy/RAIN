# RAIN repository maintenance

This repository is the ManiSkill-only implementation of RAIN. Its public
remote is `https://github.com/rain-policy/RAIN`; its public project page is
`https://rain-policy.github.io/`. The paper title is **Refine Actions If Needed:
Adaptive Action Chunk Refinement for Robot Manipulation**.

- Preserve the eight-task H8 deployment contract and the 0/1/2/3 refinement choices.
- Keep the proposal, uncertainty, FM refinement, error ensemble and calibration
  components. Standalone baselines, ablations and real-robot code are outside
  this repository's release scope.
- Preserve checkpoint-compatible module/configuration identifiers. Do not rename
  core modules just to improve their appearance.
- `manifests/tasks.json` and `datasets.json` record frozen model/data identities.
  Do not replace these with new training inputs or claim a new experiment is the
  archived 533/800 evaluation.
- The model Release v0.1.0 archive is immutable. The reference simulator source
  is a separate archive described by `manifests/simulator.json`.
- Use `run.py`, `train.py` and the installation/download tools as user-facing
  entry points. Keep docs aligned with their actual options and outputs.
- Keep arXiv and paper BibTeX marked pending until the confirmed public reference
  is supplied. `CITATION.cff` describes the software release, not a paper venue.
- The user explicitly authorized public code, model and manuscript PDF release
  on 2026-10-14. Keep release links aligned with public availability.
  The user selected MIT for original RAIN code, documentation and all eight-task
  v0.1.0 model weights on 2026-10-14. Preserve upstream dependency, simulator
  asset and dataset terms; do not apply the RAIN license to those materials.
- Do not add datasets, models, raw experiment logs, private paths or manuscript
  review IDs to Git. Keep internal audits under the local parent workspace's
  reports directory.
- Run the existing CPU regression suite for code changes; GPU evaluation is a
  separate check and must be described accurately.
- Server access follows the user's local Codex → ordinary SSH policy. No Codex
  or account tokens may be installed, copied or forwarded to shared servers.
