# Changelog

## v0.1.1 — 2026-10-14

- Published the ManiSkill code, pretrained models and manuscript PDF.
- Enabled model and simulator downloads without GitHub CLI or authentication.
- Linked the manuscript PDF and project homepage.
- Passed 21 CPU checks; original model and simulator archives are unchanged.

- Adopted the MIT License for original RAIN code, documentation and the
  eight-task v0.1.0 model weights on 2026-10-14.
- Organized the repository homepage and documentation around pretrained
  evaluation, data preparation, training and development.
- Added project-page navigation and a pending arXiv reference slot.
- Added software citation metadata, issue forms and contribution guidance.
- Moved the private code/model repository to the `rain-policy` organization.
- Replaced the unavailable upstream commit URL with a verified archive of the
  retained reference ManiSkill source and a source installer.

## v0.1.0 — 2026-09-18

- Released RAIN training, calibration and evaluation code for eight ManiSkill tasks.
- Packaged 32 portable model artifacts with archive and individual SHA-256 checks.
- Verified one GPU integration episode per task against the archived episodes.
- Passed 17 CPU regression/packaging tests and training-stage argument parsing.
- Removed private filesystem paths from model metadata while preserving tensors
  and numeric route settings.

The v0.1.0 model artifact remains unchanged and is publicly downloadable.
Its original RAIN model weights use MIT, as selected on 2026-10-14.
