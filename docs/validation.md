# Release validation — 2026-09-18

## Completed

- All eight RAIN tasks completed one full episode each using the released source
  and portable model artifacts on an NVIDIA A800, FP32, CUDA Graph execution,
  parallel proposal streams and the H8 policy configuration.
- The simulation installation came from the clean pinned ManiSkill commit,
  outside the original experiment checkout. A separate virtual environment used
  the recorded experiment dependencies through system-site-packages. This checks
  source isolation and compatibility; it is **not** a completely fresh dependency
  installation from an empty machine.
- The container required its existing Mesa Lavapipe Vulkan libraries and ICD,
  selected as `pci:0000:00:00.0`. Policy inference remained on CUDA. No shared
  driver installation or original experiment source was changed.
- 17 CPU regression/packaging tests passed, covering ensemble routing, temporal
  aggregation, inference equivalence, configuration and data split consistency.
- Both the regular and Peg-specific training paths passed argument parsing for
  all nine stages. Full training was not repeated.
- 61 retained algorithm, data and runtime definitions matched the original
  source AST exactly before formatting.
- The 32 deployment artifacts were verified against the original experiment
  identities. Portable exports remove private filesystem paths and update linked
  hashes: all 3,104 serialized tensors and every numeric route setting were
  unchanged. Source hashes and release hashes are recorded separately.
- The release archive was downloaded, unpacked locally and every file verified.
- All eight smoke episodes matched their corresponding original archived
  episodes in success, final success, return, episode length, refinement-count
  histogram and total FM evaluations. Timing samples were not compared.

## One-episode integration results

| Task | Seed | Completed steps | Success |
|---|---:|---|---|
| PickCube | 40000 | Full configured episode | Yes |
| PushCube | 41000 | Full configured episode | Yes |
| StackCube | 42000 | Full configured episode | Yes |
| PegInsertionSide | 43000 | Full configured episode | No |
| PlugCharger | 44000 | Full configured episode | No |
| PullCubeTool | 45000 | Full configured episode | Yes |
| PushT | 46000 | Full configured episode | Yes |
| StackPyramid | 47000 | Full configured episode | No |

These are integration checks, not estimates of task success rates. The archived
533/800 reference result was not remeasured in this release session. Full
retraining, 800-episode reevaluation, an empty-machine install and video output
are outside the completed validation. Software rendering was not used to claim
reference policy latency.

## Model packaging

The downloadable archive contains only `models/<task>/{uact.pt,flow.pt,heads.pt,route.json}`.
The checksum of the complete archive is in `manifests/release.json` and individual
file checksums are in `manifests/tasks.json`. Private path metadata was sanitized
without changing the learned tensors. The original immutable artifacts remain
in the internal experiment archive.
