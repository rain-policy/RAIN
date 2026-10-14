# Evaluate RAIN in ManiSkill

Install the environment and [pretrained models](models.md) first. Run commands
from the repository root. The default policy backend is FP32 CUDA Graph, with
eight actions executed for every refinement choice.

## One-task smoke run

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --task pickcube --episodes 1 --output outputs/pickcube-smoke
```

## Evaluate all tasks

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --task all --episodes 100 --output outputs/evaluation-100
```

`manifests/tasks.json` defines the environment, control mode, initial seed and
maximum episode length for each task. Seeds increase by one for successive
episodes. The public wrapper deliberately uses these recorded task settings.

## Useful options

| Option | Effect |
|---|---|
| `--verify-only` | Verify the selected model files and exit |
| `--dry-run` | Verify models and print evaluation commands without simulation |
| `--episodes N` | Run N complete episodes for each selected task |
| `--video` | Save per-episode video when rendering is configured |
| `--backend eager` | Use eager execution for runtime diagnosis |
| `--render-backend DEVICE` | Select the configured Vulkan device |

```bash
python run.py --task all --dry-run --output outputs/preview
CUDA_VISIBLE_DEVICES=0 python run.py \
  --task peginsertionside --episodes 1 --video --output outputs/peg-video
```

Choose a new output directory for each actual evaluation. A dry run can use
an existing directory because it does not write results.

## Read the outputs

```text
outputs/evaluation-100/
├── summary.json
├── pickcube.route.json
├── pickcube/
│   ├── results.json
│   └── videos/             # When --video is enabled
└── ...                     # Equivalent outputs for the other tasks
```

The top-level `summary.json` records successes and episode counts. Each task's
`results.json` contains:

| Field | Meaning |
|---|---|
| `actual_episodes` | Number of completed evaluation episodes |
| `success_once_count` | Episodes reaching success at any point |
| `success_at_end_count` | Episodes successful at their final step |
| `c_block_histogram` | Counts of decisions choosing 0, 1, 2 and 3 refinements |
| `mean_c_per_block` | Average FM evaluations per decision block |
| `policy_latency` | Summary of isolated policy timing samples |
| `boundary_latency` | Summary of measured decision-boundary timing samples |
| `episodes_detail` | Per-episode outcomes, return and refinement diagnostics |

The reference task success rates use `success_once_count / actual_episodes`.
`policy_latency` summarizes timing samples from every executed action,
including cached actions between decision boundaries. `boundary_latency`
uses only decision-boundary samples. Environment execution is outside the
policy timer; input transfer inclusion is recorded by the runtime settings.
Compare timing only under the same device, precision and backend configuration.

The archived eight-task result is 533/800 successes. The release's separate
one-episode integration checks are documented in [validation](validation.md).
