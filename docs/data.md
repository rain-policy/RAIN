# Demonstration data

Pretrained evaluation only needs the released model bundle. Training needs
task-matched state observations and actions stored in HDF5.

## Dataset contract

Each task uses 500 successful trajectories with a fixed split seed of 31001:

| Partition | Trajectories | Use |
|---|---:|---|
| Training | 400 | Proposal, refiner and error-ensemble fitting |
| Calibration | 50 | Refinement cost and transition-parameter selection |
| Held out | 50 | Held-out checks |

[datasets.json](../manifests/datasets.json) records the processed HDF5 SHA-256
and exact trajectory IDs for all eight tasks. The training wrapper uses the
matching split fractions and seed.

## HDF5 format

```text
demo.h5
├── traj_0/
│   ├── obs       # [T + 1, observation_dim] or [T + 1, 1, observation_dim]
│   └── actions   # [T, action_dim]
├── traj_1/
└── ...
```

Trajectory keys must use the `traj_<integer>` convention. Each observation
sequence has one extra frame relative to its action sequence. The loader
converts observations/actions to float32 and creates two-frame observation
windows. Normalization statistics are fitted from training trajectories.

PickCube, PushCube and StackCube use `pd_ee_delta_pos`. The other tasks use
`pd_ee_delta_pose`. Keep the demonstration's controller and action dimensions
consistent with the task configuration.

## Obtain and replay ManiSkill demonstrations

The upstream [ManiSkill demonstrations](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations)
provide task trajectories. After installing the reference simulator, its tools
can download and replay a task into state observations:

```bash
python -m mani_skill.utils.download_demo PickCube-v1 --output_dir data/raw
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path /path/to/downloaded/trajectory.h5 \
  --obs-mode state --target-control-mode pd_ee_delta_pos \
  --sim-backend physx_cpu --save-traj --count 500
```

Use the actual downloaded HDF5 path and the task's control mode. The replay
tool writes a derived file next to the input; preserve the original trajectories
and inspect replay outcomes before choosing the successful training set.

These commands prepare a new training input. The original processed `paper500`
files are identified by their manifest hashes and are not included in this
release. PushT's original data used RL trajectories, state recovery and native
action clipping; reproducing that input also requires its original preprocessing.

## Start training

```bash
python train.py --task pickcube --demo data/pickcube.h5 --output runs/pickcube
```

Confirm the generated commands, then add `--run` to execute them. See
[training](training.md) for stage dependencies and [codebase](codebase.md) for
the dataset implementation.
