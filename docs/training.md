# Training RAIN

Pretrained evaluation does not need demonstration data. Training requires a
state-observation HDF5 demonstration file in the correct task/control mode.
See the [data guide](data.md) for the schema, acquisition tools and split contract.
The original experiment used 500 successful trajectories per task, with a fixed
400/50/50 training/calibration/held-out split (split seed 31001). Keep the original
trajectory IDs and preprocessing when seeking exact experiment reproduction.

PickCube, PushCube and StackCube use `pd_ee_delta_pos`; the remaining five tasks
use `pd_ee_delta_pose`. PushT used an RL demonstration source with state recovery
and native action clipping; its raw demonstrations are not interchangeable with
the processed HDF5 file. Large datasets are not included in this code release.
The dataset index, where supplied, records the processed source hash and split
IDs. A newly obtained or independently prepared dataset is a new training input,
not automatically the original experiment dataset.

Print the complete training command sequence without starting any training:

```bash
python train.py --task pickcube --demo data/pickcube.h5 --output runs/pickcube
```

Run all stages explicitly, or select one stage:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --task pickcube --demo data/pickcube.h5 --output runs/pickcube --run
CUDA_VISIBLE_DEVICES=0 python train.py --task pickcube --demo data/pickcube.h5 --output runs/pickcube --stage router --run
```

Stages are `proposal`, `cache`, `refiner`, `joint`, `joint-cache`, `score`,
`features`, `router`, and `calibrate`. They train the proposal for 25k updates,
pretrain the FM refiner for 25k, jointly train for 50k, then build the frozen-model
features, fit the risk ensemble and calibrate its cost/transition parameters.
The feature-builder head is a training intermediate; the deployed decision uses
the error ensemble before running the refiner. The `score` stage computes counterfactual
training targets, not an independent comparison experiment.

The calibration implementation retains the original offline H2/H8 sequence
selection convention; deployment uses the frozen selected coefficients with H8
for every choice, as in the released models. It does not recalibrate on evaluation
outcomes. Peg's historical model source is separately registered; the wrapper
selects its matching pretraining/joint stage identifiers.

Full training was not repeated when packaging this release. Hardware, random
number behavior and data versions can affect results. Low-level stage commands
provide their own `--help`, checkpoint/resume options and validation. The high-level
wrapper does not silently skip existing stages or overwrite a completed run.

For a newly trained model, use its `joint/deployment/uact_ema.pt`,
`joint/deployment/fm_ema.pt`, `router.pt` and `route.json` with the direct evaluator:

```bash
python -m scripts.evaluate_rain --uact-checkpoint runs/pickcube/joint/deployment/uact_ema.pt --flow-checkpoint runs/pickcube/joint/deployment/fm_ema.pt --route runs/pickcube/route.json --out outputs/retrained-pickcube --env-id PickCube-v1 --control-mode pd_ee_delta_pos --max-episode-steps 160 --seed-start 40000 --episodes 1 --device cuda --parallel-uact-streams --cache-first-query-self-attention --posthoc-diagnostic --runtime-precision fp32 --runtime-backend cudagraph --runtime-warmup-calls 30
```

The frozen release manifest is intentionally not overwritten by training.
