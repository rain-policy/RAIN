"""Print or execute the RAIN training stages for an existing state-demo HDF5 file."""

from __future__ import annotations
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from run import ROOT, load_tasks

STAGES = [
    "proposal",
    "cache",
    "refiner",
    "joint",
    "joint-cache",
    "score",
    "features",
    "router",
    "calibrate",
]


def commands(task, demo, output):
    config = load_tasks()[task]
    peg = task == "peginsertionside"
    proposal = output / "proposal/checkpoints/step_025000.pt"
    cache = output / "proposal-cache.pt"
    joint = output / "joint"
    uact, flow = joint / "deployment/uact_ema.pt", joint / "deployment/fm_ema.pt"
    joint_cache = output / "joint-cache.pt"
    score, features = output / "scores.pt", output / "features.pt"
    feature_head, heads = output / "feature-builder.pt", output / "router.pt"

    def cmd(module, *args):
        return [sys.executable, "-m", "scripts." + module, *map(str, args)]

    def cache_cmd(checkpoint, target, flag):
        return cmd(
            "cache_maniskill_uact_ta_action_chunks",
            "--demo-path",
            demo,
            "--uact-checkpoint",
            checkpoint,
            "--out",
            target,
            "--num-demos",
            500,
            "--split-seed",
            31001,
            "--train-fraction",
            0.8,
            "--calibration-fraction",
            0.1,
            "--aggregation-k",
            0.01,
            "--chunk-horizon",
            8,
            "--batch-size",
            512,
            flag,
            "--device",
            "cuda",
        )

    common = [
        "--kind",
        "fm",
        "--demo-path",
        demo,
        "--cache",
        cache,
        "--uact-checkpoint",
        proposal,
        "--control-mode",
        config["control_mode"],
        "--seed",
        1,
        "--joint-act-loss-weight",
        0.5,
        "--device",
        "cuda",
    ]
    if not peg:
        common += ["--maniskill-joint-suite", "--task-slug", task]
    return {
        "proposal": cmd(
            "train_maniskill_uncertain_act_gpu_resident_v2",
            "--out",
            output / "proposal",
            "--demo-path",
            demo,
            "--num-demos",
            500,
            "--control-mode",
            config["control_mode"],
            "--total-iters",
            25000,
            "--seed",
            1,
            "--device",
            "cuda",
            "--no-variance-detach-features",
            "--peg-joint-pretrain-25k" if peg else "--joint-pretrain-25k-suite",
        ),
        "cache": cache_cmd(proposal, cache, "--allow-peg-joint-uact-25k"),
        "refiner": cmd(
            "train_peg_uact25_joint_refiner",
            "--mode",
            "pretrain",
            "--out",
            output / "refiner",
            *common,
        ),
        "joint": cmd(
            "train_peg_uact25_joint_refiner",
            "--mode",
            "joint",
            "--out",
            joint,
            "--refiner-checkpoint",
            output / "refiner/checkpoints/step_025000.pt",
            *common,
        ),
        "joint-cache": cache_cmd(uact, joint_cache, "--allow-peg-joint-uact-50k"),
        "score": cmd(
            "score_maniskill_uact_mars_counterfactual_benefits",
            "--cache",
            joint_cache,
            "--flow-checkpoint",
            flow,
            "--out",
            score,
            "--batch-size",
            512,
            "--frontload-gamma",
            0.8,
            "--axis-scale-floor",
            0.001,
            "--device",
            "cuda",
            "--posthoc-diagnostic",
        ),
        "features": cmd(
            "train_peg_uact_fm_stop_head",
            "--cache",
            joint_cache,
            "--score",
            score,
            "--flow-checkpoint",
            flow,
            "--feature-cache",
            features,
            "--out",
            feature_head,
            "--batch-size",
            1024,
            "--feature-batch-size",
            512,
            "--updates",
            10000,
            "--seed",
            20260808,
            "--device",
            "cuda",
            "--posthoc-diagnostic",
        ),
        "router": cmd(
            "train_peg_uact_fm_fully_learned_route",
            "--source-head-checkpoint",
            feature_head,
            "--out",
            heads,
            "--ensemble-size",
            3,
            "--updates",
            5000,
            "--batch-size",
            512,
            "--hidden-dim",
            128,
            "--seed",
            20260809,
            "--device",
            "cuda",
            "--posthoc-diagnostic",
        ),
        "calibrate": cmd(
            "fit_maniskill_uact_fm_preonly_route",
            "--router-checkpoint",
            heads,
            "--out",
            output / "route.json",
            "--selection-objective",
            "best_c_error",
            "--beta-grid",
            "0,0.25,0.5,1,1.5,2",
            "--price-points",
            31,
            "--transition-strength-grid",
            "0,0.01,0.025,0.05,0.1,0.25,0.5,1",
            "--device",
            "cuda",
            "--posthoc-diagnostic",
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=list(load_tasks()), required=True)
    p.add_argument("--demo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["all", *STAGES], default="all")
    p.add_argument(
        "--run", action="store_true", help="Execute instead of only printing commands"
    )
    a = p.parse_args()
    if a.run and not a.demo.is_file():
        p.error("The state-demo HDF5 file does not exist")
    stages = commands(a.task, a.demo.resolve(), a.output.resolve())
    for stage in STAGES if a.stage == "all" else [a.stage]:
        print(f"[{stage}] {shlex.join(stages[stage])}", flush=True)
        if a.run:
            subprocess.run(stages[stage], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
