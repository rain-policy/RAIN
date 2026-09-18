from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.optim as optim
from diffusers.training_utils import EMAModel
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from scripts._record_meta import record_meta
from scripts.run_directory import prepare_run_directory
from src.data.maniskill_uncertain_act_dataset import (
    UncertainACTTrajectoryDataset,
    make_trajectory_splits,
)
from src.policies.maniskill_uncertain_act import (
    ATTACHED_VARIANCE_ACT_METHOD,
    UNCERTAIN_ACT_METHOD,
    CleanUncertainACT,
    UncertainACTConfig,
    uncertain_act_loss,
)

SCHEMA_VERSION = 3
DEFAULT_SAVE_STEPS = (10000, 30000, 60000, 100000)


def attached_variance_50k_mode(args: argparse.Namespace) -> bool:
    return bool(False)


def joint_pretrain_25k_mode(args: argparse.Namespace) -> bool:
    return bool(args.peg_joint_pretrain_25k or args.joint_pretrain_25k_suite)


def method_name(args: argparse.Namespace) -> str:
    return (
        ATTACHED_VARIANCE_ACT_METHOD
        if attached_variance_50k_mode(args) or joint_pretrain_25k_mode(args)
        else UNCERTAIN_ACT_METHOD
    )


def stage_name(args: argparse.Namespace) -> str:
    if args.peg_joint_pretrain_25k:
        return "peg_joint_uact_pretrain_25k"
    if args.joint_pretrain_25k_suite:
        return "maniskill_paper500_joint_uact_pretrain_25k"
    return "formal_variance_isolated_uact"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def count_trajectories(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return sum((key.startswith("traj_") for key in handle))


def tensor_stats(stats: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32)
        for (key, value) in stats.items()
    }


def rng_state(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def restore_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def config_identity(args: argparse.Namespace) -> dict:
    return {
        "method": method_name(args),
        "demo_path": str(Path(args.demo_path).expanduser().resolve()),
        "num_demos": args.num_demos,
        "control_mode": args.control_mode,
        "obs_horizon": args.obs_horizon,
        "num_queries": args.num_queries,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "min_logvar": args.min_logvar,
        "max_logvar": args.max_logvar,
        "variance_detach_features": args.variance_detach_features,
        "continuous_mean_weight": args.continuous_mean_weight,
        "variance_loss_weight": args.variance_loss_weight,
        "variance_warmup_iters": args.variance_warmup_iters,
        "gripper_loss_weight": args.gripper_loss_weight,
        "huber_delta": args.huber_delta,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_fraction": args.train_fraction,
        "calibration_fraction": args.calibration_fraction,
        "total_iters": args.total_iters,
    }


@torch.no_grad()
def evaluate_subset(
    model: CleanUncertainACT,
    dataset: UncertainACTTrajectoryDataset,
    *,
    batch_size: int,
    maximum_samples: int,
    device: torch.device,
    continuous_mean_weight: float,
    variance_weight: float,
    gripper_weight: float,
    huber_delta: float,
) -> dict[str, float]:
    model.eval()
    subset = Subset(dataset, range(min(len(dataset), maximum_samples)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    totals = {
        "loss": 0.0,
        "continuous_mean_huber": 0.0,
        "variance_nll": 0.0,
        "gripper_huber": 0.0,
        "mean_logvar": 0.0,
        "lower_clamp_fraction": 0.0,
        "upper_clamp_fraction": 0.0,
    }
    count = 0
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        valid = batch["action_valid"].to(device)
        output = model(observations)
        parts = uncertain_act_loss(
            output,
            actions,
            action_mean=model.action_mean,
            action_std=model.action_std,
            valid_mask=valid,
            continuous_mean_weight=continuous_mean_weight,
            variance_weight=variance_weight,
            gripper_weight=gripper_weight,
            huber_delta=huber_delta,
            min_logvar=model.config.min_logvar,
            max_logvar=model.config.max_logvar,
        )
        current = observations.shape[0]
        for key in totals:
            totals[key] += float(getattr(parts, key).item()) * current
        count += current
    model.train()
    return {key: value / count for (key, value) in totals.items()}


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: CleanUncertainACT,
    ema_model: CleanUncertainACT,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    ema: EMAModel,
    iteration: int,
    dataset_sha256: str,
    splits: dict,
    sampler_generator: torch.Generator,
    device: torch.device,
) -> dict:
    ema.copy_to(ema_model.parameters())
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method_name(args),
        "stage": stage_name(args),
        "training_complete": iteration == args.total_iters,
        "identity": config_identity(args),
        "iteration": iteration,
        "model_config": model.config.to_dict(),
        "agent": model.state_dict(),
        "ema_agent": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "dataset": {
            "path": str(Path(args.demo_path).expanduser().resolve()),
            "sha256": dataset_sha256,
            "num_demos": args.num_demos,
            "splits": splits,
        },
        "rng_state": rng_state(device),
        "sampler_generator_state": sampler_generator.get_state(),
    }


def validate_resume(state: dict, args: argparse.Namespace, dataset_sha256: str) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("uncertain ACT checkpoint schema mismatch")
    if state.get("method") != method_name(args):
        raise RuntimeError("checkpoint is not a clean uncertain ACT")
    if state.get("identity") != config_identity(args):
        raise RuntimeError("uncertain ACT resume configuration drift")
    if state.get("dataset", {}).get("sha256") != dataset_sha256:
        raise RuntimeError("uncertain ACT resume dataset hash mismatch")
    if not 0 < int(state.get("iteration", 0)) < args.total_iters:
        raise RuntimeError("uncertain ACT resume iteration is invalid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--num-demos", type=int, default=647)
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--num-queries", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-logvar", type=float, default=-8.0)
    parser.add_argument("--max-logvar", type=float, default=2.0)
    parser.add_argument(
        "--variance-detach-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--continuous-mean-weight", type=float, default=1.0)
    parser.add_argument("--variance-loss-weight", type=float, default=0.1)
    parser.add_argument("--variance-warmup-iters", type=int, default=20000)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--total-iters", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=31001)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--peg-joint-pretrain-25k",
        action="store_true",
        help="Registered Peg post-hoc source: attached-variance U-ACT trained from scratch for exactly 25k updates before refiner pretraining.",
    )
    parser.add_argument(
        "--joint-pretrain-25k-suite",
        action="store_true",
        help="Registered eight-task source: attached-variance U-ACT trained for 25k updates.",
    )
    return parser.parse_args()


def prepare_output_directory(args: argparse.Namespace) -> Path:
    if not args.resume_checkpoint:
        return prepare_run_directory(args.out)
    output = Path(args.out).expanduser().resolve()
    checkpoint = Path(args.resume_checkpoint).expanduser().resolve()
    if not output.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_file() else output)
    if checkpoint.parent != output / "checkpoints":
        raise RuntimeError("resume checkpoint must belong to the requested U-ACT run")
    return output


def main() -> None:
    args = parse_args()
    registered_modes = sum(
        (
            int(value)
            for value in (
                False,
                False,
                args.peg_joint_pretrain_25k,
                args.joint_pretrain_25k_suite,
            )
        )
    )
    if registered_modes > 1:
        raise ValueError("select exactly one attached-variance registration")
    attached_50k = attached_variance_50k_mode(args)
    joint_25k = joint_pretrain_25k_mode(args)
    if args.num_demos <= 0 or args.batch_size <= 0 or args.total_iters <= 0:
        raise ValueError("demo count, batch size, and training budget must be positive")
    if args.obs_horizon != 2 or args.num_queries != 30:
        raise ValueError(
            "clean uncertain ACT v2 requires obs_horizon=2 and num_queries=30"
        )
    if (
        min(
            args.continuous_mean_weight,
            args.variance_loss_weight,
            args.gripper_loss_weight,
        )
        < 0
        or args.gradient_clip <= 0
        or args.huber_delta <= 0
    ):
        raise ValueError("loss weight and gradient clip are invalid")
    if not 0 <= args.variance_warmup_iters < args.total_iters:
        raise ValueError("variance warmup must lie in [0, total_iters)")
    if args.smoke and (attached_50k or joint_25k):
        raise ValueError(
            "smoke and attached-variance registered modes are mutually exclusive"
        )
    if attached_50k:
        if args.variance_detach_features:
            raise ValueError(
                "attached-variance diagnostic requires --no-variance-detach-features"
            )
        if args.total_iters != 50000:
            raise ValueError(
                "attached-variance diagnostic requires exactly 50k updates"
            )
        if (
            args.num_demos != 500
            or args.batch_size != 1024
            or args.learning_rate != 0.0001
            or (args.weight_decay != 0.0001)
            or (args.variance_warmup_iters != 20000)
            or (args.variance_loss_weight != 0.1)
        ):
            raise ValueError("attached-variance diagnostic contract drift")
    if joint_25k:
        if args.variance_detach_features:
            raise ValueError("joint pretrain requires --no-variance-detach-features")
        if (
            args.total_iters != 25000
            or args.num_demos != 500
            or args.batch_size != 1024
            or (args.learning_rate != 0.0001)
            or (args.weight_decay != 0.0001)
            or (args.variance_warmup_iters != 20000)
            or (args.variance_loss_weight != 0.1)
        ):
            raise ValueError("joint U-ACT25k pretrain contract drift")
    if (
        not args.smoke
        and (not attached_50k)
        and (not joint_25k)
        and (args.variance_warmup_iters != 20000)
    ):
        raise ValueError("formal clean uncertain ACT v2 requires 20k mean-only updates")
    if (
        not args.smoke
        and (not attached_50k)
        and (not joint_25k)
        and (args.total_iters != 100000)
    ):
        raise ValueError("formal clean uncertain ACT v2 requires exactly 100k updates")
    if (
        not args.smoke
        and (not attached_50k)
        and (not joint_25k)
        and (not args.variance_detach_features)
    ):
        raise ValueError(
            "formal clean uncertain ACT v2 requires isolated variance features"
        )
    demo_path = Path(args.demo_path).expanduser().resolve()
    if not demo_path.is_file():
        raise FileNotFoundError(demo_path)
    available = count_trajectories(demo_path)
    if available < args.num_demos:
        raise RuntimeError(
            f"expected at least {args.num_demos} demos, found {available}"
        )
    dataset_sha256 = sha256(demo_path)
    output = prepare_output_directory(args)
    record_meta(
        output,
        {"entrypoint": "scripts/train_maniskill_uncertain_act.py", "args": vars(args)},
    )
    seed_everything(args.seed)
    if args.device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    splits = make_trajectory_splits(
        args.num_demos,
        seed=args.split_seed,
        train_fraction=args.train_fraction,
        calibration_fraction=args.calibration_fraction,
    )
    split_dict = splits.to_dict()
    split_path = output / "trajectory_splits.json"
    if (
        split_path.exists()
        and json.loads(split_path.read_text(encoding="utf-8")) != split_dict
    ):
        raise RuntimeError("uncertain ACT split drift during resume")
    atomic_json_save(split_dict, split_path)
    train_dataset = UncertainACTTrajectoryDataset(
        demo_path,
        splits.train,
        obs_horizon=args.obs_horizon,
        num_queries=args.num_queries,
        control_mode=args.control_mode,
    )
    calibration_dataset = UncertainACTTrajectoryDataset(
        demo_path,
        splits.calibration,
        obs_horizon=args.obs_horizon,
        num_queries=args.num_queries,
        control_mode=args.control_mode,
    )
    stats = tensor_stats(train_dataset.norm_stats())
    model_config = UncertainACTConfig(
        obs_dim=train_dataset.obs_dim,
        act_dim=train_dataset.act_dim,
        obs_horizon=args.obs_horizon,
        num_queries=args.num_queries,
        d_model=args.d_model,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        min_logvar=args.min_logvar,
        max_logvar=args.max_logvar,
        variance_detach_features=args.variance_detach_features,
    )
    model = CleanUncertainACT(
        model_config,
        obs_mean=stats["obs_mean"],
        obs_std=stats["obs_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, int(2 * args.total_iters / 3)), gamma=0.1
    )
    ema = EMAModel(model.parameters(), power=0.75)
    sampler_generator = torch.Generator().manual_seed(args.seed + 100003)
    start_iteration = 0
    if args.resume_checkpoint:
        state = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        validate_resume(state, args, dataset_sha256)
        model.load_state_dict(state["agent"], strict=True)
        ema_model.load_state_dict(state["ema_agent"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema.load_state_dict(state["ema"])
        ema.to(device=device)
        restore_rng_state(state["rng_state"], device)
        sampler_generator.set_state(state["sampler_generator_state"])
        start_iteration = int(state["iteration"])
    remaining = args.total_iters - start_iteration
    sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=remaining * args.batch_size,
        generator=sampler_generator,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=BatchSampler(sampler, args.batch_size, drop_last=True),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    writer = SummaryWriter(output)
    writer.add_text(
        "contract",
        json.dumps(
            {
                "identity": config_identity(args),
                "dataset_sha256": dataset_sha256,
                "model_config": model_config.to_dict(),
                "splits": {key: len(value) for (key, value) in split_dict.items()},
            },
            indent=2,
        ),
    )
    save_steps = set(DEFAULT_SAVE_STEPS)
    if args.smoke:
        save_steps.add(args.total_iters)
    if attached_50k:
        save_steps.update((10000, 30000, 50000))
    if joint_25k:
        save_steps.update((10000, 25000))
    last_parts = None
    model.train()
    for offset, batch in enumerate(loader, start=1):
        iteration = start_iteration + offset
        observations = batch["observations"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        valid = batch["action_valid"].to(device, non_blocking=True)
        output_prediction = model(observations)
        active_variance_weight = (
            0.0
            if iteration <= args.variance_warmup_iters
            else args.variance_loss_weight
        )
        parts = uncertain_act_loss(
            output_prediction,
            actions,
            action_mean=model.action_mean,
            action_std=model.action_std,
            valid_mask=valid,
            continuous_mean_weight=args.continuous_mean_weight,
            variance_weight=active_variance_weight,
            gripper_weight=args.gripper_loss_weight,
            huber_delta=args.huber_delta,
            min_logvar=args.min_logvar,
            max_logvar=args.max_logvar,
        )
        if not torch.isfinite(parts.loss):
            raise RuntimeError("uncertain ACT produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        parts.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("uncertain ACT produced a non-finite gradient")
        optimizer.step()
        scheduler.step()
        ema.step(model.parameters())
        last_parts = parts
        if iteration % args.log_every == 0 or iteration == args.total_iters:
            for key in (
                "loss",
                "continuous_mean_huber",
                "variance_nll",
                "gripper_huber",
                "mean_logvar",
                "lower_clamp_fraction",
                "upper_clamp_fraction",
            ):
                writer.add_scalar(f"train/{key}", getattr(parts, key).item(), iteration)
            writer.add_scalar("train/gradient_norm", gradient_norm.item(), iteration)
            writer.add_scalar(
                "train/learning_rate", scheduler.get_last_lr()[0], iteration
            )
            writer.add_scalar(
                "train/variance_loss_weight", active_variance_weight, iteration
            )
        if iteration % args.eval_every == 0 or iteration == args.total_iters:
            ema.copy_to(ema_model.parameters())
            validation = evaluate_subset(
                ema_model,
                calibration_dataset,
                batch_size=min(args.batch_size, 512),
                maximum_samples=args.eval_samples,
                device=device,
                continuous_mean_weight=args.continuous_mean_weight,
                variance_weight=active_variance_weight,
                gripper_weight=args.gripper_loss_weight,
                huber_delta=args.huber_delta,
            )
            for key, value in validation.items():
                writer.add_scalar(f"calibration/{key}", value, iteration)
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "train_loss": float(parts.loss.item()),
                        "train_continuous_mean_huber": float(
                            parts.continuous_mean_huber.item()
                        ),
                        "calibration": validation,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if iteration in save_steps or iteration == args.total_iters:
            payload = checkpoint_payload(
                args=args,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                iteration=iteration,
                dataset_sha256=dataset_sha256,
                splits=split_dict,
                sampler_generator=sampler_generator,
                device=device,
            )
            atomic_torch_save(
                payload, output / "checkpoints" / f"step_{iteration:06d}.pt"
            )
        if iteration >= args.total_iters:
            break
    writer.close()
    if last_parts is None:
        raise RuntimeError("uncertain ACT training completed no updates")
    final_checkpoint = output / "checkpoints" / f"step_{args.total_iters:06d}.pt"
    results = {
        "schema_version": SCHEMA_VERSION,
        "method": method_name(args),
        "stage": stage_name(args),
        "training_complete": True,
        "status": "complete",
        "iteration": args.total_iters,
        "checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_sha256": sha256(final_checkpoint),
        "dataset_sha256": dataset_sha256,
        "split_sizes": {key: len(value) for (key, value) in split_dict.items()},
        "last_train": {
            key: float(getattr(last_parts, key).item())
            for key in (
                "loss",
                "continuous_mean_huber",
                "variance_nll",
                "gripper_huber",
                "mean_logvar",
                "lower_clamp_fraction",
                "upper_clamp_fraction",
            )
        },
    }
    atomic_json_save(results, output / "results.json")
    print(json.dumps(results, sort_keys=True), flush=True)
