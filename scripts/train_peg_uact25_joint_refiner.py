from __future__ import annotations
import argparse
import copy
from dataclasses import fields
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.optim as optim
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.tensorboard import SummaryWriter
from scripts._record_meta import record_meta
from scripts.cache_maniskill_uact_ta_current_actions import (
    observation_windows,
    sha256,
    state_observations,
)
from scripts.run_directory import prepare_run_directory
from scripts.training_utils import (
    atomic_json,
    atomic_torch_save,
    restore_rng,
    rng_state,
    seed_everything,
)
from src.data.maniskill_uncertain_act_dataset import make_trajectory_splits
from src.policies.maniskill_uact_fresh_flow import load_frozen_uact
from src.policies.maniskill_uact_joint_refiner import (
    PEG_JOINT_FM_METHOD,
    PEG_JOINT_UACT_STAGE_FM,
    MANISKILL_JOINT_FM_METHOD,
    MANISKILL_JOINT_UACT_STAGE_FM,
    build_differentiable_causal_sources,
    flow_matching_loss_with_source_gradient,
    normalized_joint_loss,
)
from src.policies.maniskill_uact_mars_action_flow import (
    MARSActionFlowConfig,
    UACTMARSActionFlow,
    UACT_MARS_ACTION_FLOW_METHOD,
)
from src.policies.maniskill_uncertain_act import (
    ATTACHED_VARIANCE_ACT_METHOD,
    UncertainACTOutput,
    uncertain_act_loss,
)

SCHEMA_VERSION = 1
PRETRAIN_ITERS = 25000
JOINT_ITERS = 50000
SAVE_STEPS = (10000, 25000, 50000)
TRAJECTORIES_PER_UPDATE = 4
TARGETS_PER_TRAJECTORY = 256


def method(kind: str, suite: bool = False) -> str:
    if suite:
        return MANISKILL_JOINT_FM_METHOD
    return PEG_JOINT_FM_METHOD


def pretrain_method(kind: str, suite: bool = False) -> str:
    prefix = "maniskill" if suite else "peg"
    return f"{prefix}_uact25_{'fm'}_refiner_pretrain25_diagnostic_v1"


def prepare_output(args: argparse.Namespace) -> Path:
    if not args.resume_checkpoint:
        return prepare_run_directory(args.out)
    output = Path(args.out).expanduser().resolve()
    checkpoint = Path(args.resume_checkpoint).expanduser().resolve()
    if checkpoint.parent != output / "checkpoints" or not checkpoint.is_file():
        raise RuntimeError("resume checkpoint must belong to the requested run")
    return output


def validate_uact25(provenance: dict, *, suite: bool = False) -> None:
    expected_stage = (
        "maniskill_paper500_joint_uact_pretrain_25k"
        if suite
        else "peg_joint_uact_pretrain_25k"
    )
    if not (
        provenance.get("method")
        == "clean_uncertain_act_v2_attached_variance_diagnostic"
        and provenance.get("stage") == expected_stage
        and (provenance.get("iteration") == PRETRAIN_ITERS)
        and (
            provenance.get("model_config", {}).get("variance_detach_features") is False
        )
    ):
        raise RuntimeError("joint experiment requires the registered attached U-ACT25k")


def validate_cache(cache: dict, provenance: dict) -> None:
    expected_source = {
        "kind": "ta_current_plus_current_query_future_proposal",
        "chunk_horizon": 8,
        "uses_future_observations": False,
        "gaussian_source": False,
        "history_actions": False,
    }
    if (
        cache.get("schema_version") != 1
        or cache.get("method") != "uact_official_ta_causal_action_chunk_cache_v1"
        or cache.get("source_contract") != expected_source
        or (cache.get("diagnostic_peg_joint_uact_25k") is not True)
        or (cache.get("uact_source", {}).get("sha256") != provenance["sha256"])
    ):
        raise RuntimeError("joint experiment cache identity is invalid")


def build_refiner(kind: str, uact, cache: dict):
    config = MARSActionFlowConfig(
        obs_dim=uact.config.obs_dim,
        act_dim=uact.config.act_dim,
        arm_dim=uact.config.act_dim - 1,
    )
    model = UACTMARSActionFlow(
        config,
        obs_mean=uact.obs_mean,
        obs_std=uact.obs_std,
        action_mean=uact.action_mean,
        action_std=uact.action_std,
    )
    statistics = None
    return (model, config, statistics)


def cache_batch(partition: dict, indices: torch.Tensor, device: torch.device) -> dict:
    keys = ("observations", "source_chunks", "expert_chunks", "valid_mask")
    device_indices = indices.to(device, non_blocking=True)
    return {key: partition[key].index_select(0, device_indices) for key in keys}


def gpu_cache_partition(partition: dict, device: torch.device) -> dict:
    keys = ("observations", "source_chunks", "expert_chunks", "valid_mask")
    return {key: partition[key].to(device) for key in keys}


def frozen_refiner_loss(kind: str, refiner, batch: dict) -> dict:
    return refiner.flow_matching_loss(
        batch["observations"],
        batch["source_chunks"],
        batch["expert_chunks"],
        batch["valid_mask"],
    )


def train_refiner_pretrain(args, output, device, uact, provenance, cache, cache_hash):
    total_iters = args.smoke_iters if args.smoke else PRETRAIN_ITERS
    refiner, config, residual_statistics = build_refiner("fm", uact, cache)
    refiner.to(device).train()
    ema_refiner = copy.deepcopy(refiner).eval()
    optimizer = optim.AdamW(
        refiner.parameters(), lr=0.0001, betas=(0.95, 0.999), weight_decay=1e-06
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(500, total_iters),
        num_training_steps=total_iters,
    )
    ema = EMAModel(refiner.parameters(), power=0.75)
    sampler = torch.Generator().manual_seed(args.seed + 910003)
    identity = {
        "method": pretrain_method("fm", args.maniskill_joint_suite),
        "kind": "fm",
        "task_slug": args.task_slug,
        "uact_sha256": provenance["sha256"],
        "cache_sha256": cache_hash,
        "model_config": config.to_dict(),
        "batch_size": 1024,
        "learning_rate": 0.0001,
        "weight_decay": 1e-06,
        "warmup_steps": min(500, total_iters),
        "total_iters": total_iters,
        "smoke": bool(args.smoke),
        "seed": args.seed,
    }
    start = 0
    if args.resume_checkpoint:
        state = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if state.get("identity") != identity or state.get("training_complete"):
            raise RuntimeError("refiner25 resume identity is invalid")
        refiner.load_state_dict(state["agent"], strict=True)
        ema_refiner.load_state_dict(state["ema_agent"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema.load_state_dict(state["ema"])
        ema.to(device=device)
        sampler.set_state(state["sampler_state"])
        restore_rng(state["rng_state"], device)
        start = int(state["iteration"])
    train = gpu_cache_partition(cache["partitions"]["train"], device)
    writer = SummaryWriter(output)
    last = None
    for iteration in range(start + 1, total_iters + 1):
        indices = torch.randint(
            train["observations"].shape[0], (1024,), generator=sampler
        )
        batch = cache_batch(train, indices, device)
        parts = frozen_refiner_loss("fm", refiner, batch)
        optimizer.zero_grad(set_to_none=True)
        parts["loss"].backward()
        grad = torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        if not torch.isfinite(parts["loss"]) or not torch.isfinite(grad):
            raise RuntimeError("refiner25 encountered a non-finite update")
        optimizer.step()
        scheduler.step()
        ema.step(refiner.parameters())
        last = parts
        if iteration == 1 or iteration % args.log_every == 0:
            row = {
                "iteration": iteration,
                "gradient_norm": float(grad),
                "lr": optimizer.param_groups[0]["lr"],
                **{key: float(value) for (key, value) in parts.items()},
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            for key, value in row.items():
                if key != "iteration":
                    writer.add_scalar(f"train/{key}", value, iteration)
        if iteration in ({10000, 25000, total_iters} if args.smoke else {10000, 25000}):
            ema.copy_to(ema_refiner.parameters())
            complete = iteration == total_iters
            payload = {
                "schema_version": SCHEMA_VERSION,
                "method": pretrain_method("fm", args.maniskill_joint_suite),
                "stage": (
                    f"engineering_smoke_peg_{'fm'}_refiner"
                    if args.smoke
                    else (
                        f"maniskill_{args.task_slug}_{'fm'}_refiner_pretrain_25k"
                        if args.maniskill_joint_suite
                        else f"peg_{'fm'}_refiner_pretrain_25k"
                    )
                ),
                "training_complete": complete,
                "iteration": iteration,
                "identity": identity,
                "model_config": config.to_dict(),
                "agent": refiner.state_dict(),
                "ema_agent": ema_refiner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict(),
                "sampler_state": sampler.get_state(),
                "rng_state": rng_state(device),
                "uact_source": provenance,
                "cache": {
                    "path": str(Path(args.cache).resolve()),
                    "sha256": cache_hash,
                },
                "residual_statistics": residual_statistics,
            }
            atomic_torch_save(
                payload, output / "checkpoints" / f"step_{iteration:06d}.pt"
            )
    if last is None:
        raise RuntimeError("refiner25 completed no updates")
    checkpoint = output / "checkpoints" / f"step_{total_iters:06d}.pt"
    result = {
        "schema_version": SCHEMA_VERSION,
        "method": pretrain_method("fm", args.maniskill_joint_suite),
        "training_complete": True,
        "iteration": total_iters,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "uact_sha256": provenance["sha256"],
        "cache_sha256": cache_hash,
        "last_train": {key: float(value) for (key, value) in last.items()},
    }
    atomic_json(result, output / "results.json")
    writer.close()
    print(json.dumps(result, sort_keys=True), flush=True)


class TrajectoryCorpus:

    def __init__(
        self,
        demo_path: Path,
        indices: list[int],
        *,
        control_mode: str,
        device: torch.device,
    ):
        self.rows = []
        with h5py.File(demo_path, "r") as handle:
            available = {
                int(key.rsplit("_", 1)[1]): key
                for key in handle
                if key.startswith("traj_")
            }
            for trajectory_index in indices:
                group = handle[available[trajectory_index]]
                observations = state_observations(
                    np.asarray(group["obs"], dtype=np.float32)
                )
                actions = torch.from_numpy(
                    np.asarray(group["actions"], dtype=np.float32).copy()
                )
                windows = torch.from_numpy(observation_windows(observations, 2))
                targets30, valid30 = padded_targets(actions, 30, control_mode)
                targets8, valid8 = padded_targets(actions, 8, control_mode)
                self.rows.append(
                    {
                        "trajectory_index": trajectory_index,
                        "observations": windows.to(device),
                        "targets30": targets30.to(device),
                        "valid30": valid30.to(device),
                        "targets8": targets8.to(device),
                        "valid8": valid8.to(device),
                    }
                )
        self.lengths = torch.tensor(
            [row["observations"].shape[0] for row in self.rows], dtype=torch.float64
        )
        self.probabilities = self.lengths / self.lengths.sum()

    def sample(self, generator: torch.Generator, *, trajectories: int, targets: int):
        selected = torch.multinomial(
            self.probabilities, trajectories, replacement=True, generator=generator
        )
        for local_index in selected.tolist():
            row = self.rows[local_index]
            timesteps = torch.randint(
                row["observations"].shape[0], (targets,), generator=generator
            )
            yield (row, timesteps)


def padded_targets(actions: torch.Tensor, horizon: int, control_mode: str):
    count, action_dim = actions.shape
    indices = torch.arange(count)[:, None] + torch.arange(horizon)[None]
    valid = indices < count
    targets = actions[indices.clamp(max=count - 1)].clone()
    if "delta_pos" in control_mode:
        pad = torch.zeros(action_dim, dtype=actions.dtype)
        pad[-1] = actions[-1, -1]
        targets = torch.where(valid[:, :, None], targets, pad[None, None])
    return (targets, valid)


def select_output(
    output: UncertainACTOutput, indices: torch.Tensor
) -> UncertainACTOutput:
    return UncertainACTOutput(
        **{
            field.name: getattr(output, field.name).index_select(0, indices)
            for field in fields(UncertainACTOutput)
        }
    )


def row_select(row: dict, key: str, indices: torch.Tensor, device: torch.device):
    return row[key].index_select(0, indices.to(device, non_blocking=True))


def joint_refiner_loss(kind: str, refiner, observations, source, expert, valid):
    return flow_matching_loss_with_source_gradient(
        refiner, observations, source, expert, valid
    )


@torch.no_grad()
def estimate_scales(kind, uact, refiner, corpus, device, seed):
    generator = torch.Generator().manual_seed(seed + 930003)
    act_total = refiner_total = 0.0
    count = 0
    uact.train()
    refiner.train()
    for row, indices in corpus.sample(generator, trajectories=4, targets=250):
        observations_all = row["observations"]
        output = uact(observations_all)
        source_all = build_differentiable_causal_sources(output.mean)
        device_indices = indices.to(device)
        selected_output = select_output(output, device_indices)
        act_parts = uncertain_act_loss(
            selected_output,
            row_select(row, "targets30", indices, device),
            action_mean=uact.action_mean,
            action_std=uact.action_std,
            valid_mask=row_select(row, "valid30", indices, device),
            variance_weight=0.1,
        )
        ref_parts = joint_refiner_loss(
            "fm",
            refiner,
            observations_all.index_select(0, device_indices),
            source_all.index_select(0, device_indices),
            row_select(row, "targets8", indices, device),
            row_select(row, "valid8", indices, device),
        )
        act_total += abs(float(act_parts.loss)) * indices.numel()
        refiner_total += abs(float(ref_parts["loss"])) * indices.numel()
        count += indices.numel()
    scales = {
        "act": act_total / count,
        "refiner": refiner_total / count,
        "samples": count,
    }
    if min(scales["act"], scales["refiner"]) <= 0 or not all(
        (np.isfinite(v) for v in scales.values())
    ):
        raise RuntimeError("joint loss scale calibration failed")
    return scales


def train_joint(args, output, device, uact, provenance, cache, cache_hash):
    total_iters = args.smoke_iters if args.smoke else JOINT_ITERS
    pretrain_path = Path(args.refiner_checkpoint).expanduser().resolve()
    state = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    if not (
        state.get("method") == pretrain_method("fm", args.maniskill_joint_suite)
        and state.get("training_complete") is True
        and (
            state.get("iteration")
            == (args.smoke_iters if args.smoke else PRETRAIN_ITERS)
        )
        and (state.get("uact_source", {}).get("sha256") == provenance["sha256"])
        and (state.get("cache", {}).get("sha256") == cache_hash)
    ):
        raise RuntimeError("joint50 refiner25 checkpoint identity is invalid")
    refiner, config, residual_statistics = build_refiner("fm", uact, cache)
    refiner.load_state_dict(state["ema_agent"], strict=True)
    uact.train()
    refiner.to(device).train()
    for parameter in uact.parameters():
        parameter.requires_grad_(True)
    ema_uact = copy.deepcopy(uact).eval()
    ema_refiner = copy.deepcopy(refiner).eval()
    act_optimizer = optim.AdamW(uact.parameters(), lr=3e-05, weight_decay=0.0001)
    refiner_optimizer = optim.AdamW(
        refiner.parameters(), lr=5e-05, betas=(0.95, 0.999), weight_decay=1e-06
    )
    act_scheduler = get_scheduler(
        "cosine",
        optimizer=act_optimizer,
        num_warmup_steps=min(500, total_iters),
        num_training_steps=total_iters,
    )
    refiner_scheduler = get_scheduler(
        "cosine",
        optimizer=refiner_optimizer,
        num_warmup_steps=min(500, total_iters),
        num_training_steps=total_iters,
    )
    act_ema = EMAModel(uact.parameters(), power=0.75)
    refiner_ema = EMAModel(refiner.parameters(), power=0.75)
    splits = make_trajectory_splits(
        500, seed=31001, train_fraction=0.8, calibration_fraction=0.1
    )
    corpus = TrajectoryCorpus(
        Path(args.demo_path).expanduser().resolve(),
        list(splits.train),
        control_mode=args.control_mode,
        device=device,
    )
    scales = estimate_scales("fm", uact, refiner, corpus, device, args.seed)
    sampler = torch.Generator().manual_seed(args.seed + 940003)
    identity = {
        "method": method("fm", args.maniskill_joint_suite),
        "kind": "fm",
        "task_slug": args.task_slug,
        "uact25_sha256": provenance["sha256"],
        "refiner25_sha256": sha256(pretrain_path),
        "cache_sha256": cache_hash,
        "dataset_sha256": sha256(args.demo_path),
        "split_seed": 31001,
        "aggregation_k": 0.01,
        "trajectories_per_update": TRAJECTORIES_PER_UPDATE,
        "targets_per_trajectory": TARGETS_PER_TRAJECTORY,
        "effective_batch_size": 1024,
        "act_learning_rate": 3e-05,
        "refiner_learning_rate": 5e-05,
        "warmup_steps": min(500, total_iters),
        "total_iters": total_iters,
        "act_loss_weight": args.joint_act_loss_weight,
        "loss_scales": scales,
        "smoke": bool(args.smoke),
        "seed": args.seed,
    }
    start = 0
    if args.resume_checkpoint:
        resume = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if resume.get("identity") != identity or resume.get("training_complete"):
            raise RuntimeError("joint50 resume identity is invalid")
        uact.load_state_dict(resume["uact_agent"], strict=True)
        ema_uact.load_state_dict(resume["uact_ema_agent"], strict=True)
        refiner.load_state_dict(resume["refiner_agent"], strict=True)
        ema_refiner.load_state_dict(resume["refiner_ema_agent"], strict=True)
        act_optimizer.load_state_dict(resume["act_optimizer"])
        refiner_optimizer.load_state_dict(resume["refiner_optimizer"])
        act_scheduler.load_state_dict(resume["act_scheduler"])
        refiner_scheduler.load_state_dict(resume["refiner_scheduler"])
        act_ema.load_state_dict(resume["act_ema"])
        act_ema.to(device=device)
        refiner_ema.load_state_dict(resume["refiner_ema"])
        refiner_ema.to(device=device)
        sampler.set_state(resume["sampler_state"])
        restore_rng(resume["rng_state"], device)
        start = int(resume["iteration"])
    writer = SummaryWriter(output)
    last = None
    for iteration in range(start + 1, total_iters + 1):
        act_optimizer.zero_grad(set_to_none=True)
        refiner_optimizer.zero_grad(set_to_none=True)
        act_sum = ref_sum = 0.0
        for row, indices in corpus.sample(
            sampler,
            trajectories=TRAJECTORIES_PER_UPDATE,
            targets=TARGETS_PER_TRAJECTORY,
        ):
            observations_all = row["observations"]
            output_all = uact(observations_all)
            source_all = build_differentiable_causal_sources(output_all.mean)
            device_indices = indices.to(device)
            selected_output = select_output(output_all, device_indices)
            act_parts = uncertain_act_loss(
                selected_output,
                row_select(row, "targets30", indices, device),
                action_mean=uact.action_mean,
                action_std=uact.action_std,
                valid_mask=row_select(row, "valid30", indices, device),
                variance_weight=0.1,
            )
            ref_parts = joint_refiner_loss(
                "fm",
                refiner,
                observations_all.index_select(0, device_indices),
                source_all.index_select(0, device_indices),
                row_select(row, "targets8", indices, device),
                row_select(row, "valid8", indices, device),
            )
            loss = (
                normalized_joint_loss(
                    ref_parts["loss"],
                    act_parts.loss,
                    refiner_scale=scales["refiner"],
                    act_scale=scales["act"],
                    act_weight=args.joint_act_loss_weight,
                )
                / TRAJECTORIES_PER_UPDATE
            )
            loss.backward()
            act_sum += float(act_parts.loss) / TRAJECTORIES_PER_UPDATE
            ref_sum += float(ref_parts["loss"]) / TRAJECTORIES_PER_UPDATE
        act_grad = torch.nn.utils.clip_grad_norm_(uact.parameters(), 0.5)
        ref_grad = torch.nn.utils.clip_grad_norm_(refiner.parameters(), 0.5)
        if not all((torch.isfinite(value) for value in (act_grad, ref_grad))):
            raise RuntimeError("joint50 encountered a non-finite gradient")
        if float(act_grad) <= 0 or float(ref_grad) <= 0:
            raise RuntimeError("joint50 gradient routing is inactive")
        act_optimizer.step()
        refiner_optimizer.step()
        act_scheduler.step()
        refiner_scheduler.step()
        act_ema.step(uact.parameters())
        refiner_ema.step(refiner.parameters())
        last = {
            "act_loss": act_sum,
            "refiner_loss": ref_sum,
            "act_gradient_norm": float(act_grad),
            "refiner_gradient_norm": float(ref_grad),
        }
        if iteration == 1 or iteration % args.log_every == 0:
            print(
                json.dumps({"iteration": iteration, **last}, sort_keys=True), flush=True
            )
            for key, value in last.items():
                writer.add_scalar(f"train/{key}", value, iteration)
        if iteration in (
            set(SAVE_STEPS) | {total_iters} if args.smoke else set(SAVE_STEPS)
        ):
            act_ema.copy_to(ema_uact.parameters())
            refiner_ema.copy_to(ema_refiner.parameters())
            complete = iteration == total_iters
            payload = {
                "schema_version": SCHEMA_VERSION,
                "method": method("fm", args.maniskill_joint_suite),
                "stage": (
                    f"engineering_smoke_peg_uact_{'fm'}_joint"
                    if args.smoke
                    else (
                        f"maniskill_{args.task_slug}_uact25_{'fm'}25_joint50"
                        if args.maniskill_joint_suite
                        else f"peg_uact25_{'fm'}25_joint50"
                    )
                ),
                "training_complete": complete,
                "iteration": iteration,
                "identity": identity,
                "uact_model_config": uact.config.to_dict(),
                "refiner_model_config": config.to_dict(),
                "uact_agent": uact.state_dict(),
                "uact_ema_agent": ema_uact.state_dict(),
                "refiner_agent": refiner.state_dict(),
                "refiner_ema_agent": ema_refiner.state_dict(),
                "act_optimizer": act_optimizer.state_dict(),
                "refiner_optimizer": refiner_optimizer.state_dict(),
                "act_scheduler": act_scheduler.state_dict(),
                "refiner_scheduler": refiner_scheduler.state_dict(),
                "act_ema": act_ema.state_dict(),
                "refiner_ema": refiner_ema.state_dict(),
                "sampler_state": sampler.get_state(),
                "rng_state": rng_state(device),
                "uact25_source": provenance,
                "refiner25_source": {
                    "path": str(pretrain_path),
                    "sha256": sha256(pretrain_path),
                },
                "cache": {
                    "path": str(Path(args.cache).resolve()),
                    "sha256": cache_hash,
                },
                "residual_statistics": residual_statistics,
            }
            atomic_torch_save(
                payload, output / "checkpoints" / f"step_{iteration:06d}.pt"
            )
    if last is None:
        raise RuntimeError("joint50 completed no updates")
    checkpoint = output / "checkpoints" / f"step_{total_iters:06d}.pt"
    deployment = output / "deployment"
    uact_deployment = deployment / "uact_ema.pt"
    refiner_deployment = deployment / f"{'fm'}_ema.pt"
    uact_stage = (
        MANISKILL_JOINT_UACT_STAGE_FM
        if args.maniskill_joint_suite
        else PEG_JOINT_UACT_STAGE_FM
    )
    uact_payload = {
        "schema_version": 3,
        "method": ATTACHED_VARIANCE_ACT_METHOD,
        "stage": uact_stage,
        "training_complete": True,
        "iteration": total_iters,
        "model_config": uact.config.to_dict(),
        "agent": uact.state_dict(),
        "ema_agent": ema_uact.state_dict(),
        "joint_source": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
    }
    atomic_torch_save(uact_payload, uact_deployment)
    uact_deployment_source = {
        "path": str(uact_deployment.resolve()),
        "sha256": sha256(uact_deployment),
        "state_key": "ema_agent",
        "method": ATTACHED_VARIANCE_ACT_METHOD,
        "stage": uact_stage,
        "iteration": total_iters,
        "model_config": uact.config.to_dict(),
    }
    refiner_payload = {
        "schema_version": 1,
        "method": UACT_MARS_ACTION_FLOW_METHOD,
        "stage": (
            f"maniskill_{args.task_slug}_uact25_fm25_joint50_deployment"
            if args.maniskill_joint_suite
            else "peg_uact25_fm25_joint50_deployment"
        ),
        "training_objective": "standard_conditional_flow_matching",
        "training_complete": True,
        "iteration": total_iters,
        "model_config": config.to_dict(),
        "agent": refiner.state_dict(),
        "ema_agent": ema_refiner.state_dict(),
        "uact_source": uact_deployment_source,
        "joint_source": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
    }
    atomic_torch_save(refiner_payload, refiner_deployment)
    result = {
        "schema_version": SCHEMA_VERSION,
        "method": method("fm", args.maniskill_joint_suite),
        "training_complete": True,
        "iteration": total_iters,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "identity": identity,
        "last_train": last,
        "deployment": {
            "uact_checkpoint": str(uact_deployment.resolve()),
            "uact_sha256": sha256(uact_deployment),
            "refiner_checkpoint": str(refiner_deployment.resolve()),
            "refiner_sha256": sha256(refiner_deployment),
        },
    }
    atomic_json(result, output / "results.json")
    writer.close()
    print(json.dumps(result, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pretrain", "joint"), required=True)
    parser.add_argument("--kind", choices=("fm",), required=True)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--uact-checkpoint", required=True)
    parser.add_argument("--refiner-checkpoint", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-iters", type=int, default=2)
    parser.add_argument(
        "--joint-act-loss-weight",
        type=float,
        default=0.5,
        help="ACT anchor coefficient after independent loss-scale normalization",
    )
    parser.add_argument("--maniskill-joint-suite", action="store_true")
    parser.add_argument("--task-slug", default="peginsertionside")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_iters <= 0:
        raise ValueError("smoke-iters must be positive")
    if (
        not np.isfinite(args.joint_act_loss_weight)
        or not 0 < args.joint_act_loss_weight <= 1
    ):
        raise ValueError("joint-act-loss-weight must be finite and in (0,1]")
    registered_slugs = {
        "pickcube",
        "pushcube",
        "stackcube",
        "peginsertionside",
        "plugcharger",
        "pullcubetool",
        "pusht",
        "stackpyramid",
    }
    if args.maniskill_joint_suite and args.task_slug not in registered_slugs:
        raise ValueError("maniskill joint suite requires a registered task slug")
    if args.mode == "joint" and (not args.refiner_checkpoint):
        raise ValueError("joint mode requires --refiner-checkpoint")
    if args.mode == "pretrain" and args.refiner_checkpoint:
        raise ValueError("pretrain mode does not accept --refiner-checkpoint")
    demo_path = Path(args.demo_path).expanduser().resolve()
    cache_path = Path(args.cache).expanduser().resolve()
    if not demo_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError("ManiSkill dataset or U-ACT25 cache is missing")
    if args.device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    output = prepare_output(args)
    record_meta(
        output,
        {"entrypoint": "scripts/train_peg_uact25_joint_refiner.py", "args": vars(args)},
    )
    seed_everything(args.seed)
    uact, provenance = load_frozen_uact(args.uact_checkpoint, device=device)
    validate_uact25(provenance, suite=args.maniskill_joint_suite)
    cache_hash = sha256(cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    validate_cache(cache, provenance)
    if args.mode == "pretrain":
        train_refiner_pretrain(
            args, output, device, uact, provenance, cache, cache_hash
        )
    else:
        train_joint(args, output, device, uact, provenance, cache, cache_hash)


if __name__ == "__main__":
    main()
