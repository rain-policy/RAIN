from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import h5py
import numpy as np
import torch
from scripts.cache_maniskill_uact_ta_current_actions import (
    observation_windows,
    official_ta,
    sha256,
    state_observations,
)
from src.data.maniskill_uncertain_act_dataset import make_trajectory_splits
from src.policies.maniskill_uact_fresh_flow import load_frozen_uact
from src.policies.maniskill_uact_mars_action_flow import build_causal_source_chunk

SCHEMA_VERSION = 1
METHOD = "uact_official_ta_causal_action_chunk_cache_v1"


def expert_chunks(
    actions: torch.Tensor, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.ndim != 2 or horizon <= 0:
        raise ValueError("expert actions or horizon are invalid")
    count = actions.shape[0]
    starts = torch.arange(count)[:, None]
    offsets = torch.arange(horizon)[None]
    indices = starts + offsets
    valid = indices < count
    indices = indices.clamp(max=count - 1)
    return (actions[indices], valid)


@torch.no_grad()
def process_partition(
    handle: h5py.File,
    available: dict[int, str],
    indices: list[int],
    model,
    *,
    device: torch.device,
    batch_size: int,
    aggregation_k: float,
    chunk_horizon: int,
) -> dict[str, torch.Tensor]:
    fields: dict[str, list[torch.Tensor]] = {
        "observations": [],
        "source_chunks": [],
        "expert_chunks": [],
        "valid_mask": [],
        "base_actions": [],
        "expert_actions": [],
        "uncertainty": [],
        "trajectory_index": [],
        "timestep": [],
    }
    for sequence_index, trajectory_index in enumerate(indices):
        group = handle[available[trajectory_index]]
        observations = state_observations(np.asarray(group["obs"], dtype=np.float32))
        expert_np = np.asarray(group["actions"], dtype=np.float32)
        if observations.shape[0] != expert_np.shape[0] + 1:
            raise RuntimeError("trajectory observation/action lengths disagree")
        windows_np = observation_windows(observations, model.config.obs_horizon)
        proposal_parts, uncertainty_parts = ([], [])
        for start in range(0, windows_np.shape[0], batch_size):
            batch = torch.from_numpy(windows_np[start : start + batch_size]).to(device)
            output = model(batch)
            proposal_parts.append(output.mean)
            uncertainty_parts.append(output.uncertainty)
        proposals = torch.cat(proposal_parts)
        uncertainty = torch.cat(uncertainty_parts)
        if proposals.shape[1] < chunk_horizon:
            raise RuntimeError("U-ACT proposal is shorter than the registered chunk")
        base = official_ta(proposals, aggregation_k).clamp(-1, 1)
        source = build_causal_source_chunk(base, proposals, chunk_horizon=chunk_horizon)
        expert = torch.from_numpy(expert_np.copy()).to(source.device)
        targets, valid = expert_chunks(expert, chunk_horizon)
        count = expert.shape[0]
        if source.shape != (count, chunk_horizon, expert.shape[1]):
            raise RuntimeError("cached causal source has an invalid shape")
        fields["observations"].append(torch.from_numpy(windows_np))
        fields["source_chunks"].append(source.cpu())
        fields["expert_chunks"].append(targets.cpu())
        fields["valid_mask"].append(valid.cpu())
        fields["base_actions"].append(base.cpu())
        fields["expert_actions"].append(expert.cpu())
        fields["uncertainty"].append(uncertainty.cpu())
        fields["trajectory_index"].append(
            torch.full((count,), trajectory_index, dtype=torch.long)
        )
        fields["timestep"].append(torch.arange(count, dtype=torch.long))
        if sequence_index % 25 == 0 or sequence_index + 1 == len(indices):
            print(
                json.dumps(
                    {
                        "trajectory": sequence_index + 1,
                        "partition_trajectories": len(indices),
                        "original_index": trajectory_index,
                    }
                ),
                flush=True,
            )
    return {key: torch.cat(parts) for (key, parts) in fields.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--uact-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-demos", type=int, default=647)
    parser.add_argument("--split-seed", type=int, default=31001)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--aggregation-k", type=float, default=0.01)
    parser.add_argument("--chunk-horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-intermediate-uact-checkpoint",
        action="store_true",
        help="Diagnostic only: allow a frozen 30k or 60k clean U-ACT checkpoint.",
    )
    parser.add_argument(
        "--allow-attached-variance-uact-50k",
        action="store_true",
        help="Allow a registered complete attached-variance U-ACT50k checkpoint.",
    )
    parser.add_argument(
        "--allow-peg-joint-uact-25k",
        action="store_true",
        help="Allow the registered complete Peg attached-variance U-ACT25k source.",
    )
    parser.add_argument(
        "--allow-peg-joint-uact-50k",
        action="store_true",
        help="Allow an exported complete Peg FM/DP joint50 U-ACT checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        min(args.num_demos, args.batch_size, args.chunk_horizon) <= 0
        or args.aggregation_k <= 0
    ):
        raise ValueError("cache counts and aggregation k must be positive")
    if args.chunk_horizon != 8:
        raise ValueError("MARS parity cache requires an 8-step chunk")
    demo_path = Path(args.demo_path).expanduser().resolve()
    checkpoint = Path(args.uact_checkpoint).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if not demo_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("dataset or U-ACT checkpoint is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    model, provenance = load_frozen_uact(checkpoint, device=device)
    intermediate_uact = provenance["iteration"] in (30000, 60000)
    attached_uact_50k = (
        provenance.get("method")
        == "clean_uncertain_act_v2_attached_variance_diagnostic"
        and provenance.get("stage")
        in {"peg_attached_variance_50k", "maniskill_paper500_attached_variance_50k"}
        and (provenance["iteration"] == 50000)
        and (provenance["model_config"].get("variance_detach_features") is False)
    )
    joint_uact_25k = (
        provenance.get("method")
        == "clean_uncertain_act_v2_attached_variance_diagnostic"
        and provenance.get("stage")
        in {"peg_joint_uact_pretrain_25k", "maniskill_paper500_joint_uact_pretrain_25k"}
        and (provenance["iteration"] == 25000)
        and (provenance["model_config"].get("variance_detach_features") is False)
    )
    joint_uact_50k = (
        provenance.get("method")
        == "clean_uncertain_act_v2_attached_variance_diagnostic"
        and provenance.get("stage")
        in {
            "peg_uact25_fm25_joint50_uact",
            "peg_uact25_dp25_joint50_uact",
            "maniskill_uact25_fm25_joint50_uact",
            "maniskill_uact25_dp25_joint50_uact",
        }
        and (provenance["iteration"] == 50000)
        and (provenance["model_config"].get("variance_detach_features") is False)
    )
    allowed_nonformal = (
        args.allow_intermediate_uact_checkpoint
        and intermediate_uact
        or (args.allow_attached_variance_uact_50k and attached_uact_50k)
        or (args.allow_peg_joint_uact_25k and joint_uact_25k)
        or (args.allow_peg_joint_uact_50k and joint_uact_50k)
    )
    if provenance["iteration"] != 100000 and (not allowed_nonformal):
        raise RuntimeError("paper500 MARS cache requires the frozen 100k U-ACT")
    splits = make_trajectory_splits(
        args.num_demos,
        seed=args.split_seed,
        train_fraction=args.train_fraction,
        calibration_fraction=args.calibration_fraction,
    )
    split_dict = splits.to_dict()
    partitions = {}
    with h5py.File(demo_path, "r") as handle:
        available = {
            int(key.rsplit("_", 1)[1]): key for key in handle if key.startswith("traj_")
        }
        missing = set(range(args.num_demos)) - set(available)
        if missing:
            raise RuntimeError(
                f"dataset is missing trajectories: {sorted(missing)[:5]}"
            )
        for name, indices in split_dict.items():
            partitions[name] = process_partition(
                handle,
                available,
                indices,
                model,
                device=device,
                batch_size=args.batch_size,
                aggregation_k=args.aggregation_k,
                chunk_horizon=args.chunk_horizon,
            )
    train_uncertainty = partitions["train"]["uncertainty"]
    train_log = (
        torch.log(train_uncertainty.clamp_min(1e-12)).sort().values.to(torch.float64)
    )
    arm_dim = model.config.act_dim - 1
    train_source_arm = partitions["train"]["source_chunks"][..., :arm_dim]
    train_expert_arm = partitions["train"]["expert_chunks"][..., :arm_dim]
    train_valid = partitions["train"]["valid_mask"][:, :, None].expand_as(
        train_source_arm
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "dataset": {
            "path": str(demo_path),
            "sha256": sha256(demo_path),
            "num_demos": args.num_demos,
        },
        "uact_source": provenance,
        "diagnostic_intermediate_uact_checkpoint": bool(intermediate_uact),
        "diagnostic_attached_variance_uact_50k": bool(attached_uact_50k),
        "diagnostic_peg_joint_uact_25k": bool(joint_uact_25k),
        "diagnostic_peg_joint_uact_50k": bool(joint_uact_50k),
        "aggregation": {
            "kind": "official_temporal_aggregation",
            "aggregation_k": args.aggregation_k,
            "query_frequency": 1,
            "clip_base_action": True,
        },
        "source_contract": {
            "kind": "ta_current_plus_current_query_future_proposal",
            "chunk_horizon": args.chunk_horizon,
            "uses_future_observations": False,
            "gaussian_source": False,
            "history_actions": False,
        },
        "routing": {
            "kind": "external_fully_offline_monotone_route_fit",
            "choices": [0, 1, 2, 3],
        },
        "splits": split_dict,
        "train_log_uncertainty_sorted": train_log,
        "statistics": {
            "arm_dim": arm_dim,
            "train_uncertainty_q90": float(torch.quantile(train_uncertainty, 0.9)),
            "train_uncertainty_q95": float(torch.quantile(train_uncertainty, 0.95)),
            "train_chunk_arm_residual_rms": float(
                (train_source_arm - train_expert_arm)
                .square()[train_valid]
                .mean()
                .sqrt()
            ),
        },
        "partitions": partitions,
    }
    if not math.isfinite(payload["statistics"]["train_chunk_arm_residual_rms"]):
        raise RuntimeError("cache statistics are non-finite")
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    summary = {
        "output": str(output),
        "sha256": sha256(output),
        "partition_samples": {
            name: int(value["observations"].shape[0])
            for (name, value) in partitions.items()
        },
        "statistics": payload["statistics"],
        "source_contract": payload["source_contract"],
        "uact_source": provenance,
        "diagnostic_intermediate_uact_checkpoint": bool(intermediate_uact),
        "diagnostic_attached_variance_uact_50k": bool(attached_uact_50k),
        "diagnostic_peg_joint_uact_25k": bool(joint_uact_25k),
        "diagnostic_peg_joint_uact_50k": bool(joint_uact_50k),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
