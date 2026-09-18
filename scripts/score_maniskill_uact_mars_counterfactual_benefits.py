from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from scripts.checkpoint_cache_identity import validate_cache_identity
from scripts.load_refiner import load_fixed_flow
from scripts.score_maniskill_uact_mars_c0123_offline import (
    closed_loop_ta_chunks,
    sha256,
)

METHOD = "uact_mars_c0123_counterfactual_benefit_score_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--frontload-gamma", type=float, default=0.8)
    parser.add_argument("--axis-scale-floor", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--posthoc-diagnostic", action="store_true")
    return parser.parse_args()


def expert_axis_scale(cache: dict, arm_dim: int, floor: float) -> torch.Tensor:
    partition = cache["partitions"]["train"]
    expert = partition["expert_chunks"][..., :arm_dim]
    valid = partition["valid_mask"]
    flattened = expert[valid]
    if flattened.ndim != 2 or flattened.shape[1] != arm_dim:
        raise RuntimeError("failed to gather valid expert arm actions")
    scale = flattened.to(torch.float64).std(dim=0, unbiased=False).clamp_min(floor)
    if not torch.isfinite(scale).all():
        raise RuntimeError("expert action scale is non-finite")
    return scale.to(torch.float32)


def detailed_losses(
    candidate: torch.Tensor,
    expert: torch.Tensor,
    valid: torch.Tensor,
    *,
    arm_dim: int,
    axis_scale: torch.Tensor,
    frontload_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if candidate.shape != expert.shape or candidate.shape[-1] <= arm_dim:
        raise ValueError("candidate/expert action contract changed")
    difference = candidate[..., :arm_dim] - expert[..., :arm_dim]
    raw_step = difference.square().mean(dim=-1)
    normalized_step = (difference / axis_scale).square().mean(dim=-1)
    valid_weight = valid.to(raw_step.dtype)
    uniform = (raw_step * valid_weight).sum(dim=-1) / valid_weight.sum(
        dim=-1
    ).clamp_min(1)
    horizon = candidate.shape[1]
    time_weight = torch.pow(
        torch.as_tensor(frontload_gamma, device=candidate.device, dtype=raw_step.dtype),
        torch.arange(horizon, device=candidate.device, dtype=raw_step.dtype),
    )
    front_weight = valid_weight * time_weight[None]
    frontloaded = (normalized_step * front_weight).sum(dim=-1) / front_weight.sum(
        dim=-1
    ).clamp_min(1e-12)
    if not bool(valid[:, 0].all()):
        raise RuntimeError("first action unexpectedly invalid")
    first = normalized_step[:, 0]
    return (uniform, frontloaded, first)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or not 0 < args.frontload_gamma <= 1
        or args.axis_scale_floor <= 0
    ):
        raise ValueError("invalid detailed score configuration")
    cache_path = Path(args.cache).expanduser().resolve()
    flow_path = Path(args.flow_checkpoint).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("method") != "uact_official_ta_causal_action_chunk_cache_v1":
        raise RuntimeError("invalid causal U-ACT cache")
    flow, flow_state = load_fixed_flow(flow_path, device, allow_incomplete=False)
    validate_cache_identity(flow_state, cache, cache_path)
    if flow.config.chunk_horizon != 8:
        raise RuntimeError("counterfactual benefit score requires horizon eight")
    axis_scale_cpu = expert_axis_scale(
        cache, flow.config.arm_dim, args.axis_scale_floor
    )
    axis_scale = axis_scale_cpu.to(device)
    reference = cache["train_log_uncertainty_sorted"].to(torch.float64)
    partitions = {}
    for name in ("train", "calibration", "test"):
        source_partition = cache["partitions"][name]
        count = int(source_partition["observations"].shape[0])
        c0 = closed_loop_ta_chunks(source_partition, flow.config.chunk_horizon)
        uniform = torch.empty(count, 4, dtype=torch.float32)
        frontloaded = torch.empty_like(uniform)
        first = torch.empty_like(uniform)
        teacher_frontloaded = torch.empty_like(uniform)
        teacher_first = torch.empty_like(uniform)
        for start in range(0, count, args.batch_size):
            stop = min(count, start + args.batch_size)
            observations = source_partition["observations"][start:stop].to(device)
            source = source_partition["source_chunks"][start:stop].to(device)
            expert = source_partition["expert_chunks"][start:stop].to(device)
            valid = source_partition["valid_mask"][start:stop].to(device)
            candidates = {0: c0[start:stop].to(device)}
            for c in (1, 2, 3):
                refined = flow.refine_chunk(observations, source, steps=c)
                if not torch.equal(
                    refined[..., flow.config.arm_dim :],
                    source[..., flow.config.arm_dim :],
                ):
                    raise RuntimeError("frozen Flow modified the ACT gripper")
                candidates[c] = refined
            for c, candidate in candidates.items():
                values = detailed_losses(
                    candidate,
                    expert,
                    valid,
                    arm_dim=flow.config.arm_dim,
                    axis_scale=axis_scale,
                    frontload_gamma=args.frontload_gamma,
                )
                uniform[start:stop, c] = values[0].cpu()
                frontloaded[start:stop, c] = values[1].cpu()
                first[start:stop, c] = values[2].cpu()
            teacher = candidates[3]
            for c, candidate in candidates.items():
                teacher_values = detailed_losses(
                    candidate,
                    teacher,
                    valid,
                    arm_dim=flow.config.arm_dim,
                    axis_scale=axis_scale,
                    frontload_gamma=args.frontload_gamma,
                )
                teacher_frontloaded[start:stop, c] = teacher_values[1].cpu()
                teacher_first[start:stop, c] = teacher_values[2].cpu()
            print(
                json.dumps({"partition": name, "scored": stop, "total": count}),
                flush=True,
            )
        uncertainty = source_partition["uncertainty"].clamp_min(1e-12)
        percentile = torch.searchsorted(
            reference, torch.log(uncertainty).to(torch.float64), right=True
        ).to(torch.float32) / len(reference)
        partitions[name] = {
            "expert_arm_chunk_mse": uniform,
            "expert_arm_frontloaded_normalized_mse": frontloaded,
            "expert_arm_first_normalized_mse": first,
            "c3_teacher_frontloaded_normalized_mse": teacher_frontloaded,
            "c3_teacher_first_normalized_mse": teacher_first,
            "uncertainty": uncertainty,
            "uncertainty_percentile": percentile,
            "trajectory_index": source_partition["trajectory_index"],
            "timestep": source_partition["timestep"],
        }
    payload = {
        "schema_version": 1,
        "method": METHOD,
        "choices": [0, 1, 2, 3],
        "execution_horizon": 8,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "flow_checkpoint": {"path": str(flow_path), "sha256": sha256(flow_path)},
        "posthoc_diagnostic": args.posthoc_diagnostic,
        "losses": {
            "expert_arm_chunk_mse": "valid uniform raw six-arm MSE",
            "expert_arm_frontloaded_normalized_mse": "valid per-axis train-standardized MSE with gamma^h time weights",
            "expert_arm_first_normalized_mse": "first action per-axis train-standardized MSE",
            "c3_teacher_frontloaded_normalized_mse": "front-loaded per-axis normalized deviation from the frozen C3 output",
            "c3_teacher_first_normalized_mse": "first-action per-axis normalized deviation from the frozen C3 output",
        },
        "frontload_gamma": args.frontload_gamma,
        "arm_axis_scale": axis_scale_cpu.tolist(),
        "partitions": partitions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    summary = {
        "method": METHOD,
        "output": str(output),
        "sha256": sha256(output),
        "frontload_gamma": args.frontload_gamma,
        "arm_axis_scale": axis_scale_cpu.tolist(),
        "mean_loss": {
            name: {
                key: values[key].mean(dim=0).tolist()
                for key in (
                    "expert_arm_chunk_mse",
                    "expert_arm_frontloaded_normalized_mse",
                    "expert_arm_first_normalized_mse",
                    "c3_teacher_frontloaded_normalized_mse",
                    "c3_teacher_first_normalized_mse",
                )
            }
            for (name, values) in partitions.items()
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
