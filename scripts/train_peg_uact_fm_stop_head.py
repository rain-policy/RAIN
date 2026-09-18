from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.checkpoint_cache_identity import validate_cache_identity
from scripts.load_refiner import load_fixed_flow
from src.policies.maniskill_uact_fm_hierarchical_route import (
    FMStopHead,
    FMStopHeadConfig,
    METHOD,
    fm_probe_features,
    gaussian_nll,
)

FEATURE_METHOD = "peg_frozen_fm_local_probe_features_v1"
TRAIN_METHOD = "peg_hierarchical_uact_fm_stop_head_train_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=1e-05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--posthoc-diagnostic", action="store_true")
    return parser.parse_args()


def _frontloaded_mean(
    per_step: torch.Tensor, valid: torch.Tensor, gamma: float
) -> torch.Tensor:
    weight = torch.pow(
        torch.as_tensor(gamma, dtype=per_step.dtype, device=per_step.device),
        torch.arange(per_step.shape[1], device=per_step.device, dtype=per_step.dtype),
    )[None]
    weight = weight * valid.to(per_step.dtype)
    return (per_step * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-12)


@torch.inference_mode()
def build_feature_partition(
    flow,
    cache_values: dict,
    score_values: dict,
    *,
    batch_size: int,
    frontload_gamma: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    count = int(cache_values["observations"].shape[0])
    losses = score_values["expert_arm_frontloaded_normalized_mse"].to(torch.float32)
    if losses.shape != (count, 4):
        raise RuntimeError("counterfactual score and causal cache order changed")
    gains = torch.stack(
        [
            losses[:, 0] - losses[:, 1:].amin(dim=1),
            losses[:, 1] - losses[:, 2:].amin(dim=1),
            losses[:, 2] - losses[:, 3],
        ],
        dim=-1,
    )
    feature_chunks: list[torch.Tensor] = []
    variance_feature_chunks: list[torch.Tensor] = []
    variance_target_chunks: list[torch.Tensor] = []
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        observations = cache_values["observations"][start:stop].to(device)
        source = cache_values["source_chunks"][start:stop].to(device)
        expert = cache_values["expert_chunks"][start:stop].to(device)
        valid = cache_values["valid_mask"][start:stop].to(device)
        uncertainty = cache_values["uncertainty"][start:stop].to(device)
        probe = fm_probe_features(flow, observations, source, uncertainty)
        feature_chunks.append(probe.features.cpu())
        expert_arm = flow.normalize_arm(expert)
        teacher_sample = 0.5 * probe.source_arm + 0.5 * expert_arm
        generated_sample = probe.source_arm + 0.5 * probe.first_velocity
        selector = (torch.arange(start, stop, device=device, dtype=torch.int64) % 2).to(
            torch.bool
        )[:, None, None]
        mixed_sample = torch.where(selector, teacher_sample, generated_sample)
        half_time = torch.full(
            (stop - start,), 0.5, device=device, dtype=probe.source_arm.dtype
        )
        mixed = fm_probe_features(
            flow,
            observations,
            source,
            uncertainty,
            sample_arm=mixed_sample,
            time=half_time,
        )
        variance_feature_chunks.append(mixed.features.cpu())
        target_velocity = expert_arm - probe.source_arm
        residual_square = (mixed.first_velocity - target_velocity).square().mean(dim=-1)
        first_error = residual_square[:, 0]
        front_error = _frontloaded_mean(residual_square, valid, frontload_gamma)
        variance_target_chunks.append(
            torch.stack([first_error, front_error], dim=-1).cpu()
        )
        print(json.dumps({"features": stop, "total": count}), flush=True)
    return {
        "features": torch.cat(feature_chunks),
        "variance_features": torch.cat(variance_feature_chunks),
        "variance_error_square": torch.cat(variance_target_chunks),
        "gains": gains,
        "losses": losses,
        "trajectory_index": cache_values["trajectory_index"].clone(),
        "timestep": cache_values["timestep"].clone(),
    }


def build_or_load_features(
    args: argparse.Namespace, flow, cache: dict, score: dict, device: torch.device
) -> tuple[dict, Path]:
    path = Path(args.feature_cache).expanduser().resolve()
    identities = {
        "cache_sha256": sha256(Path(args.cache).expanduser().resolve()),
        "score_sha256": sha256(Path(args.score).expanduser().resolve()),
        "flow_sha256": sha256(Path(args.flow_checkpoint).expanduser().resolve()),
    }
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("method") != FEATURE_METHOD
            or payload.get("identity") != identities
        ):
            raise RuntimeError("existing FM probe feature cache identity changed")
        return (payload, path)
    partitions = {
        name: build_feature_partition(
            flow,
            cache["partitions"][name],
            score["partitions"][name],
            batch_size=args.feature_batch_size,
            frontload_gamma=float(score["frontload_gamma"]),
            device=device,
        )
        for name in ("train", "calibration", "test")
    }
    payload = {
        "schema_version": 1,
        "method": FEATURE_METHOD,
        "identity": identities,
        "feature_dim": int(partitions["train"]["features"].shape[1]),
        "mixture": "50pct_teacher_t0.5_50pct_frozen_flow_euler_t0.5",
        "partitions": partitions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return (payload, path)


def robust_gain_scale(gains: torch.Tensor) -> torch.Tensor:
    scale = gains.abs().quantile(0.75, dim=0).clamp_min(0.0001)
    if not torch.isfinite(scale).all():
        raise RuntimeError("invalid gain normalization")
    return scale


@torch.no_grad()
def metrics(head: FMStopHead, values: dict, gain_scale: torch.Tensor, device) -> dict:
    prediction = head(values["features"].to(device))
    predicted_gain = torch.stack(
        [
            prediction["gain_enter"],
            prediction["gain_after_c1"],
            prediction["gain_after_c2"],
        ],
        dim=-1,
    )
    target = values["gains"].to(device) / gain_scale
    sign_accuracy = ((predicted_gain > 0) == (target > 0)).to(torch.float32).mean(dim=0)
    correlation = []
    for column in range(3):
        x = predicted_gain[:, column].double()
        y = target[:, column].double()
        value = torch.corrcoef(torch.stack([x, y]))[0, 1]
        correlation.append(float(value) if torch.isfinite(value) else None)
    return {
        "gain_smooth_l1": float(F.smooth_l1_loss(predicted_gain, target)),
        "gain_sign_accuracy": sign_accuracy.cpu().tolist(),
        "gain_pearson": correlation,
    }


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.feature_batch_size, args.updates) <= 0:
        raise ValueError("training sizes must be positive")
    if min(args.learning_rate, args.gradient_clip) <= 0:
        raise ValueError("optimizer configuration is invalid")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    cache_path = Path(args.cache).expanduser().resolve()
    score_path = Path(args.score).expanduser().resolve()
    flow_path = Path(args.flow_checkpoint).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    score = torch.load(score_path, map_location="cpu", weights_only=False)
    if score.get("method") != "uact_mars_c0123_counterfactual_benefit_score_v1":
        raise RuntimeError("invalid counterfactual score")
    if score["cache"]["sha256"] != sha256(cache_path):
        raise RuntimeError("score/cache identity mismatch")
    flow, flow_state = load_fixed_flow(flow_path, device)
    validate_cache_identity(flow_state, cache, cache_path)
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    features, feature_path = build_or_load_features(args, flow, cache, score, device)
    train = features["partitions"]["train"]
    calibration = features["partitions"]["calibration"]
    test = features["partitions"]["test"]
    gain_scale = robust_gain_scale(train["gains"]).to(device)
    config = FMStopHeadConfig(feature_dim=int(features["feature_dim"]))
    head = FMStopHead(config).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    best_state = None
    best_calibration = math.inf
    best_update = 0
    patience = 2000
    for update in range(1, args.updates + 1):
        index = torch.randint(
            len(train["features"]), (args.batch_size,), generator=generator
        )
        variance_index = torch.randint(
            len(train["variance_features"]), (args.batch_size,), generator=generator
        )
        prediction = head(train["features"][index].to(device))
        predicted_gain = torch.stack(
            [
                prediction["gain_enter"],
                prediction["gain_after_c1"],
                prediction["gain_after_c2"],
            ],
            dim=-1,
        )
        target_gain = train["gains"][index].to(device) / gain_scale
        gain_loss = F.smooth_l1_loss(predicted_gain, target_gain)
        ranking = F.binary_cross_entropy_with_logits(
            predicted_gain, (target_gain > 0).to(predicted_gain.dtype)
        )
        variance_prediction = head(
            train["variance_features"][variance_index].to(device)
        )
        variance_log = torch.stack(
            [
                variance_prediction["logvar_first"],
                variance_prediction["logvar_frontloaded"],
            ],
            dim=-1,
        )
        variance_target = train["variance_error_square"][variance_index].to(device)
        variance_loss = gaussian_nll(variance_target, variance_log).mean()
        loss = (
            gain_loss
            + args.ranking_weight * ranking
            + args.variance_weight * variance_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("FM stop-head loss became NaN/Inf")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            head.parameters(), args.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("FM stop-head gradient became NaN/Inf")
        optimizer.step()
        if any((parameter.grad is not None for parameter in flow.parameters())):
            raise RuntimeError("frozen Flow received a gradient")
        if update % 100 == 0 or update == args.updates:
            cal = metrics(head, calibration, gain_scale, device)
            objective = cal["gain_smooth_l1"]
            if objective < best_calibration:
                best_calibration = objective
                best_update = update
                best_state = {
                    key: value.detach().cpu().clone()
                    for (key, value) in head.state_dict().items()
                }
            print(
                json.dumps({"update": update, "loss": float(loss), "cal": cal}),
                flush=True,
            )
            if update - best_update >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    head.load_state_dict(best_state, strict=True)
    payload = {
        "schema_version": 1,
        "method": TRAIN_METHOD,
        "policy_method": METHOD,
        "posthoc_diagnostic": args.posthoc_diagnostic,
        "training_complete": True,
        "best_update": best_update,
        "model_config": asdict(config),
        "model": head.state_dict(),
        "gain_scale": gain_scale.cpu(),
        "feature_cache": {"path": str(feature_path), "sha256": sha256(feature_path)},
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "score": {"path": str(score_path), "sha256": sha256(score_path)},
        "flow_checkpoint": {"path": str(flow_path), "sha256": sha256(flow_path)},
        "optimizer_contract": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "maximum_updates": args.updates,
            "ranking_weight": args.ranking_weight,
            "variance_weight": args.variance_weight,
        },
        "metrics": {
            "calibration": metrics(head, calibration, gain_scale, device),
            "test": metrics(head, test, gain_scale, device),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                key: value
                for (key, value) in payload.items()
                if key not in {"model", "gain_scale"}
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "metrics": payload["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
