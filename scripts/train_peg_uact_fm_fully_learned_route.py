from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from src.policies.maniskill_uact_fm_learned_route import (
    LossDistributionHead,
    LossDistributionHeadConfig,
    ensemble_loss_distribution,
    normalize_features,
    uact_only_route_features,
)

METHOD = "peg_uact_fm_fully_learned_heads_train_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-head-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=1e-05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--posthoc-diagnostic", action="store_true")
    return parser.parse_args()


def feature_normalizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-05)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise RuntimeError("invalid router feature normalizer")
    return (mean, scale)


def target_normalizer(losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = losses.flatten()
    location = flattened.median()
    scale = (flattened.quantile(0.75) - flattened.quantile(0.25)).clamp_min(1e-05)
    if not torch.isfinite(location) or not torch.isfinite(scale):
        raise RuntimeError("invalid router target normalizer")
    return (location, scale)


def member_loss(
    head: LossDistributionHead, features: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    mean, log_variance = head(features)
    error_square = (target - mean).square()
    nll = 0.5 * (torch.exp(-log_variance) * error_square + log_variance)
    smooth = F.smooth_l1_loss(mean, target)
    pairs = [
        (left, right)
        for left in range(target.shape[1])
        for right in range(left + 1, target.shape[1])
    ]
    predicted_difference = torch.stack(
        [mean[:, left] - mean[:, right] for (left, right) in pairs], dim=-1
    )
    target_preference = torch.stack(
        [
            (target[:, left] < target[:, right]).to(mean.dtype)
            for (left, right) in pairs
        ],
        dim=-1,
    )
    ranking = F.binary_cross_entropy_with_logits(
        -predicted_difference / 0.2, target_preference
    )
    classification = F.cross_entropy(-mean / 0.2, target.argmin(dim=-1))
    return nll.mean() + 0.5 * smooth + 0.25 * ranking + 0.25 * classification


@torch.no_grad()
def member_objective(
    head: LossDistributionHead, features: torch.Tensor, target: torch.Tensor
) -> float:
    mean, log_variance = head(features)
    nll = 0.5 * (torch.exp(-log_variance) * (target - mean).square() + log_variance)
    selected = target[
        torch.arange(len(target), device=target.device), mean.argmin(dim=-1)
    ]
    return float(nll.mean() + 0.25 * selected.mean())


def train_ensemble(
    *,
    name: str,
    config: LossDistributionHeadConfig,
    train_features: torch.Tensor,
    calibration_features: torch.Tensor,
    train_target: torch.Tensor,
    calibration_target: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    states = []
    for member in range(args.ensemble_size):
        member_seed = args.seed + 1009 * member + (0 if name == "uact" else 100000)
        torch.manual_seed(member_seed)
        head = LossDistributionHead(config).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        generator = torch.Generator(device="cpu").manual_seed(member_seed)
        best_state = None
        best_objective = math.inf
        best_update = 0
        for update in range(1, args.updates + 1):
            index = torch.randint(
                len(train_features), (args.batch_size,), generator=generator
            )
            features = train_features[index].to(device)
            target = train_target[index].to(device)
            loss = member_loss(head, features, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} router loss became NaN/Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), args.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"{name} router gradient became NaN/Inf")
            optimizer.step()
            if update % 100 == 0 or update == args.updates:
                objective = member_objective(
                    head, calibration_features.to(device), calibration_target.to(device)
                )
                if objective < best_objective:
                    best_objective = objective
                    best_update = update
                    best_state = {
                        key: value.detach().cpu().clone()
                        for (key, value) in head.state_dict().items()
                    }
                if update % 500 == 0 or update == args.updates:
                    print(
                        json.dumps(
                            {
                                "head": name,
                                "member": member,
                                "update": update,
                                "loss": float(loss),
                                "calibration_objective": objective,
                                "best_update": best_update,
                            }
                        ),
                        flush=True,
                    )
        if best_state is None:
            raise RuntimeError(f"{name} member {member} did not produce a checkpoint")
        states.append(best_state)
    return states


@torch.no_grad()
def ensemble_metrics(
    config: LossDistributionHeadConfig,
    states: list[dict[str, torch.Tensor]],
    features: torch.Tensor,
    losses: torch.Tensor,
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    target_location: torch.Tensor,
    target_scale: torch.Tensor,
    device: torch.device,
) -> dict:
    heads = torch.nn.ModuleList([LossDistributionHead(config) for _ in states]).to(
        device
    )
    for head, state in zip(heads, states):
        head.load_state_dict(state, strict=True)
    mean, std = ensemble_loss_distribution(
        heads,
        features.to(device),
        feature_mean=feature_mean.to(device),
        feature_scale=feature_scale.to(device),
        target_location=target_location.to(device),
        target_scale=target_scale.to(device),
    )
    target = losses.to(device)
    selected = mean.argmin(dim=-1)
    chosen_loss = target[torch.arange(len(target), device=device), selected]
    oracle = target.amin(dim=-1)
    return {
        "mean_selected_offline_loss": float(chosen_loss.mean()),
        "mean_oracle_offline_loss": float(oracle.mean()),
        "mean_regret": float((chosen_loss - oracle).mean()),
        "best_choice_accuracy": float(
            (selected == target.argmin(dim=-1)).float().mean()
        ),
        "choice_fraction": [
            float((selected == index).float().mean()) for index in range(config.choices)
        ],
        "mean_predictive_std": float(std.mean()),
    }


def main() -> None:
    args = parse_args()
    if min(args.ensemble_size, args.updates, args.batch_size, args.hidden_dim) <= 0:
        raise ValueError("invalid router training size")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    source_path = Path(args.source_head_checkpoint).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("method") != "peg_hierarchical_uact_fm_stop_head_train_v1":
        raise RuntimeError("invalid source frozen-FM feature checkpoint")
    feature_path = Path(source["feature_cache"]["path"]).expanduser().resolve()
    cache_path = Path(source["cache"]["path"]).expanduser().resolve()
    if source["feature_cache"]["sha256"] != sha256(feature_path):
        raise RuntimeError("source feature cache hash changed")
    if source["cache"]["sha256"] != sha256(cache_path):
        raise RuntimeError("source causal cache hash changed")
    feature_cache = torch.load(feature_path, map_location="cpu", weights_only=False)
    causal_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    uact_features = {}
    fm_features = {}
    losses = {}
    for partition in ("train", "calibration", "test"):
        causal = causal_cache["partitions"][partition]
        frozen = feature_cache["partitions"][partition]
        if not torch.equal(causal["trajectory_index"], frozen["trajectory_index"]):
            raise RuntimeError(f"{partition} causal/FM feature order changed")
        uact_features[partition] = uact_only_route_features(
            causal["observations"],
            causal["source_chunks"],
            causal["uncertainty"],
            include_uncertainty=True,
        ).to(torch.float32)
        fm_features[partition] = frozen["features"].to(torch.float32)
        losses[partition] = frozen["losses"].to(torch.float32)
    uact_mean, uact_scale = feature_normalizer(uact_features["train"])
    fm_mean, fm_scale = feature_normalizer(fm_features["train"])
    loss_reference = losses["train"].flatten().median().clamp_min(1e-06)
    transformed_target = {
        name: torch.log1p(values / loss_reference) for (name, values) in losses.items()
    }
    target_location, target_scale = target_normalizer(transformed_target["train"])
    normalized_target = {
        name: (values - target_location) / target_scale
        for (name, values) in transformed_target.items()
    }
    normalized_uact = {
        name: normalize_features(values, uact_mean, uact_scale)
        for (name, values) in uact_features.items()
    }
    normalized_fm = {
        name: normalize_features(values, fm_mean, fm_scale)
        for (name, values) in fm_features.items()
    }
    uact_config = LossDistributionHeadConfig(
        feature_dim=uact_features["train"].shape[1],
        choices=4,
        hidden_dim=args.hidden_dim,
    )
    fm_config = LossDistributionHeadConfig(
        feature_dim=fm_features["train"].shape[1], choices=3, hidden_dim=args.hidden_dim
    )
    uact_states = train_ensemble(
        name="uact",
        config=uact_config,
        train_features=normalized_uact["train"],
        calibration_features=normalized_uact["calibration"],
        train_target=normalized_target["train"],
        calibration_target=normalized_target["calibration"],
        args=args,
        device=device,
    )
    fm_states = train_ensemble(
        name="fm",
        config=fm_config,
        train_features=normalized_fm["train"],
        calibration_features=normalized_fm["calibration"],
        train_target=normalized_target["train"][:, 1:],
        calibration_target=normalized_target["calibration"][:, 1:],
        args=args,
        device=device,
    )
    payload = {
        "schema_version": 1,
        "method": METHOD,
        "posthoc_diagnostic": args.posthoc_diagnostic,
        "training_complete": True,
        "act_uncertainty_feature": "included_log_scalar",
        "source_head_checkpoint": {
            "path": str(source_path),
            "sha256": sha256(source_path),
        },
        "feature_cache": {"path": str(feature_path), "sha256": sha256(feature_path)},
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "flow_checkpoint": source["flow_checkpoint"],
        "uact_head": {
            "config": asdict(uact_config),
            "ensemble": uact_states,
            "feature_mean": uact_mean,
            "feature_scale": uact_scale,
        },
        "fm_head": {
            "config": asdict(fm_config),
            "ensemble": fm_states,
            "feature_mean": fm_mean,
            "feature_scale": fm_scale,
        },
        "target_location": target_location,
        "target_scale": target_scale,
        "target_transform": {
            "name": "shared_log1p",
            "loss_reference": float(loss_reference),
            "preserves_per_state_choice_order": True,
        },
        "optimizer_contract": {
            "ensemble_size": args.ensemble_size,
            "updates_per_member": args.updates,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "loss": "heteroscedastic_nll+0.5_huber+0.25_pairwise+0.25_bestC",
        },
        "metrics": {},
    }
    for partition in ("calibration", "test"):
        payload["metrics"][partition] = {
            "uact": ensemble_metrics(
                uact_config,
                uact_states,
                uact_features[partition],
                losses[partition],
                feature_mean=uact_mean,
                feature_scale=uact_scale,
                target_location=target_location,
                target_scale=target_scale,
                device=device,
            ),
            "fm": ensemble_metrics(
                fm_config,
                fm_states,
                fm_features[partition],
                losses[partition][:, 1:],
                feature_mean=fm_mean,
                feature_scale=fm_scale,
                target_location=target_location,
                target_scale=target_scale,
                device=device,
            ),
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
                if key
                not in {"uact_head", "fm_head", "target_location", "target_scale"}
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": sha256(output),
                "metrics": payload["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
