from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.policies.maniskill_uact_fm_learned_route import (
    METHOD,
    LossDistributionHead,
    LossDistributionHeadConfig,
    ensemble_loss_distribution,
    uact_only_route_features,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--beta-grid", default="0,0.25,0.5,1,1.5,2")
    parser.add_argument("--price-points", type=int, default=31)
    parser.add_argument(
        "--selection-objective",
        choices=("mean_loss", "best_c_error"),
        default="mean_loss",
    )
    parser.add_argument("--transition-strength-grid", default="0")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--posthoc-diagnostic", action="store_true")
    return parser.parse_args()


def trajectory_mean(values: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(trajectory, return_inverse=True)
    total = np.zeros(len(unique), dtype=np.float64)
    count = np.zeros(len(unique), dtype=np.int64)
    np.add.at(total, inverse, values)
    np.add.at(count, inverse, 1)
    if np.any(count == 0):
        raise RuntimeError("empty offline trajectory")
    return total / count


def load_heads(values: dict, key: str, device: torch.device) -> torch.nn.ModuleList:
    specification = values[key]
    config = LossDistributionHeadConfig(**specification["config"])
    heads = torch.nn.ModuleList(
        [LossDistributionHead(config) for _ in specification["ensemble"]]
    ).to(device)
    for head, state in zip(heads, specification["ensemble"]):
        head.load_state_dict(state, strict=True)
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    return heads


@torch.inference_mode()
def predict(
    checkpoint: dict,
    uact_heads: torch.nn.ModuleList,
    fm_heads: torch.nn.ModuleList,
    causal: dict,
    frozen: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uact_features = uact_only_route_features(
        causal["observations"],
        causal["source_chunks"],
        causal["uncertainty"],
        include_uncertainty=checkpoint.get("act_uncertainty_feature") != "excluded",
    ).to(device)
    fm_features = frozen["features"].to(device)
    location = checkpoint["target_location"].to(device)
    scale = checkpoint["target_scale"].to(device)
    uact_mean, uact_std = ensemble_loss_distribution(
        uact_heads,
        uact_features,
        feature_mean=checkpoint["uact_head"]["feature_mean"].to(device),
        feature_scale=checkpoint["uact_head"]["feature_scale"].to(device),
        target_location=location,
        target_scale=scale,
    )
    fm_mean, fm_std = ensemble_loss_distribution(
        fm_heads,
        fm_features,
        feature_mean=checkpoint["fm_head"]["feature_mean"].to(device),
        feature_scale=checkpoint["fm_head"]["feature_scale"].to(device),
        target_location=location,
        target_scale=scale,
    )
    return tuple(
        (
            value.cpu().numpy().astype(np.float64)
            for value in (uact_mean, uact_std, fm_mean, fm_std)
        )
    )


def choose(
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    beta: float,
    compute_price: float,
    trajectory: np.ndarray | None = None,
    timestep: np.ndarray | None = None,
    transition_probability: np.ndarray | None = None,
    transition_strength: float = 0.0,
) -> np.ndarray:
    uact_mean, uact_std, fm_mean, fm_std = predictions
    cost4 = np.asarray([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float64)
    cost3 = cost4[1:]
    uact_risk = uact_mean + beta * uact_std + compute_price * cost4[None]
    fm_risk = fm_mean + beta * fm_std + compute_price * cost3[None]
    if transition_probability is None or transition_strength == 0:
        uact_choice = np.argmin(uact_risk, axis=1)
        fm_choice = 1 + np.argmin(fm_risk, axis=1)
        return np.where(uact_choice == 0, 0, fm_choice).astype(np.int64)
    if trajectory is None or timestep is None:
        raise ValueError("sequence route requires trajectory and timestep")
    selected = np.empty(len(uact_mean), dtype=np.int64)
    for trajectory_id in np.unique(trajectory):
        positions = np.flatnonzero(trajectory == trajectory_id)
        positions = positions[np.argsort(timestep[positions])]
        cursor = 0
        previous = None
        previous_horizon = None
        while cursor < len(positions):
            index = positions[cursor]
            uact_value = uact_risk[index].copy()
            fm_value = fm_risk[index].copy()
            if previous is not None:
                transition = np.linalg.matrix_power(
                    transition_probability, previous_horizon
                )[previous]
                transition_energy = -np.log(np.clip(transition, 1e-09, 1.0))
                uact_value += transition_strength * transition_energy
                fm_value += transition_strength * transition_energy[1:]
            preliminary = int(np.argmin(uact_value))
            choice = 0 if preliminary == 0 else 1 + int(np.argmin(fm_value))
            horizon = 2 if choice == 0 else 8
            stop = min(cursor + horizon, len(positions))
            selected[positions[cursor:stop]] = choice
            previous = choice
            previous_horizon = horizon
            cursor = stop
    return selected


def learned_transition_probability(values: dict) -> np.ndarray:
    losses = values["losses"].numpy().astype(np.float64)
    trajectory = values["trajectory_index"].numpy().astype(np.int64)
    timestep = values["timestep"].numpy().astype(np.int64)
    best = losses.argmin(axis=1)
    counts = np.ones((4, 4), dtype=np.float64)
    for trajectory_id in np.unique(trajectory):
        positions = np.flatnonzero(trajectory == trajectory_id)
        positions = positions[np.argsort(timestep[positions])]
        labels = best[positions]
        np.add.at(counts, (labels[:-1], labels[1:]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def summary(values: dict, selected: np.ndarray) -> dict:
    losses = values["losses"].numpy().astype(np.float64)
    trajectory = values["trajectory_index"].numpy().astype(np.int64)
    error = losses[np.arange(len(losses)), selected]
    oracle = losses.min(axis=1)
    best = losses.argmin(axis=1)
    classification_error = (selected != best).astype(np.float64)
    return {
        "mean_trajectory_loss": float(np.mean(trajectory_mean(error, trajectory))),
        "mean_oracle_trajectory_loss": float(
            np.mean(trajectory_mean(oracle, trajectory))
        ),
        "mean_regret": float(np.mean(trajectory_mean(error - oracle, trajectory))),
        "mean_c": float(
            np.mean(trajectory_mean(selected.astype(np.float64), trajectory))
        ),
        "best_c_accuracy": float(np.mean(selected == best)),
        "mean_trajectory_best_c_error": float(
            np.mean(trajectory_mean(classification_error, trajectory))
        ),
        "c_counts": {str(c): int(np.sum(selected == c)) for c in range(4)},
        "c_fraction": {str(c): float(np.mean(selected == c)) for c in range(4)},
    }


def main() -> None:
    args = parse_args()
    if args.price_points < 5:
        raise ValueError("insufficient automatic compute-price resolution")
    beta_grid = sorted({float(value) for value in args.beta_grid.split(",")})
    transition_grid = sorted(
        {float(value) for value in args.transition_strength_grid.split(",")}
    )
    if (
        not beta_grid
        or min(beta_grid) < 0
        or (not transition_grid)
        or (min(transition_grid) < 0)
    ):
        raise ValueError("invalid beta grid")
    checkpoint_path = Path(args.router_checkpoint).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "peg_uact_fm_fully_learned_heads_train_v1":
        raise RuntimeError("invalid learned router checkpoint")
    feature_path = Path(checkpoint["feature_cache"]["path"]).expanduser().resolve()
    cache_path = Path(checkpoint["cache"]["path"]).expanduser().resolve()
    if checkpoint["feature_cache"]["sha256"] != sha256(feature_path):
        raise RuntimeError("learned router feature cache hash changed")
    if checkpoint["cache"]["sha256"] != sha256(cache_path):
        raise RuntimeError("learned router causal cache hash changed")
    feature_cache = torch.load(feature_path, map_location="cpu", weights_only=False)
    causal_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    uact_heads = load_heads(checkpoint, "uact_head", device)
    fm_heads = load_heads(checkpoint, "fm_head", device)
    prediction = {}
    for partition in ("calibration", "test"):
        causal = causal_cache["partitions"][partition]
        frozen = feature_cache["partitions"][partition]
        if not torch.equal(causal["trajectory_index"], frozen["trajectory_index"]):
            raise RuntimeError(f"{partition} learned route order changed")
        prediction[partition] = predict(
            checkpoint, uact_heads, fm_heads, causal, frozen, device
        )
    calibration = feature_cache["partitions"]["calibration"]
    transition_probability = learned_transition_probability(
        feature_cache["partitions"]["train"]
    )
    cal_losses = calibration["losses"].numpy().astype(np.float64)
    cal_trajectory = calibration["trajectory_index"].numpy().astype(np.int64)
    base_scale = float(np.median(prediction["calibration"][0][:, 0]))
    if not np.isfinite(base_scale) or base_scale <= 0:
        raise RuntimeError("invalid automatic compute-price scale")
    price_grid = np.unique(
        np.concatenate(
            [
                np.asarray([0.0]),
                base_scale * np.geomspace(0.0001, 0.5, args.price_points - 1),
            ]
        )
    )
    candidates = []
    per_trajectory = []
    for beta in beta_grid:
        for price in price_grid:
            for transition_strength in transition_grid:
                selected = choose(
                    prediction["calibration"],
                    beta=beta,
                    compute_price=float(price),
                    trajectory=cal_trajectory,
                    timestep=calibration["timestep"].numpy().astype(np.int64),
                    transition_probability=transition_probability,
                    transition_strength=transition_strength,
                )
                error = cal_losses[np.arange(len(cal_losses)), selected]
                if args.selection_objective == "mean_loss":
                    trajectory_error = trajectory_mean(error, cal_trajectory)
                else:
                    best = cal_losses.argmin(axis=1)
                    trajectory_error = trajectory_mean(
                        (selected != best).astype(np.float64), cal_trajectory
                    )
                values = summary(calibration, selected)
                candidates.append(
                    {
                        "beta": beta,
                        "compute_price": float(price),
                        "transition_strength": transition_strength,
                        **values,
                    }
                )
                per_trajectory.append(trajectory_error)
    objective_key = (
        "mean_trajectory_loss"
        if args.selection_objective == "mean_loss"
        else "mean_trajectory_best_c_error"
    )
    best_index = min(
        range(len(candidates)),
        key=lambda index: (
            candidates[index][objective_key],
            candidates[index]["mean_c"],
        ),
    )
    best_error = per_trajectory[best_index]
    feasible = []
    for index, item in enumerate(candidates):
        difference = per_trajectory[index] - best_error
        standard_error = (
            float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
            if len(difference) > 1
            else 0.0
        )
        item["paired_excess_vs_best"] = float(np.mean(difference))
        item["paired_one_standard_error"] = standard_error
        item["within_one_standard_error"] = bool(np.mean(difference) <= standard_error)
        if item["within_one_standard_error"]:
            feasible.append(index)
    selected_index = min(
        feasible,
        key=lambda index: (
            candidates[index]["mean_c"],
            candidates[index][objective_key],
        ),
    )
    selected_candidate = candidates[selected_index]
    beta = selected_candidate["beta"]
    compute_price = selected_candidate["compute_price"]
    transition_strength = selected_candidate["transition_strength"]
    payload = {
        "schema_version": 1,
        "method": METHOD,
        "posthoc_diagnostic": args.posthoc_diagnostic,
        "selection": "fully_offline_one_standard_error_no_default_c",
        "act_uncertainty_feature": checkpoint.get(
            "act_uncertainty_feature", "included_log_scalar"
        ),
        "router_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
        },
        "flow_checkpoint": checkpoint["flow_checkpoint"],
        "feature_cache": checkpoint["feature_cache"],
        "cache": checkpoint["cache"],
        "choices": [0, 1, 2, 3],
        "costs": [0.0, 1 / 3, 2 / 3, 1.0],
        "beta": beta,
        "compute_price": compute_price,
        "transition_probability": transition_probability.tolist(),
        "transition_strength": transition_strength,
        "transition_horizon_semantics": {"0": 2, "1": 8, "2": 8, "3": 8},
        "selection_rule": {
            "description": "lowest mean C statistically within one paired standard error of the best calibration objective",
            "objective": args.selection_objective,
            "beta_grid": beta_grid,
            "transition_strength_grid": transition_grid,
            "compute_price_points": len(price_grid),
            "calibration_best": candidates[best_index],
            "selected": selected_candidate,
            "feasible_candidate_count": len(feasible),
        },
        "offline_metrics": {},
    }
    for partition in ("calibration", "test"):
        selected = choose(
            prediction[partition],
            beta=beta,
            compute_price=compute_price,
            trajectory=feature_cache["partitions"][partition]["trajectory_index"]
            .numpy()
            .astype(np.int64),
            timestep=feature_cache["partitions"][partition]["timestep"]
            .numpy()
            .astype(np.int64),
            transition_probability=transition_probability,
            transition_strength=transition_strength,
        )
        payload["offline_metrics"][partition] = summary(
            feature_cache["partitions"][partition], selected
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2))
