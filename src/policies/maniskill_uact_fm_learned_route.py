from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn

METHOD = "peg_uact_fm_fully_learned_route_v1"


@dataclass(frozen=True)
class LossDistributionHeadConfig:
    feature_dim: int
    choices: int
    hidden_dim: int = 128

    def validate(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("loss-distribution head dimensions must be positive")
        if self.choices not in (3, 4):
            raise ValueError("router head must represent C1-C3 or C0-C3")


class LossDistributionHead(nn.Module):
    """Predict normalized loss means and aleatoric log variances per C."""

    def __init__(self, config: LossDistributionHeadConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2 * config.choices),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError("router feature shape changed")
        raw = self.network(features)
        if not torch.isfinite(raw).all():
            raise RuntimeError("router head produced NaN/Inf")
        mean, log_variance = raw.chunk(2, dim=-1)
        return (mean, log_variance.clamp(-10.0, 8.0))


class PreOnlyRouterInferenceCore(nn.Module):
    """Fixed-shape deployment core for the pre-FM C0--C3 router.

    The training/evaluation helpers below deliberately retain expensive
    fail-closed checks.  This core is used only after the checkpoint and tensor
    contract have been validated.  Keeping data-dependent checks out of
    ``forward`` lets the complete three-member ensemble, predictive moments,
    transition bias, and risk minimization be captured in one CUDA graph.
    """

    def __init__(
        self,
        heads: nn.ModuleList,
        *,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        target_location: torch.Tensor,
        target_scale: torch.Tensor,
        costs: torch.Tensor,
        beta: float,
        compute_price: float,
    ) -> None:
        super().__init__()
        if len(heads) <= 0:
            raise ValueError("pre-only router requires a non-empty ensemble")
        config = heads[0].config
        if config.choices != 4 or any((head.config != config for head in heads)):
            raise ValueError(
                "pre-only ensemble members must share a four-choice config"
            )
        if feature_mean.shape != (config.feature_dim,) or feature_scale.shape != (
            config.feature_dim,
        ):
            raise ValueError("pre-only feature normalizer changed")
        if costs.shape != (config.choices,):
            raise ValueError("pre-only cost vector changed")
        if min(beta, compute_price) < 0:
            raise ValueError("pre-only risk coefficients must be non-negative")
        self.heads = heads
        self.config = config
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        self.register_buffer("target_location", target_location.detach().clone())
        self.register_buffer("target_scale", target_scale.detach().clone())
        self.register_buffer(
            "fixed_risk_cost", (float(compute_price) * costs).detach().clone()
        )
        self.beta = float(beta)

    def forward(
        self,
        observations: torch.Tensor,
        source_chunk: torch.Tensor,
        uncertainty: torch.Tensor,
        transition_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = observations.shape[0]
        base_features = torch.cat(
            [observations.flatten(1), source_chunk.flatten(1)], dim=-1
        )
        if base_features.shape[-1] == self.config.feature_dim:
            features = base_features
        elif base_features.shape[-1] + 1 == self.config.feature_dim:
            features = torch.cat(
                [
                    base_features,
                    torch.log(uncertainty.reshape(batch, 1).clamp_min(1e-12)),
                ],
                dim=-1,
            )
        else:
            raise ValueError("pre-only router feature dimension changed")
        normalized = (
            (features - self.feature_mean) / self.feature_scale.clamp_min(1e-06)
        ).clamp(-10.0, 10.0)
        raw_outputs = [head.network(normalized) for head in self.heads]
        outputs = [raw.chunk(2, dim=-1) for raw in raw_outputs]
        means = torch.stack([item[0] for item in outputs], dim=0)
        variances = torch.stack(
            [item[1].clamp(-10.0, 8.0).exp() for item in outputs], dim=0
        )
        normalized_mean = means.mean(dim=0)
        normalized_variance = variances.mean(dim=0) + means.var(dim=0, unbiased=False)
        mean = (normalized_mean * self.target_scale + self.target_location).clamp_min(
            0.0
        )
        standard_deviation = (
            normalized_variance.clamp_min(1e-12).sqrt() * self.target_scale.abs()
        )
        routed_mean = mean + transition_bias
        risk = routed_mean + self.beta * standard_deviation + self.fixed_risk_cost[None]
        choice = risk.argmin(dim=-1).to(torch.int64)
        return (choice, mean, standard_deviation)


def uact_only_route_features(
    observations: torch.Tensor,
    source_chunk: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    include_uncertainty: bool = True,
) -> torch.Tensor:
    """Cheap C0-safe features; optionally exclude explicit ACT uncertainty."""
    batch = observations.shape[0]
    if observations.ndim != 3 or source_chunk.ndim != 3:
        raise ValueError("U-ACT route inputs must be batched sequences")
    if source_chunk.shape[0] != batch or uncertainty.reshape(-1).shape != (batch,):
        raise ValueError("U-ACT route input batch changed")
    parts = [observations.flatten(1), source_chunk.flatten(1)]
    if include_uncertainty:
        parts.append(torch.log(uncertainty.reshape(batch, 1).clamp_min(1e-12)))
    result = torch.cat(parts, dim=-1)
    if not torch.isfinite(result).all():
        raise RuntimeError("U-ACT route features contain NaN/Inf")
    return result


def normalize_features(
    features: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    if mean.ndim != 1 or scale.shape != mean.shape or features.shape[-1] != len(mean):
        raise ValueError("router feature normalizer changed")
    return ((features - mean) / scale.clamp_min(1e-06)).clamp(-10.0, 10.0)


@torch.inference_mode()
def ensemble_loss_distribution(
    heads: nn.ModuleList,
    features: torch.Tensor,
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    target_location: torch.Tensor,
    target_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predictive mean/std with aleatoric and ensemble uncertainty."""
    normalized = normalize_features(features, feature_mean, feature_scale)
    outputs = [head(normalized) for head in heads]
    means = torch.stack([item[0] for item in outputs], dim=0)
    variances = torch.stack([item[1].exp() for item in outputs], dim=0)
    normalized_mean = means.mean(dim=0)
    normalized_variance = variances.mean(dim=0) + means.var(dim=0, unbiased=False)
    mean = normalized_mean * target_scale + target_location
    standard_deviation = (
        normalized_variance.clamp_min(1e-12).sqrt() * target_scale.abs()
    )
    return (mean.clamp_min(0.0), standard_deviation)


def risk_choice(
    mean: torch.Tensor,
    standard_deviation: torch.Tensor,
    costs: torch.Tensor,
    *,
    beta: float,
    compute_price: float,
    first_choice: int,
) -> torch.Tensor:
    """Choose C by learned predictive Bayes risk, with no default C."""
    if mean.shape != standard_deviation.shape or mean.ndim != 2:
        raise ValueError("router loss distribution has an invalid shape")
    if costs.shape != (mean.shape[1],):
        raise ValueError("router cost vector changed")
    if min(beta, compute_price) < 0:
        raise ValueError("router risk coefficients must be non-negative")
    risk = mean + beta * standard_deviation + compute_price * costs[None]
    return risk.argmin(dim=-1).to(torch.int64) + int(first_choice)
