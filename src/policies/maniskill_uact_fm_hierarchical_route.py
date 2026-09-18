from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn

METHOD = "peg_hierarchical_uact_fm_stop_v1"


def _modulate(
    value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return value * (1 + scale[:, None]) + shift[:, None]


@dataclass(frozen=True)
class FMStopHeadConfig:
    feature_dim: int
    hidden_dim: int = 128
    output_dim: int = 5

    def validate(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("FM stop-head dimensions must be positive")
        if self.output_dim != 5:
            raise ValueError("FM stop head must emit three gains and two log variances")


class FMStopHead(nn.Module):
    """Predict enter/C1/C2 gains and local first/front-loaded log variance."""

    def __init__(self, config: FMStopHeadConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError("FM stop-head feature shape changed")
        raw = self.network(features)
        if not torch.isfinite(raw).all():
            raise RuntimeError("FM stop head produced NaN/Inf")
        return {
            "gain_enter": raw[:, 0],
            "gain_after_c1": raw[:, 1],
            "gain_after_c2": raw[:, 2],
            "logvar_first": raw[:, 3].clamp(-10.0, 8.0),
            "logvar_frontloaded": raw[:, 4].clamp(-10.0, 8.0),
        }


@dataclass
class FrozenFMProbe:
    source_chunk: torch.Tensor
    source_arm: torch.Tensor
    static_condition: torch.Tensor
    first_velocity: torch.Tensor
    features: torch.Tensor


def _flow_hidden_and_velocity(
    flow, sample_arm: torch.Tensor, time: torch.Tensor, static_condition: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the registered DiT while exposing its final frozen token features."""
    net = flow.velocity_net
    condition = static_condition + net.time_encoder(time)
    hidden = net.action_projection(sample_arm) + net.position
    for block in net.blocks:
        hidden = block(hidden, condition)
    shift, scale = net.final_modulation(condition).chunk(2, dim=-1)
    hidden = _modulate(net.final_norm(hidden), shift, scale)
    velocity = net.output(hidden)
    return (hidden, velocity)


def fm_probe_features(
    flow,
    observations: torch.Tensor,
    source_chunk: torch.Tensor,
    uact_uncertainty: torch.Tensor,
    *,
    sample_arm: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> FrozenFMProbe:
    """Extract detached local FM evidence without changing frozen parameters."""
    batch = observations.shape[0]
    expected = (batch, flow.config.chunk_horizon, flow.config.act_dim)
    if source_chunk.shape != expected:
        raise ValueError("source chunk does not match frozen Flow")
    if uact_uncertainty.reshape(-1).shape != (batch,):
        raise ValueError("U-ACT uncertainty must contain one scalar per sample")
    source_arm = flow.normalize_arm(source_chunk)
    normalized_observation = flow.normalized_observations(observations)
    static = flow.velocity_net.encode_static_condition(
        normalized_observation, source_arm
    )
    sample = source_arm if sample_arm is None else sample_arm
    if sample.shape != source_arm.shape:
        raise ValueError("FM probe sample has an invalid shape")
    if time is None:
        time = torch.zeros(batch, device=source_arm.device, dtype=source_arm.dtype)
    if time.shape != (batch,) or torch.any((time < 0) | (time > 1)):
        raise ValueError("FM probe time must have shape [batch] in [0,1]")
    hidden, velocity = _flow_hidden_and_velocity(flow, sample, time, static)
    velocity_norm = torch.linalg.vector_norm(velocity, dim=-1)
    source_delta = source_arm[:, 1:] - source_arm[:, :-1]
    scalar = torch.stack(
        [
            torch.log(uact_uncertainty.reshape(-1).clamp_min(1e-12)),
            velocity_norm[:, 0],
            velocity_norm.mean(dim=1),
            velocity_norm.amax(dim=1),
            source_delta.square().mean(dim=(1, 2)).sqrt(),
            time,
        ],
        dim=-1,
    )
    features = torch.cat(
        [static, hidden[:, 0], hidden.mean(dim=1), scalar], dim=-1
    ).detach()
    if not torch.isfinite(features).all():
        raise RuntimeError("FM probe features contain NaN/Inf")
    return FrozenFMProbe(
        source_chunk=source_chunk,
        source_arm=source_arm,
        static_condition=static,
        first_velocity=velocity,
        features=features,
    )


def gaussian_nll(
    error_square: torch.Tensor, log_variance: torch.Tensor
) -> torch.Tensor:
    if error_square.shape != log_variance.shape or torch.any(error_square < 0):
        raise ValueError("variance target has an invalid shape or value")
    return 0.5 * (torch.exp(-log_variance) * error_square + log_variance)
