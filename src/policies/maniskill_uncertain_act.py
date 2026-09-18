from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

UNCERTAIN_ACT_METHOD = "clean_uncertain_act_v2_variance_isolated"
ATTACHED_VARIANCE_ACT_METHOD = "clean_uncertain_act_v2_attached_variance_diagnostic"
LEGACY_UNCERTAIN_ACT_METHOD = "clean_uncertain_act"


@dataclass(frozen=True)
class UncertainACTConfig:
    obs_dim: int
    act_dim: int
    obs_horizon: int = 2
    num_queries: int = 30
    d_model: int = 256
    nhead: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    min_logvar: float = -8.0
    max_logvar: float = 2.0
    variance_detach_features: bool = True

    def validate(self) -> None:
        integer_fields = (
            self.obs_dim,
            self.act_dim,
            self.obs_horizon,
            self.num_queries,
            self.d_model,
            self.nhead,
            self.encoder_layers,
            self.decoder_layers,
            self.dim_feedforward,
        )
        if any((value <= 0 for value in integer_fields)):
            raise ValueError(
                "uncertain ACT dimensions and layer counts must be positive"
            )
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.min_logvar >= self.max_logvar:
            raise ValueError("min_logvar must be smaller than max_logvar")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass
class UncertainACTOutput:
    mean: torch.Tensor
    normalized_mean: torch.Tensor
    logvar: torch.Tensor
    variance: torch.Tensor
    near_uncertainty: torch.Tensor
    future_uncertainty: torch.Tensor
    p95_uncertainty: torch.Tensor
    uncertainty: torch.Tensor


@dataclass
class UncertainACTLoss:
    loss: torch.Tensor
    continuous_mean_huber: torch.Tensor
    variance_nll: torch.Tensor
    gripper_huber: torch.Tensor
    mean_logvar: torch.Tensor
    lower_clamp_fraction: torch.Tensor
    upper_clamp_fraction: torch.Tensor


class CleanUncertainACT(nn.Module):
    """Predict an action-chunk mean and diagonal heteroscedastic variance."""

    def __init__(
        self,
        config: UncertainACTConfig,
        *,
        obs_mean: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.obs_projection = nn.Sequential(
            nn.Linear(config.obs_dim, config.d_model), nn.LayerNorm(config.d_model)
        )
        self.observation_position = nn.Parameter(
            torch.empty(config.obs_horizon, config.d_model)
        )
        self.action_queries = nn.Parameter(
            torch.empty(config.num_queries, config.d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.observation_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.mean_head = nn.Linear(config.d_model, config.act_dim)
        self.logvar_head = nn.Linear(config.d_model, config.act_dim)
        self.register_buffer("obs_mean", self._stat(obs_mean, config.obs_dim, fill=0.0))
        self.register_buffer("obs_std", self._stat(obs_std, config.obs_dim, fill=1.0))
        self.register_buffer(
            "action_mean", self._stat(action_mean, config.act_dim, fill=0.0)
        )
        self.register_buffer(
            "action_std", self._stat(action_std, config.act_dim, fill=1.0)
        )
        self.reset_parameters()

    @staticmethod
    def _stat(value: torch.Tensor | None, size: int, *, fill: float) -> torch.Tensor:
        tensor = (
            torch.full((size,), fill, dtype=torch.float32)
            if value is None
            else torch.as_tensor(value, dtype=torch.float32).clone()
        )
        if tensor.shape != (size,) or not torch.isfinite(tensor).all():
            raise ValueError("normalization statistic has invalid shape or values")
        if fill == 1.0 and torch.any(tensor <= 0):
            raise ValueError("normalization standard deviations must be positive")
        return tensor

    def reset_parameters(self) -> None:
        nn.init.normal_(self.observation_position, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.action_std + self.action_mean

    def _validate_observations(self, observations: torch.Tensor) -> None:
        expected = (self.config.obs_horizon, self.config.obs_dim)
        if observations.ndim != 3 or tuple(observations.shape[1:]) != expected:
            raise ValueError(
                f"observations must have shape [batch, {expected[0]}, {expected[1]}]"
            )
        if not torch.isfinite(observations).all():
            raise ValueError("observations must be finite")

    def forward(self, observations: torch.Tensor) -> UncertainACTOutput:
        self._validate_observations(observations)
        normalized_obs = (observations - self.obs_mean) / self.obs_std
        memory = self.obs_projection(normalized_obs)
        memory = self.observation_encoder(
            memory + self.observation_position.unsqueeze(0)
        )
        queries = self.action_queries.unsqueeze(0).expand(observations.shape[0], -1, -1)
        hidden = self.action_decoder(queries, memory)
        normalized_mean = self.mean_head(hidden)
        variance_features = (
            hidden.detach() if self.config.variance_detach_features else hidden
        )
        raw_logvar = self.logvar_head(variance_features)
        logvar = raw_logvar.clamp(self.config.min_logvar, self.config.max_logvar)
        variance = logvar.exp()
        mean = self.denormalize_actions(normalized_mean)
        near, future, p95, combined = uncertainty_summaries(variance)
        return UncertainACTOutput(
            mean=mean,
            normalized_mean=normalized_mean,
            logvar=logvar,
            variance=variance,
            near_uncertainty=near,
            future_uncertainty=future,
            p95_uncertainty=p95,
            uncertainty=combined,
        )

    @torch.no_grad()
    def get_action(self, observations: torch.Tensor) -> torch.Tensor:
        return self(observations).mean


def uncertainty_summaries(
    variance: torch.Tensor,
    *,
    near_steps: int = 4,
    risk_steps: int = 8,
    continuous_dims: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if variance.ndim != 3 or variance.shape[1] <= 0 or variance.shape[2] <= 0:
        raise ValueError("variance must have shape [batch, horizon, action_dim]")
    if not torch.isfinite(variance).all() or torch.any(variance <= 0):
        raise ValueError("variance must be finite and strictly positive")
    horizon, action_dim = variance.shape[1:]
    if continuous_dims is None:
        continuous_dims = max(1, action_dim - 1)
    if not 1 <= continuous_dims <= action_dim:
        raise ValueError("continuous_dims is outside the action dimension")
    near_count = min(near_steps, horizon)
    risk_count = min(risk_steps, horizon)
    continuous = variance[:, :, :continuous_dims]
    near = continuous[:, :near_count].mean(dim=(1, 2))
    if near_count < horizon:
        future = continuous[:, near_count:].mean(dim=(1, 2))
    else:
        future = near.clone()
    p95 = torch.quantile(continuous[:, :risk_count].flatten(start_dim=1), 0.95, dim=1)
    combined = 0.7 * near + 0.3 * p95
    return (near, future, p95, combined)


def uncertain_act_loss(
    output: UncertainACTOutput,
    target_actions: torch.Tensor,
    *,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    continuous_mean_weight: float = 1.0,
    variance_weight: float = 0.1,
    gripper_weight: float = 1.0,
    huber_delta: float = 1.0,
    continuous_dims: int | None = None,
    min_logvar: float = -8.0,
    max_logvar: float = 2.0,
) -> UncertainACTLoss:
    if min(continuous_mean_weight, variance_weight, gripper_weight) < 0:
        raise ValueError("uncertain ACT loss weights must be non-negative")
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    if target_actions.shape != output.mean.shape:
        raise ValueError("target and uncertain ACT output shapes must match")
    normalized_target = (target_actions - action_mean) / action_std
    if valid_mask is None:
        mask = torch.ones(
            target_actions.shape[:2], dtype=torch.bool, device=target_actions.device
        )
    else:
        mask = valid_mask.to(device=target_actions.device, dtype=torch.bool)
        if mask.shape != target_actions.shape[:2]:
            raise ValueError("valid_mask must have shape [batch, horizon]")
    if not mask.any():
        raise ValueError("valid_mask must retain at least one action element")
    action_dim = target_actions.shape[-1]
    if continuous_dims is None:
        continuous_dims = action_dim - 1
    if not 1 <= continuous_dims < action_dim:
        raise ValueError("continuous_dims must leave at least one gripper dimension")
    residual = normalized_target - output.normalized_mean
    continuous_mask = mask.unsqueeze(-1).expand(*mask.shape, continuous_dims)
    gripper_dims = action_dim - continuous_dims
    gripper_mask = mask.unsqueeze(-1).expand(*mask.shape, gripper_dims)
    continuous_residual = residual[..., :continuous_dims]
    continuous_logvar = output.logvar[..., :continuous_dims]
    elementwise_mean_huber = F.huber_loss(
        output.normalized_mean[..., :continuous_dims],
        normalized_target[..., :continuous_dims],
        reduction="none",
        delta=huber_delta,
    )
    elementwise_variance_nll = 0.5 * (
        continuous_residual.detach().square() * torch.exp(-continuous_logvar)
        + continuous_logvar
    )
    elementwise_gripper_huber = F.huber_loss(
        output.normalized_mean[..., continuous_dims:],
        normalized_target[..., continuous_dims:],
        reduction="none",
        delta=huber_delta,
    )
    mean_huber = elementwise_mean_huber[continuous_mask].mean()
    variance_nll = elementwise_variance_nll[continuous_mask].mean()
    gripper_huber = elementwise_gripper_huber[gripper_mask].mean()
    selected_logvar = continuous_logvar[continuous_mask]
    tolerance = 1e-06
    return UncertainACTLoss(
        loss=continuous_mean_weight * mean_huber
        + variance_weight * variance_nll
        + gripper_weight * gripper_huber,
        continuous_mean_huber=mean_huber,
        variance_nll=variance_nll,
        gripper_huber=gripper_huber,
        mean_logvar=selected_logvar.mean(),
        lower_clamp_fraction=(selected_logvar <= min_logvar + tolerance).float().mean(),
        upper_clamp_fraction=(selected_logvar >= max_logvar - tolerance).float().mean(),
    )
