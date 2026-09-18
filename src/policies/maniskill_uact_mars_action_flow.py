from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
import torch.nn as nn

UACT_MARS_ACTION_FLOW_METHOD = "uact_mars_action_space_dit_flow_v1"


@dataclass(frozen=True)
class MARSActionFlowConfig:
    obs_dim: int
    act_dim: int = 7
    arm_dim: int = 6
    obs_horizon: int = 2
    chunk_horizon: int = 8
    hidden_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    max_steps: int = 10

    def validate(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("MARS action Flow dimensions must be positive")
        if self.arm_dim >= self.act_dim:
            raise ValueError("arm_dim must leave at least one frozen action dimension")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even for sinusoidal time embedding")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension <= 0 or dimension % 2:
            raise ValueError("time embedding dimension must be positive and even")
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("time must have shape [batch]")
        half = self.dimension // 2
        exponent = torch.arange(half, device=time.device, dtype=time.dtype)
        exponent = -math.log(10000.0) * exponent / max(1, half - 1)
        angles = time[:, None] * exponent.exp()[None] * 100.0
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class DeterministicSelfAttention(nn.Module):
    """Small explicit self-attention used to avoid backend-dependent SDP kernels."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1
        )
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, tokens, hidden)
        return self.proj(output)


class DiTBlock(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-06)
        self.attention = DeterministicSelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-06)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * hidden_dim, hidden_dim),
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada_ln(
            condition
        ).chunk(6, dim=-1)
        x = x + gate_a[:, None] * self.attention(
            _modulate(self.norm1(x), shift_a, scale_a)
        )
        x = x + gate_m[:, None] * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))
        return x


class MARSActionDiT(nn.Module):
    """DiT velocity field over raw action-space tokens, conditioned on state and ACT source."""

    def __init__(self, config: MARSActionFlowConfig) -> None:
        super().__init__()
        self.config = config
        condition_input = (
            config.obs_horizon * config.obs_dim + config.chunk_horizon * config.arm_dim
        )
        self.action_projection = nn.Linear(config.arm_dim, config.hidden_dim)
        self.position = nn.Parameter(
            torch.zeros(1, config.chunk_horizon, config.hidden_dim)
        )
        nn.init.normal_(self.position, std=0.02)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_input, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.hidden_dim, config.num_heads, config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_dim, elementwise_affine=False, eps=1e-06
        )
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_dim, 2 * config.hidden_dim)
        )
        self.output = nn.Linear(config.hidden_dim, config.arm_dim)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode_static_condition(
        self, normalized_observations: torch.Tensor, source_arm: torch.Tensor
    ) -> torch.Tensor:
        """Encode observation/source terms that remain fixed across Euler NFEs."""
        batch = source_arm.shape[0]
        expected = (batch, self.config.chunk_horizon, self.config.arm_dim)
        if source_arm.shape != expected:
            raise ValueError("DiT source shape differs from the registered chunk")
        if normalized_observations.shape != (
            batch,
            self.config.obs_horizon,
            self.config.obs_dim,
        ):
            raise ValueError("DiT observations differ from the registered horizon")
        condition_input = torch.cat(
            [
                normalized_observations.flatten(start_dim=1),
                source_arm.flatten(start_dim=1),
            ],
            dim=-1,
        )
        return self.condition_encoder(condition_input)

    def forward_with_static_condition(
        self, sample: torch.Tensor, time: torch.Tensor, static_condition: torch.Tensor
    ) -> torch.Tensor:
        batch = sample.shape[0]
        expected = (batch, self.config.chunk_horizon, self.config.arm_dim)
        if sample.shape != expected:
            raise ValueError(
                "DiT action sample shape differs from the registered chunk"
            )
        if time.shape != (batch,):
            raise ValueError("DiT flow time must have shape [batch]")
        if static_condition.shape != (batch, self.config.hidden_dim):
            raise ValueError("DiT static condition has an invalid shape")
        condition = static_condition + self.time_encoder(time)
        x = self.action_projection(sample) + self.position
        for block in self.blocks:
            x = block(x, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        return self.output(_modulate(self.final_norm(x), shift, scale))

    def forward(
        self,
        sample: torch.Tensor,
        time: torch.Tensor,
        normalized_observations: torch.Tensor,
        source_arm: torch.Tensor,
    ) -> torch.Tensor:
        static_condition = self.encode_static_condition(
            normalized_observations, source_arm
        )
        return self.forward_with_static_condition(sample, time, static_condition)


class UACTMARSActionFlow(nn.Module):

    def __init__(
        self,
        config: MARSActionFlowConfig,
        *,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.register_buffer("obs_mean", self._stat(obs_mean, config.obs_dim, False))
        self.register_buffer("obs_std", self._stat(obs_std, config.obs_dim, True))
        self.register_buffer(
            "action_mean", self._stat(action_mean, config.act_dim, False)
        )
        self.register_buffer("action_std", self._stat(action_std, config.act_dim, True))
        self.velocity_net = MARSActionDiT(config)

    @staticmethod
    def _stat(value: torch.Tensor, size: int, positive: bool) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32).clone()
        if tensor.shape != (size,) or not torch.isfinite(tensor).all():
            raise ValueError("normalization statistic has invalid shape or values")
        if positive and torch.any(tensor <= 0):
            raise ValueError("normalization standard deviations must be positive")
        return tensor

    def normalize_arm(self, actions: torch.Tensor) -> torch.Tensor:
        return (
            actions[..., : self.config.arm_dim]
            - self.action_mean[: self.config.arm_dim]
        ) / self.action_std[: self.config.arm_dim]

    def denormalize_arm(self, arm: torch.Tensor) -> torch.Tensor:
        return (
            arm * self.action_std[: self.config.arm_dim]
            + self.action_mean[: self.config.arm_dim]
        )

    def normalized_observations(self, observations: torch.Tensor) -> torch.Tensor:
        expected = (observations.shape[0], self.config.obs_horizon, self.config.obs_dim)
        if observations.shape != expected or not torch.isfinite(observations).all():
            raise ValueError("observations do not match MARS action Flow")
        return (observations - self.obs_mean) / self.obs_std

    def velocity(
        self,
        observations: torch.Tensor,
        source_arm: torch.Tensor,
        sample_arm: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if not all(
            (torch.isfinite(value).all() for value in (source_arm, sample_arm, time))
        ):
            raise ValueError("MARS action Flow inputs must be finite")
        if torch.any((time < 0) | (time > 1)):
            raise ValueError("flow time must lie in [0,1]")
        return self.velocity_net(
            sample_arm, time, self.normalized_observations(observations), source_arm
        )

    def unroll_normalized(
        self, observations: torch.Tensor, source_arm: torch.Tensor, *, steps: int
    ) -> list[torch.Tensor]:
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or (not 0 <= steps <= self.config.max_steps)
        ):
            raise ValueError(f"steps must be an integer in [0,{self.config.max_steps}]")
        normalized_observations = self.normalized_observations(observations)
        static_condition = self.velocity_net.encode_static_condition(
            normalized_observations, source_arm
        )
        times = (
            torch.arange(steps, device=source_arm.device, dtype=source_arm.dtype)
            / steps
        )
        states = [source_arm]
        sample = source_arm
        for index in range(steps):
            time = times[index].expand(sample.shape[0])
            velocity = self.velocity_net.forward_with_static_condition(
                sample, time, static_condition
            )
            sample = sample + velocity / steps
            states.append(sample)
        return states

    def refine_chunk(
        self, observations: torch.Tensor, source_chunk: torch.Tensor, *, steps: int
    ) -> torch.Tensor:
        expected = (
            observations.shape[0],
            self.config.chunk_horizon,
            self.config.act_dim,
        )
        if source_chunk.shape != expected or not torch.isfinite(source_chunk).all():
            raise ValueError("source chunk does not match MARS action Flow")
        if steps == 0:
            return source_chunk
        source_arm = self.normalize_arm(source_chunk)
        refined_arm = self.denormalize_arm(
            self.unroll_normalized(observations, source_arm, steps=steps)[-1]
        )
        refined = torch.cat(
            [refined_arm, source_chunk[..., self.config.arm_dim :]], dim=-1
        )
        if not torch.equal(
            refined[..., self.config.arm_dim :],
            source_chunk[..., self.config.arm_dim :],
        ):
            raise RuntimeError("MARS action Flow changed the frozen gripper")
        return refined

    def flow_matching_loss(
        self,
        observations: torch.Tensor,
        source_chunk: torch.Tensor,
        expert_chunk: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Standard conditional FM from frozen ACT chunks to expert chunks.

        This objective is intentionally independent of deployment integration
        count: no C label, route, or unrolled endpoint enters optimization.
        """
        batch = observations.shape[0]
        expected = (batch, self.config.chunk_horizon, self.config.act_dim)
        if source_chunk.shape != expected or expert_chunk.shape != expected:
            raise ValueError("training chunks do not match MARS action Flow")
        if valid_mask.shape != (batch, self.config.chunk_horizon):
            raise ValueError("valid mask does not match the action chunk")
        source_arm = self.normalize_arm(source_chunk.detach())
        expert_arm = self.normalize_arm(expert_chunk)
        if time is None:
            time = torch.rand(
                batch, device=observations.device, dtype=observations.dtype
            )
        if (
            time.shape != (batch,)
            or not torch.isfinite(time).all()
            or torch.any((time < 0) | (time > 1))
        ):
            raise ValueError("flow-matching time must have shape [batch] in [0,1]")
        path = (1 - time[:, None, None]) * source_arm + time[:, None, None] * expert_arm
        target = expert_arm - source_arm
        prediction = self.velocity(observations, source_arm, path, time)
        arm_mask = valid_mask.to(torch.bool)[:, :, None].expand_as(prediction)
        if not arm_mask.any():
            raise ValueError("flow-matching batch has no valid arm targets")
        error = prediction - target
        flow_matching = error.square()[arm_mask].mean()
        return {
            "loss": flow_matching,
            "flow_matching": flow_matching,
            "target_rms": target.square()[arm_mask].mean().sqrt(),
            "prediction_rms": prediction.square()[arm_mask].mean().sqrt(),
        }

    def routed_training_loss(
        self,
        observations: torch.Tensor,
        source_chunk: torch.Tensor,
        expert_chunk: torch.Tensor,
        valid_mask: torch.Tensor,
        route_c: torch.Tensor,
        *,
        flow_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = observations.shape[0]
        expected = (batch, self.config.chunk_horizon, self.config.act_dim)
        if source_chunk.shape != expected or expert_chunk.shape != expected:
            raise ValueError("training chunks do not match MARS action Flow")
        if valid_mask.shape != (batch, self.config.chunk_horizon):
            raise ValueError("valid mask does not match the action chunk")
        if route_c.shape != (batch,) or torch.any(
            (route_c < 0) | (route_c > self.config.max_steps)
        ):
            raise ValueError("route C does not match C0/C1/C2")
        if min(flow_weight, reconstruction_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        source_arm = self.normalize_arm(source_chunk.detach())
        expert_arm = self.normalize_arm(expert_chunk)
        if time is None:
            time = torch.rand(
                batch, device=observations.device, dtype=observations.dtype
            )
        path = (1 - time[:, None, None]) * source_arm + time[:, None, None] * expert_arm
        target = expert_arm - source_arm
        prediction = self.velocity(observations, source_arm, path, time)
        arm_mask = valid_mask.to(torch.bool)[:, :, None].expand_as(prediction)
        flow_matching = (prediction - target).square()[arm_mask].mean()
        endpoint = source_arm
        for steps in (1, 2):
            mask = route_c == steps
            if mask.any():
                candidate = self.unroll_normalized(
                    observations, source_arm, steps=steps
                )[-1]
                endpoint = torch.where(mask[:, None, None], candidate, endpoint)
        active = route_c > 0
        active_mask = active[:, None, None] & arm_mask
        reconstruction = (
            (endpoint - expert_arm).abs()[active_mask].mean()
            if active_mask.any()
            else flow_matching.new_zeros(())
        )
        all_endpoint = (endpoint - expert_arm).abs()[arm_mask].mean()
        return {
            "loss": flow_weight * flow_matching
            + reconstruction_weight * reconstruction,
            "flow_matching": flow_matching,
            "routed_reconstruction_active": reconstruction,
            "routed_reconstruction_all": all_endpoint,
            "active_fraction": active.float().mean(),
            "mean_c": route_c.float().mean(),
        }


def build_causal_source_chunk(
    ta_current: torch.Tensor, act_proposal: torch.Tensor, *, chunk_horizon: int
) -> torch.Tensor:
    """Build [TA current, current-query future proposal] without future observations."""
    if ta_current.ndim != 2 or act_proposal.ndim != 3:
        raise ValueError("TA current action and ACT proposal have invalid ranks")
    if (
        ta_current.shape[0] != act_proposal.shape[0]
        or ta_current.shape[1] != act_proposal.shape[2]
    ):
        raise ValueError("TA current action and ACT proposal dimensions disagree")
    if chunk_horizon < 1 or act_proposal.shape[1] < chunk_horizon:
        raise ValueError("ACT proposal does not cover the requested source chunk")
    return torch.cat([ta_current[:, None], act_proposal[:, 1:chunk_horizon]], dim=1)


def fill_causal_source_chunk(
    output: torch.Tensor, ta_current: torch.Tensor, act_proposal: torch.Tensor
) -> torch.Tensor:
    """Allocation-free equivalent of ``build_causal_source_chunk``."""
    if output.ndim != 3 or ta_current.ndim != 2 or act_proposal.ndim != 3:
        raise ValueError("causal source tensors have invalid ranks")
    batch, chunk_horizon, action_dim = output.shape
    if ta_current.shape != (batch, action_dim):
        raise ValueError("TA current action does not match output buffer")
    if act_proposal.shape[0] != batch or act_proposal.shape[2] != action_dim:
        raise ValueError("ACT proposal does not match output buffer")
    if act_proposal.shape[1] < chunk_horizon:
        raise ValueError("ACT proposal does not cover the output horizon")
    output[:, 0].copy_(ta_current)
    if chunk_horizon > 1:
        output[:, 1:].copy_(act_proposal[:, 1:chunk_horizon])
    return output
