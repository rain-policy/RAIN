from __future__ import annotations
import torch

PEG_JOINT_FM_METHOD = "peg_uact25_mars25_joint50_diagnostic_v1"
PEG_JOINT_UACT_STAGE_FM = "peg_uact25_fm25_joint50_uact"
MANISKILL_JOINT_FM_METHOD = "maniskill_uact25_mars25_joint50_diagnostic_v1"
MANISKILL_JOINT_UACT_STAGE_FM = "maniskill_uact25_fm25_joint50_uact"


def differentiable_official_temporal_aggregation(
    proposals: torch.Tensor, *, aggregation_k: float = 0.01
) -> torch.Tensor:
    """Return ManiSkill ACT temporal aggregation for one complete trajectory.

    ``proposals[q, d]`` predicts the action for environment time ``q + d``.
    The implementation exactly matches ManiSkill's query-frequency-one
    evaluator, including its oldest-to-newest exponential weight ordering.
    It is vectorized and retains gradients to every contributing proposal.
    """
    if proposals.ndim != 3:
        raise ValueError("proposals must have shape [time,horizon,action_dim]")
    time_count, horizon, action_dim = proposals.shape
    if min(time_count, horizon, action_dim) <= 0 or aggregation_k <= 0:
        raise ValueError("TA dimensions and aggregation_k must be positive")
    if not torch.isfinite(proposals).all():
        raise ValueError("TA proposals must be finite")
    target_time = torch.arange(time_count, device=proposals.device)[:, None]
    slot = torch.arange(horizon, device=proposals.device)[None, :]
    query_time = target_time - horizon + 1 + slot
    valid = query_time >= 0
    safe_query = query_time.clamp(min=0)
    offset = (target_time - safe_query).clamp(max=horizon - 1)
    candidates = proposals[safe_query, offset]
    oldest = (target_time - horizon + 1).clamp(min=0)
    rank = query_time - oldest
    weights = torch.exp(-aggregation_k * rank.to(proposals.dtype))
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (candidates * weights[..., None]).sum(dim=1)


def build_differentiable_causal_sources(
    proposals: torch.Tensor,
    *,
    chunk_horizon: int = 8,
    aggregation_k: float = 0.01,
    clip_base_action: bool = True,
) -> torch.Tensor:
    """Build [TA current, current-query future actions] with source gradients."""
    if chunk_horizon <= 0 or proposals.shape[1] < chunk_horizon:
        raise ValueError("ACT proposal does not cover the refiner horizon")
    base = differentiable_official_temporal_aggregation(
        proposals, aggregation_k=aggregation_k
    )
    if clip_base_action:
        base = base.clamp(-1, 1)
    return torch.cat([base[:, None], proposals[:, 1:chunk_horizon]], dim=1)


def flow_matching_loss_with_source_gradient(
    model,
    observations: torch.Tensor,
    source_chunk: torch.Tensor,
    expert_chunk: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    time: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Standard conditional FM objective without detaching the ACT source."""
    batch = observations.shape[0]
    expected = (batch, model.config.chunk_horizon, model.config.act_dim)
    if source_chunk.shape != expected or expert_chunk.shape != expected:
        raise ValueError("joint FM chunks do not match the registered model")
    if valid_mask.shape != (batch, model.config.chunk_horizon):
        raise ValueError("joint FM valid mask has an invalid shape")
    source_arm = model.normalize_arm(source_chunk)
    expert_arm = model.normalize_arm(expert_chunk)
    if time is None:
        time = torch.rand(batch, device=observations.device, dtype=observations.dtype)
    if (
        time.shape != (batch,)
        or not torch.isfinite(time).all()
        or torch.any((time < 0) | (time > 1))
    ):
        raise ValueError("joint FM time must have shape [batch] in [0,1]")
    path = (1 - time[:, None, None]) * source_arm + time[:, None, None] * expert_arm
    target = expert_arm - source_arm
    prediction = model.velocity(observations, source_arm, path, time)
    arm_mask = valid_mask.to(torch.bool)[:, :, None].expand_as(prediction)
    if not arm_mask.any():
        raise ValueError("joint FM batch contains no valid targets")
    loss = (prediction - target).square()[arm_mask].mean()
    return {
        "loss": loss,
        "flow_matching": loss,
        "target_rms": target.square()[arm_mask].mean().sqrt(),
        "prediction_rms": prediction.square()[arm_mask].mean().sqrt(),
    }


def normalized_joint_loss(
    refiner_loss: torch.Tensor,
    act_loss: torch.Tensor,
    *,
    refiner_scale: float,
    act_scale: float,
    act_weight: float = 0.5,
) -> torch.Tensor:
    """Combine frozen, positive loss scales under the registered objective."""
    if min(refiner_scale, act_scale) <= 0 or act_weight < 0:
        raise ValueError("joint loss scales must be positive")
    return refiner_loss / refiner_scale + act_weight * act_loss / act_scale
