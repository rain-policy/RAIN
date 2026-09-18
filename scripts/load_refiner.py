from __future__ import annotations
from pathlib import Path
import torch
from src.policies.maniskill_uact_mars_action_flow import (
    MARSActionFlowConfig,
    UACTMARSActionFlow,
    UACT_MARS_ACTION_FLOW_METHOD,
)

EXECUTION_HORIZON = 8


def load_fixed_flow(
    checkpoint: Path, device: torch.device, *, allow_incomplete: bool = False
) -> tuple[UACTMARSActionFlow, dict]:
    """Load a completed standard-CFM checkpoint without a router contract."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("method") != UACT_MARS_ACTION_FLOW_METHOD:
        raise RuntimeError("checkpoint is not a MARS-style action-space Flow")
    if not state.get("training_complete") and (not allow_incomplete):
        raise RuntimeError("MARS-style action Flow training is incomplete")
    if not state.get("training_complete") and int(state.get("iteration", 0)) <= 0:
        raise RuntimeError(
            "incomplete MARS-style checkpoint has no valid training step"
        )
    if state.get("training_objective") != "standard_conditional_flow_matching":
        raise RuntimeError(
            "fixed chunk evaluation requires the standard CFM checkpoint"
        )
    weights = state["ema_agent"]
    model = UACTMARSActionFlow(
        MARSActionFlowConfig(**state["model_config"]),
        obs_mean=weights["obs_mean"],
        obs_std=weights["obs_std"],
        action_mean=weights["action_mean"],
        action_std=weights["action_std"],
    )
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return (model, state)
