from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from src.policies.maniskill_uncertain_act import (
    ATTACHED_VARIANCE_ACT_METHOD,
    LEGACY_UNCERTAIN_ACT_METHOD,
    UNCERTAIN_ACT_METHOD,
    CleanUncertainACT,
    UncertainACTConfig,
)


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_uact(
    checkpoint: str | Path, *, device: torch.device | str, state_key: str = "ema_agent"
) -> tuple[CleanUncertainACT, dict]:
    path = Path(checkpoint).expanduser().resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    method = state.get("method")
    accepted_methods = {
        UNCERTAIN_ACT_METHOD,
        ATTACHED_VARIANCE_ACT_METHOD,
        LEGACY_UNCERTAIN_ACT_METHOD,
    }
    if method not in accepted_methods:
        raise RuntimeError("checkpoint is not a clean uncertain ACT")
    if state_key not in state:
        raise RuntimeError(f"U-ACT checkpoint does not contain {state_key!r}")
    model_config = dict(state["model_config"])
    if method == LEGACY_UNCERTAIN_ACT_METHOD:
        model_config.setdefault("variance_detach_features", False)
    config = UncertainACTConfig(**model_config)
    if method == UNCERTAIN_ACT_METHOD and (not config.variance_detach_features):
        raise RuntimeError("variance-isolated U-ACT checkpoint has attached features")
    if method == ATTACHED_VARIANCE_ACT_METHOD and config.variance_detach_features:
        raise RuntimeError("attached-variance U-ACT checkpoint has detached features")
    model = CleanUncertainACT(config)
    model.load_state_dict(state[state_key], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return (
        model,
        {
            "path": str(path),
            "sha256": checkpoint_sha256(path),
            "state_key": state_key,
            "method": method,
            "stage": state.get("stage"),
            "iteration": int(state["iteration"]),
            "model_config": config.to_dict(),
        },
    )
