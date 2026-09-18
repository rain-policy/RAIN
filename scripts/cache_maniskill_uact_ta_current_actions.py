from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_observations(value: np.ndarray) -> np.ndarray:
    if value.ndim == 2:
        return value
    if value.ndim == 3 and value.shape[1] == 1:
        return value[:, 0]
    raise RuntimeError("state observations have an invalid shape")


def observation_windows(observations: np.ndarray, horizon: int = 2) -> np.ndarray:
    windows = []
    for timestep in range(observations.shape[0] - 1):
        start = max(0, timestep - horizon + 1)
        window = observations[start : timestep + 1]
        if window.shape[0] < horizon:
            window = np.concatenate(
                [np.repeat(window[:1], horizon - window.shape[0], axis=0), window],
                axis=0,
            )
        windows.append(window)
    return np.asarray(windows, dtype=np.float32)


def official_ta(chunks: torch.Tensor, aggregation_k: float) -> torch.Tensor:
    if chunks.ndim != 3 or chunks.shape[0] <= 0 or chunks.shape[1] <= 0:
        raise ValueError("chunks must have shape [time,horizon,action_dim]")
    time_count, horizon, _ = chunks.shape
    aggregated = []
    for timestep in range(time_count):
        start = max(0, timestep + 1 - horizon)
        query_times = torch.arange(start, timestep + 1, device=chunks.device)
        offsets = timestep - query_times
        candidates = chunks[query_times, offsets]
        weights = torch.exp(
            -aggregation_k
            * torch.arange(
                candidates.shape[0], device=chunks.device, dtype=chunks.dtype
            )
        )
        aggregated.append((candidates * (weights / weights.sum())[:, None]).sum(0))
    return torch.stack(aggregated)
