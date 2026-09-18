from __future__ import annotations
import torch


def aggregate_uact_chunk(
    storage: torch.Tensor,
    chunk: torch.Tensor,
    *,
    timestep: int,
    aggregation_k: float,
    weight_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Official ACT TA over raw U-ACT chunks; refined actions never enter storage."""
    if chunk.ndim != 2 or storage.ndim != 3:
        raise ValueError("TA storage and chunk must have ranks 3 and 2")
    if not 0 <= timestep < storage.shape[0] or aggregation_k <= 0:
        raise ValueError("TA timestep and aggregation_k are invalid")
    horizon = chunk.shape[0]
    if storage.shape[2] != chunk.shape[1] or timestep + horizon > storage.shape[1]:
        raise ValueError("TA storage does not fit the U-ACT chunk")
    storage[timestep, timestep : timestep + horizon] = chunk
    start = max(0, timestep + 1 - horizon)
    candidates = storage[start : timestep + 1, timestep]
    if weight_table is None:
        weights = torch.exp(
            -aggregation_k
            * torch.arange(candidates.shape[0], device=chunk.device, dtype=chunk.dtype)
        )
        weights = weights / weights.sum()
    else:
        if (
            weight_table.ndim != 2
            or weight_table.shape[0] < horizon
            or weight_table.shape[1] < horizon
            or (weight_table.device != chunk.device)
            or (weight_table.dtype != chunk.dtype)
        ):
            raise ValueError("TA weight table does not match the action chunk")
        weights = weight_table[candidates.shape[0] - 1, : candidates.shape[0]]
    return (candidates * weights[:, None]).sum(dim=0)


def temporal_aggregation_weight_table(
    horizon: int,
    aggregation_k: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if horizon <= 0 or aggregation_k <= 0:
        raise ValueError("TA horizon and aggregation k must be positive")
    table = torch.zeros(horizon, horizon, device=device, dtype=dtype)
    for count in range(1, horizon + 1):
        weights = torch.exp(
            -aggregation_k * torch.arange(count, device=device, dtype=dtype)
        )
        table[count - 1, :count] = weights / weights.sum()
    return table
