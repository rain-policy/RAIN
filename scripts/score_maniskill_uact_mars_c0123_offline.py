from __future__ import annotations
import hashlib
from pathlib import Path
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def closed_loop_ta_chunks(partition: dict, horizon: int) -> torch.Tensor:
    """Gather eight consecutive already-aggregated ACT actions per trajectory."""
    base = partition["base_actions"]
    count = len(base)
    indices = torch.arange(count)[:, None] + torch.arange(horizon)[None]
    indices = indices.clamp(max=count - 1)
    valid = partition["valid_mask"]
    trajectory = partition["trajectory_index"]
    expected = trajectory[:, None].expand(-1, horizon)
    if not torch.equal(trajectory[indices][valid], expected[valid]):
        raise RuntimeError("valid closed-loop TA chunk crossed a trajectory boundary")
    return base[indices]
