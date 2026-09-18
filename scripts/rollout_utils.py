from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def numpy_value(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_bool(value) -> bool:
    if value is None:
        return False
    return bool(numpy_value(value).reshape(-1)[0])


def scalar_float(value) -> float:
    return float(numpy_value(value).reshape(-1)[0])


def state_vector(observation) -> np.ndarray:
    value = numpy_value(observation).astype(np.float32, copy=False)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1 or not np.isfinite(value).all():
        raise RuntimeError(f"invalid state observation {value.shape}")
    return value


def current_info(info: dict) -> dict:
    final = info.get("final_info")
    return final if isinstance(final, dict) else info
