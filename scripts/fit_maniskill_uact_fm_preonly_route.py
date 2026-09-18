from __future__ import annotations
import numpy as np
from scripts import fit_peg_uact_fm_fully_learned_route as base

METHOD = "peg_uact_fm_preonly_route_v1"


def choose_preonly(
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    beta: float,
    compute_price: float,
    trajectory: np.ndarray | None = None,
    timestep: np.ndarray | None = None,
    transition_probability: np.ndarray | None = None,
    transition_strength: float = 0.0,
) -> np.ndarray:
    """Choose C0--C3 directly from the pre-FM loss distribution."""
    uact_mean, uact_std, _, _ = predictions
    costs = np.asarray([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float64)
    risk = uact_mean + beta * uact_std + compute_price * costs[None]
    if transition_probability is None or transition_strength == 0:
        return np.argmin(risk, axis=1).astype(np.int64)
    if trajectory is None or timestep is None:
        raise ValueError("sequence route requires trajectory and timestep")
    selected = np.empty(len(uact_mean), dtype=np.int64)
    for trajectory_id in np.unique(trajectory):
        positions = np.flatnonzero(trajectory == trajectory_id)
        positions = positions[np.argsort(timestep[positions])]
        cursor = 0
        previous = None
        previous_horizon = None
        while cursor < len(positions):
            index = positions[cursor]
            value = risk[index].copy()
            if previous is not None:
                transition = np.linalg.matrix_power(
                    transition_probability, previous_horizon
                )[previous]
                value += transition_strength * -np.log(np.clip(transition, 1e-09, 1.0))
            choice = int(np.argmin(value))
            horizon = 2 if choice == 0 else 8
            stop = min(cursor + horizon, len(positions))
            selected[positions[cursor:stop]] = choice
            previous = choice
            previous_horizon = horizon
            cursor = stop
    return selected


def main() -> None:
    base.METHOD = METHOD
    base.choose = choose_preonly
    base.main()


if __name__ == "__main__":
    main()
