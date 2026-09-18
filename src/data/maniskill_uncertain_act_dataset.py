from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrajectorySplits:
    train: tuple[int, ...]
    calibration: tuple[int, ...]
    test: tuple[int, ...]

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "calibration": list(self.calibration),
            "test": list(self.test),
        }


def make_trajectory_splits(
    num_trajectories: int,
    *,
    seed: int,
    train_fraction: float = 0.8,
    calibration_fraction: float = 0.1,
) -> TrajectorySplits:
    if num_trajectories < 3:
        raise ValueError("at least three trajectories are required")
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("split fractions must lie in (0, 1)")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must leave a test split")
    permutation = np.random.default_rng(seed).permutation(num_trajectories)
    train_end = int(np.floor(train_fraction * num_trajectories))
    calibration_end = train_end + int(np.floor(calibration_fraction * num_trajectories))
    if (
        train_end == 0
        or calibration_end == train_end
        or calibration_end == num_trajectories
    ):
        raise ValueError("split fractions produced an empty partition")
    return TrajectorySplits(
        train=tuple(sorted((int(value) for value in permutation[:train_end]))),
        calibration=tuple(
            sorted((int(value) for value in permutation[train_end:calibration_end]))
        ),
        test=tuple(sorted((int(value) for value in permutation[calibration_end:]))),
    )


def _unbatch_state(observations: np.ndarray) -> np.ndarray:
    if observations.ndim == 2:
        return observations
    if observations.ndim == 3 and observations.shape[1] == 1:
        return observations[:, 0]
    raise RuntimeError(
        "state observations must have shape [time, dim] or [time, 1, dim]"
    )


class UncertainACTTrajectoryDataset(Dataset):
    """Two-frame observations and masked 30-query action targets."""

    def __init__(
        self,
        demo_path: str | Path,
        trajectory_indices: tuple[int, ...] | list[int],
        *,
        obs_horizon: int = 2,
        num_queries: int = 30,
        control_mode: str,
    ) -> None:
        if obs_horizon <= 0 or num_queries <= 0:
            raise ValueError("observation horizon and query count must be positive")
        selected = tuple((int(index) for index in trajectory_indices))
        if not selected or len(set(selected)) != len(selected) or min(selected) < 0:
            raise ValueError("trajectory_indices must be unique non-negative indices")
        self.trajectory_indices = selected
        self.obs_horizon = obs_horizon
        self.num_queries = num_queries
        self.control_mode = control_mode
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        with h5py.File(Path(demo_path).expanduser(), "r") as handle:
            available = {
                int(key.rsplit("_", 1)[1]): key
                for key in handle
                if key.startswith("traj_")
            }
            missing = sorted(set(selected) - set(available))
            if missing:
                raise RuntimeError(
                    f"trajectory indices are missing from the demo: {missing[:5]}"
                )
            for original_index in selected:
                group = handle[available[original_index]]
                observations = _unbatch_state(
                    np.asarray(group["obs"], dtype=np.float32)
                )
                actions = np.asarray(group["actions"], dtype=np.float32)
                if observations.shape[0] != actions.shape[0] + 1:
                    raise RuntimeError(
                        "each trajectory must contain one extra observation"
                    )
                self.observations.append(observations)
                self.actions.append(actions)
        self.indices = [
            (local_index, timestep)
            for (local_index, actions) in enumerate(self.actions)
            for timestep in range(actions.shape[0])
        ]

    @property
    def obs_dim(self) -> int:
        return int(self.observations[0].shape[-1])

    @property
    def act_dim(self) -> int:
        return int(self.actions[0].shape[-1])

    def norm_stats(self) -> dict[str, np.ndarray]:
        observations = np.concatenate(self.observations, axis=0)
        actions = np.concatenate(self.actions, axis=0)
        return {
            "obs_mean": observations.mean(axis=0),
            "obs_std": np.maximum(observations.std(axis=0), 1e-06),
            "action_mean": actions.mean(axis=0),
            "action_std": np.maximum(actions.std(axis=0), 1e-06),
        }

    def __len__(self) -> int:
        return len(self.indices)

    def _observation_window(
        self, observations: np.ndarray, timestep: int
    ) -> np.ndarray:
        start = max(0, timestep - self.obs_horizon + 1)
        window = observations[start : timestep + 1]
        missing = self.obs_horizon - window.shape[0]
        if missing:
            window = np.concatenate(
                [np.repeat(window[:1], missing, axis=0), window], axis=0
            )
        return window

    def _action_target(
        self, actions: np.ndarray, timestep: int
    ) -> tuple[np.ndarray, np.ndarray]:
        target = actions[timestep : timestep + self.num_queries]
        valid = np.zeros((self.num_queries,), dtype=np.bool_)
        valid[: target.shape[0]] = True
        missing = self.num_queries - target.shape[0]
        if missing:
            if "delta_pos" in self.control_mode:
                pad = np.zeros((actions.shape[-1],), dtype=np.float32)
                pad[-1] = target[-1, -1]
            else:
                pad = target[-1]
            target = np.concatenate(
                [target, np.repeat(pad[None], missing, axis=0)], axis=0
            )
        return (target, valid)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        local_index, timestep = self.indices[index]
        observation = self._observation_window(self.observations[local_index], timestep)
        actions, valid = self._action_target(self.actions[local_index], timestep)
        return {
            "observations": torch.from_numpy(observation.copy()),
            "actions": torch.from_numpy(actions.copy()),
            "action_valid": torch.from_numpy(valid),
            "trajectory_index": torch.tensor(
                self.trajectory_indices[local_index], dtype=torch.long
            ),
            "timestep": torch.tensor(timestep, dtype=torch.long),
        }
