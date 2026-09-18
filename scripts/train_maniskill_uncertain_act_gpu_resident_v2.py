from __future__ import annotations
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from scripts import train_maniskill_uncertain_act as trainer


class _IndexDataset(Dataset):

    def __init__(self, length: int) -> None:
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return int(index)


_RESIDENT_DATASETS: dict[tuple, dict[str, torch.Tensor]] = {}


def _dataset_key(dataset) -> tuple:
    if isinstance(dataset, Subset):
        indices = dataset.indices
        if isinstance(indices, range):
            signature = (indices.start, indices.stop, indices.step)
        else:
            signature = tuple((int(value) for value in indices))
        return ("subset", id(dataset.dataset), signature)
    return ("dataset", id(dataset), len(dataset))


def _materialize(dataset, device: torch.device) -> dict[str, torch.Tensor]:
    key = _dataset_key(dataset)
    cached = _RESIDENT_DATASETS.get(key)
    if cached is not None:
        return cached
    cpu_rng = torch.get_rng_state()
    try:
        sequential = TorchDataLoader(
            dataset, batch_size=4096, shuffle=False, num_workers=0, pin_memory=False
        )
        chunks: dict[str, list[torch.Tensor]] = {}
        for batch in sequential:
            for name, value in batch.items():
                chunks.setdefault(name, []).append(value)
        cpu_tensors = {
            name: torch.cat(values, dim=0).contiguous()
            for (name, values) in chunks.items()
        }
    finally:
        torch.set_rng_state(cpu_rng)
    if not cpu_tensors or any(
        (value.shape[0] != len(dataset) for value in cpu_tensors.values())
    ):
        raise RuntimeError("GPU-resident U-ACT materialization is incomplete")
    resident = {
        name: value.to(device, non_blocking=False)
        for (name, value) in cpu_tensors.items()
    }
    _RESIDENT_DATASETS[key] = resident
    return resident


class GPUResidentTrainingLoader:
    """Drop-in loader that retains legacy sampling and RNG behavior."""

    def __init__(self, dataset, *, batch_sampler) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU-resident U-ACT loading requires CUDA")
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.device = torch.device("cuda")
        self.tensors = _materialize(dataset, self.device)
        self.prefetch_stream = torch.cuda.Stream(device=self.device)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self):
        index_loader = TorchDataLoader(
            _IndexDataset(len(self.dataset)),
            batch_sampler=self.batch_sampler,
            num_workers=0,
            pin_memory=True,
        )
        iterator = iter(index_loader)

        def schedule(indices: torch.Tensor) -> dict[str, torch.Tensor]:
            with torch.cuda.stream(self.prefetch_stream):
                gpu_indices = indices.to(self.device, non_blocking=True)
                return {
                    name: value.index_select(0, gpu_indices)
                    for (name, value) in self.tensors.items()
                }

        try:
            pending = schedule(next(iterator))
        except StopIteration:
            return
        while True:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(self.prefetch_stream)
            for value in pending.values():
                value.record_stream(current_stream)
            try:
                following_indices = next(iterator)
            except StopIteration:
                yield pending
                break
            following = schedule(following_indices)
            yield pending
            pending = following


class GPUResidentSequentialLoader:
    """Exact sequential calibration loader backed by cached CUDA tensors."""

    def __init__(self, dataset, *, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.device = torch.device("cuda")
        self.tensors = _materialize(dataset, self.device)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = TorchDataLoader(
            _IndexDataset(len(self.dataset)),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        for batch_indices in indices:
            gpu_indices = batch_indices.to(self.device, non_blocking=False)
            yield {
                name: value.index_select(0, gpu_indices)
                for (name, value) in self.tensors.items()
            }


def _data_loader_dispatch(dataset, *args, **kwargs):
    batch_sampler = kwargs.get("batch_sampler")
    if batch_sampler is not None and kwargs.get("pin_memory", False):
        if args:
            raise RuntimeError("unexpected positional U-ACT DataLoader arguments")
        return GPUResidentTrainingLoader(dataset, batch_sampler=batch_sampler)
    batch_size = kwargs.get("batch_size", args[0] if args else None)
    if (
        torch.cuda.is_available()
        and batch_size is not None
        and (kwargs.get("shuffle", False) is False)
        and (kwargs.get("num_workers", 0) == 0)
    ):
        return GPUResidentSequentialLoader(dataset, batch_size=int(batch_size))
    return TorchDataLoader(dataset, *args, **kwargs)


def main() -> None:
    trainer.DataLoader = _data_loader_dispatch
    trainer.main()


if __name__ == "__main__":
    main()
