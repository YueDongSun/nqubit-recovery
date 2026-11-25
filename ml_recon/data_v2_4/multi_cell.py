"""Homogeneous multi-cell V2.4 training datasets and batch samplers."""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from typing import Any, Dict, List, Sequence

import random

from torch.utils.data import Dataset, Sampler


def parse_cell_keyed_paths(values: Sequence[str]) -> Dict[str, str]:
    """Parse repeated ``cell_id=path`` CLI values; bare values use the default key."""

    result: Dict[str, str] = {}
    for value in values:
        if not value:
            continue
        cell_id, path = value.split("=", 1) if "=" in value else ("", value)
        if cell_id in result:
            raise ValueError(
                f"duplicate cell-keyed path for {cell_id or 'default'}")
        result[cell_id] = path
    return result


def resolve_cell_keyed_paths(
    cell_ids: Sequence[str],
    *,
    legacy_path: str,
    cell_paths: Dict[str, str] | None,
) -> Dict[str, str]:
    """Resolve one legacy path or per-cell paths into a per-cell mapping."""

    result = dict(cell_paths or {})
    if legacy_path:
        if result:
            raise ValueError(
                "provide either one shared path or per-cell paths, not both")
        result = {cell_id: legacy_path for cell_id in cell_ids}
    return result


class V24MultiCellRangeDataset(Dataset):
    """Join per-cell datasets while keeping every microbatch cell-homogeneous.

    Each underlying dataset owns one immutable V2.4 cell.  Global indices are
    contiguous per cell; ``__getitems__`` groups batched fetches by cell so
    HDF5 shard reads stay batched and no batch ever mixes cell shapes.
    """

    def __init__(self, datasets: Sequence[Dataset], cell_ids: Sequence[str]):
        if not datasets or len(datasets) != len(cell_ids):
            raise ValueError("multi-cell dataset requires matching non-empty datasets/cell ids")
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("multi-cell dataset cell ids must be unique")
        lengths = [len(dataset) for dataset in datasets]
        if any(length <= 0 for length in lengths):
            raise ValueError("multi-cell dataset cells must be non-empty")
        self.datasets = tuple(datasets)
        self.cell_ids = tuple(cell_ids)
        self.cell_lengths = tuple(lengths)
        self.cell_offsets = tuple(self._offsets(lengths))
        self.total = sum(lengths)

    @staticmethod
    def _offsets(lengths: Sequence[int]) -> List[int]:
        offsets: List[int] = []
        cursor = 0
        for length in lengths:
            offsets.append(cursor)
            cursor += length
        return offsets

    def cell_ranges(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (cell_id, start, start + length)
            for cell_id, start, length in zip(
                self.cell_ids, self.cell_offsets, self.cell_lengths))

    def cell_dataset(self, cell_id: str) -> Dataset:
        position = self.cell_ids.index(cell_id)
        return self.datasets[position]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.total:
            raise IndexError(index)
        position = bisect_right(self.cell_offsets, index) - 1
        return position, index - self.cell_offsets[position]

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> Dict[str, Any]:
        position, local_index = self._locate(index)
        row = self.datasets[position][local_index]
        return {**row, "cell_id": self.cell_ids[position]}

    def __getitems__(self, indices: Sequence[int]) -> list[Dict[str, Any]]:
        grouped: Dict[int, list[tuple[int, int]]] = {}
        for batch_position, index in enumerate(indices):
            position, local_index = self._locate(index)
            grouped.setdefault(position, []).append((batch_position, local_index))
        results: list[Dict[str, Any] | None] = [None] * len(indices)
        for position, entries in grouped.items():
            dataset = self.datasets[position]
            locals_ = [local_index for _, local_index in entries]
            rows = (
                dataset.__getitems__(locals_)
                if hasattr(dataset, "__getitems__") else
                [dataset[local_index] for local_index in locals_])
            for (batch_position, _), row in zip(entries, rows):
                results[batch_position] = {**row, "cell_id": self.cell_ids[position]}
        return [row for row in results if row is not None]

    def close(self) -> None:
        for dataset in self.datasets:
            close = getattr(dataset, "close", None)
            if close is not None:
                close()


class V24CellHomogeneousBatchSampler(Sampler):
    """Yield homogeneous per-cell batches with round-robin cell rotation.

    Every yielded batch contains rows from exactly one cell, so model forward
    shapes stay homogeneous within each optimizer step.  Batches rotate across
    cells between steps and the shared model weights update on every cell.
    """

    def __init__(self, dataset: Dataset, batch_size: int, *, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(dataset, V24MultiCellRangeDataset):
            raise TypeError("homogeneous batch sampler requires V24MultiCellRangeDataset")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

    def _cell_batch_count(self, length: int) -> int:
        full = length // self.batch_size
        if self.drop_last:
            return full
        return full + int(length % self.batch_size != 0)

    def __iter__(self):
        rng = random.Random(self.seed)
        ranges = list(self.dataset.cell_ranges())
        order = list(range(len(ranges)))
        if self.shuffle:
            rng.shuffle(order)
        queues: List[deque] = []
        for position in order:
            _, start, stop = ranges[position]
            indices = list(range(start, stop))
            if self.shuffle:
                rng.shuffle(indices)
            if self.drop_last:
                indices = indices[:len(indices) // self.batch_size * self.batch_size]
            batches = [
                indices[index:index + self.batch_size]
                for index in range(0, len(indices), self.batch_size)]
            if batches:
                queues.append(deque(batches))
        while any(queues):
            for queue in queues:
                if queue:
                    yield queue.popleft()

    def __len__(self) -> int:
        return sum(
            self._cell_batch_count(stop - start)
            for _, start, stop in self.dataset.cell_ranges())
