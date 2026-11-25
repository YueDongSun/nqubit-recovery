"""Deterministic virtual sample ranges; no materialized synthetic sample store.

Only recipes, sample identities and hashes are durable. Rows are regenerated on
each access, including validation. There is deliberately no path/cache argument.
The existing registered physics renderer and named visibility views are reused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import NQubitDatasetContractV240
from .hashing import sha256_json
from .multi_cell import V24MultiCellRangeDataset
from .rung_plan import V24CellAllocationV2, V24RungPlanV2
from .sample_keys import V24SampleKey, make_sample_key
from .schema_registry import schema_sha256
from .version import GENERATOR_SHA
from .views import CLASSIFIER_TRUTH, classifier_input_for_family


STREAM_VERSION = "v24_keyed_stream_v1"
MAX_FETCH_ROWS = 16
DEFAULT_SEED = 20261001
STREAM_SPLITS = frozenset({"train", "validation", "test", "locked_ood",
                           "negative_control", "audit"})


@dataclass(frozen=True)
class V24_stream_spec:
    cell: V24CellAllocationV2
    split: str
    global_seed: int = DEFAULT_SEED
    sample_count: int | None = None
    start_index: int = 0

    @property
    def count(self) -> int:
        if self.sample_count is not None:
            return self.sample_count
        if self.split == "train":
            return self.cell.train_count
        if self.split == "validation":
            return self.cell.validation_count
        raise ValueError("non-training stream splits require an explicit sample count")

    def validate(self) -> None:
        for name, value in (("global_seed", self.global_seed),
                            ("start_index", self.start_index), ("sample_count", self.count)):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer, not a boolean/coercion")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.split not in STREAM_SPLITS or self.count == 0:
            raise ValueError("invalid stream split or empty sample range")
        classifier_input_for_family(self.cell.dataset_family)
        if self.cell.N not in (3, 4) or not self.cell.cell_id:
            raise ValueError("this bounded stream supports the registered 3Q/4Q classifier only")
        self.sample_key(0).validate()

    def sample_key(self, index: int) -> V24SampleKey:
        return make_sample_key(
            global_seed=self.global_seed, split=self.split,
            dataset_family=self.cell.dataset_family, N=self.cell.N,
            index=self.start_index + index, generator_sha=GENERATOR_SHA,
            cell_id=self.cell.cell_id)

    def recipe(self) -> dict[str, Any]:
        self.validate()
        return {"stream_version": STREAM_VERSION, "cell": asdict(self.cell),
                "split": self.split, "global_seed": self.global_seed,
                "sample_count": self.count, "start_index": self.start_index,
                "generator_sha": GENERATOR_SHA,
                "contract_sha": NQubitDatasetContractV240.default().sha256(),
                "schema_sha": schema_sha256()}


def _row_tensor(value: Any) -> torch.Tensor:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError("named model views must contain numeric arrays only")
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    elif array.dtype in (np.dtype("uint32"), np.dtype("uint64")):
        array = array.astype(np.int64)
    return torch.from_numpy(array.copy())


class V24_stream_dataset(Dataset):
    """Map-style, prefix-stable stream with at most one requested batch resident.

    The caller owns returned tensors. This dataset retains only an immutable
    recipe and visibility flag, not previously rendered samples or RNG state.
    """

    def __init__(self, spec: V24_stream_spec, *, include_truth: bool = True):
        spec.validate()
        if type(include_truth) is not bool:
            raise TypeError("include_truth must be explicitly Boolean")
        self.spec = spec
        self.include_truth = include_truth

    def __len__(self) -> int:
        return self.spec.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("sample index must be an integer")
        if not 0 <= index < len(self):
            raise IndexError(index)
        from .synthetic import generator

        key = self.spec.sample_key(int(index))
        rendered = generator._render_dataset_family(
            key, cell=self.spec.cell.to_generator_cell())
        observation = generator._stack([rendered["observation"]], [key.sample_id])
        truth = generator._stack([rendered["truth"]], [key.sample_id])
        fields = classifier_input_for_family(self.spec.cell.dataset_family)
        missing = fields - observation.keys()
        if self.include_truth:
            missing |= CLASSIFIER_TRUTH - truth.keys()
        if missing:
            raise ValueError(f"renderer omitted required named-view fields: {sorted(missing)}")
        result = {name: _row_tensor(observation[name][0]) for name in fields}
        if self.include_truth:
            result.update({name: _row_tensor(truth[name][0]) for name in CLASSIFIER_TRUTH})
        result.update(sample_id=key.sample_id,
                      rendered_row_sha256=generator._row_sha(observation, truth, 0),
                      row_valid_mask=torch.tensor(True))
        return result

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        if len(indices) > MAX_FETCH_ROWS:
            raise ValueError(f"stream fetch must be bounded to {MAX_FETCH_ROWS} rows")
        return [self[index] for index in indices]

    def storage_contract(self) -> dict[str, Any]:
        return {"input_delivery": "stream_only", "persisted_sample_rows": 0,
                "max_fetch_rows": MAX_FETCH_ROWS, "recipe": self.spec.recipe()}


def classifier_stream_datasets(
    plan: V24RungPlanV2, *, global_seed: int = DEFAULT_SEED,
) -> tuple[V24MultiCellRangeDataset, V24MultiCellRangeDataset, dict[str, Any]]:
    """Construct virtual ranges without evaluating or writing a sample.

    Rung and runtime authorization still belongs to the trainer. This function
    alone cannot grant a pilot, larger scale, selection or hardware authority.
    """
    plan.validate()
    if plan.task_pool != "small_n_classifier":
        raise ValueError("stream classifier requires the small_n_classifier plan")
    specifications = {
        split: tuple(V24_stream_spec(cell, split, global_seed=global_seed)
                     for cell in plan.cell_allocations)
        for split in ("train", "validation")}
    datasets = {split: V24MultiCellRangeDataset(
        [V24_stream_dataset(spec) for spec in specs],
        cell_ids=[spec.cell.cell_id for spec in specs])
        for split, specs in specifications.items()}
    recipe = {"stream_version": STREAM_VERSION, "input_delivery": "stream_only",
              "persisted_sample_rows": 0, "rung_plan": plan.to_dict(),
              "splits": {split: [spec.recipe() for spec in specs]
                         for split, specs in specifications.items()},
              "selection_evidence": False}
    recipe["recipe_sha256"] = sha256_json(recipe)
    return datasets["train"], datasets["validation"], recipe
