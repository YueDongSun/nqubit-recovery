"""Global-total, prefix-stable V2.4 rung plans split by task pool."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .contracts import NQubitDatasetContractV240
from .enums import N_PANEL
from .hashing import read_canonical_json, sha256_json, write_canonical_json
from .schema_registry import schema_sha256
from .version import GENERATOR_SHA


# Single-N structure rungs (external review follow-up 2026-10-03): one
# registered pool per general-N structure cell so a single-N training or
# evaluation rung needs no new plumbing per N.  Purely additive — membership
# is not hashed content, and the parametric branch below emits exactly the
# same cell spec the pilot pool uses, so every pre-existing pool identity is
# byte-identical.
SINGLE_N_STRUCTURE_POOLS = tuple(
    f"large_n_structure_n{N}" for N in N_PANEL if N not in (3, 4))
SINGLE_N_STRUCTURE_NS = tuple(N for N in N_PANEL if N not in (3, 4))

TASK_POOLS = (
    "small_n_classifier",
    "large_n_structure_pilot",
    *SINGLE_N_STRUCTURE_POOLS,
    "general_n_structure",
    "w_control",
    "feedback",
)
SHOT_MODES = ("S_total", "S_setting", "S_ideal")


@dataclass(frozen=True)
class V24CellAllocationV2:
    cell_id: str
    dataset_family: str
    N: int
    train_count: int
    validation_count: int
    noise_family: str
    eta: float
    kappa: float
    shot_mode: str
    shot_budget: int
    measurement_schedule: str

    def to_generator_cell(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "dataset_family": self.dataset_family,
            "N": self.N,
            "system_type": f"{self.N}qubit" if self.N in (3, 4) else "nqubit",
            "measurement_domain": "global_pauli_v24",
            "train": self.train_count,
            "validation": self.validation_count,
            "noise_family": self.noise_family,
            "eta": self.eta,
            "kappa": self.kappa,
            "kappa_inj": self.kappa,
            "shot_mode": self.shot_mode,
            "shot_budget": self.shot_budget,
            "measurement_schedule": self.measurement_schedule,
        }


@dataclass(frozen=True)
class V24RungPlanV2:
    plan_id: str
    task_pool: str
    total_train_samples: int
    total_validation_samples: int
    cell_allocations: Tuple[V24CellAllocationV2, ...]
    generator_sha: str
    contract_sha: str
    schema_sha: str
    plan_version: str = "v24_rung_plan_v2"

    def validate(self) -> None:
        if self.task_pool not in TASK_POOLS or self.plan_version != "v24_rung_plan_v2":
            raise ValueError("invalid V2.4 rung plan identity")
        if self.generator_sha != GENERATOR_SHA \
                or self.contract_sha != NQubitDatasetContractV240.default().sha256() \
                or self.schema_sha != schema_sha256():
            raise ValueError("V2.4 rung plan dependency mismatch")
        if self.total_train_samples <= 0 or self.total_validation_samples <= 0:
            raise ValueError("V2.4 rung totals must be positive")
        if sum(cell.train_count for cell in self.cell_allocations) != self.total_train_samples:
            raise ValueError("cell train allocations do not sum to the global rung total")
        if sum(cell.validation_count for cell in self.cell_allocations) \
                != self.total_validation_samples:
            raise ValueError("cell validation allocations do not sum to the global total")
        if len({cell.cell_id for cell in self.cell_allocations}) != len(self.cell_allocations):
            raise ValueError("V2.4 rung cell IDs must be unique")
        for cell in self.cell_allocations:
            if cell.train_count < 0 or cell.validation_count < 0 \
                    or cell.N not in N_PANEL or cell.shot_mode not in SHOT_MODES:
                raise ValueError("invalid V2.4 cell allocation")
        expected_plan_id = sha256_json(self.content_dict(include_plan_id=False))[:24]
        if self.plan_id != expected_plan_id:
            raise ValueError("V2.4 rung plan ID is not canonical")

    def content_dict(self, *, include_plan_id: bool = True) -> Dict[str, Any]:
        value = asdict(self)
        value["cell_allocations"] = [asdict(cell) for cell in self.cell_allocations]
        if not include_plan_id:
            value.pop("plan_id")
        return value

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return self.content_dict()

    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "V24RungPlanV2":
        payload = dict(value)
        payload["cell_allocations"] = tuple(
            V24CellAllocationV2(**item) for item in payload["cell_allocations"])
        plan = cls(**payload)
        plan.validate()
        return plan


def _balanced_counts(total: int, cell_count: int) -> Tuple[int, ...]:
    quotient, remainder = divmod(int(total), int(cell_count))
    return tuple(quotient + int(index < remainder) for index in range(cell_count))


_SINGLE_N_STRUCTURE_POOL_VALUES = dict(
    zip(SINGLE_N_STRUCTURE_POOLS, SINGLE_N_STRUCTURE_NS))


def _single_n_structure_pool_value(task_pool: str) -> int | None:
    """Return the N of a registered ``large_n_structure_n{N}`` pool id.

    Exact registered-id lookup: zero-padded or otherwise non-canonical
    aliases (``large_n_structure_n05``) are not pool ids and return ``None``
    so the unknown-pool error path applies (external review follow-up).
    """
    return _SINGLE_N_STRUCTURE_POOL_VALUES.get(task_pool)


def _task_cells(task_pool: str) -> Tuple[Dict[str, Any], ...]:
    if task_pool == "small_n_classifier":
        return tuple(
            {"cell_id": f"classifier_n{N}_{noise}", "dataset_family": "small_n_family",
             "N": N, "noise_family": noise, "eta": 0.0 if noise == "standard" else 0.08,
             "kappa": 0.0 if noise == "standard" else 0.15,
             "shot_mode": "S_setting", "shot_budget": 64,
             "measurement_schedule": "full_global_pauli"}
            for noise in ("standard", "noisy") for N in (3, 4)
        )
    if task_pool == "general_n_structure":
        return tuple(
            {"cell_id": f"structure_n{N}", "dataset_family": "general_n_structure",
             "N": N, "noise_family": "structure_mixed", "eta": 0.05,
             "kappa": 0.10, "shot_mode": "S_total", "shot_budget": 4096,
             "measurement_schedule": "registered_subsets_v1"}
            for N in N_PANEL
        )
    if task_pool == "large_n_structure_pilot":
        return tuple(
            {"cell_id": f"structure_n{N}", "dataset_family": "general_n_structure",
             "N": N, "noise_family": "structure_mixed", "eta": 0.05,
             "kappa": 0.10, "shot_mode": "S_total", "shot_budget": 4096,
             "measurement_schedule": "registered_subsets_v1"}
            for N in (5, 6)
        )
    single_n = _single_n_structure_pool_value(task_pool)
    if single_n is not None:
        # Same cell spec and cell_id as the pilot pool's cell for this N, so
        # sample-key namespaces (and therefore row identity) are shared with
        # the pilot rung — a single-N rung trains on exactly the rows the
        # pilot pool allocates to that cell.
        return ({"cell_id": f"structure_n{single_n}",
                 "dataset_family": "general_n_structure",
                 "N": single_n, "noise_family": "structure_mixed", "eta": 0.05,
                 "kappa": 0.10, "shot_mode": "S_total", "shot_budget": 4096,
                 "measurement_schedule": "registered_subsets_v1"},)
    if task_pool == "w_control":
        return tuple(
            {"cell_id": f"w_control_n{N}_{shot.lower()}", "dataset_family": "w_control",
             "N": N, "noise_family": "control_mixed", "eta": 0.05,
             "kappa": 0.10, "shot_mode": shot, "shot_budget": 4096,
             "measurement_schedule": "registered_subsets_v1"}
            for N in N_PANEL for shot in SHOT_MODES
        )
    if task_pool == "feedback":
        families = ("feedback_one_step", "feedback_history", "feedback_noop")
        return tuple(
            {"cell_id": f"{family}_n{N}", "dataset_family": family, "N": N,
             "noise_family": "feedback_mixed", "eta": 0.08, "kappa": 0.15,
             "shot_mode": "S_setting", "shot_budget": 256,
             "measurement_schedule": "phase_sensitive_w_v1"}
            for family in families for N in (3, 4)
        )
    raise ValueError(f"unknown V2.4 task pool: {task_pool}")


def registered_task_cells(task_pool: str) -> Tuple[Dict[str, Any], ...]:
    """Canonical registered cell declarations for a task pool."""

    return _task_cells(task_pool)


def plan_registered_rung(task_pool: str, total_train_samples: int,
                         total_validation_samples: int | None = None) -> V24RungPlanV2:
    cells = _task_cells(task_pool)
    if total_validation_samples is None:
        total_validation_samples = 200 if total_train_samples <= 1_000 else 2_000
    train_counts = _balanced_counts(total_train_samples, len(cells))
    validation_counts = _balanced_counts(total_validation_samples, len(cells))
    allocations = tuple(
        V24CellAllocationV2(
            train_count=train_counts[index], validation_count=validation_counts[index],
            **cell)
        for index, cell in enumerate(cells)
    )
    base = {
        "task_pool": task_pool,
        "total_train_samples": int(total_train_samples),
        "total_validation_samples": int(total_validation_samples),
        "cell_allocations": [asdict(cell) for cell in allocations],
        "generator_sha": GENERATOR_SHA,
        "contract_sha": NQubitDatasetContractV240.default().sha256(),
        "schema_sha": schema_sha256(),
        "plan_version": "v24_rung_plan_v2",
    }
    plan = V24RungPlanV2(plan_id=sha256_json(base)[:24], **{
        **base, "cell_allocations": allocations})
    plan.validate()
    return plan


def write_rung_plan(path: str | Path, plan: V24RungPlanV2) -> Dict[str, str]:
    plan_sha = plan.sha256()
    file_sha = write_canonical_json(path, plan.to_dict())
    if file_sha != plan_sha:
        raise RuntimeError("V2.4 rung plan canonical SHA mismatch")
    return {"path": str(path), "rung_plan_sha": plan_sha, "plan_id": plan.plan_id}


def read_rung_plan(path: str | Path) -> V24RungPlanV2:
    return V24RungPlanV2.from_dict(read_canonical_json(path))


def validate_registered_cell_materialization(
    plan: V24RungPlanV2,
    *,
    expected_task_pool: str,
    requested_sample_rung: int,
    cell_id: str,
    train_count: int,
    validation_count: int,
) -> V24CellAllocationV2:
    """Bind one homogeneous training job to its global-rung allocation."""

    plan.validate()
    if plan.task_pool != expected_task_pool:
        raise ValueError(
            f"rung plan task_pool must be {expected_task_pool}, got {plan.task_pool}"
        )
    if plan.total_train_samples != int(requested_sample_rung):
        raise ValueError("rung plan global train total does not match sample rung")
    matches = [cell for cell in plan.cell_allocations if cell.cell_id == cell_id]
    if len(matches) != 1:
        raise ValueError(f"dataset cell is not registered exactly once: {cell_id}")
    allocation = matches[0]
    if int(train_count) != allocation.train_count:
        raise ValueError("READY V2.4 cell train rows do not match rung allocation")
    if int(validation_count) != allocation.validation_count:
        raise ValueError("READY V2.4 cell validation rows do not match rung allocation")
    return allocation


def validate_registered_multi_cell_materialization(
    plan: V24RungPlanV2,
    *,
    expected_task_pool: str,
    requested_sample_rung: int,
    cell_ids: Sequence[str],
    train_counts: Mapping[str, int],
    validation_counts: Mapping[str, int],
) -> Dict[str, V24CellAllocationV2]:
    """Bind one homogeneous multi-cell training job to its global rung."""

    plan.validate()
    if plan.task_pool != expected_task_pool:
        raise ValueError(
            f"rung plan task_pool must be {expected_task_pool}, got {plan.task_pool}"
        )
    if plan.total_train_samples != int(requested_sample_rung):
        raise ValueError("rung plan global train total does not match sample rung")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("V2.4 multi-cell job cell IDs must be unique")
    registered = {cell.cell_id: cell for cell in plan.cell_allocations}
    missing = [cell_id for cell_id in cell_ids if cell_id not in registered]
    if missing:
        raise ValueError(f"dataset cells are not registered exactly once: {missing}")
    result: Dict[str, V24CellAllocationV2] = {}
    for cell_id in cell_ids:
        allocation = registered[cell_id]
        if int(train_counts[cell_id]) != allocation.train_count:
            raise ValueError(
                f"READY V2.4 cell {cell_id} train rows do not match rung allocation")
        if int(validation_counts[cell_id]) != allocation.validation_count:
            raise ValueError(
                f"READY V2.4 cell {cell_id} validation rows do not match rung allocation")
        result[cell_id] = allocation
    # --- Full-rung coverage gate ---
    registered_ids = {cell.cell_id for cell in plan.cell_allocations}
    selected_ids = set(cell_ids)
    if selected_ids != registered_ids:
        missing = registered_ids - selected_ids
        extra = selected_ids - registered_ids
        parts: list[str] = []
        if missing:
            parts.append(f"missing: {sorted(missing)}")
        if extra:
            parts.append(f"extra: {sorted(extra)}")
        raise ValueError(
            "multi-cell rung must cover all registered cells; " + "; ".join(parts))
    selected_train_total = sum(int(train_counts[cid]) for cid in cell_ids)
    selected_validation_total = sum(int(validation_counts[cid]) for cid in cell_ids)
    if selected_train_total != plan.total_train_samples:
        raise ValueError(
            f"selected cells train total {selected_train_total} != "
            f"rung plan total {plan.total_train_samples}")
    if selected_validation_total != plan.total_validation_samples:
        raise ValueError(
            f"selected cells validation total {selected_validation_total} != "
            f"rung plan total {plan.total_validation_samples}")
    return result
