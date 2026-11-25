"""V10 multi-task gradient conflict audit.

Migrated from ml_recon/physics_hybrid_v9/monitoring.py.
Records per-task-group gradient norms and pairwise cosine on shared
encoder parameters.  Persistent negative-cosine conflict is evidence
only; this module never enables PCGrad, GradNorm, or dynamic weighting.

Pure torch/numpy, JSON-safe outputs, never writes parameter.grad.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .training_telemetry import append_training_record, atomic_training_json, archive_training_summary


# ── Task groups (weighted sub-losses) ────────────────────────────────────────

AUDIT_GROUPS: Dict[str, Dict[str, float]] = {
    "measurement": {"measurement": 1.0, "denoise": 0.5},
    "family": {"small_n_family": 0.4, "coarse": 0.3, "calibration": 0.1},
    "structure": {
        "pair_map": 0.4, "hyperedge": 0.2, "partition": 0.3,
        "depth": 0.3, "intactness": 0.3, "stretchability": 0.2,
    },
    "w_physics": {"w_r1": 0.8, "w_sector": 0.2, "psd": 0.1},
    "noise_ood": {"operational_noise": 0.2, "ood": 0.1},
}


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class V10GradientAuditConfig:
    enabled: bool = True
    interval_steps: int = 20
    first_step: bool = True
    final_step: bool = True
    persistent_window: int = 20
    conflict_threshold: float = -0.05
    persistent_negative_fraction: float = 0.50


# ── Record ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class V10GradientAuditRecord:
    step: int
    gradient_norms: Dict[str, float]
    cosine_matrix: Dict[str, float]
    negative_cosine_count: int
    total_pairs: int
    zero_gradient_groups: Tuple[str, ...]
    all_finite: bool
    audit_seconds: float
    observation_kind: str = "PRE_BACKWARD_ATTEMPT"
    attempt_index: int = 0
    run_id: str | None = None
    execution_id: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Core math ────────────────────────────────────────────────────────────────

def _flatten_grads(
    params: Sequence[torch.Tensor],
    grads: Sequence[Optional[torch.Tensor]],
) -> torch.Tensor:
    """Concatenate gradient tensors into one flat float64 vector.

    None gradients (from allow_unused) become zeros on the same device
    as the parameter to avoid CPU/CUDA mixing.
    """
    parts = []
    for param, grad in zip(params, grads):
        if grad is None:
            parts.append(torch.zeros(
                param.numel(), dtype=torch.float64, device=param.device))
        else:
            parts.append(grad.detach().reshape(-1).to(torch.float64))
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)


def compute_gradient_norms(grads: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """L2 norm per group."""
    return {
        name: float(torch.linalg.vector_norm(g).item())
        for name, g in grads.items()
    }


def compute_gradient_cosines(grads: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Pairwise cosine similarity between all group pairs."""
    names = sorted(grads.keys())
    result: Dict[str, float] = {}
    eps = 1e-12
    for i, ni in enumerate(names):
        for nj in names[i + 1:]:
            gi, gj = grads[ni], grads[nj]
            dot = float(torch.dot(gi, gj).item())
            ni_norm = float(torch.linalg.vector_norm(gi).item())
            nj_norm = float(torch.linalg.vector_norm(gj).item())
            denom = ni_norm * nj_norm + eps
            result[f"{ni}__{nj}"] = max(-1.0, min(1.0, dot / denom))
    return result


def persistent_gradient_conflict(
    cosine_history: Sequence[Dict[str, float]],
    *,
    window: int = 20,
    neg_fraction: float = 0.5,
    threshold: float = -0.05,
) -> Dict[str, Any]:
    """Check if recent gradient cosines show persistent conflict."""
    recent = list(cosine_history[-window:])
    if len(recent) < max(1, window // 2):
        return {"persistent_conflict": False, "negative_pairs": 0,
                "total_pairs": 0, "neg_fraction": 0.0, "insufficient_data": True}
    neg_count = 0
    total = 0
    for record_cosines in recent:
        for cos_val in record_cosines.values():
            total += 1
            if cos_val < threshold:
                neg_count += 1
    frac = neg_count / max(total, 1)
    return {
        "persistent_conflict": frac >= neg_fraction,
        "negative_pairs": neg_count,
        "total_pairs": total,
        "neg_fraction": round(frac, 6),
    }


# ── Feedback audit groups ─────────────────────────────────────────────────────

FEEDBACK_AUDIT_GROUPS: Dict[str, Dict[str, float]] = {
    "action": {
        "action": 1.0, "action_cosine": 0.2,
        "action_response": 0.1, "distillation": 0.1,
    },
    "dynamics": {
        "next_measurement": 0.5, "history": 0.05, "noop": 0.2,
    },
    "recoverability": {
        "recoverability": 0.2, "recoverability_class": 0.1,
    },
    "risk": {
        "gain_quantile": 0.2, "harm": 0.2, "stop": 0.1, "abstention": 0.1,
    },
    "routing": {
        "moe_balance": 0.05,
    },
}


# ── Observer ─────────────────────────────────────────────────────────────────

class V10GradientAuditObserver:
    """Training observer that audits multi-task gradient conflict.

    Uses ``torch.autograd.grad`` with ``retain_graph=True`` and
    ``allow_unused=True`` to compute per-group gradients on shared
    encoder parameters.  Never modifies ``parameter.grad``.

    Parameters
    ----------
    task_groups : dict, optional
        Custom task group definitions.  Defaults to ``AUDIT_GROUPS``
        (classifier groups).  Pass ``FEEDBACK_AUDIT_GROUPS`` for
        feedback training.
    """

    def __init__(
        self,
        *,
        config: V10GradientAuditConfig,
        shared_parameters: Sequence[torch.nn.Parameter],
        output_dir: Path,
        task_groups: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.config = config
        self.shared_params = list(shared_parameters)
        self.output_dir = Path(output_dir)
        self.task_groups = task_groups if task_groups is not None else AUDIT_GROUPS
        self.records: List[V10GradientAuditRecord] = []
        self._cosine_history: List[Dict[str, float]] = []
        self._total_train_time: float = 0.0
        self._audit_time: float = 0.0
        self._written_records = 0
        self._attempt_count = 0
        self.run_id = None
        self.execution_id = None

    def bind_run_context(self, *, run_id: str, execution_id: str, initial_attempt_index: int) -> None:
        if self.records:
            raise ValueError("cannot rebind an active gradient observer")
        self.run_id = run_id or None
        self.execution_id = execution_id
        self._attempt_count = initial_attempt_index

    def validate_loss_coverage(self, losses: Dict[str, torch.Tensor]) -> None:
        selected = {name for group in self.task_groups.values() for name, weight in group.items()
                    if weight > 0 and name in losses}
        if not selected or not any(isinstance(losses[name], torch.Tensor) and losses[name].requires_grad
                                   for name in selected):
            raise RuntimeError("gradient audit has no registered differentiable task loss")

    def _should_audit(self, step: int, is_final: bool = False) -> bool:
        if not self.config.enabled:
            return False
        if is_final:
            return self.config.final_step
        if step <= 1 and self.config.first_step:
            return True
        return step > 0 and step % self.config.interval_steps == 0

    def _compute_group_gradients(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], bool]:
        """Compute weighted sub-loss gradients for each audit group.

        Returns (grad_vectors, all_finite).
        """
        group_grads: Dict[str, torch.Tensor] = {}
        all_finite = True
        connected = False
        ref_device = self.shared_params[0].device if self.shared_params else "cpu"
        for group_name, components in self.task_groups.items():
            weighted_loss = torch.tensor(0.0, device=next(
                iter(losses.values())).device)
            for loss_name, weight in components.items():
                if loss_name in losses:
                    weighted_loss = weighted_loss + weight * losses[loss_name]
            if not weighted_loss.requires_grad:
                group_grads[group_name] = torch.zeros(
                    sum(p.numel() for p in self.shared_params),
                    dtype=torch.float64, device=ref_device)
                continue
            raw_grads = torch.autograd.grad(
                weighted_loss, self.shared_params,
                retain_graph=True, allow_unused=True)
            connected = connected or any(gradient is not None for gradient in raw_grads)
            flat = _flatten_grads(self.shared_params, raw_grads)
            if not torch.all(torch.isfinite(flat)):
                all_finite = False
            group_grads[group_name] = flat
        if not connected:
            raise RuntimeError("gradient audit parameters are disconnected from every registered loss")
        return group_grads, all_finite

    def on_before_backward(
        self,
        *,
        model: torch.nn.Module,
        batch: Dict[str, Any],
        losses: Dict[str, torch.Tensor],
        global_step: int,
        epoch: int,
        attempt_index: int | None = None,
    ) -> None:
        """Audit gradient conflict BEFORE main backward().

        Uses torch.autograd.grad(retain_graph=True) so the main loss
        can still call backward() afterward.
        """
        self.validate_loss_coverage(losses)
        self._attempt_count = self._attempt_count + 1 if attempt_index is None else attempt_index
        if not self._should_audit(global_step):
            return
        t0 = time.monotonic()
        group_grads, all_finite = self._compute_group_gradients(losses)
        norms = compute_gradient_norms(group_grads)
        cosines = compute_gradient_cosines(group_grads)
        neg_count = sum(1 for v in cosines.values() if v < self.config.conflict_threshold)
        total_pairs = len(cosines)
        zero_groups = tuple(
            name for name, norm in norms.items() if norm < 1e-15)
        elapsed = time.monotonic() - t0
        self._audit_time += elapsed
        record = V10GradientAuditRecord(
            step=global_step, gradient_norms=norms,
            cosine_matrix=cosines, negative_cosine_count=neg_count,
            total_pairs=total_pairs, zero_gradient_groups=zero_groups,
            all_finite=all_finite, audit_seconds=round(elapsed, 4),
            attempt_index=self._attempt_count, run_id=self.run_id, execution_id=self.execution_id)
        self.records.append(record)
        self._cosine_history.append(cosines)
        self._flush_records()

    def on_epoch_end(self, **kwargs) -> None:
        return None

    def finalize(self) -> Dict[str, Any]:
        """Aggregate all records into a summary."""
        if not self.records:
            return {"audited_steps": 0, "persistent_gradient_conflict": False}
        all_norms: Dict[str, List[float]] = {}
        all_cosines: Dict[str, List[float]] = {}
        zero_frac: Dict[str, int] = {}
        for record in self.records:
            for name, norm in record.gradient_norms.items():
                all_norms.setdefault(name, []).append(norm)
            for pair, cos in record.cosine_matrix.items():
                all_cosines.setdefault(pair, []).append(cos)
            for name in record.zero_gradient_groups:
                zero_frac[name] = zero_frac.get(name, 0) + 1
        n = len(self.records)
        summary: Dict[str, Any] = {
            "audited_steps": n,
            "audited_step_semantics": "PRE_BACKWARD_ATTEMPT",
            "summary_scope": "CURRENT_OBSERVER_INSTANCE",
            "journal_scope": "APPEND_ONLY_ALL_EXECUTIONS",
            "run_id": self.run_id, "execution_id": self.execution_id,
            "first_attempt_index": self.records[0].attempt_index,
            "last_attempt_index": self.records[-1].attempt_index,
            "gradient_norm_mean_by_group": {
                k: round(float(np.mean(v)), 8) for k, v in all_norms.items()},
            "gradient_norm_p95_by_group": {
                k: round(float(np.percentile(v, 95)), 8) for k, v in all_norms.items()},
            "cosine_mean_by_pair": {
                k: round(float(np.mean(v)), 8) for k, v in all_cosines.items()},
            "cosine_p05_by_pair": {
                k: round(float(np.percentile(v, 5)), 8) for k, v in all_cosines.items()},
            "negative_fraction_by_pair": {
                k: round(sum(1 for x in v if x < self.config.conflict_threshold) / len(v), 6)
                for k, v in all_cosines.items()},
            "zero_gradient_fraction_by_group": {
                k: round(v / n, 6) for k, v in zero_frac.items()},
            "audit_overhead_seconds": round(self._audit_time, 2),
        }
        # Norm ratios
        meas_norm = summary["gradient_norm_mean_by_group"].get("measurement", 0)
        for target in ("family", "structure"):
            t_norm = summary["gradient_norm_mean_by_group"].get(target, 0)
            summary[f"measurement_to_{target}_norm_ratio"] = (
                round(meas_norm / (t_norm + 1e-15), 6))
        # Persistent conflict
        conflict = persistent_gradient_conflict(
            self._cosine_history,
            window=self.config.persistent_window,
            neg_fraction=self.config.persistent_negative_fraction,
            threshold=self.config.conflict_threshold)
        summary["persistent_gradient_conflict"] = conflict["persistent_conflict"]
        summary["persistent_conflict_detail"] = conflict
        return summary

    def _flush_records(self) -> None:
        """Append only new observations; repeated finalization is idempotent."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for record in self.records[self._written_records:]:
            append_training_record(self.output_dir / "gradient_audit.jsonl", record.to_dict())
            self._written_records += 1

    def write_outputs(self) -> None:
        """Preserve incremental history and atomically publish its summary."""
        self._flush_records()
        summary = self.finalize()
        path = self.output_dir / "gradient_audit_summary.json"
        if path.exists():
            archive_training_summary(path)
        atomic_training_json(path, summary)
