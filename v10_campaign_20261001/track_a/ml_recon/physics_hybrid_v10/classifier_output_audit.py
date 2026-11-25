"""Source-bound slice diagnostics; parameter decay is not supervision evidence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch

from .published_output_registry import published_output_registry
from .training_telemetry import atomic_training_json


@dataclass(frozen=True)
class Output_slice:
    output_name: str
    module_name: str
    start: int | None = None
    stop: int | None = None


def output_slices(model) -> tuple[Output_slice, ...]:
    offset = 2 * model.w_head.factor_rank
    return (
        Output_slice("OOD_probability", "ood_head", 0, 1),
        Output_slice("model_uncertainty", "ood_head", 1, 2),
        Output_slice("R1_re", "w_head.mode", 0, offset),
        Output_slice("amplitude_error", "w_head.mode", offset, offset + 1),
        Output_slice("phase_error", "w_head.mode", offset + 1, offset + 2),
        Output_slice("W_amplitude_variance", "w_head.mode", offset + 2, offset + 3),
        Output_slice("W_phase_variance", "w_head.mode", offset + 3, offset + 4),
        Output_slice("recoverability", "w_head.global_head", 0, 1),
        Output_slice("W_uncertainty", "w_head.global_head", 1, 2),
        Output_slice("entanglement_depth_score", "w_head.global_head", 2, 3),
        Output_slice("certificate_status_probs", "certificate_status_head"),
        Output_slice("covariance_clean_diag", "denoising_head.log_variance"),
        Output_slice("nuisance_posterior_mean", "denoising_head.nuisance", 0, 8),
        Output_slice("nuisance_posterior_variance", "denoising_head.nuisance", 8, 16),
        Output_slice("abstention_probability", "structure_head.abstain"),
    )


class Classifier_output_audit:
    def __init__(self, model, output_dir: str | Path, interval_steps: int = 20):
        if type(interval_steps) is not int or interval_steps < 1:
            raise ValueError("output audit interval must be positive")
        self.model = model
        self.output_dir = Path(output_dir)
        self.interval_steps = interval_steps
        self.slices = output_slices(model)
        all_parameters = dict(model.named_parameters())
        names = {spec.module_name + suffix for spec in self.slices for suffix in (".weight", ".bias")}
        self.parameters = {name: all_parameters[name] for name in sorted(names)}
        self.initial = {name: value.detach().clone() for name, value in self.parameters.items()}
        self.records = []
        self.coverage = {}
        self.context = {}
        self.pending_coverage = []
        self.registered_cells = set()
        self.registered_cells_locked = False
        self.audited_cells = set()
        self.final_global_step = None
        self.last_committed_record = None
        self.discarded_uncommitted_attempts = 0

    def bind_training_identity(self, *, identity: dict, run_id: str, execution_id: str) -> None:
        if not isinstance(execution_id, str) or not re.fullmatch(r"[A-Za-z0-9_]+", execution_id):
            raise ValueError("output audit requires a safe execution identity")
        parameters = dict(self.model.named_parameters())
        self.parameters = {name: parameters[name] for name in self.parameters}
        self.initial = {name: value.detach().clone() for name, value in self.parameters.items()}
        self.context = {"source_identity": dict(identity), "run_id": run_id,
                        "execution_id": execution_id, "delta_reference": "CURRENT_EXECUTION_INITIAL_STATE"}

    def on_before_backward(self, *, model, batch, losses, global_step, epoch):
        """Stage coverage only; gradients are inspected after a committed step.

        This callback runs before AMP decides whether an optimizer update can
        commit.  It must never create an audit record or update coverage.
        """
        if model is not self.model:
            raise ValueError("output audit is not bound to this model")
        cell = batch.get("cell_id")
        if cell is None:
            if self.registered_cells_locked:
                if len(self.registered_cells) != 1:
                    raise ValueError("multi-cell output auditing requires explicit batch cell identity")
                cell = (next(iter(self.registered_cells)),)
            else:
                cell = ("UNSPECIFIED_CELL",)
        cell_id = str(cell[0]) if isinstance(cell, (list, tuple)) else str(cell)
        coverage = {}
        for name, spec in published_output_registry().items():
            if not spec.authoritative_targets:
                continue
            if any(target not in batch for target in spec.authoritative_targets):
                continue
            if spec.target_masks:
                masks = [batch[target] for target in spec.target_masks]
                if any(mask.dtype != torch.bool for mask in masks):
                    raise TypeError("registered output target masks must be boolean")
                mask = masks[0]
                for other in masks[1:]:
                    if other.shape != mask.shape:
                        raise ValueError("registered output target mask shapes disagree")
                    mask = mask & other
                valid = int(mask.sum())
                available = mask.numel()
                unit = "MASK_ELEMENTS"
            else:
                target = batch[spec.authoritative_targets[0]]
                valid = available = target.numel()
                unit = "UNMASKED_TARGET_ELEMENTS"
            coverage[name] = {"valid": valid, "available": available, "unit": unit}
        self.pending_coverage.append({"anticipated_global_step": global_step,
                                      "epoch": epoch, "cell_id": cell_id,
                                      "coverage": coverage})

    def configure_registered_cells(self, registered_cell_ids=()) -> None:
        values = {str(value) for value in registered_cell_ids}
        if not values:
            # Unit-level callers can rely on the first committed batch, but a
            # formal multi-cell runtime must provide its registered panel.
            return
        self.registered_cells = values
        self.registered_cells_locked = True

    def on_training_complete(self, *, final_global_step: int) -> None:
        if type(final_global_step) is not int or final_global_step < 1:
            raise ValueError("output audit final committed step must be positive")
        self.final_global_step = final_global_step
        if (self.last_committed_record is not None
                and self.last_committed_record["global_step"] == final_global_step
                and not any(row["global_step"] == final_global_step for row in self.records)):
            self.on_step_end(record={**self.last_committed_record,
                                     "is_final_committed_step": True})

    def on_step_end(self, *, record):
        if not record.get("committed_step", record.get("optimizer_step_executed", False)):
            return
        global_step = record["global_step"]
        cell_ids = tuple(str(value) for value in record.get("cell_ids", ()))
        if not cell_ids:
            staged_cells = {pending["cell_id"] for pending in self.pending_coverage
                            if pending["anticipated_global_step"] == global_step}
            if len(staged_cells) == 1:
                # Compatibility for the generic unit-test dataset. Production
                # multi-cell records carry cell_ids explicitly.
                cell_ids = tuple(staged_cells)
        if len(cell_ids) != 1:
            raise ValueError("output audit requires a homogeneous committed cell batch")
        self.last_committed_record = {**record, "cell_ids": cell_ids}
        cell_id = cell_ids[0]
        record_cells = {str(value) for value in record.get("registered_cells", ())}
        if self.registered_cells_locked:
            if record_cells and record_cells != self.registered_cells:
                raise ValueError("committed record registered cells disagree with audit configuration")
            if cell_id not in self.registered_cells:
                raise ValueError(f"committed record references unregistered cell: {cell_id}")
        else:
            self.registered_cells.update(record_cells)
            self.registered_cells.add(cell_id)
        self._commit_coverage(global_step, cell_id)
        capture = (cell_id not in self.audited_cells
                   or global_step % self.interval_steps == 0
                   or global_step == self.final_global_step
                   or bool(record.get("is_final_committed_step", False)))
        if not capture:
            return
        pending = {"global_step": global_step, "epoch": record["epoch"], "cell_id": cell_id,
                   "gradient_stage": "UNSCALED_AFTER_CLIP_AFTER_COMMITTED_STEP",
                   "capture_reason": self._capture_reason(global_step, cell_id, record)}
        values = {}
        for spec in self.slices:
            norms = []
            connected = False
            for suffix in (".weight", ".bias"):
                gradient = self.parameters[spec.module_name + suffix].grad
                if gradient is not None:
                    selected = gradient[spec.start:spec.stop] if spec.start is not None else gradient
                    if not bool(torch.isfinite(selected).all()):
                        raise FloatingPointError(f"non-finite output slice gradient: {spec.output_name}")
                    connected = True
                    norms.append(float(torch.sum(selected.detach().double().square())))
            norm = sum(norms) ** .5
            values[spec.output_name] = {"tensor_gradient_present": connected,
                                        "slice_gradient_norm": norm, "slice_gradient_nonzero": norm > 0}
        pending["outputs"] = values
        for spec in self.slices:
            squared = 0.0
            for suffix in (".weight", ".bias"):
                name = spec.module_name + suffix
                delta = self.parameters[name].detach() - self.initial[name]
                selected = delta[spec.start:spec.stop] if spec.start is not None else delta
                if not bool(torch.isfinite(selected).all()):
                    raise FloatingPointError(f"non-finite output slice update: {spec.output_name}")
                squared += float(selected.double().square().sum())
            pending["outputs"][spec.output_name]["parameter_delta_norm"] = squared ** .5
        self.records.append(pending)
        self.audited_cells.add(cell_id)
        self.write_outputs()

    def _commit_coverage(self, global_step: int, cell_id: str) -> None:
        matches = [index for index, pending in enumerate(self.pending_coverage)
                   if (pending["anticipated_global_step"] == global_step
                       and pending["cell_id"] == cell_id)]
        if not matches:
            return
        # A skipped AMP attempt retains the same anticipated global step.  The
        # final staged item is the one paired with this committed post-hook;
        # discard the earlier attempts without counting their coverage.
        pending = self.pending_coverage[matches[-1]]
        retained = []
        for index, item in enumerate(self.pending_coverage):
            stale = item["anticipated_global_step"] <= global_step
            if index in matches or stale:
                if index != matches[-1]:
                    self.discarded_uncommitted_attempts += 1
            else:
                retained.append(item)
        self.pending_coverage = retained
        coverage = self.coverage.setdefault(cell_id, {})
        for name, values in pending["coverage"].items():
            row = coverage.setdefault(name, {"valid": 0, "available": 0,
                                              "unit": values["unit"]})
            if row["unit"] != values["unit"]:
                raise ValueError("registered output coverage units disagree")
            row["valid"] += values["valid"]
            row["available"] += values["available"]

    def _capture_reason(self, global_step, cell_id, record):
        if bool(record.get("is_final_committed_step", False)) or global_step == self.final_global_step:
            return "FINAL_COMMITTED_STEP"
        if cell_id not in self.audited_cells:
            return "FIRST_COMMITTED_STEP_FOR_CELL"
        return "PERIODIC_GLOBAL_COMMITTED_STEP"

    def on_epoch_end(self, **_kwargs):
        self.write_outputs()

    def finalize(self):
        missing_cells = sorted(self.registered_cells - self.audited_cells)
        final_step_observed = any(
            row["global_step"] == self.final_global_step for row in self.records)
        complete = bool(self.final_global_step) and final_step_observed and not missing_cells
        return {"schema": "V10ClassifierOutputSliceAuditV1", **self.context,
                "audited_steps": len(self.records), "records": self.records,
                "registered_cells": sorted(self.registered_cells),
                "audited_cells": sorted(self.audited_cells),
                "final_committed_step": self.final_global_step,
                "final_committed_step_observed": final_step_observed,
                "audit_complete": complete,
                "audit_incomplete_reason": (
                    None if complete else {"missing_audited_cells": missing_cells,
                                           "final_committed_step_observed": final_step_observed}),
                "discarded_uncommitted_backward_attempts": self.discarded_uncommitted_attempts,
                "target_coverage": self.coverage, "coverage_scope": "COMMITTED_OPTIMIZER_STEPS_ONLY",
                "held_out_calibration_verified": False, "feedback_source_eligible": False,
                "scientific_gate_evaluated": False,
                "boundary": "Nonzero parameter delta alone can be optimizer decay, not trained uncertainty"}

    def write_outputs(self):
        if not self.context:
            raise ValueError("output audit must bind its source and execution before publication")
        filename = f"classifier_output_slice_audit_{self.context['execution_id']}.json"
        atomic_training_json(self.output_dir / filename, self.finalize())
