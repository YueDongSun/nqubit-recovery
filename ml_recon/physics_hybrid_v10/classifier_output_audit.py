"""Source-bound slice diagnostics; parameter decay is not supervision evidence."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import re

import torch

from .published_output_registry import output_training_registry_for_model
from .training_telemetry import atomic_training_json

LEGACY_OUTPUT_AUDIT_SCHEMA = "V10ClassifierOutputSliceAuditV3"
PHYSICAL_OUTPUT_AUDIT_SCHEMA = "V10ClassifierOutputSliceAuditV4"


def _strict_batch_strings(batch, name: str) -> tuple[str, ...]:
    value = batch.get(name)
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple)):
        values = tuple(value)
    else:
        raise ValueError(f"output audit {name} must be a string array")
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"output audit {name} requires non-empty strings")
    return values


@dataclass(frozen=True)
class Output_slice:
    output_name: str
    module_name: str
    start: int | None = None
    stop: int | None = None


def output_slices(model) -> tuple[Output_slice, ...]:
    """Return parameter slices for both legacy and physical C_N wrappers."""
    if hasattr(model, "w_head"):
        module_prefix = ""
        w_head = model.w_head
    elif hasattr(model, "core") and hasattr(model.core, "w_head"):
        module_prefix = "core."
        w_head = model.core.w_head
    else:
        raise ValueError("output audit requires a classifier with a W head")
    offset = 2 * w_head.factor_rank
    base_slices = (
        Output_slice("OOD_probability", module_prefix + "ood_head", 0, 1),
        Output_slice("model_uncertainty", module_prefix + "ood_head", 1, 2),
        Output_slice("R1_re", module_prefix + "w_head.mode", 0, offset),
        Output_slice(
            "amplitude_error", module_prefix + "w_head.mode", offset, offset + 1
        ),
        Output_slice(
            "phase_error", module_prefix + "w_head.mode", offset + 1, offset + 2
        ),
        Output_slice(
            "W_amplitude_variance",
            module_prefix + "w_head.mode",
            offset + 2,
            offset + 3,
        ),
        Output_slice(
            "W_phase_variance", module_prefix + "w_head.mode", offset + 3, offset + 4
        ),
        Output_slice("recoverability", module_prefix + "w_head.global_head", 0, 1),
        Output_slice("W_uncertainty", module_prefix + "w_head.global_head", 1, 2),
        Output_slice(
            "entanglement_depth_score", module_prefix + "w_head.global_head", 2, 3
        ),
        Output_slice(
            "certificate_status_probs", module_prefix + "certificate_status_head"
        ),
        Output_slice(
            "covariance_clean_diag", module_prefix + "denoising_head.log_variance"
        ),
        Output_slice(
            "nuisance_posterior_mean", module_prefix + "denoising_head.nuisance", 0, 8
        ),
        Output_slice(
            "nuisance_posterior_variance",
            module_prefix + "denoising_head.nuisance",
            8,
            16,
        ),
        Output_slice(
            "abstention_probability", module_prefix + "structure_head.abstain"
        ),
    )
    if module_prefix:
        mode = getattr(model, "excitation_mode", "legacy")
        sector_module = (
            "excitation_head"
            if mode == "coupled"
            else module_prefix + "w_head.sector_head"
        )
        physical_slices = base_slices + (
            Output_slice("sector_aggregate", sector_module),
            Output_slice("k2_vs_ge3", "k2_vs_ge3_head"),
            Output_slice(
                "measurement_probability",
                module_prefix + "denoising_head.probability",
            ),
        )
        if mode != "legacy":
            physical_slices += (
                Output_slice("operation_excitation_rate", "excitation_head"),
                Output_slice("operation_effective_mu", "excitation_head"),
                Output_slice("sector_intensity_scale", "excitation_head"),
            )
            if mode == "coupled":
                physical_slices += (
                    Output_slice("R1_intensity_scale", "excitation_head"),
                    Output_slice(
                        "measurement_intensity_contribution", "excitation_head"
                    ),
                )
        return physical_slices
    return base_slices


class Classifier_output_audit:
    def __init__(self, model, output_dir: str | Path, interval_steps: int = 20):
        if type(interval_steps) is not int or interval_steps < 1:
            raise ValueError("output audit interval must be positive")
        self.model = model
        self.output_dir = Path(output_dir)
        self.interval_steps = interval_steps
        self.slices = output_slices(model)
        self.output_registry = output_training_registry_for_model(model)
        self.is_physical_route = (
            type(model).__name__ == "V10_physical_aod_classifier"
            and hasattr(model, "core")
        )
        self.excitation_mode = (
            getattr(model, "excitation_mode", "legacy")
            if self.is_physical_route
            else None
        )
        self.audit_schema = (
            PHYSICAL_OUTPUT_AUDIT_SCHEMA
            if self.is_physical_route
            else LEGACY_OUTPUT_AUDIT_SCHEMA
        )
        self.physical_objective_bindings = (
            {
                name: [
                    {
                        "objective_name": binding.objective_name,
                        "authoritative_targets": list(binding.authoritative_targets),
                        "target_masks": list(binding.target_masks),
                        "effective_mask_key": binding.effective_mask_key,
                        "coverage_unit": binding.coverage_unit,
                        "relation": binding.relation,
                        "scope": binding.scope,
                    }
                    for binding in spec.objective_bindings
                ]
                for name, spec in sorted(self.output_registry.items())
                if spec.objective_bindings
            }
            if self.is_physical_route
            else {}
        )
        all_parameters = dict(model.named_parameters())
        names = {
            spec.module_name + suffix
            for spec in self.slices
            for suffix in (".weight", ".bias")
        }
        self.parameters = {name: all_parameters[name] for name in sorted(names)}
        self.initial = {
            name: value.detach().clone() for name, value in self.parameters.items()
        }
        self.records = []
        self.coverage = {}
        self.effective_objective_coverage = {}
        self.context = {}
        self.pending_coverage = []
        self.registered_cells = set()
        self.registered_cells_locked = False
        self.audited_cells = set()
        self.observed_fit_membership = {}
        self.committed_batch_memberships = []
        self.fit_membership_capture_status = "NOT_AVAILABLE_NO_PAIRED_IDENTITIES"
        self.final_global_step = None
        self.final_model_state_sha = None
        self.committed_step_ids = set()
        self.last_committed_record = None
        self.discarded_uncommitted_attempts = 0

    def bind_training_identity(
        self, *, identity: dict, run_id: str, execution_id: str
    ) -> None:
        if not isinstance(execution_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_]+", execution_id
        ):
            raise ValueError("output audit requires a safe execution identity")
        parameters = dict(self.model.named_parameters())
        self.parameters = {name: parameters[name] for name in self.parameters}
        self.initial = {
            name: value.detach().clone() for name, value in self.parameters.items()
        }
        self.context = {
            "source_identity": dict(identity),
            "run_id": run_id,
            "execution_id": execution_id,
            "delta_reference": "CURRENT_EXECUTION_INITIAL_STATE",
        }

    def checkpoint_state(self) -> dict:
        """Portable state required to continue a physical committed-step audit."""
        return {
            "schema": "V10ClassifierOutputAuditCheckpointV1",
            "producer_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "excitation_mode": self.excitation_mode,
            "context": copy.deepcopy(self.context),
            "records": copy.deepcopy(self.records), "coverage": copy.deepcopy(self.coverage),
            "effective_objective_coverage": copy.deepcopy(self.effective_objective_coverage),
            "pending_coverage": copy.deepcopy(self.pending_coverage),
            "registered_cells": sorted(self.registered_cells),
            "registered_cells_locked": self.registered_cells_locked,
            "audited_cells": sorted(self.audited_cells),
            "observed_fit_membership": copy.deepcopy(self.observed_fit_membership),
            "committed_batch_memberships": copy.deepcopy(self.committed_batch_memberships),
            "fit_membership_capture_status": self.fit_membership_capture_status,
            "committed_step_ids": sorted(self.committed_step_ids),
            "last_committed_record": copy.deepcopy(self.last_committed_record),
            "discarded_uncommitted_attempts": self.discarded_uncommitted_attempts,
            "initial": {name: value.detach().cpu().clone() for name, value in self.initial.items()},
        }

    def restore_checkpoint_state(self, state: dict, *, identity: dict) -> None:
        if (not isinstance(state, dict) or state.get("schema") != "V10ClassifierOutputAuditCheckpointV1"
                or state.get("producer_sha") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or state.get("context", {}).get("source_identity") != identity):
            raise ValueError("classifier output audit checkpoint state is not source-bound")
        required = ("excitation_mode", "records", "coverage", "effective_objective_coverage", "pending_coverage", "registered_cells", "audited_cells", "observed_fit_membership", "committed_batch_memberships", "committed_step_ids", "initial")
        if any(key not in state for key in required):
            raise ValueError("classifier output audit checkpoint state is incomplete")
        if state["excitation_mode"] != self.excitation_mode:
            raise ValueError("classifier output audit excitation mode mismatch")
        if set(state["initial"]) != set(self.initial):
            raise ValueError("classifier output audit initial parameter reference mismatch")
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != self.initial[name].shape
            or value.dtype != self.initial[name].dtype
            or not bool(torch.isfinite(value).all())
            for name, value in state["initial"].items()
        ):
            raise ValueError("classifier output audit initial parameter state is invalid")
        self.initial = {name: value.to(self.parameters[name].device) for name, value in state["initial"].items()}
        for name in ("records", "coverage", "effective_objective_coverage", "pending_coverage", "observed_fit_membership", "committed_batch_memberships", "last_committed_record"):
            setattr(self, name, copy.deepcopy(state.get(name)))
        self.registered_cells, self.audited_cells = set(state["registered_cells"]), set(state["audited_cells"])
        self.registered_cells_locked = bool(state.get("registered_cells_locked"))
        self.committed_step_ids = set(state["committed_step_ids"])
        self.fit_membership_capture_status = state["fit_membership_capture_status"]
        if type(state["discarded_uncommitted_attempts"]) is not int or state["discarded_uncommitted_attempts"] < 0:
            raise ValueError("classifier output audit checkpoint counter is invalid")
        self.discarded_uncommitted_attempts = state["discarded_uncommitted_attempts"]

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
                    raise ValueError(
                        "multi-cell output auditing requires explicit batch cell identity"
                    )
                cell = (next(iter(self.registered_cells)),)
            else:
                cell = ("UNSPECIFIED_CELL",)
        cell_id = str(cell[0]) if isinstance(cell, (list, tuple)) else str(cell)
        sample_ids = _strict_batch_strings(batch, "sample_id")
        acquisition_group_ids = _strict_batch_strings(
            batch, "acquisition_group_id"
        )
        if acquisition_group_ids and not sample_ids:
            raise ValueError(
                "output audit sample and acquisition-group identities must align"
            )
        if sample_ids and not acquisition_group_ids:
            # Older non-calibration datasets expose sample IDs without a
            # group-safe acquisition identity.  Preserve their general audit,
            # but do not claim row-level calibration membership from them.
            sample_ids = ()
        if sample_ids and len(sample_ids) != len(acquisition_group_ids):
            raise ValueError(
                "output audit sample and acquisition-group identities must align"
            )
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("output audit committed batch repeats a sample identity")
        coverage = {}
        effective_coverage = {}
        objective_packet = None
        if self.is_physical_route:
            from .physical_objective_contract import (
                validate_physical_objective_supervision,
            )

            objective_packet = validate_physical_objective_supervision(
                losses.get("objective_supervision"), batch
            )
            for spec in self.output_registry.values():
                for binding in spec.objective_bindings:
                    required_fields = (
                        *binding.authoritative_targets,
                        *binding.target_masks,
                    )
                    missing_fields = [
                        field for field in required_fields if field not in batch
                    ]
                    if missing_fields:
                        raise ValueError(
                            "physical objective binding lacks target or mask fields: "
                            f"{binding.objective_name}:{sorted(missing_fields)}"
                        )
        for name, spec in self.output_registry.items():
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
                        raise ValueError(
                            "registered output target mask shapes disagree"
                        )
                    mask = mask & other
                valid = int(mask.sum())
                available = mask.numel()
                unit = "MASK_ELEMENTS"
            else:
                target = batch[spec.authoritative_targets[0]]
                valid = available = target.numel()
                unit = "UNMASKED_TARGET_ELEMENTS"
            coverage[name] = {"valid": valid, "available": available, "unit": unit}
            if objective_packet is not None:
                for binding in spec.objective_bindings:
                    objective_loss = losses.get(binding.objective_name)
                    if (
                        not isinstance(objective_loss, torch.Tensor)
                        or objective_loss.ndim != 0
                        or not bool(torch.isfinite(objective_loss.detach()).all())
                    ):
                        raise ValueError(
                            "physical objective binding lacks its finite scalar loss: "
                            f"{binding.objective_name}"
                        )
                    mask = objective_packet["effective_masks"].get(
                        binding.effective_mask_key
                    )
                    if mask is None:
                        raise ValueError(
                            "physical objective binding references an unavailable effective mask"
                        )
                    row = {
                        "valid": int(mask.sum()),
                        "available": mask.numel(),
                        "unit": binding.coverage_unit,
                        "effective_mask_key": binding.effective_mask_key,
                    }
                    existing = effective_coverage.get(binding.objective_name)
                    if existing is not None and existing != row:
                        raise ValueError(
                            "physical objective coverage differs across equivalent bindings"
                        )
                    effective_coverage[binding.objective_name] = row
        self.pending_coverage.append(
            {
                "anticipated_global_step": global_step,
                "epoch": epoch,
                "cell_id": cell_id,
                "sample_ids": sample_ids,
                "acquisition_group_ids": acquisition_group_ids,
                "coverage": coverage,
                "effective_objective_coverage": effective_coverage,
            }
        )

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
        # Keep this import local: training_runtime imports observer types, while
        # this observer only needs its established canonical state-dict digest.
        from ml_recon.physics_hybrid_v10.track_a_n8.training_runtime import _state_dict_sha

        self.final_model_state_sha = _state_dict_sha(self.model.state_dict())
        if (
            self.last_committed_record is not None
            and self.last_committed_record["global_step"] == final_global_step
            and not any(row["global_step"] == final_global_step for row in self.records)
        ):
            self.on_step_end(
                record={**self.last_committed_record, "is_final_committed_step": True}
            )

    def on_step_end(self, *, record):
        if not record.get(
            "committed_step", record.get("optimizer_step_executed", False)
        ):
            return
        global_step = record["global_step"]
        if type(global_step) is not int or global_step < 1:
            raise ValueError("output audit committed step must be a positive integer")
        previously_observed = global_step in self.committed_step_ids
        self.committed_step_ids.add(global_step)
        cell_ids = tuple(str(value) for value in record.get("cell_ids", ()))
        if not cell_ids:
            staged_cells = {
                pending["cell_id"]
                for pending in self.pending_coverage
                if pending["anticipated_global_step"] == global_step
            }
            if len(staged_cells) == 1:
                # Compatibility for the generic unit-test dataset. Production
                # multi-cell records carry cell_ids explicitly.
                cell_ids = tuple(staged_cells)
        if len(cell_ids) != 1:
            raise ValueError("output audit requires a homogeneous committed cell batch")
        committed_membership = None
        cell_id = cell_ids[0]
        record_cells = {str(value) for value in record.get("registered_cells", ())}
        if self.registered_cells_locked:
            if record_cells and record_cells != self.registered_cells:
                raise ValueError(
                    "committed record registered cells disagree with audit configuration"
                )
            if cell_id not in self.registered_cells:
                raise ValueError(
                    f"committed record references unregistered cell: {cell_id}"
                )
        else:
            self.registered_cells.update(record_cells)
            self.registered_cells.add(cell_id)
        if not previously_observed:
            committed_membership = self._commit_coverage(global_step, cell_id)
        membership_sample_ids = tuple(
            committed_membership["sample_ids"]
            if committed_membership is not None
            else record.get("sample_ids", ())
        )
        membership_group_ids = tuple(
            committed_membership["acquisition_group_ids"]
            if committed_membership is not None
            else record.get("acquisition_group_ids", ())
        )
        self.last_committed_record = {
            **record,
            "cell_ids": cell_ids,
            "sample_ids": membership_sample_ids,
            "acquisition_group_ids": membership_group_ids,
        }
        capture = (
            cell_id not in self.audited_cells
            or global_step % self.interval_steps == 0
            or global_step == self.final_global_step
            or bool(record.get("is_final_committed_step", False))
        )
        if not capture:
            return
        pending = {
            "global_step": global_step,
            "epoch": record["epoch"],
            "cell_id": cell_id,
            "committed_step": True,
            "gradient_stage": "UNSCALED_AFTER_CLIP_AFTER_COMMITTED_STEP",
            "capture_reason": self._capture_reason(global_step, cell_id, record),
            "sample_ids": list(membership_sample_ids),
            "acquisition_group_ids": list(membership_group_ids),
        }
        values = {}
        for spec in self.slices:
            norms = []
            connected = False
            for suffix in (".weight", ".bias"):
                gradient = self.parameters[spec.module_name + suffix].grad
                if gradient is not None:
                    selected = (
                        gradient[spec.start : spec.stop]
                        if spec.start is not None
                        else gradient
                    )
                    if not bool(torch.isfinite(selected).all()):
                        raise FloatingPointError(
                            f"non-finite output slice gradient: {spec.output_name}"
                        )
                    connected = True
                    norms.append(float(torch.sum(selected.detach().double().square())))
            norm = sum(norms) ** 0.5
            values[spec.output_name] = {
                "tensor_gradient_present": connected,
                "slice_gradient_norm": norm,
                "slice_gradient_nonzero": norm > 0,
            }
        pending["outputs"] = values
        for spec in self.slices:
            squared = 0.0
            for suffix in (".weight", ".bias"):
                name = spec.module_name + suffix
                delta = self.parameters[name].detach() - self.initial[name]
                selected = (
                    delta[spec.start : spec.stop] if spec.start is not None else delta
                )
                if not bool(torch.isfinite(selected).all()):
                    raise FloatingPointError(
                        f"non-finite output slice update: {spec.output_name}"
                    )
                squared += float(selected.double().square().sum())
            pending["outputs"][spec.output_name]["parameter_delta_norm"] = squared**0.5
        self.records.append(pending)
        self.audited_cells.add(cell_id)
        self.write_outputs()

    def _commit_coverage(self, global_step: int, cell_id: str):
        matches = [
            index
            for index, pending in enumerate(self.pending_coverage)
            if (
                pending["anticipated_global_step"] == global_step
                and pending["cell_id"] == cell_id
            )
        ]
        if not matches:
            return None
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
        sample_ids = tuple(pending["sample_ids"])
        group_ids = tuple(pending["acquisition_group_ids"])
        for sample_id, group_id in zip(sample_ids, group_ids, strict=True):
            existing = self.observed_fit_membership.get(sample_id)
            if existing is not None and existing != group_id:
                raise ValueError(
                    "one sample identity cannot belong to multiple acquisition groups"
                )
            self.observed_fit_membership[sample_id] = group_id
        if sample_ids:
            self.fit_membership_capture_status = (
                "CAPTURED_FROM_COMMITTED_BATCH_IDENTITIES"
            )
            self.committed_batch_memberships.append(
                {
                    "global_step": global_step,
                    "sample_ids": list(sample_ids),
                    "acquisition_group_ids": list(group_ids),
                }
            )
        coverage = self.coverage.setdefault(cell_id, {})
        for name, values in pending["coverage"].items():
            row = coverage.setdefault(
                name, {"valid": 0, "available": 0, "unit": values["unit"]}
            )
            if row["unit"] != values["unit"]:
                raise ValueError("registered output coverage units disagree")
            row["valid"] += values["valid"]
            row["available"] += values["available"]
        if self.is_physical_route:
            objective_coverage = self.effective_objective_coverage.setdefault(cell_id, {})
            for name, values in pending["effective_objective_coverage"].items():
                row = objective_coverage.setdefault(
                    name,
                    {
                        "valid": 0,
                        "available": 0,
                        "unit": values["unit"],
                        "effective_mask_key": values["effective_mask_key"],
                    },
                )
                if (
                    row["unit"] != values["unit"]
                    or row["effective_mask_key"] != values["effective_mask_key"]
                ):
                    raise ValueError("physical objective coverage bindings disagree")
                row["valid"] += values["valid"]
                row["available"] += values["available"]
        return pending

    def _capture_reason(self, global_step, cell_id, record):
        if (
            bool(record.get("is_final_committed_step", False))
            or global_step == self.final_global_step
        ):
            return "FINAL_COMMITTED_STEP"
        if cell_id not in self.audited_cells:
            return "FIRST_COMMITTED_STEP_FOR_CELL"
        return "PERIODIC_GLOBAL_COMMITTED_STEP"

    def on_epoch_end(self, **_kwargs):
        self.write_outputs()

    def finalize(self):
        missing_cells = sorted(self.registered_cells - self.audited_cells)
        final_step_observed = any(
            row["global_step"] == self.final_global_step for row in self.records
        )
        complete = (
            bool(self.final_global_step) and final_step_observed and not missing_cells
        )
        result = {
            "schema": self.audit_schema,
            **self.context,
            **(
                {"excitation_mode": self.excitation_mode}
                if self.is_physical_route
                else {}
            ),
            "audited_steps": len(self.records),
            "records": self.records,
            "committed_step_ids": sorted(self.committed_step_ids),
            "committed_step_count": len(self.committed_step_ids),
            "registered_cells": sorted(self.registered_cells),
            "audited_cells": sorted(self.audited_cells),
            "observed_fit_membership": {
                "row_ids": sorted(self.observed_fit_membership),
                "acquisition_group_ids": [
                    self.observed_fit_membership[row_id]
                    for row_id in sorted(self.observed_fit_membership)
                ],
            },
            "fit_membership_capture_status": self.fit_membership_capture_status,
            "committed_batch_memberships": self.committed_batch_memberships,
            "final_committed_step": self.final_global_step,
            "final_committed_step_observed": final_step_observed,
            "audit_complete": complete,
            "audit_incomplete_reason": (
                None
                if complete
                else {
                    "missing_audited_cells": missing_cells,
                    "final_committed_step_observed": final_step_observed,
                }
            ),
            "discarded_uncommitted_backward_attempts": self.discarded_uncommitted_attempts,
            **(
                {
                    "static_target_availability": self.coverage,
                    "effective_objective_coverage": self.effective_objective_coverage,
                    "objective_bindings": self.physical_objective_bindings,
                }
                if self.is_physical_route
                else {"target_coverage": self.coverage}
            ),
            "coverage_scope": "COMMITTED_OPTIMIZER_STEPS_ONLY",
            "held_out_calibration_verified": False,
            "feedback_source_eligible": False,
            "scientific_gate_evaluated": False,
            "boundary": "Source/checkpoint-bound measurements still require external calibration review; this artifact alone is not calibration or scientific-gate evidence",
            "producer_source_sha": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "final_model_state_sha": self.final_model_state_sha,
        }
        canonical_payload = json.dumps(
            result,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        result["audit_canonical_sha256"] = hashlib.sha256(canonical_payload).hexdigest()
        return result

    def write_outputs(self):
        if not self.context:
            raise ValueError(
                "output audit must bind its source and execution before publication"
            )
        filename = f"classifier_output_slice_audit_{self.context['execution_id']}.json"
        atomic_training_json(self.output_dir / filename, self.finalize())
