"""Strict read-only bridge from prepared experiment evidence to V8 inference.

The bridge is intentionally narrower than the V7 bridge.  It accepts only a
frozen V8 checkpoint/report pair, builds command features from the immutable
command table bound by ``PreparedSequenceArtifactV1``, and emits layered shadow
diagnostics.  It has no import or call path to the runner, AWG uploaders, FPGA
programmers, command materializers, or delta manager.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml_recon.physics_hybrid_v8.contracts import HistoryTransitionProvenanceV1
from ml_recon.physics_hybrid_v8.feedback_models import NUISANCE_TARGET_NAMES

from .data_protocol import stable_json_hash
from .models import MeasurementSnapshot, TargetStateSpecV1
from .sequence_service import PreparedSequenceArtifactV1


V8_SHADOW_DIAGNOSTIC_SCHEMA = "v8_feedback_shadow_diagnostic_v1"
V8_EXPOSURE_PROVENANCE_SCHEMA = "v8_measurement_exposure_provenance_v1"
V8_BOUND_HISTORY_SCHEMA = "v8_bound_history_transition_v1"
V8_SNAPSHOT_PROVENANCE_SCHEMA = "v8_snapshot_prepared_provenance_v1"
V8_REPORT_SCHEMA = "v8_feedback_training_report_v1"
V8_CHECKPOINT_SCHEMA = "v8_feedback_inference_checkpoint_v1"
V8_SOURCE_AUDIT_SCHEMA = "v8_source_tree_audit_v1"
V8_UNCERTAINTY_SCHEMA = "v8_real_episode_uncertainty_contract_v1"
V8_FINALIZATION_SCHEMA = "v8_feedback_frozen_candidate_v1"
V8_FINALIZATION_POINTER_SCHEMA = "v8_feedback_frozen_candidate_pointer_v1"
V8_STATIC_REFIT_MANIFEST_SCHEMA = (
    "v8_feedback_static_all_eligible_refit_manifest_v1"
)
V8_STATIC_REFIT_CHECKPOINT_SCHEMA = (
    "v8_feedback_static_all_eligible_refit_checkpoint_v1"
)
V8_STATIC_REFIT_PREREQUISITE_SCHEMA = (
    "v8_feedback_static_refit_prerequisite_audit_v1"
)
V8_LOCKED_FINAL_EVALUATION_SCHEMA = (
    "v8_feedback_locked_final_ensemble_evaluation_v1"
)
V8_LOCKED_ACCESS_LEDGER_SCHEMA = "v8_locked_ensemble_final_access_ledger_v1"
V8_FINALIZATION_FILENAME = "frozen_candidate_manifest.json"
V8_ENSEMBLE_UNSUPPORTED_MESSAGE = (
    "V8 shadow refuses a frozen representative/median member through the "
    "single-checkpoint interface; pass the hash-bound frozen candidate manifest "
    "or pointer so every ensemble member is loaded and cross-scored"
)
V8_ENSEMBLE_MINIMUM_MEMBER_COUNT = 5
V8_UTILITY_ALEATORIC_STD_PENALTY = 0.5
V8_UTILITY_EPISTEMIC_STD_PENALTY = 1.0
V8_HARMFUL_UPPER_PROBABILITY_PENALTY = 0.25
V8_NUISANCE_PROVENANCE = "synthetic_prior_no_real_label"
V8_NUISANCE_REAL_SUPERVISION_MASK = (0, 0, 0, 0, 0)
SUPPORTED_SYSTEM_TYPES = frozenset({"3qubit", "4qubit"})
EXPOSURE_SOURCES = frozenset({
    "acquisition_authoritative_duration",
    "prepared_authoritative_duration",
})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HISTORY_OUTCOME_DEFAULTS = {
    "utility_delta": 0.0,
    "fidelity_after": 0.0,
    "retrieval_delta": 0.0,
    "log_count_ratio": 0.0,
    "multi_risk_delta": 0.0,
    "accepted": 0.0,
    "rollback": 0.0,
    "uncertainty": 1.0,
    "best_so_far": 0.0,
    "step_fraction": 0.0,
}
V8_REQUIRED_SOURCE_FILES = frozenset({
    "ml_recon/physics_hybrid_v8/causal_negative_control.py",
    "ml_recon/physics_hybrid_v8/command_features.py",
    "ml_recon/physics_hybrid_v8/compare_v7_baseline.py",
    "ml_recon/physics_hybrid_v8/contracts.py",
    "ml_recon/physics_hybrid_v8/evaluate_feedback_models.py",
    "ml_recon/physics_hybrid_v8/feedback_confidence.py",
    "ml_recon/physics_hybrid_v8/feedback_data.py",
    "ml_recon/physics_hybrid_v8/feedback_models.py",
    "ml_recon/physics_hybrid_v8/feedback_physics.py",
    "ml_recon/physics_hybrid_v8/feedback_synthetic.py",
    "ml_recon/physics_hybrid_v8/finalize_feedback_candidate.py",
    "ml_recon/physics_hybrid_v8/refit_feedback_candidate.py",
    "ml_recon/physics_hybrid_v8/state_pair_transport.py",
    "ml_recon/physics_hybrid_v8/train_feedback_models.py",
    "ml_recon/physics_hybrid_v7/feedback_classifier_prior.py",
    "ml_recon/physics_hybrid_v7/feedback_data.py",
    "ml_recon/physics_hybrid_v7/feedback_metrics.py",
    "ml_recon/physics_hybrid_v7/feedback_models.py",
    "ml_recon/physics_hybrid_v7/feedback_synthetic.py",
    "ml_recon/physics_hybrid_v7/geometry.py",
    "ml_recon/physics_hybrid_v7/physical_ceiling.py",
    "ml_recon/physics_hybrid_v7/train_feedback_models.py",
    "ml_recon/physics_hybrid_v5/command_contract.py",
    "ml_recon/physics_hybrid_v5/full_synthetic_generator.py",
    "ml_recon/physics_hybrid_v5/projective_action.py",
    "ml_recon/physics_hybrid_v4/phase_geometry.py",
    "ml_recon/measured_data/contracts.py",
    "ml_recon/measured_data/datasets.py",
    "ml_recon/physics_identification/operation_point_calibration.py",
    "system_control/qubit_recon_files/matrix/retrieval_efficiency_3qubit.txt",
    "system_control/qubit_recon_files/matrix/retrieval_efficiency_4qubit.txt",
})


def aggregate_v8_ensemble_outputs(
    member_outputs: Sequence[Mapping[str, Any]],
    member_weights: Sequence[float],
) -> Dict[str, Any]:
    """Aggregate independent members without collapsing epistemic uncertainty."""

    if len(member_outputs) < V8_ENSEMBLE_MINIMUM_MEMBER_COUNT:
        raise ValueError("V8 ensemble aggregation requires at least five members")
    if len(member_outputs) != len(member_weights):
        raise ValueError("V8 ensemble member outputs and weights differ in length")
    import torch

    reference = member_outputs[0]["expected_counts"]
    weights = torch.as_tensor(
        member_weights,
        dtype=reference.real.dtype,
        device=reference.device,
    )
    if (
        not bool(torch.isfinite(weights).all())
        or bool((weights <= 0.0).any())
        or not math.isclose(float(weights.sum().cpu()), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("V8 ensemble weights must be finite, positive, and sum to one")

    def stack(name: str) -> Any:
        values = [member[name] for member in member_outputs]
        shape = tuple(values[0].shape)
        if any(tuple(value.shape) != shape for value in values[1:]):
            raise ValueError(f"V8 ensemble output shape differs for {name}")
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError(f"V8 ensemble output is non-finite for {name}")
        return torch.stack(values, dim=0)

    def weighted_mean(values: Any) -> Any:
        view = (weights.shape[0],) + (1,) * (values.ndim - 1)
        return torch.sum(values * weights.view(view), dim=0)

    def weighted_variance(values: Any, mean: Any) -> Any:
        view = (weights.shape[0],) + (1,) * (values.ndim - 1)
        difference = values - mean.unsqueeze(0)
        squared = torch.square(torch.abs(difference))
        return torch.sum(squared * weights.view(view), dim=0)

    expected_counts_members = stack("expected_counts").float()
    expected_counts = weighted_mean(expected_counts_members)
    dispersion_members = stack("nb2_dispersion").float()
    count_aleatoric_variance = weighted_mean(
        expected_counts_members
        + dispersion_members.unsqueeze(-1) * torch.square(expected_counts_members)
    )
    count_epistemic_variance = weighted_variance(
        expected_counts_members,
        expected_counts,
    )
    rate_members = stack("total_rate_per_s").float()
    total_rate_per_s = weighted_mean(rate_members)
    rate_epistemic_variance = weighted_variance(rate_members, total_rate_per_s)
    log_rate_supervision_mask = stack("log_rate_supervision_mask").amin(dim=0)
    uncalibrated_total_members = stack("uncalibrated_total").float()
    uncalibrated_total = weighted_mean(uncalibrated_total_members)
    uncalibrated_total_epistemic_variance = weighted_variance(
        uncalibrated_total_members,
        uncalibrated_total,
    )

    nuisance_members = stack("nuisance_mean").float()
    nuisance_mean = weighted_mean(nuisance_members)
    nuisance_aleatoric_variance = weighted_mean(torch.stack([
        torch.exp(2.0 * member["nuisance_log_scale"].float())
        for member in member_outputs
    ], dim=0))
    nuisance_epistemic_variance = weighted_variance(
        nuisance_members,
        nuisance_mean,
    )
    nuisance_predictive_variance = (
        nuisance_aleatoric_variance + nuisance_epistemic_variance
    )

    utility_members = stack("utility_mean").float()
    utility_mean = weighted_mean(utility_members)
    utility_aleatoric_variance = weighted_mean(torch.stack([
        torch.exp(2.0 * member["utility_log_scale"].float())
        for member in member_outputs
    ], dim=0))
    utility_epistemic_variance = weighted_variance(utility_members, utility_mean)
    utility_predictive_variance = (
        utility_aleatoric_variance + utility_epistemic_variance
    )
    harmful_members = torch.stack([
        torch.sigmoid(member["harmful_logit"].float())
        for member in member_outputs
    ], dim=0)
    harmful_probability = weighted_mean(harmful_members)
    harmful_epistemic_variance = weighted_variance(
        harmful_members,
        harmful_probability,
    )

    rho_current_members = stack("rho_current")
    rho_current = weighted_mean(rho_current_members)
    rho_current_epistemic_variance = weighted_variance(
        rho_current_members,
        rho_current,
    )
    rho_next_members = stack("rho_next")
    rho_next = weighted_mean(rho_next_members)
    rho_next_epistemic_variance = weighted_variance(rho_next_members, rho_next)

    result = {
        "member_count": len(member_outputs),
        "expected_counts": expected_counts,
        "expected_total": expected_counts.sum(dim=-1),
        "count_probability": expected_counts / expected_counts.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12),
        "operation_probability": weighted_mean(stack("operation_probability").float()),
        "nb2_dispersion": weighted_mean(dispersion_members),
        "count_aleatoric_variance": count_aleatoric_variance,
        "count_epistemic_variance": count_epistemic_variance,
        "count_predictive_variance": (
            count_aleatoric_variance + count_epistemic_variance
        ),
        "total_rate_per_s": total_rate_per_s,
        "log_rate": torch.log(total_rate_per_s.clamp_min(1e-12)),
        "rate_epistemic_variance": rate_epistemic_variance,
        "log_rate_supervision_mask": log_rate_supervision_mask,
        "authoritative_log_rate": torch.where(
            log_rate_supervision_mask > 0.5,
            torch.log(total_rate_per_s.clamp_min(1e-12)),
            torch.zeros_like(total_rate_per_s),
        ),
        "authoritative_total_rate_per_s": weighted_mean(
            stack("authoritative_total_rate_per_s")
        ),
        "uncalibrated_total": uncalibrated_total,
        "uncalibrated_log_total": torch.log(
            uncalibrated_total.clamp_min(1e-12)
        ),
        "uncalibrated_total_epistemic_variance": (
            uncalibrated_total_epistemic_variance
        ),
        "uncalibrated_total_supervision_mask": stack(
            "uncalibrated_total_supervision_mask"
        ).amin(dim=0),
        "rho_current": rho_current,
        "rho_next": rho_next,
        "rho_current_epistemic_variance": rho_current_epistemic_variance,
        "rho_next_epistemic_variance": rho_next_epistemic_variance,
        "rho_current_epistemic_frobenius_variance": (
            rho_current_epistemic_variance.flatten(start_dim=1).sum(dim=-1)
        ),
        "rho_next_epistemic_frobenius_variance": (
            rho_next_epistemic_variance.flatten(start_dim=1).sum(dim=-1)
        ),
        "nuisance_mean": nuisance_mean,
        "nuisance_log_scale": 0.5 * torch.log(
            nuisance_predictive_variance.clamp_min(1e-12)
        ),
        "nuisance_aleatoric_variance": nuisance_aleatoric_variance,
        "nuisance_epistemic_variance": nuisance_epistemic_variance,
        "nuisance_predictive_variance": nuisance_predictive_variance,
        "utility_mean": utility_mean,
        "utility_log_scale": 0.5 * torch.log(
            utility_predictive_variance.clamp_min(1e-12)
        ),
        "utility_aleatoric_variance": utility_aleatoric_variance,
        "utility_epistemic_variance": utility_epistemic_variance,
        "utility_predictive_variance": utility_predictive_variance,
        "harmful_probability": harmful_probability,
        "harmful_logit": torch.logit(
            harmful_probability.clamp(1e-6, 1.0 - 1e-6)
        ),
        "harmful_epistemic_variance": harmful_epistemic_variance,
    }
    weighted_keys = (
        "rho_current_w_subspace_mass",
        "rho_next_w_subspace_mass",
        "rho_current_leakage",
        "rho_next_leakage",
        "multi_interval_prediction",
        "ood_energy",
        "command_conditioning_fraction",
        "measurement_transport_embedding",
        "static_state_embedding",
        "static_transport_embedding",
    )
    for name in weighted_keys:
        result[name] = weighted_mean(stack(name))
    result["w_subspace_mass"] = result["rho_next_w_subspace_mass"]
    result["leakage"] = result["rho_next_leakage"]
    result["rate_equivariance_verifiable"] = torch.stack([
        member["rate_equivariance_verifiable"] for member in member_outputs
    ], dim=0).amin(dim=0)
    for name in (
        "calibration_known_mask",
        "calibration_identity_match_mask",
        "retrieval_calibration_authorized_mask",
        "nuisance_real_supervision_mask",
    ):
        result[name] = stack(name).amin(dim=0)
    result["nuisance_synthetic_prior_mask"] = stack(
        "nuisance_synthetic_prior_mask"
    ).amax(dim=0)
    return result


@dataclass(frozen=True)
class MeasurementExposureProvenanceV1:
    """Hash-bound proof that a snapshot duration is authoritative."""

    acquisition_id: str
    source: str
    exposure_s: float
    exposure_hash: str
    schema: str = V8_EXPOSURE_PROVENANCE_SCHEMA

    @classmethod
    def bind(
        cls,
        *,
        acquisition_id: str,
        source: str,
        exposure_s: float,
    ) -> "MeasurementExposureProvenanceV1":
        identity = {
            "schema": V8_EXPOSURE_PROVENANCE_SCHEMA,
            "acquisition_id": str(acquisition_id),
            "source": str(source),
            "exposure_s": float(exposure_s),
        }
        return cls(
            acquisition_id=str(acquisition_id),
            source=str(source),
            exposure_s=float(exposure_s),
            exposure_hash=stable_json_hash(identity),
        )

    @classmethod
    def from_json_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "MeasurementExposureProvenanceV1":
        return cls(
            acquisition_id=str(value.get("acquisition_id") or ""),
            source=str(value.get("source") or ""),
            exposure_s=float(value.get("exposure_s", 0.0)),
            exposure_hash=str(value.get("exposure_hash") or ""),
            schema=str(value.get("schema") or ""),
        )

    def validate(self) -> None:
        if self.schema != V8_EXPOSURE_PROVENANCE_SCHEMA:
            raise ValueError("unsupported V8 exposure provenance schema")
        if not self.acquisition_id.strip():
            raise ValueError("exposure acquisition_id is required")
        if self.source not in EXPOSURE_SOURCES:
            raise ValueError("exposure source is not authoritative")
        if not math.isfinite(float(self.exposure_s)) or float(self.exposure_s) <= 0.0:
            raise ValueError("authoritative exposure must be finite and positive")
        identity = {
            "schema": self.schema,
            "acquisition_id": self.acquisition_id,
            "source": self.source,
            "exposure_s": float(self.exposure_s),
        }
        if stable_json_hash(identity) != self.exposure_hash:
            raise ValueError("V8 exposure provenance hash mismatch")

    def to_json_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class BoundHistoryTransitionV1:
    """Typed binding of executed action values to V8 transition provenance."""

    provenance: HistoryTransitionProvenanceV1
    actual_loaded_action: Mapping[str, Any]
    outcome_features: Mapping[str, float] = field(default_factory=dict)
    outcome_hash: str = ""
    schema: str = V8_BOUND_HISTORY_SCHEMA

    @classmethod
    def bind(
        cls,
        provenance: HistoryTransitionProvenanceV1,
        actual_loaded_action: Mapping[str, Any],
        *,
        outcome_features: Optional[Mapping[str, float]] = None,
    ) -> "BoundHistoryTransitionV1":
        outcomes = dict(outcome_features or {})
        return cls(
            provenance=provenance,
            actual_loaded_action=dict(actual_loaded_action),
            outcome_features=outcomes,
            outcome_hash=stable_json_hash({
                "schema": "v8_history_outcome_features_v1",
                "values": outcomes,
            }),
        )

    def validate(self, *, system_type: str) -> Tuple[float, ...]:
        if self.schema != V8_BOUND_HISTORY_SCHEMA:
            raise ValueError("unsupported bound-history schema")
        if not isinstance(self.provenance, HistoryTransitionProvenanceV1):
            raise TypeError("history requires HistoryTransitionProvenanceV1")
        self.provenance.validate()
        if self.provenance.system_type != system_type:
            raise ValueError("history transition system differs from current system")
        action = dict(self.actual_loaded_action)
        if stable_json_hash(action) != self.provenance.actual_action_hash:
            raise ValueError("actual_loaded_action hash differs from transition provenance")
        if action.get("schema") != "GlobalWProjectiveActionV1":
            raise ValueError("history action is not GlobalWProjectiveActionV1")
        if action.get("system_type") != system_type:
            raise ValueError("history action system differs from transition provenance")
        coordinate_count = 2 if system_type == "3qubit" else 3
        amplitude = np.asarray(action.get("amplitude_coordinates"), dtype=float)
        phase = np.asarray(action.get("phase_coordinates"), dtype=float)
        if amplitude.shape != (coordinate_count,) or phase.shape != (coordinate_count,):
            raise ValueError("actual_loaded_action coordinate shape changed")
        if not np.all(np.isfinite(amplitude)) or not np.all(np.isfinite(phase)):
            raise ValueError("actual_loaded_action coordinates must be finite")
        for name in (
            "brightness_delta",
            "read_action",
            "local_residual",
            "cycle_residual",
        ):
            value = float(action.get(name, 0.0))
            if not math.isfinite(value) or abs(value) > 1e-12:
                raise ValueError(f"unsupported nonzero history action field: {name}")
        outcomes = dict(self.outcome_features)
        unknown = sorted(set(outcomes) - set(HISTORY_OUTCOME_DEFAULTS))
        if unknown:
            raise ValueError(f"unknown history outcome fields: {unknown}")
        if stable_json_hash({
            "schema": "v8_history_outcome_features_v1",
            "values": outcomes,
        }) != self.outcome_hash:
            raise ValueError("history outcome hash mismatch")
        for name, value in outcomes.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"history outcome {name} is non-finite")
        for name in ("accepted", "rollback"):
            if float(outcomes.get(name, HISTORY_OUTCOME_DEFAULTS[name])) not in {0.0, 1.0}:
                raise ValueError(f"history outcome {name} must be binary")
        padded_amplitude = np.zeros(3, dtype=np.float32)
        padded_phase = np.zeros(3, dtype=np.float32)
        padded_amplitude[:coordinate_count] = amplitude
        padded_phase[:coordinate_count] = phase
        return tuple(float(value) for value in np.concatenate((
            padded_amplitude,
            padded_phase,
        )))

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "provenance": self.provenance.to_json_dict(),
            "actual_loaded_action": dict(self.actual_loaded_action),
            "outcome_features": dict(self.outcome_features),
            "outcome_hash": self.outcome_hash,
        }


@dataclass(frozen=True)
class V8ShadowDiagnostic:
    system_type: str
    created_at: float
    target_hash: str
    prepared_artifact_id: str
    checkpoint_sha256: str
    report_hash: str
    abstain: bool
    abstain_reasons: Tuple[str, ...]
    recommended_shadow_action: Tuple[float, ...]
    hardware_action: Tuple[float, ...]
    selected_candidate: str
    gates: Mapping[str, Any]
    layered_diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    guard: Mapping[str, Any]
    schema: str = V8_SHADOW_DIAGNOSTIC_SCHEMA

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "created_at": float(self.created_at),
            "target_hash": self.target_hash,
            "prepared_artifact_id": self.prepared_artifact_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "report_hash": self.report_hash,
            "abstain": bool(self.abstain),
            "abstain_reasons": list(self.abstain_reasons),
            "recommended_shadow_action": list(self.recommended_shadow_action),
            "hardware_action": list(self.hardware_action),
            "selected_candidate": self.selected_candidate,
            "gates": dict(self.gates),
            "layered_diagnostics": dict(self.layered_diagnostics),
            "provenance": dict(self.provenance),
            "guard": dict(self.guard),
        }


class V8FeedbackBridge:
    """Frozen, fail-closed V8 shadow inference bridge."""

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        training_report_path: Optional[str | Path] = None,
        *,
        device: str = "auto",
        frozen_candidate_path: Optional[str | Path] = None,
    ) -> None:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("V8 device must be auto, cpu, or cuda")
        has_single_member = checkpoint_path is not None and training_report_path is not None
        has_frozen_ensemble = frozen_candidate_path is not None
        if has_single_member == has_frozen_ensemble:
            raise ValueError(
                "V8 bridge requires exactly one frozen ensemble manifest/pointer "
                "or one checkpoint/report pair"
            )
        if (checkpoint_path is None) != (training_report_path is None):
            raise ValueError("V8 single-member checkpoint and report must be paired")
        self.checkpoint_path = (
            Path(checkpoint_path).resolve() if checkpoint_path is not None else None
        )
        self.training_report_path = (
            Path(training_report_path).resolve()
            if training_report_path is not None
            else None
        )
        self.frozen_candidate_path = (
            Path(frozen_candidate_path).resolve()
            if frozen_candidate_path is not None
            else None
        )
        self.device_name = device
        self.repository_root = Path(__file__).resolve().parents[2]
        self._runtime: Optional[Dict[str, Any]] = None

    def diagnose(
        self,
        target_state: TargetStateSpecV1,
        snapshot: MeasurementSnapshot,
        prepared_artifact: PreparedSequenceArtifactV1,
        *,
        episode_history: Sequence[BoundHistoryTransitionV1] = (),
        exposure_provenance: Optional[MeasurementExposureProvenanceV1] = None,
    ) -> V8ShadowDiagnostic:
        runtime = self._load_runtime()
        if target_state.system_type != snapshot.system_type:
            raise ValueError("V8 target and snapshot system types differ")
        if prepared_artifact.system_type != snapshot.system_type:
            raise ValueError("V8 prepared artifact and snapshot system types differ")
        if prepared_artifact.target_hash != target_state.target_hash:
            raise ValueError("V8 target hash differs from prepared artifact")
        command_table, command_features = self._verify_prepared_artifact(
            prepared_artifact
        )
        snapshot_provenance = self._snapshot_provenance(
            snapshot,
            prepared_artifact,
            runtime["checkpoint_sha256"],
        )
        exposure = self._resolve_exposure(snapshot, exposure_provenance)
        arrays = self._build_inputs(
            runtime,
            target_state,
            snapshot,
            command_features,
            episode_history,
        )
        torch = runtime["torch"]
        device = runtime["device"]
        operation = torch.from_numpy(arrays["operation_features"]).to(device)
        target = torch.from_numpy(arrays["target_features"]).to(device)
        graph = torch.from_numpy(arrays["graph_kernel"]).to(device)
        history = torch.from_numpy(arrays["history"]).to(device)
        history_mask = torch.from_numpy(arrays["history_mask"]).to(device)
        action_mask = torch.from_numpy(arrays["action_mask"]).to(device)
        command = torch.from_numpy(arrays["command_features"]).to(device)
        command_mask = torch.from_numpy(arrays["command_mask"]).to(device)
        calibration_known_mask = torch.from_numpy(
            arrays["calibration_known_mask"]
        ).to(device)
        calibration_identity_match_mask = torch.from_numpy(
            arrays["calibration_identity_match_mask"]
        ).to(device)
        exposure_tensor = (
            torch.as_tensor([exposure["exposure_s"]], dtype=torch.float32, device=device)
            if exposure["known"]
            else None
        )
        exposure_mask = torch.as_tensor(
            [float(exposure["known"])], dtype=torch.float32, device=device
        )
        zero_action = torch.zeros_like(action_mask)
        common_model_kwargs = {
            "system_type": snapshot.system_type,
            "history": history,
            "history_mask": history_mask,
            "exposure_s": exposure_tensor,
            "exposure_known_mask": exposure_mask,
            "command_features": command,
            "command_mask": command_mask,
            "calibration_known_mask": calibration_known_mask,
            "calibration_identity_match_mask": calibration_identity_match_mask,
        }
        member_currents: List[Mapping[str, Any]] = []
        member_model_kwargs: List[Mapping[str, Any]] = []
        with torch.no_grad():
            for member in runtime["members"]:
                member_kwargs = {
                    **common_model_kwargs,
                    **self._classifier_features(member, snapshot, target),
                }
                member_currents.append(member["model"].model_a(
                    operation,
                    target,
                    graph,
                    zero_action,
                    action_mask,
                    **member_kwargs,
                ))
                member_model_kwargs.append(member_kwargs)
        is_ensemble = runtime["mode"] in {
            "frozen_multi_seed_ensemble",
            "static_refit_multi_seed_ensemble",
        }
        is_static_refit = bool(runtime.get("static_only_refit", False))
        current = (
            aggregate_v8_ensemble_outputs(
                member_currents,
                runtime["member_weights"],
            )
            if is_ensemble
            else member_currents[0]
        )

        gates = self._action_gates(runtime, snapshot.system_type)
        abstain_reason_list = [
            name for name, value in (
                ("real_action_supervision_gate_failed", gates["real_action_supervision"]),
                ("causal_negative_control_gate_failed", gates["causal_context"]),
                ("real_episode_calibration_gate_failed", gates["real_episode_calibration"]),
                (
                    "true_multi_member_inference_unavailable",
                    gates["true_multi_member_inference"],
                ),
                (
                    "frozen_action_policy_gate_failed",
                    gates["frozen_action_policy_eligible"],
                ),
            )
            if not value
        ]
        if is_static_refit:
            abstain_reason_list.append("static_refit_action_surface_invalidated")
        abstain_reasons = tuple(dict.fromkeys(abstain_reason_list))
        selected_candidate = "no_op"
        recommended_action = np.zeros(6, dtype=np.float32)
        candidate_diagnostics: Sequence[Mapping[str, Any]] = ()
        hypothetical_candidate = "no_op"
        if is_ensemble and not is_static_refit:
            hypothetical_candidate, hypothetical_action, candidate_diagnostics = (
                self._rank_ensemble_shadow_candidates(
                    runtime,
                    arrays,
                    member_currents,
                    member_model_kwargs,
                )
            )
            if not abstain_reasons:
                selected_candidate = hypothetical_candidate
                recommended_action = hypothetical_action

        static_layer = {
            "evidence_scope": "static_measured_snapshot",
            "observed_total_counts": float(np.sum(snapshot.coin_bins)),
            "expected_total_counts": float(current["expected_total"][0].cpu()),
            "expected_counts": current["expected_counts"][0].float().cpu().tolist(),
            "log_rate": (
                float(current["authoritative_log_rate"][0].cpu())
                if exposure["known"]
                else None
            ),
            "log_rate_authoritative": bool(exposure["known"]),
            "log_rate_supervision_mask": float(
                current["log_rate_supervision_mask"][0].cpu()
            ),
            "uncalibrated_log_total": (
                float(current["uncalibrated_log_total"][0].cpu())
                if not exposure["known"]
                else None
            ),
            "uncalibrated_total_supervision_mask": float(
                current["uncalibrated_total_supervision_mask"][0].cpu()
            ),
            "nb2_dispersion": float(current["nb2_dispersion"][0].cpu()),
            "conditional_w_core_rho_real": (
                current["rho_current"][0].real.float().cpu().tolist()
            ),
            "conditional_w_core_rho_imag": (
                current["rho_current"][0].imag.float().cpu().tolist()
            ),
            "w_subspace_mass": float(
                current["rho_current_w_subspace_mass"][0].cpu()
            ),
            "leakage": float(current["rho_current_leakage"][0].cpu()),
            "nuisance_mean": current["nuisance_mean"][0].float().cpu().tolist(),
            "nuisance": {
                "names": list(NUISANCE_TARGET_NAMES),
                "mean": current["nuisance_mean"][0].float().cpu().tolist(),
                "provenance": V8_NUISANCE_PROVENANCE,
                "real_supervision_mask": list(V8_NUISANCE_REAL_SUPERVISION_MASK),
                "synthetic_prior_mask": (
                    current["nuisance_synthetic_prior_mask"][0]
                    .float()
                    .cpu()
                    .tolist()
                ),
                "may_be_reported_as_real_measured_nuisance": False,
            },
            "real_static_supervision": {
                "nb2_dispersion": {
                    "mask": 1,
                    "provenance": "measured_count_likelihood_dispersion_proxy",
                    "direct_label": False,
                },
                "global_multi_click_upper_bound_proxy": {
                    "mask": [
                        int(value is not None)
                        for value in (
                            snapshot.multi2_count,
                            snapshot.multi3_count,
                            snapshot.multi4_count,
                        )
                    ],
                    "provenance": (
                        "global_multi_click_upper_bound_masked_proxy_"
                        "not_operation_local_label"
                    ),
                    "prediction": (
                        current["multi_interval_prediction"][0]
                        .float()
                        .cpu()
                        .tolist()
                    ),
                    "direct_label": False,
                },
                "nuisance": {
                    "mask": list(V8_NUISANCE_REAL_SUPERVISION_MASK),
                    "provenance": V8_NUISANCE_PROVENANCE,
                },
            },
            "retrieval_calibration": {
                "calibration_known_mask": float(
                    current["calibration_known_mask"][0].cpu()
                ),
                "calibration_identity_match_mask": float(
                    current["calibration_identity_match_mask"][0].cpu()
                ),
                "retrieval_relative_feature_authorized_mask": float(
                    current["retrieval_calibration_authorized_mask"][0].cpu()
                ),
                "provenance": (
                    "unavailable_no_hash_bound_per_snapshot_retrieval_efficiency_identity"
                ),
                "declared_snapshot_calibration_hash": snapshot.calibration_snapshot_hash,
                "declared_snapshot_calibration_scope": (
                    "phase_or_operation_calibration_not_retrieval_efficiency_map"
                ),
            },
            "command_feature_availability": (
                np.mean(command_features.feature_mask, axis=0).tolist()
            ),
            "command_conditioning_fraction": float(
                current["command_conditioning_fraction"][0].cpu()
            ),
            "exposure": exposure,
            "rate_equivariance_verifiable": bool(
                current["rate_equivariance_verifiable"][0].cpu() > 0.5
            ),
        }
        if is_ensemble:
            static_layer["ensemble"] = {
                "member_count": len(runtime["members"]),
                "member_weights": list(runtime["member_weights"]),
                "count_aleatoric_variance": (
                    current["count_aleatoric_variance"][0].float().cpu().tolist()
                ),
                "count_epistemic_variance": (
                    current["count_epistemic_variance"][0].float().cpu().tolist()
                ),
                "count_predictive_variance": (
                    current["count_predictive_variance"][0].float().cpu().tolist()
                ),
                "rate_epistemic_variance": (
                    float(current["rate_epistemic_variance"][0].cpu())
                    if exposure["known"]
                    else None
                ),
                "uncalibrated_total_epistemic_variance": (
                    float(
                        current["uncalibrated_total_epistemic_variance"][0].cpu()
                    )
                    if not exposure["known"]
                    else None
                ),
                "rho_epistemic_frobenius_variance": float(
                    current["rho_current_epistemic_frobenius_variance"][0].cpu()
                ),
                "nuisance_aleatoric_variance": (
                    current["nuisance_aleatoric_variance"][0].float().cpu().tolist()
                ),
                "nuisance_epistemic_variance": (
                    current["nuisance_epistemic_variance"][0].float().cpu().tolist()
                ),
                "member_expected_total_counts": [
                    float(value["expected_total"][0].cpu())
                    for value in member_currents
                ],
                "member_total_rate_per_s": (
                    [
                        float(value["total_rate_per_s"][0].cpu())
                        for value in member_currents
                    ]
                    if exposure["known"]
                    else None
                ),
                "member_uncalibrated_total": (
                    [
                        float(value["uncalibrated_total"][0].cpu())
                        for value in member_currents
                    ]
                    if not exposure["known"]
                    else None
                ),
            }
        weak_layer = {
            "evidence_scope": "weak_state_pair_transport_only",
            "measurement_transport_embedding": (
                current["measurement_transport_embedding"][0].float().cpu().tolist()
            ),
            "real_action_supervision_mask": 0,
            "real_jacobian_supervision_mask": 0,
            "counts_as_real_causality_evidence": False,
            "contract": runtime["contract"].get("state_pair_transport_weak", {}),
        }
        synthetic_layer = {
            "evidence_scope": "synthetic_action_response_only",
            "candidate_diagnostics": list(candidate_diagnostics),
            "hypothetical_cross_scored_candidate": hypothetical_candidate,
            "shared_candidate_set_used": is_ensemble,
            "all_candidates_scored_by_all_members": is_ensemble,
            "command_feature_support": runtime["contract"].get(
                "synthetic_command_feature_support", {}
            ),
            "cannot_authorize_real_action": True,
            "action_surface_invalidated": is_static_refit,
            "action_surface_invalidation_reason": (
                "static_only_all_eligible_refit"
                if is_static_refit
                else None
            ),
        }
        real_layer = {
            "evidence_scope": "real_paired_transition_or_signed_probe_only",
            "history_transition_count": int(np.sum(arrays["history_mask"])),
            "gates": gates,
            "abstain": bool(abstain_reasons),
            "abstain_reasons": list(abstain_reasons),
            "uncertainty_calibration": runtime["uncertainty_calibration"],
            "causal_context_gate": runtime["causal_context_gate"],
        }
        provenance = {
            "snapshot": snapshot_provenance,
            "exposure": exposure,
            "command": command_features.to_json_dict(),
            "command_table_id": command_table.table_id,
            "command_table_hash": command_table.table_hash,
            "checkpoint_hash_verified": True,
            "report_hash_verified": True,
            "source_tree_hash_verified": True,
            "data_hash_verified": True,
            "split_hash_verified": True,
            "model_hash_verified": True,
            "contract_hash_verified": True,
            "inference_scope": runtime["mode"],
            "runtime_member_count": len(runtime["members"]),
            "ensemble_inference_supported": bool(
                runtime["ensemble_inference_supported"]
            ),
                "representative_member_inference_forbidden": True,
            "static_only_refit": is_static_refit,
            "action_surface_invalidated": is_static_refit,
            "ensemble_sha256": runtime.get("ensemble_sha256"),
            "manifest_hash": runtime.get("manifest_hash"),
            "manifest_file_sha256": runtime.get("manifest_file_sha256"),
            "manifest_path": runtime.get("manifest_path"),
            "pointer_path": runtime.get("pointer_path"),
            "members": [
                {
                    "seed": (
                        int(member["report"].get("seed", -1))
                        if member["report"].get("seed") is not None
                        else None
                    ),
                    "weight": float(runtime["member_weights"][index]),
                    "checkpoint_sha256": member["checkpoint_sha256"],
                    "report_hash": member["report"]["report_hash"],
                }
                for index, member in enumerate(runtime["members"])
            ],
            "history_actions": [item.to_json_dict() for item in episode_history],
        }
        zero_hardware_action = (0.0,) * 6
        return V8ShadowDiagnostic(
            system_type=snapshot.system_type,
            created_at=time.time(),
            target_hash=target_state.target_hash,
            prepared_artifact_id=prepared_artifact.artifact_id,
            checkpoint_sha256=runtime["checkpoint_sha256"],
            report_hash=runtime["report"]["report_hash"],
            abstain=bool(abstain_reasons),
            abstain_reasons=abstain_reasons,
            recommended_shadow_action=tuple(float(value) for value in recommended_action),
            hardware_action=zero_hardware_action,
            selected_candidate=selected_candidate,
            gates=gates,
            layered_diagnostics={
                "static": static_layer,
                "weak": weak_layer,
                "synthetic": synthetic_layer,
                "real": real_layer,
            },
            provenance=provenance,
            guard={
                "accepted": False,
                "shadow_only": True,
                "frozen_ensemble_candidate_supported": is_ensemble,
                "static_refit_inference_supported": is_static_refit,
                "action_surface_invalidated": is_static_refit,
                "hardware_apply_blocked": True,
                "hardware_apply_reason": "v8_shadow has no write authority",
                "materialized": False,
                "bundle_staged": False,
                "uploaded": False,
                "delta_written": False,
                "active_command_changed": False,
            },
        )

    def _load_runtime(self) -> Dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        if self.frozen_candidate_path is not None:
            self._runtime = self._load_ensemble_runtime(self.frozen_candidate_path)
            return self._runtime
        if (
            self.checkpoint_path is None
            or self.training_report_path is None
            or not self.checkpoint_path.is_file()
            or not self.training_report_path.is_file()
        ):
            raise FileNotFoundError("V8 checkpoint and training report are both required")
        self._reject_unsupported_ensemble_candidate()
        member = self._load_member_runtime(
            self.checkpoint_path,
            self.training_report_path,
            require_declared_checkpoint_path=True,
        )
        member.update({
            "mode": "single_member_static_shadow_only",
            "members": (member,),
            "member_weights": (1.0,),
            "ensemble_inference_supported": False,
            "input_cache": {},
        })
        self._runtime = member
        return self._runtime

    def _load_member_runtime(
        self,
        checkpoint_path: Path,
        training_report_path: Path,
        *,
        require_declared_checkpoint_path: bool,
    ) -> Dict[str, Any]:
        if not checkpoint_path.is_file() or not training_report_path.is_file():
            raise FileNotFoundError("V8 frozen ensemble member is incomplete")
        import torch

        from ml_recon.physics_hybrid_v8.feedback_models import (
            V8FeedbackModelBundle,
            V8FeedbackModelConfig,
        )

        report = self._read_json(training_report_path)
        if report.get("schema") != V8_REPORT_SCHEMA:
            raise ValueError("V8 training report schema changed")
        report_hash = str(report.get("report_hash") or "")
        unsigned_report = dict(report)
        unsigned_report.pop("report_hash", None)
        if stable_json_hash(unsigned_report) != report_hash:
            raise ValueError("V8 training report hash mismatch")
        checkpoint_sha256 = self._sha256_file(checkpoint_path)
        report_checkpoint = dict(report.get("checkpoint") or {})
        if report_checkpoint.get("sha256") != checkpoint_sha256:
            raise ValueError("V8 checkpoint hash differs from training report")
        if report_checkpoint.get("schema") != V8_CHECKPOINT_SCHEMA:
            raise ValueError("V8 report checkpoint schema changed")
        declared_checkpoint = Path(str(report_checkpoint.get("path") or "")).resolve()
        if require_declared_checkpoint_path and declared_checkpoint != checkpoint_path:
            raise ValueError("V8 report points to a different checkpoint path")

        device_name = self.device_name
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("V8 CUDA inference was requested but CUDA is unavailable")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device_name,
            weights_only=False,
        )
        if checkpoint.get("schema") != V8_CHECKPOINT_SCHEMA:
            raise ValueError("V8 inference checkpoint schema changed")
        self._verify_research_only_contract(report, checkpoint)
        contract = self._verify_training_contract(report, checkpoint)
        config_path, config = self._verify_source_tree_and_config(
            contract["source_tree_audit"],
            str(contract["config_sha256"]),
        )
        self._verify_measured_data(contract["measured_file_audit"], config, config_path)
        self._verify_split_hashes(contract["split_hash_by_system"])

        model_config = V8FeedbackModelConfig.from_mapping(checkpoint["model_config"])
        model = V8FeedbackModelBundle(model_config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        expected_model_audit_hash = str(contract["model_audit_sha256"])
        if stable_json_hash(checkpoint.get("audit") or {}) != expected_model_audit_hash:
            raise ValueError("V8 checkpoint model audit hash mismatch")
        if stable_json_hash(model.audit_dict()) != expected_model_audit_hash:
            raise ValueError("V8 runtime model audit differs from checkpoint")
        device = torch.device(device_name)
        model.to(device).eval()
        classifier_prior, classifier_contract = self._load_classifier_prior(
            checkpoint,
            model,
            device,
        )
        return {
            "torch": torch,
            "device": device,
            "model": model,
            "classifier_prior": classifier_prior,
            "classifier_contract": classifier_contract,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_path": str(checkpoint_path),
            "report": report,
            "training_report_path": str(training_report_path),
            "contract": contract,
            "config": config,
            "config_path": str(config_path),
            "uncertainty_calibration": dict(
                checkpoint.get("uncertainty_calibration") or {}
            ),
            "causal_context_gate": dict(checkpoint.get("causal_context_gate") or {}),
        }

    def _load_ensemble_runtime(self, candidate_path: Path) -> Dict[str, Any]:
        if not candidate_path.is_file():
            raise FileNotFoundError("V8 frozen candidate manifest/pointer is missing")
        candidate = self._read_json(candidate_path)
        pointer: Optional[Dict[str, Any]] = None
        if candidate.get("schema") == V8_FINALIZATION_POINTER_SCHEMA:
            pointer = candidate
            pointer_hash = str(pointer.get("pointer_hash") or "")
            unsigned_pointer = dict(pointer)
            unsigned_pointer.pop("pointer_hash", None)
            if stable_json_hash(unsigned_pointer) != pointer_hash:
                raise ValueError("V8 frozen candidate pointer hash mismatch")
            if (
                pointer.get("promotion_ready") is not False
                or pointer.get("hardware_action_authorized") is not False
            ):
                raise ValueError("V8 frozen candidate pointer grants forbidden authority")
            manifest_path = self._resolve_manifest_path(
                str(pointer.get("version_manifest_path") or ""),
                candidate_path.parent,
            )
            if manifest_path.parent.parent != candidate_path.parent:
                raise ValueError("V8 frozen candidate pointer escapes its output root")
            expected_manifest_file_hash = str(
                pointer.get("version_manifest_sha256") or ""
            )
            self._require_sha256(
                expected_manifest_file_hash,
                "V8 frozen manifest file hash",
            )
            if self._sha256_file(manifest_path) != expected_manifest_file_hash:
                raise ValueError("V8 frozen candidate manifest file hash mismatch")
            manifest = self._read_json(manifest_path)
        elif candidate.get("schema") == V8_FINALIZATION_SCHEMA:
            manifest_path = candidate_path
            manifest = candidate
            expected_manifest_file_hash = self._sha256_file(manifest_path)
        elif candidate.get("schema") == V8_STATIC_REFIT_MANIFEST_SCHEMA:
            return self._load_static_refit_runtime(candidate_path, candidate)
        else:
            raise ValueError("unsupported V8 frozen candidate schema")

        manifest_hash = str(manifest.get("manifest_hash") or "")
        unsigned_manifest = dict(manifest)
        unsigned_manifest.pop("manifest_hash", None)
        if stable_json_hash(unsigned_manifest) != manifest_hash:
            raise ValueError("V8 frozen candidate manifest hash mismatch")
        if (
            manifest.get("schema") != V8_FINALIZATION_SCHEMA
            or manifest.get("selection_scope") != "development_only"
            or manifest.get("frozen_inference") is not True
            or manifest.get("online_training_allowed") is not False
            or manifest.get("research_only") is not True
            or manifest.get("promotion_ready") is not False
            or manifest.get("hardware_action_authorized") is not False
        ):
            raise ValueError("V8 frozen candidate violates research-only contract")

        records = list(manifest.get("ensemble_members") or ())
        member_count = int(manifest.get("ensemble_member_count", 0))
        seeds = [int(value) for value in manifest.get("seeds") or ()]
        if (
            member_count < V8_ENSEMBLE_MINIMUM_MEMBER_COUNT
            or len(records) != member_count
            or int(manifest.get("seed_count", 0)) != member_count
            or len(seeds) != member_count
            or len(set(seeds)) != member_count
        ):
            raise ValueError("V8 frozen ensemble member count/seeds are invalid")
        records = sorted(records, key=lambda value: int(value.get("seed", -1)))
        if [int(value.get("seed", -1)) for value in records] != sorted(seeds):
            raise ValueError("V8 frozen ensemble member seeds differ from manifest")
        weights = tuple(float(value.get("weight", 0.0)) for value in records)
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in weights)
            or not math.isclose(sum(weights), 1.0, abs_tol=1e-6)
            or any(
                not math.isclose(value, 1.0 / float(member_count), abs_tol=1e-9)
                for value in weights
            )
            or not math.isclose(
                float(manifest.get("ensemble_member_weights_sum", float("nan"))),
                sum(weights),
                abs_tol=1e-6,
            )
        ):
            raise ValueError("V8 frozen ensemble weights are invalid")

        ensemble_identity = {
            "profile": manifest.get("profile"),
            "ablation_id": manifest.get("ablation_id"),
            "members": [
                {
                    "seed": int(record["seed"]),
                    "checkpoint_sha256": str(record["checkpoint_sha256"]),
                    "weight": float(record["weight"]),
                }
                for record in records
            ],
        }
        ensemble_sha256 = stable_json_hash(ensemble_identity)
        if manifest.get("ensemble_sha256") != ensemble_sha256:
            raise ValueError("V8 frozen ensemble identity hash mismatch")
        if pointer is not None:
            self._verify_pointer_matches_manifest(pointer, manifest, records)

        rows = {
            int(value.get("seed", -1)): dict(value)
            for value in manifest.get("rows") or ()
        }
        if len(rows) != member_count or set(rows) != set(seeds):
            raise ValueError("V8 frozen ensemble rows do not cover every member")
        version_root = manifest_path.parent
        members: List[Dict[str, Any]] = []
        for record in records:
            seed = int(record["seed"])
            member_root = version_root / "members" / f"seed_{seed}"
            checkpoint_path = self._resolve_manifest_path(
                str(record.get("checkpoint_path") or ""),
                version_root,
            )
            training_report_path = self._resolve_manifest_path(
                str(record.get("training_report_path") or ""),
                version_root,
            )
            if (
                checkpoint_path != (member_root / "feedback_model_bundle.pt").resolve()
                or training_report_path != (member_root / "training_report.json").resolve()
            ):
                raise ValueError("V8 frozen member path differs from finalizer layout")
            checkpoint_sha256 = str(record.get("checkpoint_sha256") or "")
            report_file_sha256 = str(record.get("training_report_sha256") or "")
            self._require_sha256(checkpoint_sha256, f"V8 member {seed} checkpoint")
            self._require_sha256(report_file_sha256, f"V8 member {seed} report")
            if self._sha256_file(checkpoint_path) != checkpoint_sha256:
                raise ValueError(f"V8 frozen member {seed} checkpoint hash mismatch")
            if self._sha256_file(training_report_path) != report_file_sha256:
                raise ValueError(f"V8 frozen member {seed} report hash mismatch")
            member = self._load_member_runtime(
                checkpoint_path,
                training_report_path,
                require_declared_checkpoint_path=False,
            )
            report = member["report"]
            report_profile = dict(report.get("profile") or {})
            checkpoint_declaration = dict(report.get("checkpoint") or {})
            row = rows[seed]
            if (
                int(report.get("seed", -1)) != seed
                or report_profile.get("name") != manifest.get("profile")
                or report_profile.get("ablation_id") != manifest.get("ablation_id")
                or checkpoint_declaration.get("sha256") != checkpoint_sha256
                or row.get("checkpoint_sha256") != checkpoint_sha256
                or Path(str(row.get("checkpoint_path") or "")).resolve()
                != Path(str(checkpoint_declaration.get("path") or "")).resolve()
                or bool(record.get("static_seed_gate_passed"))
                != bool(row.get("static_seed_gate_passed"))
                or bool(record.get("action_seed_gate_passed"))
                != bool(row.get("action_seed_gate_passed"))
            ):
                raise ValueError(f"V8 frozen member {seed} manifest/report binding differs")
            members.append(member)

        invariant_manifest_fields = (
            "config_sha256",
            "split_hash_by_system",
            "source_tree_audit",
            "measured_audit_sha256",
            "model_audit_sha256",
            "contract_audit_sha256",
            "locked_final_test",
        )
        for name in invariant_manifest_fields:
            if any(member["report"].get(name) != manifest.get(name) for member in members):
                raise ValueError(f"V8 ensemble member {name} differs from manifest")
        classifier_contract_hashes = {
            stable_json_hash(member["classifier_contract"]) for member in members
        }
        if len(classifier_contract_hashes) != 1:
            raise ValueError("V8 ensemble classifier-prior contracts differ")

        causal_hashes = dict(
            dict(manifest.get("causal_context_gate") or {}).get(
                "member_gate_sha256"
            ) or {}
        )
        uncertainty_hashes = dict(
            dict(manifest.get("uncertainty_calibration") or {}).get(
                "member_state_sha256"
            ) or {}
        )
        for member, seed in zip(members, sorted(seeds)):
            if causal_hashes.get(str(seed)) != stable_json_hash(
                member["causal_context_gate"]
            ):
                raise ValueError(f"V8 ensemble member {seed} causal gate hash mismatch")
            if uncertainty_hashes.get(str(seed)) != stable_json_hash(
                member["uncertainty_calibration"]
            ):
                raise ValueError(
                    f"V8 ensemble member {seed} uncertainty hash mismatch"
                )

        static_comparison_gate = dict(manifest.get("static_comparison_gate") or {})
        gate_hash = str(static_comparison_gate.get("gate_sha256") or "")
        unsigned_gate = dict(static_comparison_gate)
        unsigned_gate.pop("gate_sha256", None)
        if (
            static_comparison_gate.get("schema")
            != "v8_measured_static_comparison_gate_v1"
            or stable_json_hash(unsigned_gate) != gate_hash
            or bool(manifest.get("static_shadow_ready"))
            != bool(static_comparison_gate.get("static_shadow_ready"))
        ):
            raise ValueError("V8 static comparison gate hash/state mismatch")
        representative = dict(manifest.get("representative_member") or {})
        representative_seed = int(representative.get("seed", -1))
        representative_record = next(
            (record for record in records if int(record["seed"]) == representative_seed),
            None,
        )
        if (
            representative.get("not_used_for_ensemble_selection_or_confidence")
            is not True
            or representative_record is None
            or representative.get("checkpoint_sha256")
            != representative_record.get("checkpoint_sha256")
        ):
            raise ValueError("V8 representative diagnostic member binding changed")

        first = members[0]
        return {
            "mode": "frozen_multi_seed_ensemble",
            "torch": first["torch"],
            "device": first["device"],
            "members": tuple(members),
            "member_weights": weights,
            "member_seeds": tuple(sorted(seeds)),
            "ensemble_inference_supported": True,
            "ensemble_sha256": ensemble_sha256,
            "manifest": manifest,
            "manifest_hash": manifest_hash,
            "manifest_file_sha256": expected_manifest_file_hash,
            "manifest_path": str(manifest_path),
            "pointer": pointer,
            "pointer_path": str(candidate_path) if pointer is not None else None,
            "checkpoint_sha256": ensemble_sha256,
            "report": {"report_hash": manifest_hash},
            "contract": first["contract"],
            "config": first["config"],
            "config_path": first["config_path"],
            "uncertainty_calibration": dict(
                manifest.get("uncertainty_calibration") or {}
            ),
            "causal_context_gate": dict(manifest.get("causal_context_gate") or {}),
            "input_cache": {},
        }

    def _load_static_refit_runtime(
        self,
        manifest_path: Path,
        manifest: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Load a metric-free static refit while invalidating every action surface."""

        import torch

        from ml_recon.physics_hybrid_v8.feedback_models import (
            V8FeedbackModelBundle,
            V8FeedbackModelConfig,
        )

        payload = dict(manifest)
        manifest_hash = str(payload.pop("manifest_hash", ""))
        if stable_json_hash(payload) != manifest_hash:
            raise ValueError("V8 static-refit manifest hash mismatch")
        if (
            manifest.get("schema") != V8_STATIC_REFIT_MANIFEST_SCHEMA
            or manifest.get("lineage_kind")
            != "post_locked_evaluation_static_all_eligible_refit"
            or manifest.get("static_refit_complete") is not True
            or manifest.get(
                "action_policy_parameters_bitwise_frozen_for_every_member"
            )
            is not True
            or manifest.get("frozen_inference") is not True
            or manifest.get("online_training_allowed") is not False
            or manifest.get("action_policy_ready") is not False
            or manifest.get("proposal_behavior") != "forced_no_op_abstain"
            or manifest.get("research_only") is not True
            or manifest.get("promotion_ready") is not False
            or manifest.get("hardware_action_authorized") is not False
        ):
            raise ValueError("V8 static-refit manifest violates static-only contract")
        metric_lineage = dict(manifest.get("metric_lineage") or {})
        if (
            metric_lineage.get("parent_locked_final_metrics_inherited") is not False
            or metric_lineage.get("refit_unbiased_locked_metric_available") is not False
            or metric_lineage.get("new_evaluation_required") is not True
        ):
            raise ValueError("V8 static-refit metric lineage is unsafe")

        records = sorted(
            list(manifest.get("ensemble_members") or ()),
            key=lambda value: int(value.get("seed", -1)),
        )
        member_count = int(manifest.get("ensemble_member_count", 0))
        weights = tuple(float(value.get("weight", 0.0)) for value in records)
        seeds = [int(value.get("seed", -1)) for value in records]
        if (
            member_count < V8_ENSEMBLE_MINIMUM_MEMBER_COUNT
            or len(records) != member_count
            or len(set(seeds)) != member_count
            or any(not math.isfinite(value) or value <= 0.0 for value in weights)
            or not math.isclose(sum(weights), 1.0, abs_tol=1e-6)
            or not math.isclose(
                float(manifest.get("ensemble_member_weights_sum", float("nan"))),
                1.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("V8 static-refit ensemble membership/weights are invalid")

        prerequisite = dict(manifest.get("prerequisite_audit") or {})
        prerequisite_hash = str(prerequisite.get("audit_sha256") or "")
        unsigned_prerequisite = dict(prerequisite)
        unsigned_prerequisite.pop("audit_sha256", None)
        if (
            prerequisite.get("schema") != V8_STATIC_REFIT_PREREQUISITE_SCHEMA
            or stable_json_hash(unsigned_prerequisite) != prerequisite_hash
            or int(prerequisite.get("locked_access_count", 0)) != 1
            or prerequisite.get(
                "former_locked_rows_may_enter_static_gradient_after_this_audit"
            )
            is not True
            or prerequisite.get("locked_metrics_may_be_inherited") is not False
        ):
            raise ValueError("V8 static-refit prerequisite audit is invalid")

        def bound_artifact_path(field_name: str) -> Path:
            path = self._resolve_manifest_path(
                str(prerequisite.get(field_name) or ""),
                manifest_path.parent,
            )
            try:
                path.relative_to(self.repository_root)
            except ValueError as exc:
                raise ValueError(
                    f"V8 static-refit {field_name} escapes repository"
                ) from exc
            return path

        parent_manifest_path = bound_artifact_path("parent_manifest_path")
        if self._sha256_file(parent_manifest_path) != prerequisite.get(
            "parent_manifest_file_sha256"
        ):
            raise ValueError("V8 static-refit parent manifest file hash mismatch")
        parent_runtime = self._load_ensemble_runtime(parent_manifest_path)
        if (
            parent_runtime["mode"] != "frozen_multi_seed_ensemble"
            or parent_runtime["manifest_hash"]
            != manifest.get("parent_ensemble_manifest_hash")
            or parent_runtime["manifest_hash"]
            != prerequisite.get("parent_manifest_hash")
            or parent_runtime["ensemble_sha256"]
            != manifest.get("parent_ensemble_sha256")
            or len(parent_runtime["members"]) != member_count
        ):
            raise ValueError("V8 static-refit parent ensemble binding changed")

        locked_path = bound_artifact_path("locked_evaluation_path")
        ledger_path = bound_artifact_path("locked_access_ledger_path")
        if (
            self._sha256_file(locked_path)
            != prerequisite.get("locked_evaluation_file_sha256")
            or self._sha256_file(ledger_path)
            != prerequisite.get("locked_access_ledger_file_sha256")
        ):
            raise ValueError("V8 static-refit locked evidence file hash mismatch")
        locked = self._read_json(locked_path)
        locked_hash = str(locked.get("report_hash") or "")
        unsigned_locked = dict(locked)
        unsigned_locked.pop("report_hash", None)
        ledger = self._read_json(ledger_path)
        ledger_hash = str(ledger.get("ledger_sha256") or "")
        unsigned_ledger = dict(ledger)
        unsigned_ledger.pop("ledger_sha256", None)
        if (
            locked.get("schema") != V8_LOCKED_FINAL_EVALUATION_SCHEMA
            or locked.get("evaluation_scope") != "locked_final_single_use"
            or locked.get("allowed_for_model_selection") is not False
            or locked.get("weights_changed") is not False
            or locked.get("online_training_used") is not False
            or locked.get("selection_score_emitted") is not False
            or locked.get("promotion_ready") is not False
            or locked.get("hardware_action_authorized") is not False
            or stable_json_hash(unsigned_locked) != locked_hash
            or locked_hash != prerequisite.get("locked_evaluation_report_hash")
            or ledger.get("schema") != V8_LOCKED_ACCESS_LEDGER_SCHEMA
            or stable_json_hash(unsigned_ledger) != ledger_hash
            or ledger_hash != prerequisite.get("locked_access_ledger_sha256")
            or int(ledger.get("access_count", 0)) != 1
            or int(ledger.get("maximum_access_count", 0)) != 1
            or ledger.get("allowed_for_model_selection") is not False
        ):
            raise ValueError("V8 static-refit locked evidence semantics changed")

        parent_manifest = dict(parent_runtime["manifest"])
        parent_audit_hashes = dict(
            manifest.get("parent_data_and_contract_audit_hashes") or {}
        )
        expected_audit_hashes = {
            name: str(parent_manifest.get(name) or "")
            for name in (
                "measured_audit_sha256",
                "model_audit_sha256",
                "contract_audit_sha256",
            )
        }
        measured_bundle_audit_sha256 = stable_json_hash(
            parent_runtime["contract"]["measured_bundle_audit"]
        )
        if (
            parent_audit_hashes != expected_audit_hashes
            or dict(prerequisite.get("parent_audit_hashes") or {})
            != expected_audit_hashes
            or manifest.get("config_sha256") != parent_manifest.get("config_sha256")
            or manifest.get("split_hash_by_system")
            != parent_manifest.get("split_hash_by_system")
            or manifest.get("parent_source_tree_audit")
            != parent_manifest.get("source_tree_audit")
            or manifest.get("measured_bundle_audit_sha256")
            != measured_bundle_audit_sha256
        ):
            raise ValueError("V8 static-refit data/model/contract lineage changed")
        self._verify_split_hashes(manifest["split_hash_by_system"])
        config_path, config = self._verify_source_tree_and_config(
            dict(manifest.get("refit_source_tree_audit") or {}),
            str(manifest.get("config_sha256") or ""),
        )
        if config_path.resolve() != Path(parent_runtime["config_path"]).resolve():
            raise ValueError("V8 static-refit config path differs from parent")
        architecture = dict(manifest.get("frozen_architecture_audit") or {})
        architecture_hash = str(architecture.get("audit_sha256") or "")
        unsigned_architecture = dict(architecture)
        unsigned_architecture.pop("audit_sha256", None)
        if (
            architecture.get("schema")
            != "v8_feedback_refit_frozen_architecture_audit_v1"
            or stable_json_hash(unsigned_architecture) != architecture_hash
            or int(architecture.get("member_count", 0)) != member_count
            or architecture.get("architecture_shared_by_all_members") is not True
            or architecture.get("ensemble_membership_and_weights_frozen") is not True
        ):
            raise ValueError("V8 static-refit architecture audit changed")

        parent_members = {
            int(member["report"]["seed"]): member
            for member in parent_runtime["members"]
        }
        members: List[Dict[str, Any]] = []
        for record in records:
            seed = int(record["seed"])
            parent_member = parent_members.get(seed)
            if parent_member is None:
                raise ValueError("V8 static-refit seed is absent from parent ensemble")
            checkpoint_path = self._resolve_manifest_path(
                str(record.get("checkpoint_path") or ""),
                manifest_path.parent,
            )
            expected_path = (
                manifest_path.parent
                / "members"
                / f"seed_{seed}"
                / "static_refit_checkpoint.pt"
            ).resolve()
            if checkpoint_path != expected_path:
                raise ValueError("V8 static-refit member path differs from refit layout")
            checkpoint_sha256 = str(record.get("checkpoint_sha256") or "")
            self._require_sha256(checkpoint_sha256, "V8 static-refit checkpoint")
            if self._sha256_file(checkpoint_path) != checkpoint_sha256:
                raise ValueError("V8 static-refit member checkpoint hash mismatch")
            checkpoint = torch.load(
                checkpoint_path,
                map_location=parent_runtime["device"],
                weights_only=False,
            )
            refit_contract = dict(checkpoint.get("refit_contract") or {})
            refit_contract_hash = str(refit_contract.get("contract_sha256") or "")
            unsigned_refit_contract = dict(refit_contract)
            unsigned_refit_contract.pop("contract_sha256", None)
            parent_declaration = dict(checkpoint.get("parent_member_checkpoint") or {})
            if (
                checkpoint.get("schema") != V8_STATIC_REFIT_CHECKPOINT_SCHEMA
                or checkpoint.get("frozen_inference") is not True
                or checkpoint.get("online_training_allowed") is not False
                or checkpoint.get("action_policy_ready") is not False
                or checkpoint.get("proposal_behavior") != "forced_no_op_abstain"
                or checkpoint.get("hardware_action_authorized") is not False
                or checkpoint.get("parent_ensemble_manifest_hash")
                != manifest.get("parent_ensemble_manifest_hash")
                or checkpoint.get("prerequisite_audit_sha256") != prerequisite_hash
                or checkpoint.get("config_sha256") != manifest.get("config_sha256")
                or checkpoint.get("split_hash_by_system")
                != manifest.get("split_hash_by_system")
                or checkpoint.get("refit_source_tree_audit")
                != manifest.get("refit_source_tree_audit")
                or checkpoint.get("frozen_architecture_audit") != architecture
                or checkpoint.get("measured_bundle_audit_sha256")
                != measured_bundle_audit_sha256
                or record.get("checkpoint_schema")
                != V8_STATIC_REFIT_CHECKPOINT_SCHEMA
                or parent_declaration.get("sha256")
                != record.get("parent_checkpoint_sha256")
                or parent_declaration.get("sha256")
                != parent_member["checkpoint_sha256"]
                or stable_json_hash(unsigned_refit_contract) != refit_contract_hash
                or refit_contract_hash != record.get("refit_contract_sha256")
                or float(refit_contract.get("real_action_loss_contribution", -1.0))
                != 0.0
                or float(refit_contract.get("real_jacobian_loss_contribution", -1.0))
                != 0.0
                or float(refit_contract.get("policy_or_ranking_loss_contribution", -1.0))
                != 0.0
                or record.get("action_policy_parameters_bitwise_unchanged") is not True
            ):
                raise ValueError("V8 static-refit member lineage/action freeze changed")
            for name, expected in expected_audit_hashes.items():
                checkpoint_name = (
                    "parent_contract_audit_sha256"
                    if name == "contract_audit_sha256"
                    else name
                )
                if str(checkpoint.get(checkpoint_name) or "") != expected:
                    raise ValueError(
                        f"V8 static-refit member {checkpoint_name} differs from parent"
                    )
            model_config = V8FeedbackModelConfig.from_mapping(
                checkpoint["model_config"]
            )
            model = V8FeedbackModelBundle(model_config)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            if stable_json_hash(model.audit_dict()) != expected_audit_hashes[
                "model_audit_sha256"
            ]:
                raise ValueError("V8 static-refit model audit differs from parent")
            model.to(parent_runtime["device"]).eval()
            member = dict(parent_member)
            member.update({
                "model": model,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_path": str(checkpoint_path),
            })
            members.append(member)

        refit_identity = {
            "schema": V8_STATIC_REFIT_MANIFEST_SCHEMA,
            "parent_ensemble_manifest_hash": manifest.get(
                "parent_ensemble_manifest_hash"
            ),
            "members": [
                {
                    "seed": int(record["seed"]),
                    "checkpoint_sha256": str(record["checkpoint_sha256"]),
                    "weight": float(record["weight"]),
                }
                for record in records
            ],
        }
        refit_ensemble_sha256 = stable_json_hash(refit_identity)
        return {
            "mode": "static_refit_multi_seed_ensemble",
            "torch": parent_runtime["torch"],
            "device": parent_runtime["device"],
            "members": tuple(members),
            "member_weights": weights,
            "member_seeds": tuple(seeds),
            "ensemble_inference_supported": True,
            "ensemble_sha256": refit_ensemble_sha256,
            "manifest": dict(manifest),
            "manifest_hash": manifest_hash,
            "manifest_file_sha256": self._sha256_file(manifest_path),
            "manifest_path": str(manifest_path),
            "pointer": None,
            "pointer_path": None,
            "checkpoint_sha256": refit_ensemble_sha256,
            "report": {"report_hash": manifest_hash},
            "contract": parent_runtime["contract"],
            "config": config,
            "config_path": str(config_path),
            "uncertainty_calibration": {
                "status": "invalidated_by_static_all_eligible_refit",
                "abstain": True,
                "policy_execution_allowed": False,
            },
            "causal_context_gate": {
                "status": "action_surface_invalidated_by_static_refit",
                "causal_context_gate": False,
            },
            "static_only_refit": True,
            "action_surface_invalidated": True,
            "input_cache": {},
        }

    @staticmethod
    def _resolve_manifest_path(value: str, relative_root: Path) -> Path:
        if not value.strip():
            raise ValueError("V8 frozen manifest path is empty")
        path = Path(value)
        return (
            path.resolve()
            if path.is_absolute()
            else (relative_root / path).resolve()
        )

    @staticmethod
    def _verify_pointer_matches_manifest(
        pointer: Mapping[str, Any],
        manifest: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        pointer_members = sorted(
            list(pointer.get("ensemble_members") or ()),
            key=lambda value: int(value.get("seed", -1)),
        )
        expected_members = [
            {
                "seed": int(record["seed"]),
                "weight": float(record["weight"]),
                "checkpoint_path": str(record["checkpoint_path"]),
                "checkpoint_sha256": str(record["checkpoint_sha256"]),
            }
            for record in records
        ]
        if (
            pointer.get("ensemble_sha256") != manifest.get("ensemble_sha256")
            or int(pointer.get("ensemble_member_count", 0))
            != int(manifest.get("ensemble_member_count", -1))
            or pointer_members != expected_members
            or pointer.get("candidate_stage") != manifest.get("candidate_stage")
            or int(pointer.get("representative_seed_for_diagnostics_only", -1))
            != int(dict(manifest.get("representative_member") or {}).get("seed", -2))
            or pointer.get("static_shadow_ready")
            != manifest.get("static_shadow_ready")
            or pointer.get("action_shadow_ready")
            != manifest.get("action_shadow_ready")
            or pointer.get("inference_abstains") != manifest.get("inference_abstains")
        ):
            raise ValueError("V8 frozen candidate pointer differs from manifest")

    @staticmethod
    def _verify_research_only_contract(
        report: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
    ) -> None:
        for name, value in (("report", report), ("checkpoint", checkpoint)):
            if (
                value.get("research_only") is not True
                or value.get("promotion_ready") is not False
                or value.get("hardware_action_authorized") is not False
                or value.get("selection_scope") != "development_only"
            ):
                raise ValueError(f"V8 {name} violates research-only selection scope")
        if (
            checkpoint.get("frozen_inference") is not True
            or checkpoint.get("online_training_allowed") is not False
            or report.get("locked_final_metrics_present") is not False
        ):
            raise ValueError("V8 frozen/locked-final contract changed")

    def _reject_unsupported_ensemble_candidate(
        self,
        *,
        report: Optional[Mapping[str, Any]] = None,
        checkpoint: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Fail before a frozen ensemble can degrade to one representative seed."""

        for path in (self.checkpoint_path, self.training_report_path):
            if path is None:
                continue
            if path.name in {V8_FINALIZATION_FILENAME, "selected_candidate.json"}:
                raise ValueError(V8_ENSEMBLE_UNSUPPORTED_MESSAGE)
            for ancestor in path.parents:
                if ancestor.name.lower() != "members":
                    continue
                version_root = ancestor.parent
                if (
                    version_root.name.lower().startswith("ensemble_")
                    or (version_root / V8_FINALIZATION_FILENAME).is_file()
                ):
                    raise ValueError(V8_ENSEMBLE_UNSUPPORTED_MESSAGE)

        declarations = tuple(
            value for value in (report, checkpoint) if value is not None
        )
        for value in declarations:
            if value.get("schema") in {
                V8_FINALIZATION_SCHEMA,
                V8_FINALIZATION_POINTER_SCHEMA,
            }:
                raise ValueError(V8_ENSEMBLE_UNSUPPORTED_MESSAGE)
            member_count = value.get("ensemble_member_count")
            if member_count is not None and int(member_count) > 1:
                raise ValueError(V8_ENSEMBLE_UNSUPPORTED_MESSAGE)
            if value.get("ensemble_members") or value.get("representative_member"):
                raise ValueError(V8_ENSEMBLE_UNSUPPORTED_MESSAGE)
            confidence = dict(value.get("uncertainty_calibration") or {})
            if (
                int(confidence.get("ensemble_member_count", 0)) > 1
                or confidence.get("true_multi_member_inference_required") is True
            ):
                raise ValueError(V8_ENSEMBLE_UNSUPPORTED_MESSAGE)

    def _verify_training_contract(
        self,
        report: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
    ) -> Dict[str, Any]:
        report_contract = dict(report.get("v8_training_contract") or {})
        checkpoint_contract = dict(checkpoint.get("v8_training_contract") or {})
        if report_contract != checkpoint_contract or not report_contract:
            raise ValueError("V8 report/checkpoint training contracts differ")
        declared_hash = str(report_contract.get("contract_audit_sha256") or "")
        unsigned = dict(report_contract)
        unsigned.pop("contract_audit_sha256", None)
        if stable_json_hash(unsigned) != declared_hash:
            raise ValueError("V8 training contract hash mismatch")
        required = (
            "config_sha256",
            "split_hash_by_system",
            "measured_file_audit",
            "source_tree_audit",
            "model_audit_sha256",
            "measured_bundle_audit",
            "causal_context_gate",
            "locked_final_test",
        )
        missing = sorted(name for name in required if name not in report_contract)
        if missing:
            raise ValueError(f"V8 training contract is incomplete: {missing}")
        cross_bound = (
            "source_tree_audit",
            "config_sha256",
            "split_hash_by_system",
            "model_audit_sha256",
            "contract_audit_sha256",
            "uncertainty_calibration",
            "causal_context_gate",
            "state_pair_transport_weak",
            "locked_final_test",
        )
        for name in cross_bound:
            report_value = report.get(name)
            checkpoint_value = checkpoint.get(name)
            contract_value = report_contract.get(name)
            if name in report_contract:
                if report_value != checkpoint_value or report_value != contract_value:
                    raise ValueError(f"V8 {name} copies are not hash-bound consistently")
            elif report_value != checkpoint_value:
                raise ValueError(f"V8 {name} report/checkpoint copies differ")
        measured_audit_hash = str(
            dict(report_contract["measured_file_audit"]).get("audit_sha256") or ""
        )
        if (
            report.get("measured_audit_sha256") != measured_audit_hash
            or checkpoint.get("measured_audit_sha256") != measured_audit_hash
        ):
            raise ValueError("V8 measured data audit hash copies differ")
        for name in ("model_audit_sha256", "contract_audit_sha256"):
            if report.get(name) != report_contract.get(name) or checkpoint.get(name) != report_contract.get(name):
                raise ValueError(f"V8 {name} copies differ")
        state_pair = dict(report_contract.get("state_pair_transport_weak") or {})
        if state_pair:
            state_pair_hash = str(state_pair.get("contract_sha256") or "")
            unsigned_pair = dict(state_pair)
            unsigned_pair.pop("contract_sha256", None)
            if stable_json_hash(unsigned_pair) != state_pair_hash:
                raise ValueError("V8 weak state-pair contract hash mismatch")
            if (
                int(state_pair.get("real_action_supervision_mask", -1)) != 0
                or int(state_pair.get("real_jacobian_supervision_mask", -1)) != 0
                or state_pair.get("counts_as_real_causality_evidence") is not False
            ):
                raise ValueError("V8 weak state pairs leaked into real action supervision")
        locked = dict(report_contract["locked_final_test"])
        declaration = locked.get("declaration")
        if declaration is not None and locked.get("declaration_sha256") != stable_json_hash(declaration):
            raise ValueError("V8 locked-final declaration hash mismatch")
        return report_contract

    def _verify_source_tree_and_config(
        self,
        value: Mapping[str, Any],
        config_sha256: str,
    ) -> Tuple[Path, Dict[str, Any]]:
        audit = dict(value)
        if audit.get("schema") != V8_SOURCE_AUDIT_SCHEMA:
            raise ValueError("V8 source-tree audit schema changed")
        audit_hash = str(audit.get("audit_sha256") or "")
        unsigned = dict(audit)
        unsigned.pop("audit_sha256", None)
        if stable_json_hash(unsigned) != audit_hash:
            raise ValueError("V8 source-tree audit hash mismatch")
        source_hashes = dict(audit.get("source_file_sha256") or {})
        if not source_hashes or audit.get("source_tree_sha256") != stable_json_hash(source_hashes):
            raise ValueError("V8 source-tree hash mismatch")
        normalized_sources = {str(name).replace("\\", "/") for name in source_hashes}
        missing_sources = sorted(V8_REQUIRED_SOURCE_FILES - normalized_sources)
        if missing_sources:
            raise ValueError(f"V8 source audit omits runtime dependencies: {missing_sources}")
        config_candidates = []
        for relative, expected in source_hashes.items():
            self._require_sha256(str(expected), f"source file {relative}")
            path = self._safe_repository_path(str(relative))
            if self._sha256_file(path) != expected:
                raise ValueError(f"V8 audited source file changed: {relative}")
            if str(expected) == config_sha256 and path.suffix.lower() == ".json":
                payload = self._read_json(path)
                if payload.get("schema") == "physics_hybrid_v8_feedback_training_v1":
                    config_candidates.append((path, payload))
        self._require_sha256(config_sha256, "config_sha256")
        if len(config_candidates) != 1:
            raise ValueError("V8 audited training config cannot be identified uniquely")
        return config_candidates[0]

    def _verify_measured_data(
        self,
        value: Mapping[str, Any],
        config: Mapping[str, Any],
        config_path: Path,
    ) -> None:
        audit = dict(value)
        file_hashes = dict(audit.get("file_sha256") or {})
        waiver = dict(audit.get("waiver") or {})
        if audit.get("audit_sha256") != stable_json_hash({
            "file_sha256": file_hashes,
            "waiver": waiver,
        }):
            raise ValueError("V8 measured data audit hash mismatch")
        required = {
            "measured_data_summary.json",
            "measured_splits_v1.json",
            "split_audit.json",
        }
        if set(file_hashes) != required:
            raise ValueError("V8 measured data audit must bind all three manifest files")
        manifest_value = str(dict(config.get("data") or {}).get("manifest_dir") or "")
        if not manifest_value:
            raise ValueError("V8 config lacks data.manifest_dir")
        manifest_dir = Path(manifest_value)
        if not manifest_dir.is_absolute():
            manifest_dir = (self.repository_root / manifest_dir).resolve()
        else:
            manifest_dir = manifest_dir.resolve()
        try:
            manifest_dir.relative_to(self.repository_root)
        except ValueError as exc:
            raise ValueError("V8 measured manifest directory escapes repository") from exc
        for name, expected in file_hashes.items():
            self._require_sha256(str(expected), f"measured data file {name}")
            path = (manifest_dir / name).resolve()
            try:
                path.relative_to(manifest_dir)
            except ValueError as exc:
                raise ValueError("V8 measured data path traversal") from exc
            if self._sha256_file(path) != expected:
                raise ValueError(f"V8 measured data file changed: {name}")
        expected_summary = str(waiver.get("summary_sha256") or "")
        if expected_summary and expected_summary != file_hashes["measured_data_summary.json"]:
            raise ValueError("V8 measured audit waiver summary hash differs")

    def _verify_split_hashes(self, value: Mapping[str, Any]) -> None:
        hashes = dict(value)
        if set(hashes) != SUPPORTED_SYSTEM_TYPES:
            raise ValueError("V8 split hashes must cover both systems")
        for system_type, digest in hashes.items():
            self._require_sha256(str(digest), f"split hash {system_type}")

    def _load_classifier_prior(
        self,
        checkpoint: Mapping[str, Any],
        model: Any,
        device: Any,
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        contract = dict(checkpoint.get("classifier_prior") or {
            "source": "none",
            "frozen": True,
        })
        if contract.get("frozen") is not True:
            raise ValueError("V8 classifier prior must be frozen")
        source = str(contract.get("source") or "none")
        if source == "none":
            return None, contract
        from ml_recon.physics_hybrid_v7.feedback_classifier_prior import (
            FrozenFeedbackClassifierPrior,
        )

        prior = FrozenFeedbackClassifierPrior(
            source=source,
            model_dir=self.repository_root / "ml_recon" / "models",
            expected_latent_dim=model.config.classifier_latent_dim,
            device=device,
        )
        expected_hashes = dict(contract.get("checkpoints") or {})
        actual_hashes = prior.audit_dict()["checkpoints"]
        for system_type in SUPPORTED_SYSTEM_TYPES:
            if (
                dict(expected_hashes.get(system_type) or {}).get("sha256")
                != actual_hashes[system_type]["sha256"]
            ):
                raise ValueError("V8 classifier prior checkpoint hash mismatch")
        return prior, contract

    def _verify_prepared_artifact(
        self,
        artifact: PreparedSequenceArtifactV1,
    ) -> Tuple[Any, Any]:
        if not isinstance(artifact, PreparedSequenceArtifactV1):
            raise TypeError("V8 requires PreparedSequenceArtifactV1")
        if not bool(artifact.fourier_report.get("passed", False)):
            raise ValueError("V8 prepared artifact failed the Fourier guard")
        if not bool(artifact.materialization.get("accepted", False)):
            raise ValueError("V8 prepared artifact materialization was not accepted")
        identity = {
            "schema": "PreparedSequenceArtifactV1",
            "system_type": artifact.system_type,
            "target_hash": artifact.target_hash,
            "control_hash": artifact.control_hash,
            "parent_control_hash": artifact.parent_control_hash,
            "requested_action_hash": artifact.requested_action_hash,
            "projected_action_hash": artifact.projected_action_hash,
            "actual_action_hash": artifact.actual_action_hash,
            "matrix_hash": artifact.matrix_hash,
            "registry_hash": artifact.registry_hash,
            "command_table_hash": artifact.command_table_hash,
            "waveform_hash": artifact.waveform_hash,
            "phase_cal_manifest_hash": artifact.phase_cal_manifest_hash,
        }
        if stable_json_hash(identity) != artifact.artifact_id:
            raise ValueError("V8 prepared artifact ID mismatch")
        materialization = dict(artifact.materialization)
        for name, expected in (
            ("requested_action", artifact.requested_action_hash),
            ("projected_action", artifact.projected_action_hash),
            ("actual_loaded_action", artifact.actual_action_hash),
        ):
            payload = materialization.get(name)
            if not isinstance(payload, Mapping) or stable_json_hash(dict(payload)) != expected:
                raise ValueError(f"V8 prepared {name} hash mismatch")
        from ml_recon.physics_hybrid_v5.command_contract import PhysicsHybridCommandTableV1
        from ml_recon.physics_hybrid_v8.command_features import (
            build_prepared_command_features,
        )

        command_root = Path(artifact.command_table_directory).resolve()
        command_manifest = (command_root / "command_table.json").resolve()
        try:
            command_manifest.relative_to(command_root)
        except ValueError as exc:
            raise ValueError("V8 command manifest escapes immutable table directory") from exc
        command_table = PhysicsHybridCommandTableV1.load(command_manifest, mmap_mode=None)
        if (
            command_table.table_id != artifact.command_table_id
            or command_table.table_hash != artifact.command_table_hash
            or command_table.operation_registry_hash != artifact.registry_hash
            or command_table.matrix_snapshot_hash != artifact.matrix_hash
        ):
            raise ValueError("V8 prepared command-table identity mismatch")
        matrix_hashes = self._verify_files(
            Path(artifact.matrix_workspace),
            artifact.matrix_files,
        )
        waveform_hashes = self._verify_files(
            Path(artifact.sequence_dir),
            artifact.waveform_files,
        )
        if stable_json_hash(matrix_hashes) != artifact.matrix_hash:
            raise ValueError("V8 prepared matrix bundle hash mismatch")
        if stable_json_hash(waveform_hashes) != artifact.waveform_hash:
            raise ValueError("V8 prepared waveform bundle hash mismatch")
        if self._sha256_file(Path(artifact.phase_cal_manifest_path)) != artifact.phase_cal_manifest_hash:
            raise ValueError("V8 phase calibration manifest hash mismatch")
        if artifact.upload_receipt is not None:
            receipt = dict(artifact.upload_receipt)
            receipt_hash = str(receipt.pop("receipt_hash", ""))
            if not receipt_hash or stable_json_hash(receipt) != receipt_hash:
                raise ValueError("V8 prepared upload receipt hash mismatch")
        command_features = build_prepared_command_features(command_table)
        if command_features.provenance.get("command_table_hash") != artifact.command_table_hash:
            raise ValueError("V8 command feature provenance hash mismatch")
        return command_table, command_features

    def _snapshot_provenance(
        self,
        snapshot: MeasurementSnapshot,
        artifact: PreparedSequenceArtifactV1,
        checkpoint_sha256: str,
    ) -> Dict[str, Any]:
        diagnostics = dict(snapshot.diagnostics or {})
        value = diagnostics.get("v8_snapshot_provenance")
        if not isinstance(value, Mapping):
            raise ValueError("V8 snapshot lacks hash-bound prepared provenance")
        provenance = dict(value)
        declared_hash = str(provenance.pop("provenance_hash", ""))
        expected = {
            "schema": V8_SNAPSHOT_PROVENANCE_SCHEMA,
            "system_type": snapshot.system_type,
            "prepared_artifact_id": artifact.artifact_id,
            "command_table_id": artifact.command_table_id,
            "command_table_hash": artifact.command_table_hash,
            "target_state_hash": artifact.target_hash,
            "control_state_hash": artifact.control_hash,
            "matrix_bundle_hash": artifact.matrix_hash,
            "waveform_hash": artifact.waveform_hash,
            "requested_action_hash": artifact.requested_action_hash,
            "projected_action_hash": artifact.projected_action_hash,
            "actual_action_hash": artifact.actual_action_hash,
            "model_checkpoint_hash": checkpoint_sha256,
        }
        if any(provenance.get(name) != expected_value for name, expected_value in expected.items()):
            raise ValueError("V8 snapshot/prepared/model provenance mismatch")
        upload_receipt_hash = str(provenance.get("upload_receipt_hash") or "")
        self._require_sha256(upload_receipt_hash, "snapshot upload receipt hash")
        if artifact.upload_receipt is not None:
            if dict(artifact.upload_receipt).get("receipt_hash") != upload_receipt_hash:
                raise ValueError("V8 snapshot upload receipt differs from prepared artifact")
        if stable_json_hash(provenance) != declared_hash:
            raise ValueError("V8 snapshot provenance hash mismatch")
        return {**provenance, "provenance_hash": declared_hash, "complete": True}

    def _resolve_exposure(
        self,
        snapshot: MeasurementSnapshot,
        explicit: Optional[MeasurementExposureProvenanceV1],
    ) -> Dict[str, Any]:
        value = explicit
        if value is None:
            raw = dict(snapshot.diagnostics or {}).get("v8_exposure_provenance")
            if raw is not None:
                if not isinstance(raw, Mapping):
                    raise TypeError("V8 exposure provenance must be a mapping")
                value = MeasurementExposureProvenanceV1.from_json_dict(raw)
        if value is None:
            return {
                "known": False,
                "exposure_s": None,
                "effective_source": "model_reference_exposure_normalization_only",
                "rate_equivariance_verifiable": False,
            }
        if not isinstance(value, MeasurementExposureProvenanceV1):
            raise TypeError("V8 exposure requires MeasurementExposureProvenanceV1")
        value.validate()
        tolerance = max(1e-6, 1e-6 * float(value.exposure_s))
        if abs(float(snapshot.duration_s) - float(value.exposure_s)) > tolerance:
            raise ValueError("V8 authoritative exposure differs from snapshot duration")
        return {
            "known": True,
            "exposure_s": float(value.exposure_s),
            "effective_source": value.source,
            "rate_equivariance_verifiable": True,
            "provenance": value.to_json_dict(),
        }

    def _build_inputs(
        self,
        runtime: Dict[str, Any],
        target_state: TargetStateSpecV1,
        snapshot: MeasurementSnapshot,
        command_features: Any,
        episode_history: Sequence[BoundHistoryTransitionV1],
    ) -> Dict[str, np.ndarray]:
        from ml_recon.physics_hybrid_v7.feedback_data import (
            build_operation_features,
            build_target_features,
        )
        from ml_recon.physics_hybrid_v7.feedback_synthetic import canonical_action_mask
        from ml_recon.physics_hybrid_v7.geometry import build_operation_frequency_geometry

        system_type = snapshot.system_type
        cached = runtime["input_cache"].get(system_type)
        if cached is None:
            geometry = build_operation_frequency_geometry(system_type)
            static, _ = geometry.static_features()
            graph = geometry.coupling_kernel(
                x_length_mhz=8.0,
                y_length_mhz=8.0,
                same_local_cross_cycle_weight=0.05,
            )
            graph += np.eye(geometry.operation_count)
            graph /= np.maximum(graph.sum(axis=1, keepdims=True), 1e-12)
            cached = (geometry, static.astype(np.float32), graph.astype(np.float32))
            runtime["input_cache"][system_type] = cached
        geometry, static, graph = cached
        operation_count = geometry.operation_count
        coin = np.asarray(snapshot.coin_bins, dtype=np.float32).reshape(1, -1)
        signal = np.asarray(snapshot.signal_bins, dtype=np.float32).reshape(1, -1)
        idler = np.asarray(snapshot.idler_bins, dtype=np.float32).reshape(1, -1)
        if (
            coin.shape != (1, operation_count)
            or signal.shape != coin.shape
            or idler.shape != coin.shape
        ):
            raise ValueError("V8 snapshot operation counts have the wrong shape")
        if command_features.features.shape[0] != operation_count:
            raise ValueError("V8 command table operation count differs from snapshot geometry")
        amplitude = np.zeros((1, 4), dtype=np.float32)
        phase = np.zeros_like(amplitude)
        target_mask = np.zeros_like(amplitude)
        qubit_count = geometry.qubit_count
        amplitude[0, :qubit_count] = np.asarray(target_state.desired_amplitudes)
        phase[0, :qubit_count] = np.asarray(target_state.desired_phases_rad)
        target_mask[0, :qubit_count] = 1.0
        target_features = build_target_features(amplitude, phase, target_mask)
        operation_features = build_operation_features(
            coin_counts=coin,
            signal_counts=signal,
            idler_counts=idler,
            read_count=np.asarray([max(int(snapshot.read_count), 1)], dtype=np.float32),
            target_amplitude=amplitude,
            target_phase_rad=phase,
            geometry=geometry,
            static_operation_features=static,
            retrieval_relative_map=np.ones(operation_count, dtype=np.float32),
            observed_coin_mask=np.ones_like(coin, dtype=np.float32),
        )
        history, history_mask = self._history_tensor(
            episode_history,
            system_type=system_type,
            checkpoint_sha256=runtime["checkpoint_sha256"],
        )
        return {
            "operation_features": operation_features.astype(np.float32),
            "target_features": target_features.astype(np.float32),
            "graph_kernel": graph,
            "history": history,
            "history_mask": history_mask,
            "action_mask": canonical_action_mask(system_type)[None, :].astype(np.float32),
            "command_features": np.asarray(command_features.features, dtype=np.float32)[None, :, :],
            "command_mask": np.asarray(command_features.feature_mask, dtype=np.float32)[None, :, :],
            # Prepared artifacts bind phase/command calibration, but the current
            # snapshot contract has no immutable retrieval-efficiency identity.
            # Do not treat the process-global retrieval file as historical fact.
            "calibration_known_mask": np.zeros(1, dtype=np.float32),
            "calibration_identity_match_mask": np.zeros(1, dtype=np.float32),
        }

    @staticmethod
    def _history_tensor(
        episode_history: Sequence[BoundHistoryTransitionV1],
        *,
        system_type: str,
        checkpoint_sha256: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from ml_recon.physics_hybrid_v7.feedback_synthetic import HISTORY_FEATURE_NAMES

        selected = list(episode_history)[-9:]
        history = np.zeros((1, 9, len(HISTORY_FEATURE_NAMES)), dtype=np.float32)
        mask = np.zeros((1, 9), dtype=np.float32)
        previous: Optional[HistoryTransitionProvenanceV1] = None
        for row, item in enumerate(selected):
            if not isinstance(item, BoundHistoryTransitionV1):
                raise TypeError(
                    "V8 history rejects mappings; use BoundHistoryTransitionV1"
                )
            action = list(item.validate(system_type=system_type))
            outcomes = dict(HISTORY_OUTCOME_DEFAULTS)
            outcomes.update({
                name: float(value) for name, value in item.outcome_features.items()
            })
            provenance = item.provenance
            derived = {
                "model_hash_changed": float(provenance.model_hash != checkpoint_sha256),
                "calibration_hash_changed": float(
                    previous is not None
                    and provenance.calibration_hash != previous.calibration_hash
                ),
                "command_hash_changed": float(
                    previous is not None
                    and provenance.actual_command_hash != previous.actual_command_hash
                ),
            }
            values = action + [
                float(derived.get(name, outcomes.get(name, 0.0)))
                for name in HISTORY_FEATURE_NAMES[6:]
            ]
            history[0, row] = np.asarray(values, dtype=np.float32)
            mask[0, row] = 1.0
            previous = provenance
        return history, mask

    def _classifier_features(
        self,
        runtime: Mapping[str, Any],
        snapshot: MeasurementSnapshot,
        target: Any,
    ) -> Dict[str, Any]:
        prior = runtime["classifier_prior"]
        if prior is None:
            return {}
        torch = runtime["torch"]
        from ml_recon.physics_hybrid_v7.feedback_classifier_prior import embed_core_density

        qubit_count = 3 if snapshot.system_type == "3qubit" else 4
        density = snapshot.density_matrix
        if density is None:
            full = np.eye(2 ** qubit_count, dtype=np.complex64)[None, :, :] / float(
                2 ** qubit_count
            )
            rho_mask = torch.zeros(1, device=runtime["device"])
        else:
            value = np.asarray(density, dtype=np.complex64)
            if value.shape == (qubit_count, qubit_count):
                value = embed_core_density(value[None, :, :], snapshot.system_type)[0]
            if value.shape != (2 ** qubit_count, 2 ** qubit_count):
                raise ValueError("V8 classifier rho dimension is unsupported")
            full = value[None, :, :]
            rho_mask = torch.ones(1, device=runtime["device"])
        latent, uncertainty = prior.encode(
            current_full_rho=torch.from_numpy(full).to(runtime["device"]),
            target_features=target,
            rho_available_mask=rho_mask,
            system_type=snapshot.system_type,
        )
        return {
            "classifier_latent": latent,
            "classifier_uncertainty": uncertainty,
        }

    @staticmethod
    def _action_gates(runtime: Mapping[str, Any], system_type: str) -> Dict[str, Any]:
        if runtime.get("static_only_refit") is True:
            return {
                "real_action_supervision": False,
                "real_action_supervision_count": 0,
                "causal_context": False,
                "real_episode_calibration": False,
                "true_multi_member_inference": True,
                "frozen_action_policy_eligible": False,
                "ensemble_member_count": len(runtime["members"]),
                "static_only_refit": True,
                "action_surface_valid": False,
                "action_surface_invalidated": True,
                "action_surface_invalidation_reason": (
                    "static_all_eligible_refit_changed_shared_static_representation"
                ),
                "shadow_only": True,
                "hardware_action_authorized": False,
            }
        if runtime.get("mode") == "frozen_multi_seed_ensemble":
            member_gates = [
                V8FeedbackBridge._action_gates(member, system_type)
                for member in runtime["members"]
            ]
            manifest = dict(runtime["manifest"])
            confidence = dict(manifest.get("uncertainty_calibration") or {})
            if (
                confidence.get("schema")
                != "v8_ensemble_real_episode_uncertainty_state_v1"
                or int(confidence.get("ensemble_member_count", 0))
                != len(member_gates)
                or confidence.get("true_multi_member_inference_required") is not True
            ):
                raise ValueError("V8 ensemble uncertainty contract changed")
            return {
                "real_action_supervision": all(
                    value["real_action_supervision"] for value in member_gates
                ),
                "real_action_supervision_count": min(
                    int(value["real_action_supervision_count"])
                    for value in member_gates
                ),
                "causal_context": all(
                    value["causal_context"] for value in member_gates
                ),
                "real_episode_calibration": all(
                    value["real_episode_calibration"] for value in member_gates
                ),
                "true_multi_member_inference": True,
                "frozen_action_policy_eligible": (
                    manifest.get("action_policy_eligible") is True
                    and manifest.get("action_shadow_ready") is True
                    and confidence.get("abstain") is False
                ),
                "ensemble_member_count": len(member_gates),
                "member_gates": member_gates,
                "shadow_only": True,
                "hardware_action_authorized": False,
            }
        contract = dict(runtime["contract"])
        measured = dict(contract.get("measured_bundle_audit") or {})
        counts = dict(measured.get("real_action_supervision_count_by_system") or {})
        if set(counts) != SUPPORTED_SYSTEM_TYPES:
            raise ValueError("V8 real action evidence counts must cover both systems")
        real_count = int(counts[system_type])
        causal = dict(runtime["causal_context_gate"])
        uncertainty = dict(runtime["uncertainty_calibration"])
        if uncertainty.get("schema") != V8_UNCERTAINTY_SCHEMA:
            raise ValueError("V8 uncertainty calibration schema changed")
        if causal.get("metrics_sha256"):
            if causal.get("metrics_sha256") != stable_json_hash(causal.get("metrics") or {}):
                raise ValueError("V8 causal negative-control metrics hash mismatch")
        calibration_ready = bool(
            uncertainty.get("grouped_bootstrap_fitted") is True
            and uncertainty.get("conformal_fitted") is True
            and uncertainty.get("abstain") is False
        )
        return {
            "real_action_supervision": real_count > 0,
            "real_action_supervision_count": real_count,
            "causal_context": causal.get("causal_context_gate") is True,
            "real_episode_calibration": calibration_ready,
            "true_multi_member_inference": False,
            "frozen_action_policy_eligible": False,
            "shadow_only": True,
            "hardware_action_authorized": False,
        }

    def _rank_ensemble_shadow_candidates(
        self,
        runtime: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
        member_currents: Sequence[Mapping[str, Any]],
        member_model_kwargs: Sequence[Mapping[str, Any]],
    ) -> Tuple[str, np.ndarray, Sequence[Mapping[str, Any]]]:
        """Build one candidate union, then cross-score it with every member."""

        torch = runtime["torch"]
        device = runtime["device"]
        members = runtime["members"]
        if not (
            len(members)
            == len(member_currents)
            == len(member_model_kwargs)
            == len(runtime["member_weights"])
        ):
            raise ValueError("V8 ensemble candidate inputs do not cover every member")
        operation = torch.from_numpy(arrays["operation_features"]).to(device)
        target = torch.from_numpy(arrays["target_features"]).to(device)
        graph = torch.from_numpy(arrays["graph_kernel"]).to(device)
        history = torch.from_numpy(arrays["history"]).to(device)
        history_mask = torch.from_numpy(arrays["history_mask"]).to(device)
        action_mask = torch.from_numpy(arrays["action_mask"]).to(device)
        family_names = ("no_op", "phase_25", "phase_50", "phase_full", "amplitude", "joint")
        candidate_actions: List[Any] = [torch.zeros_like(action_mask[0])]
        candidate_names = ["no_op"]
        candidate_sources: List[Optional[int]] = [None]
        with torch.no_grad():
            for member, current, seed in zip(
                members,
                member_currents,
                runtime["member_seeds"],
            ):
                policy = member["model"].model_b(
                    current["state_embedding"],
                    history,
                    history_mask,
                    action_mask,
                    system_type=str(member_model_kwargs[0]["system_type"]),
                )
                proposed = policy["candidate_actions"][0]
                if proposed.shape != (len(family_names), action_mask.shape[-1]):
                    raise ValueError("V8 ensemble member candidate shape changed")
                if not bool(torch.isfinite(proposed).all()):
                    raise ValueError("V8 ensemble member candidate is non-finite")
                if not bool(torch.equal(proposed[0], torch.zeros_like(proposed[0]))):
                    raise ValueError("V8 ensemble member no-op candidate is not exact zero")
                for family_index, family_name in enumerate(family_names[1:], start=1):
                    candidate_actions.append(proposed[family_index])
                    candidate_names.append(f"seed_{seed}_{family_name}")
                    candidate_sources.append(int(seed))
            candidates = torch.stack(candidate_actions, dim=0)
            candidate_count = candidates.shape[0]
            member_predictions = []
            for member, model_kwargs in zip(members, member_model_kwargs):
                repeated_kwargs: Dict[str, Any] = {}
                for name, value in model_kwargs.items():
                    if name == "system_type":
                        repeated_kwargs[name] = value
                    elif value is None:
                        repeated_kwargs[name] = None
                    elif hasattr(value, "shape") and value.shape[0] == 1:
                        repeated_kwargs[name] = value.expand(
                            candidate_count,
                            *value.shape[1:],
                        )
                    else:
                        repeated_kwargs[name] = value
                member_predictions.append(member["model"].model_a(
                    operation.expand(candidate_count, -1, -1),
                    target.expand(candidate_count, -1),
                    graph,
                    candidates,
                    action_mask.expand(candidate_count, -1),
                    **repeated_kwargs,
                ))
            aggregate = aggregate_v8_ensemble_outputs(
                member_predictions,
                runtime["member_weights"],
            )
            utility_aleatoric_std = torch.sqrt(
                aggregate["utility_aleatoric_variance"].clamp_min(0.0)
            )
            utility_epistemic_std = torch.sqrt(
                aggregate["utility_epistemic_variance"].clamp_min(0.0)
            )
            harmful_epistemic_std = torch.sqrt(
                aggregate["harmful_epistemic_variance"].clamp_min(0.0)
            )
            harmful_upper = (
                aggregate["harmful_probability"] + harmful_epistemic_std
            ).clamp(0.0, 1.0)
            score = (
                aggregate["utility_mean"]
                - V8_UTILITY_ALEATORIC_STD_PENALTY * utility_aleatoric_std
                - V8_UTILITY_EPISTEMIC_STD_PENALTY * utility_epistemic_std
                - V8_HARMFUL_UPPER_PROBABILITY_PENALTY * harmful_upper
            )
            if not bool(torch.isfinite(score).all()):
                raise ValueError("V8 ensemble candidate score is non-finite")
            selected = int(torch.argmax(score).cpu())

        diagnostics = []
        for index, name in enumerate(candidate_names):
            member_utility = [
                float(value["utility_mean"][index].cpu())
                for value in member_predictions
            ]
            member_harmful = [
                float(torch.sigmoid(value["harmful_logit"][index]).cpu())
                for value in member_predictions
            ]
            diagnostics.append({
                "name": name,
                "source_seed": candidate_sources[index],
                "action": candidates[index].float().cpu().tolist(),
                "scored_member_count": len(members),
                "scored_member_seeds": list(runtime["member_seeds"]),
                "member_utility_mean": member_utility,
                "member_harmful_probability": member_harmful,
                "utility_mean": float(aggregate["utility_mean"][index].cpu()),
                "utility_aleatoric_std": float(utility_aleatoric_std[index].cpu()),
                "utility_epistemic_std": float(utility_epistemic_std[index].cpu()),
                "harmful_probability": float(
                    aggregate["harmful_probability"][index].cpu()
                ),
                "harmful_epistemic_std": float(harmful_epistemic_std[index].cpu()),
                "harmful_upper_probability": float(harmful_upper[index].cpu()),
                "risk_adjusted_score": float(score[index].cpu()),
                "expected_total_counts": float(
                    aggregate["expected_total"][index].cpu()
                ),
                "expected_total_count_epistemic_variance": float(
                    aggregate["count_epistemic_variance"][index].sum().cpu()
                ),
            })
        return (
            candidate_names[selected],
            candidates[selected].float().cpu().numpy(),
            tuple(diagnostics),
        )

    def _rank_shadow_candidates(
        self,
        runtime: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
        current: Mapping[str, Any],
        model_kwargs: Mapping[str, Any],
    ) -> Tuple[str, np.ndarray, Sequence[Mapping[str, Any]]]:
        torch = runtime["torch"]
        model = runtime["model"]
        device = runtime["device"]
        operation = torch.from_numpy(arrays["operation_features"]).to(device)
        target = torch.from_numpy(arrays["target_features"]).to(device)
        graph = torch.from_numpy(arrays["graph_kernel"]).to(device)
        history = torch.from_numpy(arrays["history"]).to(device)
        history_mask = torch.from_numpy(arrays["history_mask"]).to(device)
        action_mask = torch.from_numpy(arrays["action_mask"]).to(device)
        with torch.no_grad():
            policy = model.model_b(
                current["state_embedding"],
                history,
                history_mask,
                action_mask,
                system_type=str(model_kwargs["system_type"]),
            )
            candidates = policy["candidate_actions"][0]
            count = candidates.shape[0]
            repeated_kwargs: Dict[str, Any] = {}
            for name, value in model_kwargs.items():
                if name == "system_type":
                    repeated_kwargs[name] = value
                elif value is None:
                    repeated_kwargs[name] = None
                elif hasattr(value, "shape") and value.shape[0] == 1:
                    repeated_kwargs[name] = value.expand(count, *value.shape[1:])
                else:
                    repeated_kwargs[name] = value
            prediction = model.model_a(
                operation.expand(count, -1, -1),
                target.expand(count, -1),
                graph,
                candidates,
                action_mask.expand(count, -1),
                **repeated_kwargs,
            )
            utility = prediction["utility_mean"].float()
            utility_std = torch.exp(prediction["utility_log_scale"].float())
            harmful = torch.sigmoid(prediction["harmful_logit"].float())
            score = utility - 0.5 * utility_std - 0.25 * harmful
            selected = int(torch.argmax(score).cpu())
        names = ("no_op", "phase_25", "phase_50", "phase_full", "amplitude", "joint")
        diagnostics = []
        for index, name in enumerate(names):
            diagnostics.append({
                "name": name,
                "action": candidates[index].float().cpu().tolist(),
                "utility_mean": float(utility[index].cpu()),
                "utility_std": float(utility_std[index].cpu()),
                "harmful_probability": float(harmful[index].cpu()),
                "risk_adjusted_score": float(score[index].cpu()),
                "expected_total_counts": float(
                    prediction["expected_total"][index].cpu()
                ),
            })
        return (
            names[selected],
            candidates[selected].float().cpu().numpy(),
            tuple(diagnostics),
        )

    def _verify_files(
        self,
        root: Path,
        hashes: Mapping[str, str],
    ) -> Dict[str, str]:
        resolved_root = root.resolve()
        values: Dict[str, str] = {}
        if not hashes:
            raise ValueError("V8 prepared artifact file hash mapping is empty")
        for name, expected in hashes.items():
            path = (resolved_root / str(name)).resolve()
            try:
                path.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError("V8 prepared artifact file path traversal") from exc
            self._require_sha256(str(expected), f"prepared file {name}")
            digest = self._sha256_file(path)
            if digest != expected:
                raise ValueError(f"V8 prepared artifact file hash mismatch: {name}")
            values[str(name)] = digest
        return values

    def _safe_repository_path(self, relative: str) -> Path:
        path_value = Path(relative)
        if path_value.is_absolute():
            raise ValueError("V8 source audit paths must be repository-relative")
        path = (self.repository_root / path_value).resolve()
        try:
            path.relative_to(self.repository_root)
        except ValueError as exc:
            raise ValueError("V8 source audit path escapes repository") from exc
        return path

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"JSON object required: {path}")
        return dict(value)

    @staticmethod
    def _require_sha256(value: str, field_name: str) -> None:
        if not SHA256_PATTERN.fullmatch(str(value).lower()):
            raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


__all__ = [
    "BoundHistoryTransitionV1",
    "MeasurementExposureProvenanceV1",
    "V8FeedbackBridge",
    "V8_REQUIRED_SOURCE_FILES",
    "V8ShadowDiagnostic",
    "V8_BOUND_HISTORY_SCHEMA",
    "V8_EXPOSURE_PROVENANCE_SCHEMA",
    "V8_SHADOW_DIAGNOSTIC_SCHEMA",
    "V8_SNAPSHOT_PROVENANCE_SCHEMA",
    "aggregate_v8_ensemble_outputs",
]
