"""State-machine runner for normal and feedback W-state experiments."""

from __future__ import annotations

import json
import hashlib
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ml_recon.feedback_model import TrainedFeedbackModel
from ml_recon.aod_calibration.multifrequency_coupling import (
    MultiFrequencyCouplingModel,
    build_default_features,
)

from system_control.qubit_recon_files.system_specs import SYSTEM_SPECS

from .awg_uploader import AwgUploader, NoOpAwgUploader
from .data_source import MeasurementAccumulator, MockMeasurementSource
from .data_protocol import SnapshotManifestV4, stable_json_hash, validate_snapshot_files
from .dataset_store import FeedbackDatasetStore, matrix_hash_snapshot
from .delta_manager import DeltaManager
from .model_paths import resolve_classifier_model_path, resolve_feedback_model_path
from .models import (
    CalibrationDelta,
    ControlStateSpecV1,
    ExperimentRunConfig,
    FeedbackPrediction,
    FeedbackSample,
    MeasurementSnapshot,
    TargetStateSpecV1,
)
from .noise_context import build_noise_context
from .operation_calibration import OperationCalibrationStore
from .operation_points import build_operation_registry, operation_registry_hash
from .sequence_service import PreparedSequenceArtifactV1, SequenceService
from ml_recon.physics_hybrid_v4.probe_design import (
    EphemeralPhaseProbeStore,
    build_phase_probe_tensor,
    phase_probe_tensor_to_cycle_matrices,
)


class ExperimentState(Enum):
    IDLE = "Idle"
    SEQUENCE_UI_OPEN = "SequenceUiOpen"
    SEQUENCE_READY = "SequenceGenerated"
    WAITING_AWG_UPLOAD = "WaitingAwgUpload"
    WAITING_AWG_READY = "WaitingAwgReady"
    DATA_UI_OPEN = "DataUiOpen"
    COLLECTING = "Collecting"
    RECONSTRUCTING = "Reconstructing"
    INFERRING_DELTA = "InferringDelta"
    GUARDING_DELTA = "GuardingDelta"
    APPLYING_DELTA = "ApplyingDelta"
    REGENERATING_SEQUENCE = "RegeneratingSequence"
    SAVING = "Saving"
    WAITING_NEXT_STATE = "WaitingNextState"
    PAUSED = "Paused"


class ExperimentRunner:
    """Coordinate sequence generation, data collection, ML calibration, and saving."""

    def __init__(
        self,
        config: ExperimentRunConfig,
        sequence_service: Optional[SequenceService] = None,
        delta_manager: Optional[DeltaManager] = None,
        awg_uploader: Optional[AwgUploader] = None,
        coupling_model: Optional[MultiFrequencyCouplingModel] = None,
        feedback_model: Optional[TrainedFeedbackModel] = None,
        dataset_store: Optional[FeedbackDatasetStore] = None,
        v6_feedback_bridge: Optional[Any] = None,
        prepared_artifact: Optional[PreparedSequenceArtifactV1] = None,
        v7_feedback_bridge: Optional[Any] = None,
        v8_feedback_bridge: Optional[Any] = None,
    ):
        self.config = config
        self.sequence_service = sequence_service or SequenceService()
        self.delta_manager = delta_manager or DeltaManager(str(self.sequence_service.matrix_dir))
        self.awg_uploader = awg_uploader or NoOpAwgUploader()
        n = 3 if config.system_type == "3qubit" else 4
        self.coupling_model = coupling_model or MultiFrequencyCouplingModel(num_modes=n)
        self.feedback_model = feedback_model
        self.dataset_store = dataset_store or FeedbackDatasetStore()
        self.v6_feedback_bridge = v6_feedback_bridge
        self.v7_feedback_bridge = v7_feedback_bridge
        self.v8_feedback_bridge = v8_feedback_bridge
        self.prepared_artifact = prepared_artifact
        self._classifier = None  # lazy-loaded QuantumStateFamilyTorchClassifier
        self._classifier_load_attempted = False
        self.state = ExperimentState.IDLE
        self.current_state_index = 0
        self.last_sequence_dir: Optional[str] = None
        self.last_sample_path: Optional[Path] = None
        self.last_prediction: Optional[FeedbackPrediction] = None
        self.operation_calibration = OperationCalibrationStore(config.system_type)
        self.phase_probe_store: Optional[EphemeralPhaseProbeStore] = None
        self.last_phase_probe: Optional[Dict] = None
        self.probe_recovery_reset_pending = False

    def prepare_sequence(self) -> str:
        self.state = ExperimentState.SEQUENCE_UI_OPEN
        state_params = self.current_state_params()
        probe_overlay = None
        probe_tensor = None
        probe_store = self._probe_store()
        if probe_store is not None and probe_store.active_record() is not None:
            # A previous process stopped between stage and authoritative save.
            # Reset hardware to the unprobed sequence before staging any new probe.
            probe_store.recover_and_verify_cleared()
            reset_dir = self.sequence_service.generate(
                self.config.system_type,
                state_params,
                self.config.excitation_mode,
                self.config.output_dir,
                digital_delay_offset_ns=self.config.digital_delay_offset_ns,
            )
            if isinstance(self.awg_uploader, NoOpAwgUploader):
                self.last_sequence_dir = reset_dir
                self.probe_recovery_reset_pending = True
                self.state = ExperimentState.SEQUENCE_READY
                return reset_dir
            self.awg_uploader.upload(
                reset_dir,
                {
                    "system_type": self.config.system_type,
                    "workflow_mode": self.config.workflow_mode,
                    "state": state_params,
                    "phase_probe_recovery_reset": True,
                },
            )
        if (
            self.config.system_type == "4qubit"
            and self.config.enable_phase_probes
            and state_params.get("state_role") == "anchor"
        ):
            if probe_store is None:
                raise RuntimeError("4Q anchor probe requires an ephemeral campaign store")
            # The probe directions are independent of matrix values.  Build the
            # in-memory overlay now, but bind/persist its contract only after
            # phase_cal_4qubit has generated the *new target state's* matrices.
            # Otherwise a normal coverage-to-anchor transition looks like an
            # unsafe base-matrix mutation.
            probe_tensor = build_phase_probe_tensor(
                state_params,
                [{} for _ in range(256)],
            )
            probe_overlay = phase_probe_tensor_to_cycle_matrices(probe_tensor)
        strict_env_before = os.environ.get("NQUBIT_STRICT_MATRIX_LOADING")
        if self.config.strict_hardware_mode:
            os.environ["NQUBIT_STRICT_MATRIX_LOADING"] = "1"
        try:
            generate_kwargs = {
                "digital_delay_offset_ns": self.config.digital_delay_offset_ns,
            }
            if probe_overlay is not None:
                generate_kwargs["phase_probe_overlay"] = probe_overlay
            self.last_sequence_dir = self.sequence_service.generate(
                self.config.system_type,
                state_params,
                self.config.excitation_mode,
                self.config.output_dir,
                **generate_kwargs,
            )
        finally:
            if strict_env_before is None:
                os.environ.pop("NQUBIT_STRICT_MATRIX_LOADING", None)
            else:
                os.environ["NQUBIT_STRICT_MATRIX_LOADING"] = strict_env_before
        if probe_store is not None and probe_overlay is not None and probe_tensor is not None:
            phase_cal_hash = str(
                state_params.get("base_w_ch1_matrix_hash")
                or getattr(self.sequence_service, "last_4qubit_base_matrix_hash", "")
                or ""
            )
            realized_hash = self._w_ch1_base_matrix_hash()
            if not phase_cal_hash:
                # Compatibility for test/dry services that do not yet emit the
                # phase-calculation manifest identity.
                phase_cal_hash = realized_hash
            if realized_hash != phase_cal_hash:
                raise RuntimeError(
                    "base W_CH1 matrices changed after phase_cal_4qubit completed"
                )
            registry = build_operation_registry(
                "4qubit",
                matrix_dir=self.sequence_service.matrix_dir,
                include_ml_delta=True,
            )
            contract = probe_store.stage(
                plan_id=str(
                    state_params.get("plan_id")
                    or self.config.campaign_id
                    or "v4_4q_1000"
                ),
                schedule_state=state_params,
                operation_registry=registry,
                requested_awg_matrix_hash=phase_cal_hash,
                actual_awg_matrix_hash=realized_hash,
            )
            stored_tensor = probe_store.load_active_tensor()
            if not np.array_equal(stored_tensor, probe_tensor):
                raise RuntimeError(
                    "persisted phase-probe tensor differs from the generated waveform overlay"
                )
            self.last_phase_probe = contract.to_json_dict()
            probe_store.mark_lifecycle(
                "generated",
                base_matrix_hash=realized_hash,
                waveform_hash=self._waveform_directory_hash(self.last_sequence_dir),
            )
        self.state = ExperimentState.SEQUENCE_READY
        self.state = ExperimentState.WAITING_AWG_UPLOAD
        self.awg_uploader.upload(self.last_sequence_dir, {
            "system_type": self.config.system_type,
            "mode": self.config.mode,
            "workflow_mode": self.config.workflow_mode,
            "state": state_params,
            "excitation_mode": self.config.excitation_mode,
            "strict_hardware_mode": self.config.strict_hardware_mode,
            "digital_delay_offset_ns": self.config.digital_delay_offset_ns,
            "phase_probe": self.last_phase_probe,
        })
        if (
            probe_store is not None
            and probe_overlay is not None
            and not isinstance(self.awg_uploader, NoOpAwgUploader)
        ):
            probe_store.mark_lifecycle(
                "uploaded",
                waveform_hash=self._waveform_directory_hash(self.last_sequence_dir),
            )
        self.state = ExperimentState.WAITING_AWG_READY if self.config.awg_ready_manual else ExperimentState.DATA_UI_OPEN
        return self.last_sequence_dir

    def collect_mock_window(self, window_s: Optional[float] = None, frames: int = 5) -> MeasurementSnapshot:
        self.state = ExperimentState.COLLECTING
        source = MockMeasurementSource(self.config.system_type)
        accumulator = MeasurementAccumulator(self.config.system_type, window_s or 0.0)
        for _ in range(frames):
            accumulator.add_frame(source.read_frame())
            time.sleep(0.001)
        return accumulator.snapshot()

    def calibrate(self, snapshot: MeasurementSnapshot) -> MeasurementSnapshot:
        self.state = ExperimentState.RECONSTRUCTING
        self._reconstruct(snapshot)
        if self.config.workflow_mode == "feedback_optimize":
            prediction = self._predict_feedback(snapshot)
            self.state = ExperimentState.GUARDING_DELTA
            guarded = self._guard_prediction_with_steps(prediction, snapshot)
            snapshot.deltas = guarded.deltas
            shadow = self.config.noise_aware_mode == "shadow"
            gated_without_promotion = self.config.noise_aware_mode == "gated" and not self.config.promotion_ready
            # Shadow mode is strictly read-only.  Candidate files begin only
            # after the operator switches to gated/strict evaluation.
            candidate_workflow = self.config.strict_hardware_mode or self.config.noise_aware_mode == "gated"
            actionable = any(delta.accepted for delta in guarded.deltas)
            staged = None
            if candidate_workflow and actionable:
                staged = self.delta_manager.stage_candidate(
                    self.config.system_type,
                    guarded.deltas,
                    metadata={
                        "snapshot_timestamp": float(snapshot.timestamp),
                        "prediction_source": guarded.prediction_source,
                        "prediction_confidence": float(guarded.confidence),
                        "strict_hardware_mode": bool(self.config.strict_hardware_mode),
                        "noise_aware_mode": self.config.noise_aware_mode,
                    },
                )
                snapshot.deltas = staged["deltas"]
                guarded.guard_result = dict(guarded.guard_result)
                guarded.guard_result.update({
                    "candidate_id": staged["candidate_id"],
                    "candidate_manifest_path": staged["manifest_path"],
                    "candidate_status": "candidate",
                })

            requires_confirmation = candidate_workflow and actionable and not self.config.promotion_ready
            if shadow or gated_without_promotion or requires_confirmation:
                guarded.guard_result = dict(guarded.guard_result)
                guarded.guard_result["hardware_apply_blocked"] = True
                guarded.guard_result["hardware_apply_reason"] = (
                    "noise-aware shadow mode"
                    if shadow
                    else "candidate requires explicit human promotion_ready=true"
                )
                snapshot.diagnostics["feedback_apply_status"] = guarded.guard_result["hardware_apply_reason"]
            elif staged is not None:
                self.state = ExperimentState.APPLYING_DELTA
                promotion = self.delta_manager.promote_candidate(
                    staged["candidate_id"],
                    human_approved=bool(self.config.promotion_ready),
                )
                for delta, active_path in zip(snapshot.deltas, promotion["active_paths"]):
                    delta.path = active_path
                guarded.guard_result = dict(guarded.guard_result)
                guarded.guard_result.update({
                    "candidate_status": "promoted",
                    "hardware_apply_blocked": False,
                    "human_approved": True,
                })
                snapshot.diagnostics["feedback_apply_status"] = "candidate promoted"
            else:
                self.state = ExperimentState.APPLYING_DELTA
                snapshot.deltas = self.delta_manager.save_deltas(self.config.system_type, guarded.deltas)
            self.last_prediction = guarded
        return snapshot

    def save_snapshot(self, snapshot: MeasurementSnapshot) -> Path:
        self.state = ExperimentState.SAVING
        payload = snapshot.to_json_dict()
        payload["run_config"] = {
            "system_type": self.config.system_type,
            "mode": self.config.mode,
            "workflow_mode": self.config.workflow_mode,
            "excitation_mode": self.config.excitation_mode,
            "target_fidelity": self.config.target_fidelity,
            "state": self.current_state_params(),
        }
        if self.config.workflow_mode == "collect_dataset":
            return self._write_normal_snapshot(payload)
        return self.delta_manager.write_snapshot_metadata(self.config.system_type, payload)

    def run_dry_cycle(self) -> MeasurementSnapshot:
        if self.config.workflow_mode == "collect_dataset":
            snapshot = self.run_dataset_collection_cycle()
            return snapshot
        return self.run_feedback_window_cycle()

    def run_dataset_collection_cycle(self, snapshot: Optional[MeasurementSnapshot] = None) -> MeasurementSnapshot:
        self.prepare_sequence()
        snapshot = snapshot or self.collect_mock_window(window_s=self.config.collection_duration_s, frames=8)
        return self.process_dataset_snapshot(snapshot)

    def process_dataset_snapshot(self, snapshot: MeasurementSnapshot) -> MeasurementSnapshot:
        self.state = ExperimentState.RECONSTRUCTING
        self._reconstruct(snapshot)
        if self.last_phase_probe:
            snapshot.diagnostics["phase_probe"] = dict(self.last_phase_probe)
        if self.config.use_mock_data or self.config.legacy_feedback_sample_export:
            self.save_snapshot(snapshot)
        if self.config.legacy_feedback_sample_export:
            sample = self._build_feedback_sample(snapshot)
            self.last_sample_path = self.dataset_store.save_sample(sample)
        self.state = ExperimentState.WAITING_NEXT_STATE
        return snapshot

    def finalize_authoritative_snapshot(
        self,
        snapshot_path: str | Path,
        *,
        sample_id: Optional[str] = None,
    ) -> None:
        """Validate an authoritative V4 snapshot and mark its probe saved.

        Clearing is deliberately a separate transaction.  The campaign ledger
        must be committed first so a crash cannot lose both the active probe and
        the only pointer to the completed snapshot.
        """

        store = self._probe_store()
        path = Path(snapshot_path)
        if not path.exists() or not (path / "manifest.json").exists():
            raise RuntimeError("phase probe cannot clear before an authoritative V4 manifest exists")
        manifest = validate_snapshot_files(path)
        if not isinstance(manifest, SnapshotManifestV4):
            raise RuntimeError("the authoritative 4Q campaign snapshot must use manifest V4")
        if sample_id is not None and str(sample_id) != manifest.sample_id:
            raise RuntimeError("authoritative snapshot sample_id does not match the commit request")
        if manifest.system_type != "4qubit":
            raise RuntimeError("the authoritative V4 campaign snapshot must be 4qubit")
        if float(manifest.duration_s) != 1200.0:
            raise RuntimeError("the authoritative V4 campaign snapshot must contain 1200 s")
        if self.config.campaign_id and manifest.campaign_id != self.config.campaign_id:
            raise RuntimeError("authoritative snapshot campaign_id does not match the active run")
        target = self.current_state_params()
        expected_plan_id = target.get("plan_id")
        campaign_manifest_path = (
            Path(self.config.campaign_run_dir) / "campaign_manifest.json"
            if self.config.campaign_run_dir
            else None
        )
        if expected_plan_id in (None, "") and campaign_manifest_path is not None:
            try:
                expected_plan_id = json.loads(
                    campaign_manifest_path.read_text(encoding="utf-8")
                ).get("plan_id")
            except (OSError, ValueError, TypeError):
                expected_plan_id = None
        expected_plan_id = str(expected_plan_id or self.config.campaign_id or "")
        if expected_plan_id and manifest.plan_id != expected_plan_id:
            raise RuntimeError("authoritative snapshot plan_id does not match the scheduled state")
        expected_role = str(target.get("state_role") or "unspecified")
        if manifest.state_role != expected_role:
            raise RuntimeError("authoritative snapshot state_role does not match the scheduled state")
        expected_anchor = target.get("anchor_id")
        if (manifest.anchor_id or None) != (None if expected_anchor is None else str(expected_anchor)):
            raise RuntimeError("authoritative snapshot anchor_id does not match the scheduled state")
        expected_index = target.get("schedule_index")
        if expected_index is not None and int(manifest.state_index or -1) != int(expected_index):
            raise RuntimeError("authoritative snapshot state_index does not match the scheduled state")
        required_arrays = {
            "signal_bins",
            "idler_bins",
            "coin_bins",
            "signal_bins_raw",
            "idler_bins_raw",
            "coin_bins_raw",
        }
        missing_arrays = sorted(required_arrays - set(manifest.arrays))
        if missing_arrays:
            raise RuntimeError(
                "authoritative snapshot is missing required count arrays: "
                + ", ".join(missing_arrays)
            )
        missing_hashes = [
            name
            for name, value in {
                "schema": manifest.schema_hash,
                "registry": manifest.operation_registry_hash,
                "calibration": manifest.calibration_snapshot_hash,
                "matrix": manifest.matrix_snapshot_hash,
                "model_schema": manifest.model_schema_hash,
                "awg": (manifest.awg_state or {}).get("waveform_hash"),
            }.items()
            if value in (None, "")
        ]
        if missing_hashes:
            raise RuntimeError(
                "authoritative snapshot is missing required hashes: " + ", ".join(missing_hashes)
            )
        blocking_flags = [
            flag
            for flag in manifest.quality_flags
            if flag.startswith("missing_required_array:")
            or flag.startswith("missing_raw_count_array:")
            or flag.startswith("missing_context:")
        ]
        if blocking_flags:
            raise RuntimeError(
                "authoritative snapshot failed quality checks: " + ", ".join(blocking_flags)
            )

        if store is None or store.active_record() is None:
            if expected_role == "anchor" or manifest.probe:
                raise RuntimeError("anchor snapshot is missing its active phase-probe contract")
            return
        pointer = store.active_record() or {}
        if pointer.get("lifecycle_status") != "collecting":
            raise RuntimeError("phase probe must be collecting before its snapshot can be saved")
        contract = store.load_active_contract()
        for field_name in (
            "plan_id",
            "anchor_id",
            "repeat_index",
            "code_pair_id",
            "pair_sign",
            "phase_tensor_sha256",
            "operation_registry_hash",
        ):
            if manifest.probe.get(field_name) != getattr(contract, field_name):
                raise RuntimeError(
                    f"authoritative snapshot probe mismatch: {field_name}"
                )
        if "phase_probe_tensor" not in manifest.arrays:
            raise RuntimeError("anchor snapshot is missing phase_probe_tensor.npy")
        if manifest.arrays["phase_probe_tensor"].get("sha256") != contract.phase_tensor_sha256:
            raise RuntimeError("snapshot phase-probe tensor hash does not match the active probe")
        waveform_hash = (manifest.awg_state or {}).get("waveform_hash")
        if pointer.get("waveform_hash") not in (None, waveform_hash):
            raise RuntimeError("snapshot AWG waveform hash does not match the uploaded probe")
        store.mark_lifecycle(
            "saved",
            snapshot_path=str(path),
            sample_id=manifest.sample_id,
        )

    def mark_active_probe_uploaded(self) -> None:
        """Record completion of the real UI-driven AWG upload."""

        store = self._probe_store()
        if store is None or store.active_record() is None:
            return
        store.load_active_contract()
        store.mark_lifecycle(
            "uploaded",
            waveform_hash=self._waveform_directory_hash(self.last_sequence_dir or ""),
        )

    def mark_active_probe_collecting(self) -> None:
        """Record that the FPGA output window has actually started."""

        store = self._probe_store()
        if store is None or store.active_record() is None:
            return
        store.load_active_contract()
        store.mark_lifecycle("collecting", collection_started_at=time.time())

    def clear_finalized_probe(self) -> None:
        """Clear only a hash-validated probe already marked as saved."""

        store = self._probe_store()
        if store is not None and store.active_record() is not None:
            store.clear(require_saved=True)

    def run_feedback_window_cycle(self, snapshot: Optional[MeasurementSnapshot] = None) -> MeasurementSnapshot:
        self.prepare_sequence()
        snapshot = snapshot or self.collect_mock_window(window_s=self.config.feedback_interval_s, frames=8)
        return self.process_feedback_snapshot(snapshot)

    def process_feedback_snapshot(self, snapshot: MeasurementSnapshot) -> MeasurementSnapshot:
        snapshot = self.calibrate(snapshot)
        self.save_snapshot(snapshot)
        can_regenerate = bool(
            self.last_prediction
            and self.last_prediction.guard_result.get("candidate_status") == "promoted"
        ) or (
            self.config.noise_aware_mode == "disabled"
            and not self.config.strict_hardware_mode
        )
        if self.config.auto_apply and can_regenerate and any(delta.accepted and delta.path for delta in snapshot.deltas):
            self.state = ExperimentState.REGENERATING_SEQUENCE
            self.prepare_sequence()
        self.state = ExperimentState.IDLE
        return snapshot

    def advance_state_schedule(self) -> None:
        self.current_state_index = (self.current_state_index + 1) % len(self.config.state_schedule)

    def current_state_params(self) -> Dict:
        return self.config.state_schedule[self.current_state_index]

    def _write_normal_snapshot(self, payload: Dict) -> Path:
        root = Path(self.config.output_dir) if self.config.output_dir else Path(self.last_sequence_dir or self.sequence_service.qubit_recon_dir)
        snapshot_dir = root / "normal_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(payload.get("timestamp", time.time())))
        path = snapshot_dir / f"snapshot_{self.config.system_type}_{stamp}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def _build_feedback_sample(self, snapshot: MeasurementSnapshot) -> FeedbackSample:
        basis = self._load_basis_config()
        matrix_snapshot = matrix_hash_snapshot(self.sequence_service.matrix_dir)
        target_rho = self._target_density_matrix(self.current_state_params(), self.config.system_type)
        return self.dataset_store.create_sample(
            system_type=self.config.system_type,
            target_state_params=self.current_state_params(),
            rho_target=target_rho,
            snapshot=snapshot,
            basis_config=basis,
            matrix_snapshot=matrix_snapshot,
            notes="collect_dataset",
            data_provenance="synthetic" if self.config.use_mock_data else "real",
        )

    def _predict_feedback(self, snapshot: MeasurementSnapshot) -> FeedbackPrediction:
        if self.config.feedback_backend == "v8_shadow":
            return self._predict_v8_feedback(snapshot)
        if self.config.feedback_backend == "v7_shadow":
            return self._predict_v7_feedback(snapshot)
        if self.config.feedback_backend != "legacy":
            return self._predict_v6_feedback(snapshot)
        if snapshot.density_matrix is None:
            self._reconstruct(snapshot)
        target_rho = self._target_density_matrix(self.current_state_params(), self.config.system_type)
        model = self._load_feedback_model()
        self.state = ExperimentState.INFERRING_DELTA
        context = {
            "system_type": self.config.system_type,
            "target_state_params": self.current_state_params(),
            "retrieval_rate": snapshot.retrieval_rate,
            "retrieval_metrics": {
                "retrieval_rate": snapshot.retrieval_rate,
                "fidelity": snapshot.fidelity,
                "purity": snapshot.purity,
                "diagnostics": snapshot.diagnostics,
            },
            "classifier_diagnostics": (
                snapshot.diagnostics.get("classifier_diagnostics")
                or snapshot.diagnostics.get("diagnosis")
                or snapshot.diagnostics.get("quantum_diagnosis")
            ),
            "matrix_snapshot": matrix_hash_snapshot(self.sequence_service.matrix_dir),
            "retrieval_map_path": self.config.retrieval_map_path,
            "excitation_mode": self.config.excitation_mode,
            "phase_feedback_strategy": self.config.phase_feedback_strategy,
            "strict_hardware_mode": self.config.strict_hardware_mode,
            "noise_context": snapshot.diagnostics.get("noise_context", {}),
            "feature_schema_version": snapshot.feature_schema_version,
        }

        n_passes = max(1, int(self.config.ensemble_n_passes))
        if n_passes > 1 and self._supports_dropout_ensemble(model):
            prediction = self._ensemble_predict(model, target_rho, snapshot, context, n_passes)
        else:
            prediction = model.predict(target_rho, snapshot.density_matrix, context)

        # New pointwise heads are shadow diagnostics until a separately trained
        # checkpoint is explicitly promoted.  Keep them in the established
        # FeedbackPrediction surface without changing CalibrationDelta.
        shadow = snapshot.diagnostics.get("noise_aware_reconstruction", {}) or {}
        if shadow:
            prediction.physics_diagnostics["operation_point_shadow"] = {
                "feature_schema_version": 2,
                "mode": self.config.noise_aware_mode,
                "conditional_fidelity_median": shadow.get("conditional_fidelity_median"),
                "conditional_fidelity_ci": shadow.get("conditional_fidelity_ci"),
                "leakage_adjusted_fidelity_median": shadow.get("leakage_adjusted_fidelity_median"),
                "leakage_adjusted_fidelity_ci": shadow.get("leakage_adjusted_fidelity_ci"),
                "point_diagnostics": shadow.get("point_diagnostics", {}),
                "model_uncertainty": shadow.get("elementwise_std"),
                "feedback_apply_forbidden": self.config.noise_aware_mode == "shadow",
            }

        threshold = self.config.uncertainty_threshold
        if threshold is not None:
            unc = float((prediction.physics_diagnostics or {}).get("ensemble_std", 0.0) or 0.0)
            if unc > float(threshold):
                reason = (
                    f"ensemble_std {unc:.4f} above uncertainty_threshold {threshold:.4f} — "
                    "skipping delta this round"
                )
                prediction.guard_result = {
                    "accepted": False,
                    "reason": reason,
                    "ensemble_skipped": True,
                    "ensemble_std": unc,
                    "uncertainty_threshold": float(threshold),
                }
                prediction.deltas = [self._reject_delta(delta, reason) for delta in prediction.deltas]
        return prediction

    def set_v6_prepared_artifact(
        self, artifact: PreparedSequenceArtifactV1,
    ) -> None:
        """Bind the exact pre-measurement command artifact for read-only V6 inference."""
        self.prepared_artifact = artifact

    def _predict_v6_feedback(self, snapshot: MeasurementSnapshot) -> FeedbackPrediction:
        if self.config.feedback_backend != "v6_shadow":
            raise RuntimeError(
                f"{self.config.feedback_backend} requires the gated feedback episode orchestrator"
            )
        artifact = self.prepared_artifact
        if artifact is None:
            raise RuntimeError("v6_shadow requires an exact PreparedSequenceArtifactV1")
        bridge = self.v6_feedback_bridge
        if bridge is None:
            if not self.config.v6_checkpoint_path or not self.config.v6_normalization_path:
                raise RuntimeError(
                    "v6_shadow requires v6_checkpoint_path and v6_normalization_path"
                )
            from .v6_feedback_bridge import V6FeedbackBridge

            bridge = V6FeedbackBridge(
                self.config.v6_checkpoint_path,
                self.config.v6_normalization_path,
                config_path=self.config.v6_config_path,
            )
            self.v6_feedback_bridge = bridge
        target_state = TargetStateSpecV1.from_state(
            self.config.system_type,
            self.current_state_params(),
            excitation_mode=self.config.excitation_mode,
        )
        actual_control_payload = dict(
            (artifact.materialization or {}).get("actual_control_state") or {}
        )
        if not actual_control_payload:
            raise RuntimeError("prepared artifact is missing actual_control_state")
        control_state = ControlStateSpecV1.from_state(
            self.config.system_type, actual_control_payload,
        )
        proposal = bridge.propose(
            target_state,
            control_state,
            snapshot,
            artifact,
            episode_history=(),
        )
        proposal_payload = proposal.to_json_dict()
        proposal_path = self._save_v6_shadow_proposal(snapshot, proposal_payload)
        snapshot.diagnostics["v6_shadow_proposal"] = proposal_payload
        snapshot.diagnostics["v6_shadow_proposal_path"] = str(proposal_path)
        return FeedbackPrediction(
            deltas=[],
            safe_deltas=[],
            confidence=0.0,
            heuristic_gain_proxy=0.0,
            physics_diagnostics={"v6_shadow_proposal": proposal_payload},
            guard_result={
                "accepted": False,
                "hardware_apply_blocked": True,
                "hardware_apply_reason": "v6_shadow is read-only",
                "materialized": False,
                "bundle_staged": False,
                "uploaded": False,
                "active_command_changed": False,
            },
            prediction_source="v6_shadow",
        )

    def _save_v6_shadow_proposal(
        self,
        snapshot: MeasurementSnapshot,
        proposal: Dict[str, Any],
    ) -> Path:
        if self.config.v6_shadow_output_dir:
            root = Path(self.config.v6_shadow_output_dir)
        elif self.config.output_dir:
            root = Path(self.config.output_dir) / "v6_shadow"
        else:
            root = self.sequence_service.qubit_recon_dir.parents[1] / "outputs" / "v6_shadow"
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(snapshot.timestamp))
        destination = root / f"v6_shadow_{self.config.system_type}_{stamp}.json"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(proposal, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
        return destination

    def _predict_v7_feedback(self, snapshot: MeasurementSnapshot) -> FeedbackPrediction:
        """Run frozen V7 Model A/B inference and persist diagnostics only."""

        artifact = self.prepared_artifact
        if artifact is None:
            raise RuntimeError("v7_shadow requires an exact PreparedSequenceArtifactV1")
        bridge = self.v7_feedback_bridge
        if bridge is None:
            if not self.config.v7_checkpoint_path or not self.config.v7_training_report_path:
                raise RuntimeError(
                    "v7_shadow requires v7_checkpoint_path and v7_training_report_path"
                )
            from .v7_feedback_bridge import V7FeedbackBridge

            bridge = V7FeedbackBridge(
                self.config.v7_checkpoint_path,
                self.config.v7_training_report_path,
                device=self.config.v7_device,
                measurement_window_s=self.config.v7_measurement_window_s,
            )
            self.v7_feedback_bridge = bridge
        target_state = TargetStateSpecV1.from_state(
            self.config.system_type,
            self.current_state_params(),
            excitation_mode=self.config.excitation_mode,
        )
        history = tuple((snapshot.diagnostics or {}).get("v7_episode_history") or ())
        if len(history) > 9:
            history = history[-9:]
        proposal = bridge.propose(
            target_state,
            snapshot,
            artifact,
            episode_history=history,
        )
        proposal_payload = proposal.to_json_dict()
        proposal_path = self._save_v7_shadow_proposal(snapshot, proposal_payload)
        snapshot.diagnostics["v7_shadow_proposal"] = proposal_payload
        snapshot.diagnostics["v7_shadow_proposal_path"] = str(proposal_path)
        return FeedbackPrediction(
            deltas=[],
            safe_deltas=[],
            confidence=0.0,
            heuristic_gain_proxy=float(proposal.predicted_gain_mean),
            physics_diagnostics={"v7_shadow_proposal": proposal_payload},
            guard_result={
                "accepted": False,
                "hardware_apply_blocked": True,
                "hardware_apply_reason": "v7_shadow is read-only",
                "materialized": False,
                "bundle_staged": False,
                "uploaded": False,
                "active_command_changed": False,
            },
            prediction_source="v7_shadow",
        )

    def _save_v7_shadow_proposal(
        self,
        snapshot: MeasurementSnapshot,
        proposal: Dict[str, Any],
    ) -> Path:
        if self.config.v7_shadow_output_dir:
            root = Path(self.config.v7_shadow_output_dir)
        elif self.config.output_dir:
            root = Path(self.config.output_dir) / "v7_shadow"
        else:
            root = self.sequence_service.qubit_recon_dir.parents[1] / "outputs" / "v7_shadow"
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(snapshot.timestamp))
        destination = root / f"v7_shadow_{self.config.system_type}_{stamp}.json"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(proposal, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
        return destination

    def _predict_v8_feedback(self, snapshot: MeasurementSnapshot) -> FeedbackPrediction:
        """Run strict V8 inference and persist a read-only layered diagnostic."""

        artifact = self.prepared_artifact
        if artifact is None:
            raise RuntimeError("v8_shadow requires an exact PreparedSequenceArtifactV1")
        bridge = self.v8_feedback_bridge
        if bridge is None:
            has_single_member = bool(
                self.config.v8_checkpoint_path
                and self.config.v8_training_report_path
            )
            if not self.config.v8_frozen_candidate_path and not has_single_member:
                raise RuntimeError(
                    "v8_shadow requires a frozen candidate manifest/pointer or a "
                    "single checkpoint/report pair"
                )
            from .v8_feedback_bridge import V8FeedbackBridge

            bridge = V8FeedbackBridge(
                self.config.v8_checkpoint_path,
                self.config.v8_training_report_path,
                device=self.config.v8_device,
                frozen_candidate_path=self.config.v8_frozen_candidate_path,
            )
            self.v8_feedback_bridge = bridge
        target_state = TargetStateSpecV1.from_state(
            self.config.system_type,
            self.current_state_params(),
            excitation_mode=self.config.excitation_mode,
        )
        history = tuple((snapshot.diagnostics or {}).get("v8_episode_history") or ())
        if len(history) > 9:
            history = history[-9:]
        diagnostic = bridge.diagnose(
            target_state,
            snapshot,
            artifact,
            episode_history=history,
        )
        diagnostic_payload = diagnostic.to_json_dict()
        diagnostic_path = self._save_v8_shadow_diagnostic(
            snapshot,
            diagnostic_payload,
        )
        snapshot.diagnostics["v8_shadow_diagnostic"] = diagnostic_payload
        snapshot.diagnostics["v8_shadow_diagnostic_path"] = str(diagnostic_path)
        return FeedbackPrediction(
            deltas=[],
            safe_deltas=[],
            confidence=0.0,
            heuristic_gain_proxy=0.0,
            physics_diagnostics={"v8_shadow_diagnostic": diagnostic_payload},
            guard_result={
                "accepted": False,
                "hardware_apply_blocked": True,
                "hardware_apply_reason": "v8_shadow is read-only",
                "materialized": False,
                "bundle_staged": False,
                "uploaded": False,
                "delta_written": False,
                "active_command_changed": False,
            },
            prediction_source="v8_shadow",
        )

    def _save_v8_shadow_diagnostic(
        self,
        snapshot: MeasurementSnapshot,
        diagnostic: Dict[str, Any],
    ) -> Path:
        if self.config.v8_shadow_output_dir:
            root = Path(self.config.v8_shadow_output_dir)
        elif self.config.output_dir:
            root = Path(self.config.output_dir) / "v8_shadow"
        else:
            root = (
                self.sequence_service.qubit_recon_dir.parents[1]
                / "outputs"
                / "v8_shadow"
            )
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(snapshot.timestamp))
        destination = root / f"v8_shadow_{self.config.system_type}_{stamp}.json"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(diagnostic, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def _supports_dropout_ensemble(self, model: "TrainedFeedbackModel") -> bool:
        """MC Dropout ensemble only makes sense for models with dropout layers."""
        return getattr(model, "physics_transfer_model", None) is not None

    def _ensemble_predict(
        self,
        model: "TrainedFeedbackModel",
        target_rho: np.ndarray,
        snapshot: MeasurementSnapshot,
        context: Dict,
        n_passes: int,
    ) -> FeedbackPrediction:
        """Run model.predict() n_passes times with MC Dropout active and aggregate."""
        ensemble_context = dict(context)
        ensemble_context["ensemble_train_mode"] = True

        predictions: list[FeedbackPrediction] = []
        for _ in range(n_passes):
            predictions.append(model.predict(target_rho, snapshot.density_matrix, ensemble_context))

        # Aggregate: mean delta arrays, std as scalar uncertainty
        if not predictions:
            return model.predict(target_rho, snapshot.density_matrix, context)

        base = predictions[0]
        n_deltas = len(base.deltas)
        if n_deltas == 0:
            base.physics_diagnostics["ensemble_std"] = 0.0
            base.physics_diagnostics["ensemble_n_passes"] = n_passes
            return base

        # Stack each delta channel across the ensemble
        aggregated_deltas: list[CalibrationDelta] = []
        per_channel_stds: list[float] = []
        for ch_idx in range(n_deltas):
            stacked = np.stack(
                [np.asarray(p.deltas[ch_idx].delta, dtype=float) for p in predictions if ch_idx < len(p.deltas)],
                axis=0,
            )
            mean_delta = stacked.mean(axis=0)
            std_delta = stacked.std(axis=0)
            template = base.deltas[ch_idx]
            aggregated_deltas.append(CalibrationDelta(
                target_channel=template.target_channel,
                matrix_type=template.matrix_type,
                delta=mean_delta,
                limit=template.limit,
                accepted=template.accepted,
                reason=template.reason,
                path=template.path,
            ))
            per_channel_stds.append(float(np.mean(np.abs(std_delta))))

        ensemble_std = float(np.mean(per_channel_stds)) if per_channel_stds else 0.0

        result = FeedbackPrediction(
            deltas=aggregated_deltas,
            safe_deltas=aggregated_deltas,
            confidence=base.confidence,
            heuristic_gain_proxy=base.heuristic_gain_proxy,
            prediction_source=f"{base.prediction_source}+ensemble{n_passes}",
            physics_diagnostics={
                **(base.physics_diagnostics or {}),
                "ensemble_std": ensemble_std,
                "ensemble_n_passes": n_passes,
                "ensemble_per_channel_std": per_channel_stds,
            },
            guard_result=base.guard_result,
        )
        return result

    def _load_feedback_model(self) -> TrainedFeedbackModel:
        if self.feedback_model is not None:
            return self.feedback_model
        model = TrainedFeedbackModel(self.config.system_type)
        model_path = resolve_feedback_model_path(self.config.system_type, self.config.model_path)
        if model_path.exists():
            model.load(str(model_path))
        self.feedback_model = model
        return model

    def _load_classifier(self):
        """Lazily load the pretrained state-family classifier.

        Returns the wrapper or None if loading fails. In strict_hardware_mode
        a load failure raises; otherwise it is recorded and we fall back to
        running without classifier diagnostics (matching pre-F.3 behaviour).
        """
        if self._classifier is not None:
            return self._classifier
        if self._classifier_load_attempted:
            return None
        self._classifier_load_attempted = True
        path = resolve_classifier_model_path(
            self.config.system_type, self.config.classifier_model_path
        )
        v4_collection_campaign = bool(
            self.config.workflow_mode == "collect_dataset"
            and self.config.system_type == "4qubit"
            and (self.config.campaign_id or self.config.campaign_run_dir)
        )
        classifier_required = (
            self.config.strict_hardware_mode and not v4_collection_campaign
        )
        if not path.exists():
            if classifier_required:
                raise RuntimeError(
                    f"strict_hardware_mode requires a classifier model at {path}, but it is missing"
                )
            return None
        try:
            from ml_recon.quantum_representation.state_family import (
                QuantumStateFamilyTorchClassifier,
            )

            self._classifier = QuantumStateFamilyTorchClassifier.load(str(path))
        except Exception as exc:  # noqa: BLE001
            if classifier_required:
                raise
            # Soft failure: record the reason and disable classifier for the rest of the run.
            self._classifier = None
            self._classifier_error = f"{type(exc).__name__}: {exc}"
        return self._classifier

    def _probe_store(self) -> Optional[EphemeralPhaseProbeStore]:
        if self.config.system_type != "4qubit" or not self.config.enable_phase_probes:
            return None
        if self.phase_probe_store is None:
            if self.config.campaign_run_dir:
                root = Path(self.config.campaign_run_dir)
            elif self.config.output_dir:
                root = Path(self.config.output_dir) / ".v4_campaign"
            else:
                root = Path(__file__).resolve().parents[2] / "data" / "v4_probe_lifecycle"
            root.mkdir(parents=True, exist_ok=True)
            self.phase_probe_store = EphemeralPhaseProbeStore(root)
        return self.phase_probe_store

    def _w_ch1_base_matrix_hash(self) -> str:
        payload = {}
        for cycle in range(1, 5):
            path = self.sequence_service.matrix_dir / f"W_CH1_phase_{cycle}_4qubit.txt"
            if not path.exists():
                payload[path.name] = None
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            payload[path.name] = digest
        return stable_json_hash(payload)

    @staticmethod
    def _waveform_directory_hash(output_dir: str | Path) -> str:
        root = Path(output_dir)
        payload = []
        if root.exists():
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                payload.append(
                    {
                        "path": str(path.relative_to(root)).replace("\\", "/"),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        return stable_json_hash(payload)

    def _guard_prediction_with_steps(self, prediction: FeedbackPrediction, snapshot: MeasurementSnapshot) -> FeedbackPrediction:
        if _is_heuristic_prediction_source(prediction.prediction_source) and not self.config.allow_heuristic_deltas:
            prediction.guard_result = {
                "accepted": False,
                "reason": (
                    f"{prediction.prediction_source} predictions disabled "
                    "(allow_heuristic_deltas=False)"
                ),
            }
            prediction.deltas = [self._reject_delta(delta, prediction.guard_result["reason"]) for delta in prediction.deltas]
            return prediction

        if prediction.confidence < self.config.model_confidence_threshold:
            prediction.guard_result = {
                "accepted": False,
                "reason": (
                    f"confidence {prediction.confidence:.3f} below "
                    f"{self.config.model_confidence_threshold:.3f}"
                ),
            }
            prediction.deltas = [self._reject_delta(delta, prediction.guard_result["reason"]) for delta in prediction.deltas]
            return prediction

        gain_proxy = float(prediction.heuristic_gain_proxy or 0.0)
        min_gain_proxy = float(self.config.min_heuristic_gain_proxy or 0.0)
        if gain_proxy < min_gain_proxy:
            prediction.guard_result = {
                "accepted": False,
                "reason": (
                    f"heuristic gain proxy {gain_proxy:.4f} below "
                    f"{min_gain_proxy:.4f}"
                ),
            }
            prediction.deltas = [self._reject_delta(delta, prediction.guard_result["reason"]) for delta in prediction.deltas]
            return prediction

        reasons = []
        for step in self.config.delta_step_schedule:
            scaled = [self._scale_delta(delta, step) for delta in prediction.deltas]
            guarded = self.delta_manager.guard_amp_deltas(
                self.config.system_type,
                scaled,
                snapshot.retrieval_rate,
                self.config.efficiency_drop_threshold,
            )
            if any(delta.accepted for delta in guarded):
                prediction.deltas = guarded
                prediction.guard_result = {"accepted": True, "step": float(step), "reason": "accepted"}
                return prediction
            reasons.extend(delta.reason for delta in guarded if delta.reason)

        reason = "; ".join(reasons) or "all guard step attempts rejected"
        prediction.guard_result = {"accepted": False, "reason": reason}
        prediction.deltas = [self._reject_delta(delta, reason) for delta in prediction.deltas]
        self.state = ExperimentState.PAUSED
        return prediction

    def _scale_delta(self, delta: CalibrationDelta, step: float) -> CalibrationDelta:
        return CalibrationDelta(
            target_channel=delta.target_channel,
            matrix_type=delta.matrix_type,
            delta=np.asarray(delta.delta, dtype=float) * float(step),
            limit=delta.limit,
            accepted=delta.accepted,
            reason=delta.reason,
            path=delta.path,
        )

    def _reject_delta(self, delta: CalibrationDelta, reason: str) -> CalibrationDelta:
        return CalibrationDelta(
            target_channel=delta.target_channel,
            matrix_type=delta.matrix_type,
            delta=np.zeros_like(delta.delta),
            limit=delta.limit,
            accepted=False,
            reason=reason,
            path=delta.path,
        )

    def _predict_deltas(self, snapshot: MeasurementSnapshot) -> list[CalibrationDelta]:
        state_params = self.current_state_params()
        features = build_default_features(self.config.system_type, state_params)
        prediction = self.coupling_model.predict(features)
        spec = SYSTEM_SPECS.get(self.config.system_type, SYSTEM_SPECS["3qubit"])
        spec_n = spec["freq_steps"]
        group_n = spec["groups_per_row"]

        source = prediction.get("prediction_source", "unknown")
        if _is_heuristic_prediction_source(source):
            import warnings
            warnings.warn(
                "AOD coupling model is using heuristic fallback predictions, not trained neural model. "
                "Delta corrections may be inaccurate. Consider training the model or disabling auto-apply.",
                UserWarning,
                stacklevel=2,
            )

        def square_from_vector(values) -> np.ndarray:
            arr = np.zeros((spec_n, spec_n), dtype=float)
            values = np.asarray(values, dtype=float)
            arr[:values.size, :values.size] = np.diag(values)
            return arr

        common_w = np.full((spec_n, spec_n), float(prediction["W_CH1_global_phase"]), dtype=float)
        common_r = np.full((spec_n, spec_n), float(prediction["R_CH1_global_phase"]), dtype=float)

        return [
            CalibrationDelta("W_CH3", "phase", square_from_vector(prediction["W_CH3_phase"]), 0.08),
            CalibrationDelta("R_CH3", "phase", square_from_vector(prediction["R_CH3_phase"]), 0.08),
            CalibrationDelta("W_CH3", "amp", square_from_vector(prediction["W_CH3_amp"]), 0.03),
            CalibrationDelta("R_CH3", "amp", square_from_vector(prediction["R_CH3_amp"]), 0.03),
            CalibrationDelta("W_CH1", "phase", common_w, 0.20),
            CalibrationDelta("R_CH1", "phase", common_r, 0.20),
        ]

    def _reconstruct(self, snapshot: MeasurementSnapshot) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        recon_dir = repo_root / "system_control" / "qubit_recon_files"
        if str(recon_dir) not in sys.path:
            sys.path.insert(0, str(recon_dir))
        import calibrated_density_reconstruction

        basis_name = "basis_config_3qubit.json" if snapshot.system_type == "3qubit" else "basis_config_4qubit.json"
        basis_path = recon_dir / "matrix" / basis_name
        efficiency_map_path = (
            Path(self.config.retrieval_map_path)
            if self.config.retrieval_map_path
            else calibrated_density_reconstruction.default_efficiency_map_path(snapshot.system_type)
        )
        # Development/check-code trees do not always contain the optional CSV
        # source-characterization export.  Retain the historical tracked text
        # map as an explicit compatibility fallback rather than failing the
        # entire acquisition window.
        if not efficiency_map_path.exists() and self.config.retrieval_map_path is None:
            efficiency_map_path = calibrated_density_reconstruction.legacy_efficiency_map_path(snapshot.system_type)
        try:
            result = calibrated_density_reconstruction.reconstruct_density_matrix(
                snapshot.system_type,
                np.maximum(snapshot.segment_values, 0.0),
                str(basis_path),
                efficiency_map_path=str(efficiency_map_path),
            )
            snapshot.density_matrix = result["rho_full"]
            snapshot.purity = float(np.real(np.trace(snapshot.density_matrix @ snapshot.density_matrix)))
            snapshot.diagnostics.update(result.get("diagnostics", {}))
            target_rho = self._target_density_matrix(self.current_state_params(), snapshot.system_type)
            snapshot.fidelity = self._density_fidelity(target_rho, snapshot.density_matrix)
        except Exception as exc:
            snapshot.diagnostics["reconstruction_error"] = str(exc)
            return

        # Build the operation registry and partial calibration snapshot beside
        # the legacy reconstruction.  This does not touch timing, sequences, or
        # ml_delta files.  A missing calibration remains explicit.
        registry = build_operation_registry(snapshot.system_type, matrix_dir=self.sequence_service.matrix_dir)
        snapshot.operation_point_registry = [point.to_json_dict() for point in registry]
        snapshot.operation_point_registry_hash = operation_registry_hash(registry)
        snapshot.calibration_snapshot = self.operation_calibration.to_json_dict()
        calibration_payload = json.dumps(snapshot.calibration_snapshot, sort_keys=True, separators=(",", ":"))
        snapshot.calibration_snapshot_hash = hashlib.sha256(calibration_payload.encode("utf-8")).hexdigest()
        snapshot.phase_sign = -1.0
        snapshot.basis_phase_convention = "read_aod_negative_order"
        snapshot.feature_schema_version = 2
        if snapshot.signed_segment_values is None:
            snapshot.signed_segment_values = np.asarray(snapshot.segment_values, dtype=float).copy()
        if snapshot.legacy_clipped_segment_values is None:
            snapshot.legacy_clipped_segment_values = np.maximum(snapshot.signed_segment_values, 0.0)
        snapshot.diagnostics["noise_context"] = build_noise_context(
            snapshot,
            pair_model_mode=self.config.pair_model_mode,
            external_dlcz_gsi_reference=self.config.external_dlcz_gsi_reference,
        ).to_json_dict()

        if self.config.noise_aware_mode != "disabled":
            try:
                from system_control.qubit_recon_files.noise_aware_reconstruction import reconstruct_noise_aware
                shadow_result = reconstruct_noise_aware(
                    snapshot.system_type,
                    signal_bins=snapshot.signal_bins,
                    idler_bins=snapshot.idler_bins,
                    coin_bins=snapshot.coin_bins,
                    read_count=snapshot.read_count,
                    registry=registry,
                    calibration=self.operation_calibration,
                    noise_sampling_mode=self.config.noise_sampling_mode,
                    projection_mode="hardware_response",
                    mc_samples=self.config.mc_samples_realtime,
                )
                snapshot.diagnostics["noise_aware_reconstruction"] = _json_ready(shadow_result)
                snapshot.diagnostics["noise_aware_mode"] = self.config.noise_aware_mode
            except Exception as exc:  # shadow diagnostics must not stop legacy acquisition
                snapshot.diagnostics["noise_aware_reconstruction_error"] = f"{type(exc).__name__}: {exc}"

        # F.3: inject classifier diagnostics so the feedback model's
        # `classifier_enhanced` path can fire on real hardware. If the
        # classifier model is missing or fails, we degrade gracefully unless
        # strict_hardware_mode demands otherwise.
        if snapshot.density_matrix is not None:
            classifier = self._load_classifier()
            if classifier is not None:
                try:
                    diag = classifier.predict(snapshot.density_matrix)
                    snapshot.diagnostics["classifier_diagnostics"] = diag
                except Exception as exc:  # noqa: BLE001
                    if self.config.strict_hardware_mode and not (
                        self.config.workflow_mode == "collect_dataset"
                        and self.config.system_type == "4qubit"
                        and (self.config.campaign_id or self.config.campaign_run_dir)
                    ):
                        raise
                    snapshot.diagnostics["classifier_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            elif getattr(self, "_classifier_error", None):
                snapshot.diagnostics["classifier_error"] = self._classifier_error

    def _target_density_matrix(self, state_params: Dict, system_type: str) -> np.ndarray:
        n = 3 if system_type == "3qubit" else 4
        dim = 2 ** n
        psi = np.zeros(dim, dtype=complex)
        amps = np.asarray(state_params.get("amplitudes", [1.0] * n), dtype=float)
        if "phases_rad" in state_params:
            phases = np.asarray(state_params["phases_rad"], dtype=float)
        else:
            phases = np.deg2rad(np.asarray(state_params.get("phases_deg", [0.0] * n), dtype=float))
        amps = np.pad(amps, (0, max(0, n - amps.size)), constant_values=1.0)[:n]
        phases = np.pad(phases, (0, max(0, n - phases.size)), constant_values=0.0)[:n]
        norm = np.linalg.norm(amps)
        if norm <= 0:
            amps = np.ones(n, dtype=float)
            norm = np.linalg.norm(amps)
        for i in range(n):
            idx = 2 ** (n - 1 - i)
            psi[idx] = (amps[i] / norm) * np.exp(1j * phases[i])
        return np.outer(psi, np.conjugate(psi))

    def _density_fidelity(self, rho_target: np.ndarray, rho_measured: np.ndarray) -> float:
        try:
            eigvals, eigvecs = np.linalg.eigh(rho_target)
            eigvals = np.clip(np.real(eigvals), 0.0, None)
            sqrt_target = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
            middle = sqrt_target @ rho_measured @ sqrt_target
            vals = np.linalg.eigvalsh(0.5 * (middle + middle.conj().T))
            return float(np.real(np.sum(np.sqrt(np.clip(vals, 0.0, None))) ** 2))
        except Exception:
            return float(np.real(np.trace(rho_target @ rho_measured)))

    def _load_basis_config(self) -> Dict:
        basis_name = "basis_config_3qubit.json" if self.config.system_type == "3qubit" else "basis_config_4qubit.json"
        path = Path(self.sequence_service.matrix_dir) / basis_name
        if not path.exists():
            return {"path": str(path), "missing": True}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


def _is_heuristic_prediction_source(source: Optional[str]) -> bool:
    return "heuristic" in str(source or "").lower()


def _json_ready(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": np.real(value).tolist(), "imag": np.imag(value).tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
