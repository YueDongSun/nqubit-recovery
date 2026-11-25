"""Callable sequence generation service for 3-qubit and 4-qubit experiments."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from system_control.qubit_recon_files.digital_delay_defaults import (
    DEFAULT_DIGITAL_DELAY_OFFSET_NS,
    digital_delay_values,
)

from .data_protocol import stable_json_hash
from .feedback_action_materializer import (
    ActionMaterializationResult,
    FeedbackActionMaterializer,
)
from .models import ControlStateSpecV1, TargetStateSpecV1


@dataclass(frozen=True)
class PreparedSequenceArtifactV1:
    """Immutable, pre-upload binding of one candidate command and waveform."""

    artifact_id: str
    system_type: str
    target_hash: str
    control_hash: str
    parent_control_hash: str
    requested_action_hash: str
    projected_action_hash: str
    actual_action_hash: str
    matrix_workspace: str
    sequence_dir: str
    matrix_hash: str
    matrix_files: Mapping[str, str]
    registry_hash: str
    command_table_id: str
    command_table_hash: str
    command_table_directory: str
    waveform_hash: str
    waveform_files: Mapping[str, str]
    phase_cal_manifest_hash: str
    phase_cal_manifest_path: str
    fourier_report: Mapping[str, Any]
    materialization: Mapping[str, Any]
    upload_receipt: Optional[Mapping[str, Any]] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": "PreparedSequenceArtifactV1",
            "artifact_id": self.artifact_id,
            "system_type": self.system_type,
            "target_hash": self.target_hash,
            "control_hash": self.control_hash,
            "parent_control_hash": self.parent_control_hash,
            "requested_action_hash": self.requested_action_hash,
            "projected_action_hash": self.projected_action_hash,
            "actual_action_hash": self.actual_action_hash,
            "matrix_workspace": self.matrix_workspace,
            "sequence_dir": self.sequence_dir,
            "matrix_hash": self.matrix_hash,
            "matrix_files": dict(self.matrix_files),
            "registry_hash": self.registry_hash,
            "command_table_id": self.command_table_id,
            "command_table_hash": self.command_table_hash,
            "command_table_directory": self.command_table_directory,
            "waveform_hash": self.waveform_hash,
            "waveform_files": dict(self.waveform_files),
            "phase_cal_manifest_hash": self.phase_cal_manifest_hash,
            "phase_cal_manifest_path": self.phase_cal_manifest_path,
            "fourier_report": dict(self.fourier_report),
            "materialization": dict(self.materialization),
            "upload_receipt": None if self.upload_receipt is None else dict(self.upload_receipt),
        }


class SequenceService:
    """Wrap existing phase-calculation and sequence-generation scripts."""

    def __init__(
        self,
        qubit_recon_dir: str | None = None,
        python_executable: str | None = None,
        *,
        matrix_dir: str | Path | None = None,
        include_ml_delta: bool = True,
    ):
        if qubit_recon_dir is None:
            qubit_recon_dir = str(Path(__file__).resolve().parents[1] / "qubit_recon_files")
        self.qubit_recon_dir = Path(qubit_recon_dir).resolve()
        self.matrix_dir = (
            Path(matrix_dir).resolve()
            if matrix_dir is not None
            else (self.qubit_recon_dir / "matrix").resolve()
        )
        self.include_ml_delta = bool(include_ml_delta)
        self.python_executable = python_executable or sys.executable
        self.last_4qubit_base_matrix_hash: str | None = None
        self.last_actual_control_state: ControlStateSpecV1 | None = None
        self.last_generation_report: Dict[str, Any] = {}
        self.last_generation_state: Dict[str, Any] = {}

    def generate(
        self,
        system_type: str,
        state: Dict[str, Any] | TargetStateSpecV1,
        excitation_mode: int,
        output_dir: str | None = None,
        digital_delay_offset_ns: int = DEFAULT_DIGITAL_DELAY_OFFSET_NS,
        phase_probe_overlay: Any | None = None,
        control_state: ControlStateSpecV1 | None = None,
    ) -> str:
        if isinstance(state, TargetStateSpecV1):
            target_spec = state
            control_spec = control_state or ControlStateSpecV1.from_target(target_spec)
            state_payload: Dict[str, Any] = {
                "target_state_spec": target_spec,
                "control_state_spec": control_spec,
            }
        else:
            state_payload = state
            target_spec, control_spec = self._state_specs(
                system_type, state_payload, excitation_mode, control_state=control_state,
            )
        state_payload.update({
            "desired_amplitudes": list(target_spec.desired_amplitudes),
            "desired_phases_rad": list(target_spec.desired_phases_rad),
            "commanded_amplitudes": list(control_spec.commanded_amplitudes),
            "commanded_phases_rad": list(control_spec.commanded_phases_rad),
            "target_hash": target_spec.target_hash,
            "control_hash": control_spec.control_hash,
        })
        if system_type == "3qubit":
            if phase_probe_overlay is not None:
                raise ValueError("phase_probe_overlay is currently 4qubit-only")
            generated = self._generate_3qubit(
                state_payload, excitation_mode, output_dir, digital_delay_offset_ns,
            )
        elif system_type == "4qubit":
            if phase_probe_overlay is not None and state_payload.get("state_role") != "anchor":
                raise ValueError("phase probes are forbidden on 4Q coverage states")
            generated = self._generate_4qubit(
                state_payload,
                excitation_mode,
                output_dir,
                digital_delay_offset_ns,
                phase_probe_overlay,
            )
        else:
            raise ValueError("system_type must be '3qubit' or '4qubit'")
        self.last_generation_state = dict(state_payload)
        return generated

    def prepare_candidate(
        self,
        target_state: TargetStateSpecV1,
        control_state: ControlStateSpecV1,
        action: Mapping[str, Any] | ActionMaterializationResult,
        workspace: str | Path,
        *,
        excitation_mode: int = 1,
        digital_delay_offset_ns: int = DEFAULT_DIGITAL_DELAY_OFFSET_NS,
        materializer: FeedbackActionMaterializer | None = None,
    ) -> PreparedSequenceArtifactV1:
        """Prepare one isolated candidate and bind every pre-upload command fact.

        The method never writes the shared matrix directory and never uploads a
        waveform.  A non-empty workspace is rejected so a prepared candidate
        cannot be mutated or mixed with a previous cycle.
        """
        if target_state.system_type != control_state.system_type:
            raise ValueError("target/control system_type mismatch")
        workspace_path = Path(workspace).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        if any(workspace_path.iterdir()):
            raise FileExistsError("candidate workspace must be empty and immutable")
        matrix_workspace = workspace_path / "matrices"
        sequence_dir = workspace_path / "sequence"
        command_table_root = workspace_path / "command_table"
        matrix_workspace.mkdir()
        sequence_dir.mkdir()
        command_table_root.mkdir()
        self._copy_candidate_matrix_inputs(matrix_workspace)

        action_materializer = materializer or FeedbackActionMaterializer()
        if isinstance(action, ActionMaterializationResult):
            materialization = action
            if materialization.parent_control_state.control_hash != control_state.control_hash:
                raise ValueError("materialization parent control does not match candidate parent")
        else:
            materialization = action_materializer.materialize(
                target_state, control_state, action,
            )
        if not materialization.accepted:
            raise ValueError(
                "candidate action rejected before sequence generation: "
                + ", ".join(materialization.reasons)
            )

        candidate_service = SequenceService(
            str(self.qubit_recon_dir),
            self.python_executable,
            matrix_dir=matrix_workspace,
            include_ml_delta=False,
        )
        candidate_service.generate(
            target_state.system_type,
            target_state,
            excitation_mode,
            output_dir=str(sequence_dir),
            digital_delay_offset_ns=digital_delay_offset_ns,
            control_state=materialization.projected_control_state,
        )
        actual_control = candidate_service.last_actual_control_state
        if actual_control is None:
            raise RuntimeError("candidate generation did not report the actual control state")
        generation = dict(candidate_service.last_generation_report)
        if not generation:
            raise RuntimeError("candidate generation did not produce a generation manifest")

        from ml_recon.physics_hybrid_v5.command_contract import build_command_table

        from .operation_points import build_operation_registry, operation_registry_hash

        matrix_hash = str(generation["matrix_bundle_hash"])
        waveform_hash = str(generation["waveform_hash"])
        registry = build_operation_registry(
            target_state.system_type,
            matrix_dir=matrix_workspace,
            include_ml_delta=False,
        )
        registry_hash = operation_registry_hash(registry)
        self._atomic_write_json(
            command_table_root / "operation_registry.json",
            {
                "schema": "OperationRegistrySnapshotV1",
                "system_type": target_state.system_type,
                "registry_hash": registry_hash,
                "operations": [item.to_json_dict() for item in registry],
            },
        )
        calibration_name = (
            "phase_Cali.txt" if target_state.system_type == "3qubit" else "phase_Cali_4qubit.txt"
        )
        calibration_path = self.qubit_recon_dir / calibration_name
        calibration_hash = self._sha256_file(calibration_path)
        command_table = build_command_table(
            registry,
            matrix_snapshot_hash=matrix_hash,
            calibration_snapshot_hash=calibration_hash,
            awg_state_hash=stable_json_hash({
                "state": "prepared_not_uploaded",
                "waveform_hash": waveform_hash,
            }),
        )
        command_table_manifest_path = command_table.save(command_table_root)
        command_table_directory = command_table_manifest_path.parent
        waveform_diagnostics = dict(generation.get("waveform_diagnostics") or {})
        waveform_diagnostics.update(
            self._command_power_diagnostics(command_table, matrix_workspace)
        )
        materialization = action_materializer.bind_actual_loaded(
            materialization,
            target_state,
            actual_control,
            waveform_diagnostics=waveform_diagnostics,
            fourier_report=dict(generation.get("fourier_report") or {}),
        )

        identity = {
            "schema": "PreparedSequenceArtifactV1",
            "system_type": target_state.system_type,
            "target_hash": target_state.target_hash,
            "control_hash": actual_control.control_hash,
            "parent_control_hash": control_state.control_hash,
            "requested_action_hash": materialization.requested_action_hash,
            "projected_action_hash": materialization.projected_action_hash,
            "actual_action_hash": materialization.actual_action_hash,
            "matrix_hash": matrix_hash,
            "registry_hash": registry_hash,
            "command_table_hash": command_table.table_hash,
            "waveform_hash": waveform_hash,
            "phase_cal_manifest_hash": generation["phase_cal_manifest_hash"],
        }
        artifact = PreparedSequenceArtifactV1(
            artifact_id=stable_json_hash(identity),
            system_type=target_state.system_type,
            target_hash=target_state.target_hash,
            control_hash=actual_control.control_hash,
            parent_control_hash=control_state.control_hash,
            requested_action_hash=materialization.requested_action_hash,
            projected_action_hash=materialization.projected_action_hash,
            actual_action_hash=materialization.actual_action_hash,
            matrix_workspace=str(matrix_workspace.resolve()),
            sequence_dir=str(sequence_dir.resolve()),
            matrix_hash=matrix_hash,
            matrix_files=dict(generation["matrix_files"]),
            registry_hash=registry_hash,
            command_table_id=command_table.table_id,
            command_table_hash=command_table.table_hash,
            command_table_directory=str(command_table_directory.resolve()),
            waveform_hash=waveform_hash,
            waveform_files=dict(generation["waveform_files"]),
            phase_cal_manifest_hash=str(generation["phase_cal_manifest_hash"]),
            phase_cal_manifest_path=str(Path(generation["phase_cal_manifest_path"]).resolve()),
            fourier_report=dict(generation.get("fourier_report") or {}),
            materialization=materialization.to_json_dict(),
        )
        self._atomic_write_json(
            workspace_path / "prepared_sequence_artifact.json", artifact.to_json_dict(),
        )
        return artifact

    def _generate_3qubit(
        self,
        state: Dict[str, Any],
        excitation_mode: int,
        output_dir: str | None,
        digital_delay_offset_ns: int,
    ) -> str:
        output_dir = output_dir or str(self.qubit_recon_dir.parent.parent / "AWG_Sequence" / "entangle_sequences")
        target_spec, control_spec = self._state_specs("3qubit", state, excitation_mode)
        target_amps = list(target_spec.desired_amplitudes)
        target_phases = list(target_spec.desired_phases_rad)
        amps = list(control_spec.commanded_amplitudes)
        phases = list(control_spec.commanded_phases_rad)

        cmd = [
            self.python_executable, str(self.qubit_recon_dir / "phase_cal.py"),
            "--w_ch3_amp", str(self.matrix_dir / "W_CH3_amp.txt"),
            "--w_ch3_phase", str(self.matrix_dir / "W_CH3_phase.txt"),
            "--w_ch4_amp", str(self.matrix_dir / "W_CH4_amp.txt"),
            "--w_ch4_phase", str(self.matrix_dir / "W_CH4_phase.txt"),
            "--r_ch3_amp", str(self.matrix_dir / "R_CH3_amp.txt"),
            "--r_ch3_phase", str(self.matrix_dir / "R_CH3_phase.txt"),
            "--r_ch4_amp", str(self.matrix_dir / "R_CH4_amp.txt"),
            "--r_ch4_phase", str(self.matrix_dir / "R_CH4_phase.txt"),
            "--phase_cali", str(self.qubit_recon_dir / "phase_Cali.txt"),
            "--basis_config", str(self.matrix_dir / "basis_config_3qubit.json"),
            "--user_amp1", str(amps[0]),
            "--user_amp2", str(amps[1]),
            "--user_amp3", str(amps[2]),
            "--user_phase1", str(phases[0]),
            "--user_phase2", str(phases[1]),
            "--user_phase3", str(phases[2]),
            "--target_amp1", str(target_amps[0]),
            "--target_amp2", str(target_amps[1]),
            "--target_amp3", str(target_amps[2]),
            "--target_phase1", str(target_phases[0]),
            "--target_phase2", str(target_phases[1]),
            "--target_phase3", str(target_phases[2]),
            "--command_amp1", str(amps[0]),
            "--command_amp2", str(amps[1]),
            "--command_amp3", str(amps[2]),
            "--command_phase1", str(phases[0]),
            "--command_phase2", str(phases[1]),
            "--command_phase3", str(phases[2]),
            "--target_hash", target_spec.target_hash,
            "--control_hash", control_spec.control_hash,
            "--excitation_mode", str(excitation_mode),
            "--w_ch1_global_amp", str(self.matrix_dir / "W_CH1_global_amp.txt"),
            "--output_dir", str(self.matrix_dir),
        ]
        self._run(cmd)

        self._import_local_modules()
        from entangle_sequence import EntangleConfig, generate_entangle_sequence

        config = EntangleConfig(
            matrix_dir=str(self.matrix_dir), include_ml_delta=self.include_ml_delta,
        )
        config.output_dir = output_dir
        config.w_ch1_excitation_mode = excitation_mode
        self._apply_digital_delay_offset(config, digital_delay_offset_ns)
        channels, config = generate_entangle_sequence(
            config, return_channels=True, make_preview=False,
        )
        fourier_report = self._fourier_report(config)
        waveform_diagnostics = self._waveform_diagnostics(channels)
        state.update({
            "requested_amplitudes": list(amps),
            "actual_loaded_amplitudes": list(amps),
            "requested_phases_rad": list(phases),
            "actual_loaded_phases_rad": list(phases),
            "fourier_guard_phase_adjustment_rad": [0.0] * 3,
            "fourier_guard_passed": bool(fourier_report.get("passed", False)),
            "fourier_guard_report": fourier_report,
        })
        self.last_actual_control_state = control_spec
        self.last_generation_report = self._finalize_generation_manifest(
            "3qubit",
            target_spec,
            control_spec,
            output_dir,
            requested_amplitudes=amps,
            actual_amplitudes=amps,
            requested_phases=phases,
            actual_phases=phases,
            fourier_report=fourier_report,
            waveform_diagnostics=waveform_diagnostics,
        )
        return output_dir

    def _generate_4qubit(
        self,
        state: Dict[str, Any],
        excitation_mode: int,
        output_dir: str | None,
        digital_delay_offset_ns: int,
        phase_probe_overlay: Any | None = None,
    ) -> str:
        self.last_4qubit_base_matrix_hash = None
        output_dir = output_dir or str(self.qubit_recon_dir.parent.parent / "AWG_Sequence" / "entangle_sequences_4qubit")
        target_spec, control_spec = self._state_specs("4qubit", state, excitation_mode)
        self._active_target_state = target_spec
        self._active_control_state = control_spec
        amps = list(control_spec.commanded_amplitudes)
        phases = list(control_spec.commanded_phases_rad)
        state.update({
            "target_hash": target_spec.target_hash,
            "control_hash": control_spec.control_hash,
        })

        requested_phases = list(phases)
        attempts = self._fourier_phase_candidates(requested_phases)
        last_config = None
        for attempt_index, candidate_phases in enumerate(attempts, start=1):
            last_config = self._generate_4qubit_candidate(
                amps,
                candidate_phases,
                excitation_mode,
                output_dir,
                digital_delay_offset_ns,
                phase_probe_overlay,
            )
            if self._fourier_guard_passed(last_config):
                requested_deg = [math.degrees(value) for value in requested_phases]
                actual_deg = [math.degrees(value) for value in candidate_phases]
                state.update({
                    "requested_phases_deg": requested_deg,
                    "actual_loaded_phases_deg": actual_deg,
                    "actual_loaded_phases_rad": list(candidate_phases),
                    "fourier_guard_phase_adjustment_deg": [
                        actual - requested for actual, requested in zip(actual_deg, requested_deg)
                    ],
                    "fourier_guard_attempt_count": attempt_index,
                    "fourier_guard_passed": True,
                    "fourier_guard_report": last_config.fourier_guard_report.to_dict(),
                    "base_w_ch1_matrix_hash": last_config.base_w_ch1_matrix_hash,
                    "control_hash": getattr(self, "_last_candidate_control_hash", control_spec.control_hash),
                })
                self.last_4qubit_base_matrix_hash = last_config.base_w_ch1_matrix_hash
                actual_control = getattr(self, "_last_candidate_control_state", control_spec)
                self.last_actual_control_state = actual_control
                if self._phase_manifest_path("4qubit").is_file():
                    self.last_generation_report = self._finalize_generation_manifest(
                        "4qubit",
                        target_spec,
                        actual_control,
                        output_dir,
                        requested_amplitudes=amps,
                        actual_amplitudes=amps,
                        requested_phases=requested_phases,
                        actual_phases=candidate_phases,
                        fourier_report=self._fourier_report(last_config),
                        waveform_diagnostics=dict(
                            getattr(last_config, "waveform_diagnostics", {})
                        ),
                    )
                return output_dir

        # Restore the originally requested matrices/waveforms before refusing upload.
        restored = self._generate_4qubit_candidate(
            amps,
            requested_phases,
            excitation_mode,
            output_dir,
            digital_delay_offset_ns,
            phase_probe_overlay,
        )
        state.update({
            "requested_phases_deg": [math.degrees(value) for value in requested_phases],
            "actual_loaded_phases_deg": [math.degrees(value) for value in requested_phases],
            "actual_loaded_phases_rad": list(requested_phases),
            "fourier_guard_phase_adjustment_deg": [0.0] * 4,
            "fourier_guard_attempt_count": len(attempts),
            "fourier_guard_passed": False,
            "fourier_guard_report": restored.fourier_guard_report.to_dict(),
            "base_w_ch1_matrix_hash": restored.base_w_ch1_matrix_hash,
            "control_hash": getattr(self, "_last_candidate_control_hash", control_spec.control_hash),
        })
        self.last_4qubit_base_matrix_hash = restored.base_w_ch1_matrix_hash
        self.last_actual_control_state = getattr(self, "_last_candidate_control_state", control_spec)
        if self._phase_manifest_path("4qubit").is_file():
            self.last_generation_report = self._finalize_generation_manifest(
                "4qubit",
                target_spec,
                self.last_actual_control_state,
                output_dir,
                requested_amplitudes=amps,
                actual_amplitudes=amps,
                requested_phases=requested_phases,
                actual_phases=requested_phases,
                fourier_report=self._fourier_report(restored),
                waveform_diagnostics=dict(getattr(restored, "waveform_diagnostics", {})),
            )
        raise RuntimeError(
            f"4Q Fourier guard failed after {len(attempts)} bounded phase candidates; "
            "the requested phase was restored and AWG upload is blocked"
        )

    def _generate_4qubit_candidate(
        self,
        amps: list[float],
        phases: list[float],
        excitation_mode: int,
        output_dir: str,
        digital_delay_offset_ns: int,
        phase_probe_overlay: Any | None,
    ) -> Any:
        target_spec = getattr(self, "_active_target_state", None)
        target_amps = list(target_spec.desired_amplitudes) if target_spec is not None else list(amps)
        target_phases = list(target_spec.desired_phases_rad) if target_spec is not None else list(phases)
        base_control = getattr(self, "_active_control_state", None)
        if base_control is None:
            candidate_control = ControlStateSpecV1.from_state("4qubit", {
                "amplitudes": amps, "phases_rad": phases,
            })
        else:
            candidate_control = base_control.evolve(
                commanded_amplitudes=amps,
                commanded_phases_rad=phases,
                cumulative_amplitude_coordinates=(
                    np.asarray(base_control.cumulative_amplitude_coordinates, dtype=float)
                    + self._amplitude_coordinates(amps)
                    - self._amplitude_coordinates(base_control.commanded_amplitudes)
                ),
                cumulative_phase_coordinates=(
                    np.asarray(base_control.cumulative_phase_coordinates, dtype=float)
                    + self._phase_coordinates(phases)
                    - self._phase_coordinates(base_control.commanded_phases_rad)
                ),
            )
        self._last_candidate_control_hash = candidate_control.control_hash
        self._last_candidate_control_state = candidate_control
        cmd = [
            self.python_executable, str(self.qubit_recon_dir / "phase_cal_4qubit.py"),
            "--w_ch3_amp", str(self.matrix_dir / "W_CH3_amp_4qubit.txt"),
            "--w_ch3_phase", str(self.matrix_dir / "W_CH3_phase_4qubit.txt"),
            "--w_ch4_amp", str(self.matrix_dir / "W_CH4_amp_4qubit.txt"),
            "--w_ch4_phase", str(self.matrix_dir / "W_CH4_phase_4qubit.txt"),
            "--r_ch3_amp", str(self.matrix_dir / "R_CH3_amp_4qubit.txt"),
            "--r_ch3_phase", str(self.matrix_dir / "R_CH3_phase_4qubit.txt"),
            "--r_ch4_amp", str(self.matrix_dir / "R_CH4_amp_4qubit.txt"),
            "--r_ch4_phase", str(self.matrix_dir / "R_CH4_phase_4qubit.txt"),
            "--phase_cali", str(self.qubit_recon_dir / "phase_Cali_4qubit.txt"),
            "--basis_config", str(self.matrix_dir / "basis_config_4qubit.json"),
            "--user_amp1", str(amps[0]),
            "--user_amp2", str(amps[1]),
            "--user_amp3", str(amps[2]),
            "--user_amp4", str(amps[3]),
            "--user_phase1", str(phases[0]),
            "--user_phase2", str(phases[1]),
            "--user_phase3", str(phases[2]),
            "--user_phase4", str(phases[3]),
            "--target_amp1", str(target_amps[0]),
            "--target_amp2", str(target_amps[1]),
            "--target_amp3", str(target_amps[2]),
            "--target_amp4", str(target_amps[3]),
            "--target_phase1", str(target_phases[0]),
            "--target_phase2", str(target_phases[1]),
            "--target_phase3", str(target_phases[2]),
            "--target_phase4", str(target_phases[3]),
            "--command_amp1", str(amps[0]),
            "--command_amp2", str(amps[1]),
            "--command_amp3", str(amps[2]),
            "--command_amp4", str(amps[3]),
            "--command_phase1", str(phases[0]),
            "--command_phase2", str(phases[1]),
            "--command_phase3", str(phases[2]),
            "--command_phase4", str(phases[3]),
            "--target_hash", "" if target_spec is None else target_spec.target_hash,
            "--control_hash", candidate_control.control_hash,
            "--excitation_mode", str(excitation_mode),
            "--w_ch1_global_amp_1", str(self.matrix_dir / "W_CH1_global_amp_1_4qubit.txt"),
            "--w_ch1_global_amp_2", str(self.matrix_dir / "W_CH1_global_amp_2_4qubit.txt"),
            "--w_ch1_global_amp_3", str(self.matrix_dir / "W_CH1_global_amp_3_4qubit.txt"),
            "--w_ch1_global_amp_4", str(self.matrix_dir / "W_CH1_global_amp_4_4qubit.txt"),
            "--output_dir", str(self.matrix_dir),
        ]
        self._run(cmd)
        base_matrix_hash = self._validate_4qubit_phase_generation_manifest()

        self._import_local_modules()
        from entangle_sequence_4qubit import EntangleConfig4Qubit, generate_entangle_sequence_4qubit

        config = EntangleConfig4Qubit(
            matrix_dir=str(self.matrix_dir), include_ml_delta=self.include_ml_delta,
        )
        config.output_dir = output_dir
        config.w_ch1_excitation_mode = excitation_mode
        config.apply_phase_probe_overlay(phase_probe_overlay)
        self._apply_digital_delay_offset(config, digital_delay_offset_ns)
        channels, config = generate_entangle_sequence_4qubit(
            config, return_channels=True, make_preview=False,
        )
        config.waveform_diagnostics = self._waveform_diagnostics(channels)
        if self._validate_4qubit_phase_generation_manifest() != base_matrix_hash:
            raise RuntimeError(
                "W_CH1 base matrices changed after phase_cal_4qubit completed "
                "and before waveform rendering finished"
            )
        config.base_w_ch1_matrix_hash = base_matrix_hash
        return config

    def _validate_4qubit_phase_generation_manifest(self) -> str:
        manifest_path = self.matrix_dir / "phase_cal_4qubit_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                "phase_cal_4qubit did not produce phase_cal_4qubit_manifest.json"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "PhaseCalculation4QubitManifestV1":
            raise RuntimeError("unsupported phase_cal_4qubit generation manifest")
        recorded = dict(payload.get("w_ch1_phase_files") or {})
        current = {}
        for cycle_index in range(1, 5):
            path = self.matrix_dir / f"W_CH1_phase_{cycle_index}_4qubit.txt"
            if not path.is_file():
                raise RuntimeError(f"generated W_CH1 phase matrix is missing: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            current[path.name] = digest
        if current != recorded:
            raise RuntimeError(
                "generated W_CH1 phase matrices do not match phase_cal_4qubit manifest"
            )
        encoded = json.dumps(
            current,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        current_hash = hashlib.sha256(encoded).hexdigest()
        if current_hash != str(payload.get("w_ch1_base_matrix_hash") or ""):
            raise RuntimeError("phase_cal_4qubit W_CH1 base-matrix hash mismatch")
        return current_hash

    def _finalize_generation_manifest(
        self,
        system_type: str,
        target_state: TargetStateSpecV1,
        actual_control_state: ControlStateSpecV1,
        sequence_dir: str | Path,
        *,
        requested_amplitudes: list[float],
        actual_amplitudes: list[float],
        requested_phases: list[float],
        actual_phases: list[float],
        fourier_report: Mapping[str, Any],
        waveform_diagnostics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        manifest_path = self._phase_manifest_path(system_type)
        if not manifest_path.is_file():
            raise RuntimeError(f"phase calculation manifest is missing: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        matrix_files = self._hash_files(
            self.matrix_dir,
            lambda path: (
                path.suffix.lower() == ".txt"
                and path.name.startswith(("W_", "R_"))
                and ("_amp" in path.name or "_phase" in path.name)
            ),
        )
        waveform_files = self._hash_files(
            Path(sequence_dir), lambda path: path.suffix.lower() == ".txt",
        )
        if not matrix_files:
            raise RuntimeError("generation manifest cannot bind an empty matrix set")
        if not waveform_files:
            raise RuntimeError("generation manifest cannot bind an empty waveform set")
        matrix_bundle_hash = stable_json_hash(matrix_files)
        waveform_hash = stable_json_hash(waveform_files)
        phase_adjustment = [
            float(actual - requested)
            for actual, requested in zip(actual_phases, requested_phases)
        ]
        payload.update({
            "generation_manifest_version": 2,
            "target_hash": target_state.target_hash,
            "control_hash": actual_control_state.control_hash,
            "matrix_files": matrix_files,
            "all_file_sha256": {**matrix_files, **waveform_files},
            "matrix_bundle_hash": matrix_bundle_hash,
            "requested_amplitudes": [float(value) for value in requested_amplitudes],
            "actual_loaded_amplitudes": [float(value) for value in actual_amplitudes],
            "requested_phases_rad": [float(value) for value in requested_phases],
            "actual_loaded_phases_rad": [float(value) for value in actual_phases],
            "fourier_guard_phase_adjustment_rad": phase_adjustment,
            "fourier_report": dict(fourier_report),
            "waveform_files": waveform_files,
            "waveform_hash": waveform_hash,
            "waveform_diagnostics": dict(waveform_diagnostics),
        })
        self._atomic_write_json(manifest_path, payload)
        return {
            "phase_cal_manifest_path": str(manifest_path),
            "phase_cal_manifest_hash": self._sha256_file(manifest_path),
            "matrix_files": matrix_files,
            "matrix_bundle_hash": matrix_bundle_hash,
            "waveform_files": waveform_files,
            "waveform_hash": waveform_hash,
            "fourier_report": dict(fourier_report),
            "waveform_diagnostics": dict(waveform_diagnostics),
        }

    def _copy_candidate_matrix_inputs(self, destination: Path) -> None:
        if not self.matrix_dir.is_dir():
            raise FileNotFoundError(f"source matrix directory is missing: {self.matrix_dir}")
        for source in sorted(self.matrix_dir.iterdir(), key=lambda path: path.name):
            if not source.is_file():
                continue
            if source.suffix.lower() == ".txt" or source.name.startswith("basis_config_"):
                shutil.copy2(source, destination / source.name)

    @staticmethod
    def _command_power_diagnostics(command_table: Any, matrix_workspace: Path) -> Dict[str, Any]:
        per_channel = np.asarray(
            command_table.static_arrays["per_channel_rf_power"], dtype=float,
        )
        reduce_axes = tuple(range(per_channel.ndim - 1))
        maximum_per_channel = np.max(per_channel, axis=reduce_axes)
        total_power = float(np.max(np.asarray(
            command_table.static_arrays["total_rf_power"], dtype=float,
        )))
        matrix_min = float("inf")
        matrix_max = float("-inf")
        for path in matrix_workspace.glob("*.txt"):
            if not path.name.startswith(("W_", "R_")) or "_amp" not in path.name:
                continue
            values = np.loadtxt(path)
            matrix_min = min(matrix_min, float(np.min(values)))
            matrix_max = max(matrix_max, float(np.max(values)))
        if not np.isfinite(matrix_min) or not np.isfinite(matrix_max):
            raise RuntimeError("candidate matrix workspace contains no amplitude matrices")
        return {
            "per_channel_rf_power": maximum_per_channel.tolist(),
            "total_rf_power": total_power,
            "matrix_amplitude_min": matrix_min,
            "matrix_amplitude_max": matrix_max,
        }

    @staticmethod
    def _waveform_diagnostics(channels: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        channel_diagnostics: Dict[str, Any] = {}
        total_values = 0
        clipped_values = 0
        maximum_peak = 0.0
        maximum_crest = 0.0
        for channel, samples in channels.items():
            analog = np.asarray(samples, dtype=float)[:, 0]
            absolute = np.abs(analog)
            peak = float(np.max(absolute)) if absolute.size else 0.0
            active = absolute[absolute > 1e-12]
            rms = float(np.sqrt(np.mean(np.square(active)))) if active.size else 0.0
            crest = peak / rms if rms > 0.0 else 0.0
            channel_diagnostics[str(channel)] = {
                "peak": peak,
                "active_rms": rms,
                "crest_factor": crest,
            }
            maximum_peak = max(maximum_peak, peak)
            maximum_crest = max(maximum_crest, crest)
            total_values += int(absolute.size)
            clipped_values += int(np.count_nonzero(absolute > 1.0 + 1e-7))
        return {
            "waveform_peak": maximum_peak,
            "crest_factor": maximum_crest,
            "clipping_fraction": (
                float(clipped_values) / float(total_values) if total_values else 0.0
            ),
            "channels": channel_diagnostics,
        }

    @staticmethod
    def _fourier_report(config: Any) -> Dict[str, Any]:
        report = getattr(config, "fourier_guard_report", None)
        payload = {} if report is None else dict(report.to_dict())
        payload["passed"] = bool(
            report is not None
            and int(getattr(report, "checked_segments", 0)) > 0
            and not bool(getattr(report, "has_frequency_issues", True))
        )
        return payload

    def _phase_manifest_path(self, system_type: str) -> Path:
        name = "phase_cal_manifest.json" if system_type == "3qubit" else "phase_cal_4qubit_manifest.json"
        return self.matrix_dir / name

    @staticmethod
    def _hash_files(directory: Path, include: Any) -> Dict[str, str]:
        return {
            path.name: SequenceService._sha256_file(path)
            for path in sorted(directory.iterdir(), key=lambda item: item.name)
            if path.is_file() and include(path)
        }

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _amplitude_coordinates(amplitudes: Any) -> np.ndarray:
        from ml_recon.physics_hybrid_v5.projective_action import amplitude_to_coordinates

        return np.asarray(amplitude_to_coordinates(np.asarray(amplitudes, dtype=float)), dtype=float)

    @staticmethod
    def _phase_coordinates(phases: Any) -> np.ndarray:
        from ml_recon.physics_hybrid_v5.projective_action import phase_to_coordinates

        return np.asarray(phase_to_coordinates(np.asarray(phases, dtype=float)), dtype=float)

    @staticmethod
    def _fourier_guard_passed(config: Any) -> bool:
        report = getattr(config, "fourier_guard_report", None)
        return bool(
            report is not None
            and int(getattr(report, "checked_segments", 0)) > 0
            and not bool(getattr(report, "has_frequency_issues", True))
        )

    @staticmethod
    def _fourier_phase_candidates(requested_phases: list[float]) -> list[list[float]]:
        """Original target plus bounded relative-phase probes; Q1 is the gauge reference."""
        if len(requested_phases) != 4:
            raise ValueError("4Q phase rollback requires four phases")
        patterns = (
            (0.0, 1.0, -1.0, 1.0),
            (0.0, 1.0, 1.0, -1.0),
            (0.0, -1.0, 1.0, 1.0),
            (0.0, 1.0, -1.0, -1.0),
        )
        candidates = [list(requested_phases)]
        for magnitude_deg in (0.25, 0.5, 1.0, 2.0):
            magnitude = math.radians(magnitude_deg)
            for pattern in patterns:
                for sign in (1.0, -1.0):
                    candidates.append([
                        phase + sign * magnitude * direction
                        for phase, direction in zip(requested_phases, pattern)
                    ])
        return candidates

    def _run(self, cmd: list[str]) -> None:
        result = subprocess.run(cmd, cwd=str(self.qubit_recon_dir), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")

    def _import_local_modules(self) -> None:
        path = str(self.qubit_recon_dir)
        if path not in sys.path:
            sys.path.insert(0, path)

    def _apply_digital_delay_offset(self, config: Any, offset_ns: int) -> None:
        delays = digital_delay_values(offset_ns)
        config.digital_delay_offset_ns = int(offset_ns)
        config.w_ch1_digital_delay = delays["W_CH1"]
        config.r_ch1_digital_delay = delays["R_CH1"]
        config.channel_digital_delays.update(delays)
        if hasattr(config, "channel_digital2_delays"):
            config.channel_digital2_delays.update(delays)

    def _values(self, state: Dict[str, Any], key: str, length: int, default: float | None):
        values = state.get(key)
        if values is None:
            if default is None:
                return None
            values = [default] * length
        if len(values) != length:
            raise ValueError(f"{key} must contain {length} values")
        return [float(value) for value in values]

    @staticmethod
    def _state_specs(
        system_type: str,
        state: Dict[str, Any],
        excitation_mode: int,
        *,
        control_state: ControlStateSpecV1 | None = None,
    ) -> tuple[TargetStateSpecV1, ControlStateSpecV1]:
        target_value = state.get("target_state_spec")
        if isinstance(target_value, TargetStateSpecV1):
            target = target_value
        else:
            target_source = state.get("target_state")
            target = TargetStateSpecV1.from_state(
                system_type,
                target_source if isinstance(target_source, dict) else state,
                excitation_mode=excitation_mode,
            )
        control_value: Any = control_state or state.get("control_state_spec") or state.get("control_state")
        if isinstance(control_value, ControlStateSpecV1):
            control = control_value
        elif isinstance(control_value, dict):
            control = ControlStateSpecV1.from_state(system_type, control_value)
        else:
            control = ControlStateSpecV1.from_state(system_type, state)
        if target.system_type != system_type or control.system_type != system_type:
            raise ValueError("target/control system_type mismatch")
        return target, control
