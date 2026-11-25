"""Read-only engineering bridge from prepared commands to V6 proposals.

The bridge verifies every supplied artifact and model contract, performs frozen
inference plus same-context inverse optimization, and returns diagnostics only.
It has no matrix, bundle, AWG, FPGA, runner-state, approval, or promotion write
authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data_protocol import stable_json_hash
from .models import ControlStateSpecV1, MeasurementSnapshot, TargetStateSpecV1
from .sequence_service import PreparedSequenceArtifactV1


@dataclass(frozen=True)
class FeedbackActionProposalV2:
    target_hash: str
    parent_control_hash: str
    command_table_hash: str
    model_checkpoint_hash: str
    neural_warm_start: Tuple[float, ...]
    optimizer_action: Tuple[float, ...]
    selected_base_action: Tuple[float, ...]
    amplitude_coordinates: Tuple[float, ...]
    phase_coordinates: Tuple[float, ...]
    action_covariance: Tuple[Tuple[float, ...], ...]
    amplitude_actionable_mask: Tuple[bool, ...]
    phase_actionable_mask: Tuple[bool, ...]
    per_coordinate_actionable_mask: Tuple[bool, ...]
    block_fisher: Mapping[str, Any]
    block_conditions: Mapping[str, Any]
    posterior_std: Tuple[float, ...]
    candidate_scales: Tuple[float, ...]
    predicted_candidate_outcomes: Tuple[Mapping[str, Any], ...]
    ood_diagnostics: Mapping[str, Any]
    cycle_consistency: Mapping[str, Any]
    detector_source_diagnostics: Mapping[str, Any]
    checkpoint_metadata: Mapping[str, Any]
    same_context_integrability: Mapping[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        return _json_value({
            "schema": "FeedbackActionProposalV2",
            "target_hash": self.target_hash,
            "parent_control_hash": self.parent_control_hash,
            "command_table_hash": self.command_table_hash,
            "model_checkpoint_hash": self.model_checkpoint_hash,
            "neural_warm_start": self.neural_warm_start,
            "optimizer_action": self.optimizer_action,
            "selected_base_action": self.selected_base_action,
            "amplitude_coordinates": self.amplitude_coordinates,
            "phase_coordinates": self.phase_coordinates,
            "action_covariance": self.action_covariance,
            "amplitude_actionable_mask": self.amplitude_actionable_mask,
            "phase_actionable_mask": self.phase_actionable_mask,
            "per_coordinate_actionable_mask": self.per_coordinate_actionable_mask,
            "block_fisher": self.block_fisher,
            "block_conditions": self.block_conditions,
            "posterior_std": self.posterior_std,
            "candidate_scales": self.candidate_scales,
            "predicted_candidate_outcomes": self.predicted_candidate_outcomes,
            "ood_diagnostics": self.ood_diagnostics,
            "cycle_consistency": self.cycle_consistency,
            "detector_source_diagnostics": self.detector_source_diagnostics,
            "checkpoint_metadata": self.checkpoint_metadata,
            "same_context_integrability": self.same_context_integrability,
        })


class V6FeedbackBridge:
    """Load one strict V6 runtime and propose bounded global W actions."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        normalization_path: str | Path,
        *,
        config_path: str | Path | None = None,
        device: str = "cpu",
        inverse_optimizer_config: Any = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.normalization_path = Path(normalization_path).resolve()
        self.config_path = (
            Path(config_path).resolve()
            if config_path is not None
            else (
                Path(__file__).resolve().parents[2]
                / "ml_recon"
                / "configs"
                / "physics_hybrid_v4.json"
            )
        )
        self.device = str(device)
        self.inverse_optimizer_config = inverse_optimizer_config
        self._runtime: Optional[Dict[str, Any]] = None

    def propose(
        self,
        target_state: TargetStateSpecV1,
        control_state: ControlStateSpecV1,
        snapshot: MeasurementSnapshot,
        prepared_artifact: PreparedSequenceArtifactV1,
        episode_history: Sequence[Mapping[str, Any]],
    ) -> FeedbackActionProposalV2:
        if target_state.system_type != control_state.system_type:
            raise ValueError("target/control system_type mismatch")
        if snapshot.system_type != target_state.system_type:
            raise ValueError("snapshot system_type mismatch")
        if prepared_artifact.target_hash != target_state.target_hash:
            raise ValueError("prepared artifact target hash mismatch")
        if prepared_artifact.control_hash != control_state.control_hash:
            raise ValueError("prepared artifact actual control hash mismatch")
        self._verify_prepared_artifact(prepared_artifact)
        provenance = self._snapshot_provenance(snapshot, prepared_artifact)

        runtime = self._load_runtime(prepared_artifact)
        torch = runtime["torch"]
        model = runtime["model"]
        table = runtime["table"]
        metadata = runtime["metadata"]
        command = runtime["command_tensor_batch"](
            table,
            device=self.device,
            normalization_contract=runtime["normalization"],
            expected_normalization_sha256=metadata["command_normalization_sha256"],
        )
        supported_time_blocks = self._supported_time_blocks(model)
        observation, observation_diagnostics = self._observation(
            snapshot,
            control_state,
            episode_history,
            supported_time_blocks=supported_time_blocks,
        )
        model.eval()
        output = model(observation, command)
        neural_action = torch.cat((
            output["amplitude_action_coordinates"], output["phase_action_coordinates"],
        ), dim=-1)
        neural_mask = torch.cat((
            output["amplitude_actionable_mask"], output["phase_actionable_mask"],
        ), dim=-1)
        optimizer_config = self.inverse_optimizer_config
        if optimizer_config is None:
            optimizer_config = runtime["InverseTeacherConfig"]()
        optimizer = runtime["ConstrainedInverseTeacher"](model, optimizer_config)
        optimized = optimizer.solve(
            observation,
            command,
            target_counts=None,
            target_amplitude=target_state.desired_amplitudes,
            target_phase_rad=target_state.desired_phases_rad,
            warm_start=neural_action.detach(),
            forward_context=output["forward_context"],
            actionable_mask=neural_mask,
        )
        selected = optimized.action_coordinates.detach()
        dimensions = model.dimensions
        amp_dim = int(dimensions["amplitude"])
        fisher = optimized.fisher.detach()
        outcomes = self._candidate_outcomes(
            model,
            observation,
            command,
            output["forward_context"],
            selected,
            target_state,
        )
        cycle_consistency = self._cycle_consistency(
            outcomes[-1]["expected_counts_by_cycle"], target_state.system_type,
        )
        detector_keys = (
            "detector_regularization",
            "detector_dark_prior_deviation",
            "detector_efficiency_prior_deviation",
            "detector_time_block_residual_norm",
            "source_regularization",
            "dark_identifiable",
            "posterior_uncertainty_required",
            "true_multi_subtraction_available",
            "multi_click_amplitude_actionable",
            "source_detector_nuisance_scope",
        )
        detector_source = {
            name: _json_value(output[name]) for name in detector_keys if name in output
        }
        table_known = table.table_hash in set(metadata["command_table_hashes"])
        valid_fraction = _scalar_or_list(
            output["forward_context"].encoded_command_context.valid_tone_fraction
        )
        ood = {
            "runtime_table_seen_during_training": bool(table_known),
            "runtime_registry_seen_during_training": (
                metadata.get("registry_graph_hash") == prepared_artifact.registry_hash
            ),
            "valid_tone_fraction": valid_fraction,
            "fail_closed_missing_command": _json_value(output["fail_closed_missing_command"]),
            "fail_closed_reason": output["fail_closed_reason"],
            "snapshot_provenance": provenance,
            "observation": observation_diagnostics,
            "table_identity_used_as_numeric_input": False,
        }
        amplitude_mask = _bool_tuple(optimized.amplitude_actionable_mask[0])
        phase_mask = _bool_tuple(optimized.phase_actionable_mask[0])
        coordinate_mask = _bool_tuple(optimized.per_coordinate_actionable_mask[0])
        selected_values = _float_tuple(selected[0])
        covariance = _matrix_tuple(optimized.covariance[0])
        neural_covariance = _matrix_tuple(output["action_covariance"][0])
        return FeedbackActionProposalV2(
            target_hash=target_state.target_hash,
            parent_control_hash=control_state.control_hash,
            command_table_hash=table.table_hash,
            model_checkpoint_hash=runtime["checkpoint_sha256"],
            neural_warm_start=_float_tuple(neural_action[0]),
            optimizer_action=selected_values,
            selected_base_action=selected_values,
            amplitude_coordinates=selected_values[:amp_dim],
            phase_coordinates=selected_values[amp_dim:],
            action_covariance=covariance,
            amplitude_actionable_mask=amplitude_mask,
            phase_actionable_mask=phase_mask,
            per_coordinate_actionable_mask=coordinate_mask,
            block_fisher={
                "full": _matrix_tuple(fisher[0]),
                "amplitude": _matrix_tuple(fisher[0, :amp_dim, :amp_dim]),
                "phase": _matrix_tuple(fisher[0, amp_dim:, amp_dim:]),
                "cross_amplitude_phase": _matrix_tuple(fisher[0, :amp_dim, amp_dim:]),
            },
            block_conditions={
                "amplitude": _json_value(optimized.diagnostics.get("fisher_condition_amp")),
                "phase": _json_value(optimized.diagnostics.get("fisher_condition_phase")),
                "optimizer_converged": bool(optimized.converged),
                "optimizer_iterations": int(optimized.iterations),
            },
            posterior_std=_float_tuple(optimized.posterior_std[0]),
            candidate_scales=(0.0, 0.25, 0.5, 1.0),
            predicted_candidate_outcomes=tuple(outcomes),
            ood_diagnostics=ood,
            cycle_consistency=cycle_consistency,
            detector_source_diagnostics=detector_source,
            checkpoint_metadata={
                "metadata_sha256": metadata["metadata_sha256"],
                "architecture_id": metadata["architecture_id"],
                "command_normalization_sha256": metadata["command_normalization_sha256"],
                "research_only": metadata["research_only"],
                "promotion_ready": metadata["promotion_ready"],
                "neural_action_covariance": neural_covariance,
            },
            same_context_integrability={
                "model_context_hash": output["model_context_hash"],
                "count_jacobian_source": output["count_jacobian_source"],
                "same_forward_context_reused": optimized.diagnostics[
                    "same_forward_context_reused"
                ],
                "forward_model_frozen": optimized.diagnostics["forward_model_frozen"],
                "independent_count_jacobian_head": bool(
                    getattr(model, "independent_count_jacobian_head", True)
                ),
            },
        )

    def _load_runtime(self, artifact: PreparedSequenceArtifactV1) -> Dict[str, Any]:
        if self._runtime is not None:
            cached_table = self._runtime["table"]
            if cached_table.table_hash == artifact.command_table_hash:
                return self._runtime
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        receipt_path = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".receipt.json")
        if not receipt_path.is_file():
            raise FileNotFoundError("V6 checkpoint receipt is required for strict inference")
        checkpoint_sha256 = self._sha256_file(self.checkpoint_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("V6 checkpoint receipt hash mismatch")
        if receipt.get("research_only") is not True or receipt.get("promotion_ready") is not False:
            raise ValueError("V6 checkpoint receipt violates research-only promotion contract")
        if not self.normalization_path.is_file() or not self.config_path.is_file():
            raise FileNotFoundError("V6 normalization and config files are required")

        import torch

        from ml_recon.physics_hybrid_v5.action_conditioned_model import command_tensor_batch
        from ml_recon.physics_hybrid_v5.action_optimizer import (
            ConstrainedInverseTeacher,
            InverseTeacherConfig,
        )
        from ml_recon.physics_hybrid_v5.command_contract import PhysicsHybridCommandTableV1
        from ml_recon.physics_hybrid_v5.command_features import command_tone_features
        from ml_recon.physics_hybrid_v5.joint_training import build_research_feedback_model
        from ml_recon.physics_hybrid_v5.runtime_checkpoint import (
            canonical_sha256,
            load_joint_checkpoint,
            validate_joint_checkpoint_metadata,
        )
        from ml_recon.physics_hybrid_v5.train_joint_feedback import _enable_joint_research_config

        table_manifest = Path(artifact.command_table_directory) / "command_table.json"
        table = PhysicsHybridCommandTableV1.load(table_manifest, mmap_mode="r")
        if table.table_hash != artifact.command_table_hash or table.table_id != artifact.command_table_id:
            raise ValueError("prepared artifact command table identity mismatch")
        if table.operation_registry_hash != artifact.registry_hash:
            raise ValueError("prepared artifact operation registry hash mismatch")
        if table.matrix_snapshot_hash != artifact.matrix_hash:
            raise ValueError("prepared artifact matrix snapshot hash mismatch")
        raw = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(raw, Mapping):
            raise ValueError("V6 checkpoint payload must be a mapping")
        metadata = validate_joint_checkpoint_metadata(raw.get("metadata") or {})
        if receipt.get("metadata_sha256") != metadata["metadata_sha256"]:
            raise ValueError("V6 checkpoint receipt metadata hash mismatch")
        normalization = json.loads(self.normalization_path.read_text(encoding="utf-8"))
        config = _enable_joint_research_config(
            json.loads(self.config_path.read_text(encoding="utf-8"))
        )
        if canonical_sha256(config) != metadata["config_sha256"]:
            raise ValueError("V6 runtime config hash does not match checkpoint")
        expected_architecture = canonical_sha256({
            "hidden_dim": 64,
            "system_type": table.system_type,
        })
        expected = {
            "architecture_id": "OperationComplexFeedbackModel",
            "architecture_sha256": expected_architecture,
            "system_type": table.system_type,
            "command_schema": table.schema,
            "command_schema_hash": table.feature_schema_hash,
            "command_normalization_sha256": normalization.get("sha256"),
        }
        model = build_research_feedback_model(config, table, hidden_dim=64).to(self.device)
        loaded = load_joint_checkpoint(
            self.checkpoint_path,
            model,
            expected_metadata=expected,
            map_location=self.device,
        )
        feature_order = tuple(command_tone_features(table)[1].ordered_names)
        if feature_order != tuple(metadata["command_feature_order"]):
            raise ValueError("runtime command feature order differs from checkpoint")
        runtime = {
            "torch": torch,
            "model": model,
            "table": table,
            "metadata": loaded["metadata"],
            "checkpoint_sha256": checkpoint_sha256,
            "normalization": normalization,
            "command_tensor_batch": command_tensor_batch,
            "ConstrainedInverseTeacher": ConstrainedInverseTeacher,
            "InverseTeacherConfig": InverseTeacherConfig,
        }
        self._runtime = runtime
        return runtime

    def _verify_prepared_artifact(self, artifact: PreparedSequenceArtifactV1) -> None:
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
            raise ValueError("prepared artifact ID does not match its immutable identity")
        matrix_hashes = self._verify_files(Path(artifact.matrix_workspace), artifact.matrix_files)
        waveform_hashes = self._verify_files(Path(artifact.sequence_dir), artifact.waveform_files)
        if stable_json_hash(matrix_hashes) != artifact.matrix_hash:
            raise ValueError("prepared artifact matrix hash mismatch")
        if stable_json_hash(waveform_hashes) != artifact.waveform_hash:
            raise ValueError("prepared artifact waveform hash mismatch")
        if self._sha256_file(Path(artifact.phase_cal_manifest_path)) != artifact.phase_cal_manifest_hash:
            raise ValueError("prepared artifact phase manifest hash mismatch")

    @staticmethod
    def _snapshot_provenance(
        snapshot: MeasurementSnapshot,
        artifact: PreparedSequenceArtifactV1,
    ) -> Dict[str, Any]:
        diagnostics = dict(snapshot.diagnostics or {})
        expected = {
            "prepared_artifact_id": artifact.artifact_id,
            "command_table_hash": artifact.command_table_hash,
            "target_state_hash": artifact.target_hash,
            "control_state_hash": artifact.control_hash,
            "matrix_bundle_hash": artifact.matrix_hash,
            "waveform_hash": artifact.waveform_hash,
        }
        mismatched = {
            key: {"snapshot": diagnostics.get(key), "prepared": value}
            for key, value in expected.items()
            if diagnostics.get(key) is not None and diagnostics.get(key) != value
        }
        if mismatched:
            raise ValueError(f"snapshot/prepared artifact provenance mismatch: {mismatched}")
        present = {key: diagnostics.get(key) is not None for key in expected}
        return {
            "present": present,
            "complete": all(present.values()),
            "strict_episode_ready": all(present.values()),
            "source": "prepared_artifact_argument_not_current_matrix_directory",
        }

    @staticmethod
    def _observation(
        snapshot: MeasurementSnapshot,
        control_state: ControlStateSpecV1,
        episode_history: Sequence[Mapping[str, Any]],
        *,
        supported_time_blocks: int,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        cell_count = len(control_state.commanded_amplitudes)
        amplitude, phase, source = V6FeedbackBridge._measured_w_state(snapshot, control_state)
        temperature = float((snapshot.diagnostics or {}).get("temperature_c", 25.0) or 25.0)
        requested_time_block = int(
            (snapshot.diagnostics or {}).get("time_block_index", len(episode_history)) or 0
        )
        time_block = min(max(requested_time_block, 0), max(int(supported_time_blocks) - 1, 0))
        features = np.concatenate((
            amplitude,
            np.sin(phase),
            np.asarray([(temperature - 25.0) / 10.0, time_block / 3.0]),
        ))
        expected_operations = 75 if snapshot.system_type == "3qubit" else 256
        counts = np.stack((
            np.asarray(snapshot.signal_bins, dtype=float).reshape(-1),
            np.asarray(snapshot.idler_bins, dtype=float).reshape(-1),
            np.asarray(snapshot.coin_bins, dtype=float).reshape(-1),
        ), axis=-1)
        if counts.shape != (expected_operations, 3):
            raise ValueError(
                f"snapshot count shape {counts.shape} does not match {expected_operations} operations"
            )
        observation = {
            "observation_features": features,
            "exposure_s": np.asarray([max(float(snapshot.duration_s), 1e-9)]),
            "detector_time_block_index": time_block,
            "off_window_calibration_available": bool(
                (snapshot.diagnostics or {}).get("off_window_calibration_available", False)
            ),
        }
        gate_exposure = (snapshot.diagnostics or {}).get("gate_exposure")
        if gate_exposure is not None:
            observation["gate_exposure"] = gate_exposure
        return observation, {
            "measured_state_source": source,
            "feature_dimension": int(features.size),
            "point_count_shape": list(counts.shape),
            "temperature_c": temperature,
            "time_block_index": time_block,
            "requested_time_block_index": requested_time_block,
            "supported_time_blocks": int(supported_time_blocks),
            "time_block_out_of_distribution": requested_time_block != time_block,
            "desired_target_used_as_current_state": False,
            "cell_count": cell_count,
        }

    @staticmethod
    def _supported_time_blocks(model: Any) -> int:
        blocks = []
        source_model = getattr(model.count_forward_model, "source_model", None)
        detector_model = getattr(model.count_forward_model, "detector_model", None)
        for component in (source_model, detector_model):
            residual = getattr(component, "time_block_residual", None)
            if residual is not None and getattr(residual, "shape", None):
                blocks.append(int(residual.shape[0]))
        return max(1, min(blocks) if blocks else 1)

    @staticmethod
    def _measured_w_state(
        snapshot: MeasurementSnapshot,
        control_state: ControlStateSpecV1,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        rho = snapshot.density_matrix
        cell_count = len(control_state.commanded_amplitudes)
        if rho is None:
            amplitude = np.asarray(control_state.commanded_amplitudes, dtype=float)
            amplitude /= np.linalg.norm(amplitude)
            return amplitude, np.asarray(control_state.commanded_phases_rad, dtype=float), "actual_control_fallback"
        density = np.asarray(rho, dtype=complex)
        indices = [1 << (cell_count - index - 1) for index in range(cell_count)]
        if density.shape == (cell_count, cell_count):
            core = density
        elif density.shape == (2 ** cell_count, 2 ** cell_count):
            core = density[np.ix_(indices, indices)]
        else:
            raise ValueError("snapshot density matrix has an unsupported dimension")
        hermitian = 0.5 * (core + np.conjugate(core.T))
        values, vectors = np.linalg.eigh(hermitian)
        vector = vectors[:, int(np.argmax(values))]
        amplitude = np.abs(vector)
        if float(np.linalg.norm(amplitude)) <= 1e-12:
            raise ValueError("measured W-subspace state has zero norm")
        amplitude /= np.linalg.norm(amplitude)
        phase = np.angle(vector)
        phase -= float(np.mean(phase))
        phase = (phase + np.pi) % (2.0 * np.pi) - np.pi
        return amplitude, phase, "dominant_measured_w_subspace_eigenvector"

    @staticmethod
    def _candidate_outcomes(
        model: Any,
        observation: Mapping[str, Any],
        command: Any,
        forward_context: Any,
        selected: Any,
        target_state: TargetStateSpecV1,
    ) -> list[Dict[str, Any]]:
        import torch

        amp_dim = int(model.dimensions["amplitude"])
        phase_only = torch.cat((
            torch.zeros_like(selected[..., :amp_dim]), selected[..., amp_dim:],
        ), dim=-1)
        amplitude_only = torch.cat((
            selected[..., :amp_dim], torch.zeros_like(selected[..., amp_dim:]),
        ), dim=-1)
        candidates = (
            ("zero", selected * 0.0, 0.0),
            ("phase_only", phase_only, 1.0),
            ("amplitude_only", amplitude_only, 1.0),
            ("full_0_25", selected * 0.25, 0.25),
            ("full_0_50", selected * 0.50, 0.50),
            ("full_1_00", selected, 1.0),
        )
        outcomes = []
        target_amplitude = selected.new_tensor(target_state.desired_amplitudes).reshape(1, -1)
        target_phase = selected.new_tensor(target_state.desired_phases_rad).reshape(1, -1)
        for kind, action, scale in candidates:
            result = model.count_forward_model.forward_map(
                observation, command, action, forward_context=forward_context,
            )
            state = result["shared_state_amplitude"] * torch.exp(
                1j * result["shared_state_phase_rad"]
            )
            target = target_amplitude * torch.exp(1j * target_phase)
            infidelity = 1.0 - torch.abs(
                torch.sum(torch.conj(target) * state, dim=-1)
            ).square()
            expected = result["expected_counts"][0]
            if model.system_type == "4qubit":
                by_cycle = expected.reshape(4, 64, 3).sum(dim=1)
            else:
                by_cycle = expected.sum(dim=0, keepdim=True)
            action_energy = torch.sum(action.square(), dim=-1)
            utility = -infidelity - 0.01 * action_energy
            outcomes.append({
                "kind": kind,
                "scale": float(scale),
                "action": _float_tuple(action[0]),
                "predicted_utility": float(utility[0].detach().cpu()),
                "predicted_state_infidelity": float(infidelity[0].detach().cpu()),
                "predicted_total_counts": _float_tuple(expected.sum(dim=0)),
                "expected_counts_by_cycle": _json_value(by_cycle),
                "same_forward_context_reused": True,
            })
        return outcomes

    @staticmethod
    def _cycle_consistency(expected_by_cycle: Any, system_type: str) -> Dict[str, Any]:
        values = np.asarray(expected_by_cycle, dtype=float)
        totals = np.sum(values, axis=-1)
        mean = float(np.mean(totals)) if totals.size else 0.0
        coefficient = float(np.std(totals) / max(mean, 1e-12))
        return {
            "applicable": system_type == "4qubit",
            "per_cycle_total": totals.tolist(),
            "coefficient_of_variation": coefficient,
            "passed": bool(system_type != "4qubit" or coefficient <= 0.10),
            "threshold": 0.10,
        }

    @staticmethod
    def _verify_files(root: Path, hashes: Mapping[str, str]) -> Dict[str, str]:
        values = {}
        for name, expected in hashes.items():
            path = root / str(name)
            digest = V6FeedbackBridge._sha256_file(path)
            if digest != expected:
                raise ValueError(f"prepared artifact file hash mismatch: {name}")
            values[str(name)] = digest
        return values

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


def _float_tuple(value: Any) -> Tuple[float, ...]:
    array = np.asarray(_json_value(value), dtype=float).reshape(-1)
    return tuple(float(item) for item in array)


def _bool_tuple(value: Any) -> Tuple[bool, ...]:
    array = np.asarray(_json_value(value), dtype=bool).reshape(-1)
    return tuple(bool(item) for item in array)


def _matrix_tuple(value: Any) -> Tuple[Tuple[float, ...], ...]:
    array = np.asarray(_json_value(value), dtype=float)
    return tuple(tuple(float(item) for item in row) for row in array)


def _scalar_or_list(value: Any) -> Any:
    converted = _json_value(value)
    if isinstance(converted, list) and len(converted) == 1:
        return converted[0]
    return converted


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = ["FeedbackActionProposalV2", "V6FeedbackBridge"]
