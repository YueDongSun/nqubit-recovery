"""Read-only production shadow bridge for frozen V7 feedback models.

The bridge validates immutable model evidence and prepared-command provenance,
then runs Model B proposal generation plus Model A candidate ranking.  It never
materializes matrices, uploads AWG data, programs an FPGA, requests collection,
or mutates an experiment runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data_protocol import stable_json_hash
from .models import MeasurementSnapshot, TargetStateSpecV1
from .sequence_service import PreparedSequenceArtifactV1


V7_SHADOW_PROPOSAL_SCHEMA = "FeedbackActionProposalV7"
CANDIDATE_NAMES = (
    "no_op",
    "phase_0_25",
    "phase_0_50",
    "phase_1_00",
    "amplitude_1_00",
    "combined_1_00",
)


@dataclass(frozen=True)
class FeedbackActionProposalV7:
    system_type: str
    created_at: float
    target_hash: str
    prepared_artifact_id: str
    checkpoint_sha256: str
    report_hash: str
    history_length: int
    selected_candidate: str
    action: Tuple[float, ...]
    action_mask: Tuple[float, ...]
    model_b_mode: str
    predicted_gain_mean: float
    predicted_gain_std: float
    predicted_harmful_probability: float
    predicted_stop_probability: float
    multi_traffic_light: str
    multi_probability_mean: float
    multi_probability_lcb90: float
    multi_probability_ucb90: float
    candidate_outcomes: Tuple[Mapping[str, Any], ...]
    count_diagnostics: Mapping[str, Any]
    rho_diagnostics: Mapping[str, Any]
    classifier_diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    guard: Mapping[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["schema"] = V7_SHADOW_PROPOSAL_SCHEMA
        value["action_semantics"] = {
            "coordinates_0_2": "Aitchison relative amplitude; inactive coordinates are zero",
            "coordinates_3_5": "Helmert relative phase in radians; inactive coordinates are zero",
            "write_channel": "W_CH1 projective global action only",
            "r_channels_fixed": True,
            "brightness_action_enabled": False,
        }
        value["research_only"] = True
        value["promotion_ready"] = False
        value["hardware_action_authorized"] = False
        return value


class V7FeedbackBridge:
    """Strict frozen V7 Model A/B runtime with no hardware-write authority."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        training_report_path: str | Path,
        *,
        device: str = "cpu",
        measurement_window_s: float = 1200.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.training_report_path = Path(training_report_path).resolve()
        self.device_name = str(device)
        self.measurement_window_s = float(measurement_window_s)
        if self.measurement_window_s <= 0.0:
            raise ValueError("V7 measurement window must be positive")
        self._runtime: Optional[Dict[str, Any]] = None

    def propose(
        self,
        target_state: TargetStateSpecV1,
        snapshot: MeasurementSnapshot,
        prepared_artifact: PreparedSequenceArtifactV1,
        *,
        episode_history: Sequence[Mapping[str, Any]] = (),
    ) -> FeedbackActionProposalV7:
        if target_state.system_type != snapshot.system_type:
            raise ValueError("V7 target and measurement system types differ")
        if prepared_artifact.system_type != snapshot.system_type:
            raise ValueError("V7 prepared artifact and measurement system types differ")
        if prepared_artifact.target_hash != target_state.target_hash:
            raise ValueError("V7 target hash differs from prepared artifact")
        duration_tolerance_s = max(1.0, 0.01 * self.measurement_window_s)
        if abs(float(snapshot.duration_s) - self.measurement_window_s) > duration_tolerance_s:
            raise ValueError(
                "V7 snapshot duration differs from the frozen measurement-window contract"
            )
        self._verify_prepared_artifact(prepared_artifact)
        provenance = self._snapshot_provenance(snapshot, prepared_artifact)
        runtime = self._load_runtime()
        torch = runtime["torch"]
        model = runtime["model"]
        classifier_prior = runtime["classifier_prior"]
        from ml_recon.physics_hybrid_v7.feedback_synthetic import (
            ACTION_MODE_NAMES,
            HISTORY_FEATURE_NAMES,
            PHASE_ACTION_MIN_RAD,
        )

        system_type = snapshot.system_type
        arrays = self._build_inputs(runtime, target_state, snapshot, episode_history)
        operation = torch.from_numpy(arrays["operation_features"]).to(runtime["device"])
        target = torch.from_numpy(arrays["target_features"]).to(runtime["device"])
        graph = torch.from_numpy(arrays["graph_kernel"]).to(runtime["device"])
        history = torch.from_numpy(arrays["history"]).to(runtime["device"])
        history_mask = torch.from_numpy(arrays["history_mask"]).to(runtime["device"])
        action_mask = torch.from_numpy(arrays["action_mask"]).to(runtime["device"])
        classifier_features: Dict[str, Any] = {}
        if classifier_prior is not None:
            from ml_recon.physics_hybrid_v7.feedback_classifier_prior import (
                embed_core_density,
            )

            density = snapshot.density_matrix
            qubit_count = 3 if system_type == "3qubit" else 4
            if density is None:
                full = np.eye(2 ** qubit_count, dtype=np.complex64)[None, :, :] / float(
                    2 ** qubit_count
                )
                rho_mask = torch.zeros(1, device=runtime["device"])
            else:
                value = np.asarray(density, dtype=np.complex64)
                if value.shape == (qubit_count, qubit_count):
                    value = embed_core_density(value[None, :, :], system_type)[0]
                if value.shape != (2 ** qubit_count, 2 ** qubit_count):
                    raise ValueError("V7 classifier rho dimension is unsupported")
                full = value[None, :, :]
                rho_mask = torch.ones(1, device=runtime["device"])
            latent, uncertainty = classifier_prior.encode(
                current_full_rho=torch.from_numpy(full).to(runtime["device"]),
                target_features=target,
                rho_available_mask=rho_mask,
                system_type=system_type,
            )
            classifier_features = {
                "classifier_latent": latent,
                "classifier_uncertainty": uncertainty,
            }
        zero_action = torch.zeros_like(action_mask)
        with torch.no_grad():
            current = model.model_a(
                operation,
                target,
                graph,
                zero_action,
                action_mask,
                system_type=system_type,
                **classifier_features,
            )
            policy = model.model_b(
                current["state_embedding"],
                history,
                history_mask,
                action_mask,
                system_type=system_type,
            )
            candidate_actions = policy["candidate_actions"][0]
            history_length = int(np.sum(arrays["history_mask"]))
            history_values = arrays["history"][0, :history_length]
            if history_length > 0:
                accepted_index = HISTORY_FEATURE_NAMES.index("accepted")
                accepted = history_values[:, accepted_index] > 0.5
                cumulative_action = np.sum(
                    history_values[accepted, :6], axis=0,
                ) if np.any(accepted) else np.zeros(6, dtype=np.float32)
            else:
                cumulative_action = np.zeros(6, dtype=np.float32)
            cumulative_limits = np.asarray(
                [0.03, 0.03, 0.03, 0.80, 0.80, 0.80],
                dtype=np.float32,
            )
            lower = torch.from_numpy(
                -cumulative_limits - cumulative_action
            ).to(runtime["device"])
            upper = torch.from_numpy(
                cumulative_limits - cumulative_action
            ).to(runtime["device"])
            candidate_actions = torch.minimum(
                torch.maximum(candidate_actions, lower[None, :]),
                upper[None, :],
            )
            phase_size = candidate_actions[:, 3:].abs().amax(dim=1)
            phase_keep = (
                (phase_size == 0.0) | (phase_size >= PHASE_ACTION_MIN_RAD)
            ).to(dtype=candidate_actions.dtype)
            candidate_actions = torch.cat((
                candidate_actions[:, :3],
                candidate_actions[:, 3:] * phase_keep[:, None],
            ), dim=1)
            candidate_count = candidate_actions.shape[0]
            repeated_operation = operation.expand(candidate_count, -1, -1)
            repeated_target = target.expand(candidate_count, -1)
            repeated_mask = action_mask.expand(candidate_count, -1)
            repeated_classifier: Dict[str, Any] = {}
            if classifier_features:
                repeated_classifier = {
                    "classifier_latent": classifier_features["classifier_latent"].expand(
                        candidate_count, -1,
                    ),
                    "classifier_uncertainty": classifier_features[
                        "classifier_uncertainty"
                    ].expand(candidate_count),
                }
            candidate_prediction = model.model_a(
                repeated_operation,
                repeated_target,
                graph,
                candidate_actions,
                repeated_mask,
                system_type=system_type,
                **repeated_classifier,
            )
        outcomes, selected_index = self._rank_candidates(
            candidate_actions,
            candidate_prediction,
            target_state,
        )
        mode_index = int(torch.argmax(policy["mode_logits"], dim=-1)[0].cpu())
        mode = ACTION_MODE_NAMES[mode_index]
        stop_probability = float(torch.sigmoid(policy["stop_logit"])[0].cpu())
        policy_contract = dict(
            runtime["checkpoint"].get("policy_training_contract") or {}
        )
        threshold_by_system = dict(
            policy_contract.get("stop_probability_threshold_by_system") or {}
        )
        threshold_by_rho_availability = dict(
            policy_contract.get(
                "stop_probability_threshold_by_system_and_rho_availability"
            ) or {}
        )
        rho_availability = (
            "available" if snapshot.density_matrix is not None else "missing"
        )
        if rho_availability in threshold_by_rho_availability:
            threshold_by_system = dict(
                threshold_by_rho_availability[rho_availability]
            )
        default_stop_probability_threshold = float(
            policy_contract.get("stop_probability_threshold", 0.50)
        )
        system_threshold_contract = threshold_by_system.get(
            system_type,
            default_stop_probability_threshold,
        )
        if isinstance(system_threshold_contract, Mapping):
            history_bucket = (
                "0" if history_length == 0
                else "1" if history_length == 1
                else "2_plus"
            )
            stop_probability_threshold = float(
                system_threshold_contract.get(
                    history_bucket,
                    default_stop_probability_threshold,
                )
            )
        else:
            history_bucket = "all"
            stop_probability_threshold = float(system_threshold_contract)
        history_rollback_veto = bool(
            history_length > 0
            and arrays["history"][
                0,
                history_length - 1,
                HISTORY_FEATURE_NAMES.index("rollback"),
            ] > 0.5
        )
        plateau_veto = False
        if history_length >= 2:
            accepted_index = HISTORY_FEATURE_NAMES.index("accepted")
            utility_index = HISTORY_FEATURE_NAMES.index("utility_delta")
            tail = arrays["history"][0, history_length - 2:history_length]
            plateau_veto = bool(np.all(
                (tail[:, accepted_index] <= 0.5)
                | (tail[:, utility_index] < 0.0005)
            ))
        # Model A ranks the complete bounded candidate family.  Model B's
        # discrete mode is an episode-level veto only; using it as a second
        # candidate selector would bypass the trained Model A ranker whenever
        # the two heads disagree.
        stop_veto = stop_probability >= stop_probability_threshold
        if stop_veto or history_rollback_veto or plateau_veto:
            selected_index = 0
        selected_action = candidate_actions[selected_index].float().cpu().numpy()
        selected_outcome = outcomes[selected_index]
        nuisance_mean = candidate_prediction["nuisance_mean"][selected_index]
        nuisance_std = torch.exp(candidate_prediction["nuisance_log_scale"][selected_index])
        from ml_recon.physics_hybrid_v7.feedback_models import NUISANCE_TARGET_NAMES

        multi_index = NUISANCE_TARGET_NAMES.index("multi_excitation_probability")
        multi_mean = float(torch.clamp(nuisance_mean[multi_index], 0.0, 1.0).cpu())
        multi_std = float(torch.clamp(nuisance_std[multi_index], 0.0, 1.0).cpu())
        multi_lcb = max(0.0, multi_mean - 1.645 * multi_std)
        multi_ucb = min(1.0, multi_mean + 1.645 * multi_std)
        traffic = "green" if multi_ucb <= 0.10 else "red" if multi_lcb > 0.15 else "yellow"
        gain_std = float(torch.exp(policy["gain_log_scale"])[0].cpu())
        classifier_uncertainty = (
            float(classifier_features["classifier_uncertainty"][0].cpu())
            if classifier_features else None
        )
        return FeedbackActionProposalV7(
            system_type=system_type,
            created_at=time.time(),
            target_hash=target_state.target_hash,
            prepared_artifact_id=prepared_artifact.artifact_id,
            checkpoint_sha256=runtime["checkpoint_sha256"],
            report_hash=runtime["report"]["report_hash"],
            history_length=history_length,
            selected_candidate=CANDIDATE_NAMES[selected_index],
            action=tuple(float(value) for value in selected_action),
            action_mask=tuple(float(value) for value in arrays["action_mask"][0]),
            model_b_mode=mode,
            predicted_gain_mean=float(policy["gain_mean"][0].cpu()),
            predicted_gain_std=gain_std,
            predicted_harmful_probability=float(
                torch.sigmoid(policy["harmful_logit"])[0].cpu()
            ),
            predicted_stop_probability=stop_probability,
            multi_traffic_light=traffic,
            multi_probability_mean=multi_mean,
            multi_probability_lcb90=multi_lcb,
            multi_probability_ucb90=multi_ucb,
            candidate_outcomes=tuple(outcomes),
            count_diagnostics={
                "observed_total": float(np.sum(snapshot.coin_bins)),
                "predicted_current_total": float(current["expected_total"][0].cpu()),
                "nb2_dispersion": float(current["nb2_dispersion"][0].cpu()),
                "ood_energy": float(current["ood_energy"][0].cpu()),
                "validation_temperature": float(
                    runtime["checkpoint"]["temperature_by_system"][system_type]
                ),
                "gsi_is_diagnostic_not_subtraction_coefficient": True,
            },
            rho_diagnostics={
                "rho_available": snapshot.density_matrix is not None,
                "predicted_current_w_core_trace": float(
                    torch.diagonal(current["rho_current"][0]).real.sum().cpu()
                ),
                "selected_predicted_target_fidelity": selected_outcome[
                    "predicted_target_fidelity"
                ],
            },
            classifier_diagnostics={
                "source": runtime["classifier_contract"]["source"],
                "uncertainty": classifier_uncertainty,
                "gate_mean": float(current["classifier_gate"].abs().mean().cpu()),
                "hard_argmax_used": False,
            },
            provenance=provenance,
            guard={
                "accepted": False,
                "shadow_only": True,
                "hardware_apply_blocked": True,
                "hardware_apply_reason": "v7_shadow has no write authority",
                "model_b_always_analyzed": True,
                "model_b_mode_is_diagnostic_only": True,
                "stop_probability_threshold": stop_probability_threshold,
                "stop_probability_threshold_history_bucket": history_bucket,
                "stop_probability_threshold_rho_availability": rho_availability,
                "stop_veto_applied": stop_veto,
                "history_rollback_veto_applied": history_rollback_veto,
                "two_round_plateau_veto_applied": plateau_veto,
                "traffic_light_is_diagnostic": True,
                "per_step_phase_bound_rad": 0.10,
                "per_step_amplitude_bound": 0.005,
                "cumulative_phase_bound_requires_guarded_coordinator": 0.80,
                "cumulative_action_before_proposal": [
                    float(value) for value in cumulative_action
                ],
                "cumulative_candidate_clipped": True,
                "candidate_materialized": False,
                "uploaded": False,
                "active_command_changed": False,
            },
        )

    def _load_runtime(self) -> Dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        if not self.checkpoint_path.is_file() or not self.training_report_path.is_file():
            raise FileNotFoundError("V7 checkpoint and training report are both required")
        import torch

        from ml_recon.physics_hybrid_v7.feedback_classifier_prior import (
            FrozenFeedbackClassifierPrior,
        )
        from ml_recon.physics_hybrid_v7.feedback_models import (
            V7FeedbackModelBundle,
            V7FeedbackModelConfig,
        )

        report = json.loads(self.training_report_path.read_text(encoding="utf-8"))
        if report.get("schema") != "v7_feedback_training_report_v1":
            raise ValueError("V7 training report schema changed")
        if (
            report.get("research_only") is not True
            or report.get("promotion_ready") is not False
            or report.get("hardware_action_authorized") is not False
        ):
            raise ValueError("V7 training report violates the research-only contract")
        report_hash = str(report.get("report_hash") or "")
        unsigned_report = dict(report)
        unsigned_report.pop("report_hash", None)
        if self._hash_json(unsigned_report) != report_hash:
            raise ValueError("V7 training report hash mismatch")
        checkpoint_sha256 = self._sha256_file(self.checkpoint_path)
        if report.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
            raise ValueError("V7 checkpoint hash differs from training report")
        device_name = self.device_name
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("V7 CUDA inference was requested but CUDA is unavailable")
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=device_name,
            weights_only=False,
        )
        if checkpoint.get("schema") != "v7_feedback_inference_checkpoint_v1":
            raise ValueError("V7 inference checkpoint schema changed")
        if (
            checkpoint.get("frozen_inference") is not True
            or checkpoint.get("online_training_allowed") is not False
            or checkpoint.get("promotion_ready") is not False
            or checkpoint.get("hardware_action_authorized") is not False
        ):
            raise ValueError("V7 checkpoint violates the frozen research-only contract")
        config_payload = dict(checkpoint["model_config"])
        config_payload.pop("schema", None)
        model_config = V7FeedbackModelConfig(**config_payload)
        if (
            model_config.research_only is not True
            or model_config.promotion_ready is not False
            or model_config.hardware_action_authorized is not False
        ):
            raise ValueError("V7 model config violates the research-only contract")
        model = V7FeedbackModelBundle(model_config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        device = torch.device(device_name)
        model.to(device).eval()
        classifier_prior = None
        classifier_contract = dict(
            checkpoint.get("classifier_prior") or {"source": "none", "frozen": True}
        )
        classifier_source = classifier_contract.get("source", "none")
        if classifier_contract.get("frozen") is not True:
            raise ValueError("V7 classifier prior must be frozen")
        if classifier_source != "none":
            repository_root = Path(__file__).resolve().parents[2]
            classifier_prior = FrozenFeedbackClassifierPrior(
                source=str(classifier_source),
                model_dir=repository_root / "ml_recon" / "models",
                expected_latent_dim=model.config.classifier_latent_dim,
                device=device,
            )
            expected_hashes = classifier_contract.get("checkpoints", {})
            actual_hashes = classifier_prior.audit_dict()["checkpoints"]
            for system_type in ("3qubit", "4qubit"):
                if (
                    expected_hashes.get(system_type, {}).get("sha256")
                    != actual_hashes[system_type]["sha256"]
                ):
                    raise ValueError("V7 classifier prior checkpoint hash mismatch")
        self._runtime = {
            "torch": torch,
            "device": device,
            "model": model,
            "classifier_prior": classifier_prior,
            "classifier_contract": classifier_contract,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "report": report,
            "input_cache": {},
        }
        return self._runtime

    def _build_inputs(
        self,
        runtime: Dict[str, Any],
        target_state: TargetStateSpecV1,
        snapshot: MeasurementSnapshot,
        episode_history: Sequence[Mapping[str, Any]],
    ) -> Dict[str, np.ndarray]:
        from ml_recon.physics_hybrid_v7.feedback_data import (
            build_operation_features,
            build_target_features,
        )
        from ml_recon.physics_hybrid_v7.feedback_synthetic import canonical_action_mask
        from ml_recon.physics_hybrid_v7.geometry import build_operation_frequency_geometry
        from ml_recon.physics_hybrid_v7.physical_ceiling import load_retrieval_rate_vectors

        system_type = snapshot.system_type
        cached = runtime["input_cache"].get(system_type)
        if cached is None:
            geometry = build_operation_frequency_geometry(system_type)
            retrieval_vectors, _ = load_retrieval_rate_vectors(
                system_type, geometry.operation_count,
            )
            retrieval_vectors = np.asarray(retrieval_vectors, dtype=np.float64)
            retrieval_vectors /= max(float(np.max(retrieval_vectors)), 1e-12)
            retrieval = np.mean(retrieval_vectors, axis=1)
            retrieval /= max(float(np.max(retrieval)), 1e-12)
            static, _ = geometry.static_features()
            graph = geometry.coupling_kernel(
                x_length_mhz=8.0,
                y_length_mhz=8.0,
                same_local_cross_cycle_weight=0.05,
            )
            graph += np.eye(geometry.operation_count)
            graph /= np.maximum(graph.sum(axis=1, keepdims=True), 1e-12)
            cached = (geometry, static.astype(np.float32), graph.astype(np.float32), retrieval)
            runtime["input_cache"][system_type] = cached
        geometry, static, graph, retrieval = cached
        operation_count = geometry.operation_count
        coin = np.asarray(snapshot.coin_bins, dtype=np.float32).reshape(1, -1)
        signal = np.asarray(snapshot.signal_bins, dtype=np.float32).reshape(1, -1)
        idler = np.asarray(snapshot.idler_bins, dtype=np.float32).reshape(1, -1)
        if coin.shape[1] != operation_count or signal.shape != coin.shape or idler.shape != coin.shape:
            raise ValueError("V7 snapshot operation counts have the wrong shape")
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
            retrieval_relative_map=retrieval,
            observed_coin_mask=np.ones_like(coin, dtype=np.float32),
        )
        history, history_mask = self._history_tensor(episode_history)
        action_mask = canonical_action_mask(system_type)[None, :]
        return {
            "operation_features": operation_features,
            "target_features": target_features.astype(np.float32),
            "graph_kernel": graph,
            "history": history,
            "history_mask": history_mask,
            "action_mask": action_mask.astype(np.float32),
        }

    @staticmethod
    def _history_tensor(
        episode_history: Sequence[Mapping[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        from ml_recon.physics_hybrid_v7.feedback_synthetic import HISTORY_FEATURE_NAMES

        selected = list(episode_history)[-9:]
        history = np.zeros((1, 9, len(HISTORY_FEATURE_NAMES)), dtype=np.float32)
        mask = np.zeros((1, 9), dtype=np.float32)
        scalar_defaults = {
            "utility_delta": 0.0,
            "fidelity_after": 0.0,
            "retrieval_delta": 0.0,
            "log_count_ratio": 0.0,
            "multi_risk_delta": 0.0,
            "accepted": 0.0,
            "rollback": 0.0,
            "uncertainty": 1.0,
            "best_so_far": 0.0,
            "model_hash_changed": 0.0,
            "calibration_hash_changed": 0.0,
            "command_hash_changed": 0.0,
            "step_fraction": 0.0,
        }
        for row, item in enumerate(selected):
            if not isinstance(item, Mapping):
                raise ValueError("V7 episode history entries must be mappings")
            action = np.asarray(item.get("actual_action", item.get("action", [])), dtype=float)
            if action.shape != (6,) or not np.all(np.isfinite(action)):
                raise ValueError("V7 episode history action must contain six finite values")
            values = [float(value) for value in action]
            for name in HISTORY_FEATURE_NAMES[6:]:
                value = float(item.get(name, scalar_defaults[name]))
                if not math.isfinite(value):
                    raise ValueError(f"V7 history field {name} is non-finite")
                values.append(value)
            history[0, row] = np.asarray(values, dtype=np.float32)
            mask[0, row] = 1.0
        return history, mask

    @staticmethod
    def _rank_candidates(
        actions: Any,
        output: Mapping[str, Any],
        target_state: TargetStateSpecV1,
    ) -> Tuple[list[Dict[str, Any]], int]:
        torch = __import__("torch")
        qubit_count = 3 if target_state.system_type == "3qubit" else 4
        target = np.asarray(target_state.desired_amplitudes) * np.exp(
            1j * np.asarray(target_state.desired_phases_rad)
        )
        target /= max(float(np.linalg.norm(target)), 1e-12)
        target_tensor = torch.as_tensor(
            target,
            dtype=output["rho_next"].dtype,
            device=output["rho_next"].device,
        )
        fidelity = torch.real(torch.einsum(
            "i,bij,j->b", torch.conj(target_tensor), output["rho_next"], target_tensor,
        )).clamp(0.0, 1.0)
        utility_mean = output["utility_mean"].float()
        utility_std = torch.exp(output["utility_log_scale"].float())
        harmful = torch.sigmoid(output["harmful_logit"].float())
        risk_score = utility_mean - 0.50 * utility_std - 0.25 * harmful
        selected_index = int(torch.argmax(risk_score).cpu())
        outcomes = []
        for index, name in enumerate(CANDIDATE_NAMES):
            outcomes.append({
                "name": name,
                "action": [float(value) for value in actions[index].float().cpu().numpy()],
                "predicted_utility_mean": float(utility_mean[index].cpu()),
                "predicted_utility_std": float(utility_std[index].cpu()),
                "predicted_harmful_probability": float(harmful[index].cpu()),
                "predicted_target_fidelity": float(fidelity[index].cpu()),
                "predicted_total_counts": float(output["expected_total"][index].cpu()),
                "risk_adjusted_score": float(risk_score[index].cpu()),
                "qubit_count": qubit_count,
            })
        return outcomes, selected_index

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
            raise ValueError(f"V7 snapshot/prepared provenance mismatch: {mismatched}")
        present = {key: diagnostics.get(key) is not None for key in expected}
        present["upload_receipt_hash"] = bool(diagnostics.get("upload_receipt_hash"))
        if not all(present.values()):
            missing = sorted(key for key, value in present.items() if not value)
            raise ValueError(f"V7 snapshot provenance is incomplete: {missing}")
        return {
            "present": present,
            "complete": True,
            "source": "prepared_artifact_argument_not_current_matrix_directory",
        }

    def _verify_prepared_artifact(self, artifact: PreparedSequenceArtifactV1) -> None:
        if not bool(artifact.fourier_report.get("passed", False)):
            raise ValueError("V7 prepared artifact did not pass the Fourier guard")
        if not bool(artifact.materialization.get("accepted", False)):
            raise ValueError("V7 prepared artifact materialization was not accepted")
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
            raise ValueError("V7 prepared artifact ID mismatch")
        from ml_recon.physics_hybrid_v5.command_contract import PhysicsHybridCommandTableV1

        command_manifest = Path(artifact.command_table_directory) / "command_table.json"
        command_table = PhysicsHybridCommandTableV1.load(command_manifest, mmap_mode=None)
        if (
            command_table.table_id != artifact.command_table_id
            or command_table.table_hash != artifact.command_table_hash
        ):
            raise ValueError("V7 prepared artifact command-table identity mismatch")
        if command_table.operation_registry_hash != artifact.registry_hash:
            raise ValueError("V7 prepared artifact operation-registry hash mismatch")
        if command_table.matrix_snapshot_hash != artifact.matrix_hash:
            raise ValueError("V7 prepared artifact command-table matrix hash mismatch")
        matrix_hashes = self._verify_files(Path(artifact.matrix_workspace), artifact.matrix_files)
        waveform_hashes = self._verify_files(Path(artifact.sequence_dir), artifact.waveform_files)
        if stable_json_hash(matrix_hashes) != artifact.matrix_hash:
            raise ValueError("V7 prepared artifact matrix hash mismatch")
        if stable_json_hash(waveform_hashes) != artifact.waveform_hash:
            raise ValueError("V7 prepared artifact waveform hash mismatch")
        if self._sha256_file(Path(artifact.phase_cal_manifest_path)) != artifact.phase_cal_manifest_hash:
            raise ValueError("V7 phase calibration manifest hash mismatch")

    def _verify_files(self, root: Path, hashes: Mapping[str, str]) -> Dict[str, str]:
        values = {}
        for name, expected in hashes.items():
            path = root / str(name)
            digest = self._sha256_file(path)
            if digest != expected:
                raise ValueError(f"V7 prepared artifact file hash mismatch: {name}")
            values[str(name)] = digest
        return values

    @staticmethod
    def _hash_json(value: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


__all__ = [
    "FeedbackActionProposalV7",
    "V7FeedbackBridge",
    "V7_SHADOW_PROPOSAL_SCHEMA",
]
