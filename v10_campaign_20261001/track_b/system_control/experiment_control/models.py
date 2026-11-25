"""Shared data structures for the unified experiment feedback loop."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from system_control.qubit_recon_files.digital_delay_defaults import DEFAULT_DIGITAL_DELAY_OFFSET_NS

from .model_paths import LEGACY_FEEDBACK_MODEL_PATH, display_feedback_model_path


DEFAULT_MEASUREMENT_DURATION_S = 1800.0
V4_4Q_CAMPAIGN_DURATION_S = 1200.0


def _state_contract_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _state_values(
    state: Mapping[str, Any],
    primary_key: str,
    legacy_key: str,
    length: int,
    *,
    degrees_key: Optional[str] = None,
) -> Tuple[float, ...]:
    values = state.get(primary_key, state.get(legacy_key))
    if values is None and degrees_key is not None and state.get(degrees_key) is not None:
        values = np.deg2rad(np.asarray(state[degrees_key], dtype=float)).tolist()
    if values is None or len(values) != length:
        raise ValueError(f"{primary_key} must contain {length} values")
    result = tuple(float(value) for value in values)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{primary_key} contains non-finite values")
    return result


def _target_rho(
    system_type: str,
    amplitudes: Sequence[float],
    phases_rad: Sequence[float],
    excitation_mode: int,
) -> np.ndarray:
    dimension = 8 if system_type == "3qubit" else 16
    state_indices = (4, 2, 1) if system_type == "3qubit" else (8, 4, 2, 1)
    selected_count = {
        "3qubit": {1: 3, 2: 2, 3: 3},
        "4qubit": {1: 4, 2: 3, 3: 2, 4: 4},
    }[system_type].get(int(excitation_mode))
    if selected_count is None:
        raise ValueError(f"unsupported excitation_mode={excitation_mode} for {system_type}")
    alpha = np.asarray(amplitudes, dtype=float) * np.exp(1j * np.asarray(phases_rad, dtype=float))
    if (system_type == "3qubit" and int(excitation_mode) == 3) or (
        system_type == "4qubit" and int(excitation_mode) == 4
    ):
        rho = np.zeros((dimension, dimension), dtype=complex)
        norm_sq = float(np.sum(np.square(np.abs(alpha))))
        if norm_sq > 0.0:
            for index, basis_index in enumerate(state_indices):
                rho[basis_index, basis_index] = np.square(abs(alpha[index])) / norm_sq
        return rho
    psi = np.zeros(dimension, dtype=complex)
    selected = alpha[:selected_count]
    norm = float(np.linalg.norm(selected))
    if norm > 0.0:
        for index, basis_index in enumerate(state_indices[:selected_count]):
            psi[basis_index] = selected[index] / norm
    return np.outer(psi, psi.conj())


@dataclass(frozen=True)
class TargetStateSpecV1:
    """Immutable scientific target; feedback must never mutate this object."""

    desired_amplitudes: Tuple[float, ...]
    desired_phases_rad: Tuple[float, ...]
    immutable_target_rho: np.ndarray = field(repr=False, compare=False)
    target_hash: str
    system_type: str

    def __post_init__(self) -> None:
        cell_count = 3 if self.system_type == "3qubit" else 4 if self.system_type == "4qubit" else 0
        amplitudes = tuple(float(value) for value in self.desired_amplitudes)
        phases = tuple(float(value) for value in self.desired_phases_rad)
        if cell_count == 0 or len(amplitudes) != cell_count or len(phases) != cell_count:
            raise ValueError("target state dimension does not match system_type")
        rho = np.array(self.immutable_target_rho, dtype=complex, copy=True)
        expected_dimension = 2 ** cell_count
        if rho.shape != (expected_dimension, expected_dimension):
            raise ValueError("immutable_target_rho has the wrong shape")
        rho.setflags(write=False)
        payload = {
            "schema": "TargetStateSpecV1",
            "system_type": self.system_type,
            "desired_amplitudes": list(amplitudes),
            "desired_phases_rad": list(phases),
            "rho_real": np.real(rho).tolist(),
            "rho_imag": np.imag(rho).tolist(),
        }
        expected_hash = _state_contract_hash(payload)
        if self.target_hash and self.target_hash != expected_hash:
            raise ValueError("target_hash does not match immutable target contents")
        object.__setattr__(self, "desired_amplitudes", amplitudes)
        object.__setattr__(self, "desired_phases_rad", phases)
        object.__setattr__(self, "immutable_target_rho", rho)
        object.__setattr__(self, "target_hash", expected_hash)

    @classmethod
    def from_state(
        cls,
        system_type: str,
        state: Mapping[str, Any],
        *,
        excitation_mode: int = 1,
    ) -> "TargetStateSpecV1":
        cell_count = 3 if system_type == "3qubit" else 4
        amplitudes = _state_values(state, "desired_amplitudes", "amplitudes", cell_count)
        phases = _state_values(
            state, "desired_phases_rad", "phases_rad", cell_count,
            degrees_key="phases_deg",
        )
        rho_value = state.get("immutable_target_rho")
        rho = _target_rho(system_type, amplitudes, phases, excitation_mode) if rho_value is None else np.asarray(rho_value, dtype=complex)
        return cls(amplitudes, phases, rho, str(state.get("target_hash") or ""), system_type)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": "TargetStateSpecV1",
            "system_type": self.system_type,
            "desired_amplitudes": list(self.desired_amplitudes),
            "desired_phases_rad": list(self.desired_phases_rad),
            "immutable_target_rho_real": np.real(self.immutable_target_rho).tolist(),
            "immutable_target_rho_imag": np.imag(self.immutable_target_rho).tolist(),
            "target_hash": self.target_hash,
        }


@dataclass(frozen=True)
class ControlStateSpecV1:
    """Immutable record of one mutable-through-replacement control state."""

    commanded_amplitudes: Tuple[float, ...]
    commanded_phases_rad: Tuple[float, ...]
    cumulative_amplitude_coordinates: Tuple[float, ...]
    cumulative_phase_coordinates: Tuple[float, ...]
    parent_control_hash: Optional[str]
    control_hash: str
    system_type: str

    def __post_init__(self) -> None:
        cell_count = 3 if self.system_type == "3qubit" else 4 if self.system_type == "4qubit" else 0
        amplitudes = tuple(float(value) for value in self.commanded_amplitudes)
        phases = tuple(float(value) for value in self.commanded_phases_rad)
        amplitude_coordinates = tuple(float(value) for value in self.cumulative_amplitude_coordinates)
        phase_coordinates = tuple(float(value) for value in self.cumulative_phase_coordinates)
        if cell_count == 0 or len(amplitudes) != cell_count or len(phases) != cell_count:
            raise ValueError("control state dimension does not match system_type")
        if len(amplitude_coordinates) != cell_count - 1 or len(phase_coordinates) != cell_count - 1:
            raise ValueError("cumulative projective coordinate dimension mismatch")
        if not np.all(np.isfinite(amplitudes + phases + amplitude_coordinates + phase_coordinates)):
            raise ValueError("control state contains non-finite values")
        payload = {
            "schema": "ControlStateSpecV1",
            "system_type": self.system_type,
            "commanded_amplitudes": list(amplitudes),
            "commanded_phases_rad": list(phases),
            "cumulative_amplitude_coordinates": list(amplitude_coordinates),
            "cumulative_phase_coordinates": list(phase_coordinates),
            "parent_control_hash": self.parent_control_hash,
        }
        expected_hash = _state_contract_hash(payload)
        if self.control_hash and self.control_hash != expected_hash:
            raise ValueError("control_hash does not match control contents")
        object.__setattr__(self, "commanded_amplitudes", amplitudes)
        object.__setattr__(self, "commanded_phases_rad", phases)
        object.__setattr__(self, "cumulative_amplitude_coordinates", amplitude_coordinates)
        object.__setattr__(self, "cumulative_phase_coordinates", phase_coordinates)
        object.__setattr__(self, "control_hash", expected_hash)

    @classmethod
    def from_state(cls, system_type: str, state: Mapping[str, Any]) -> "ControlStateSpecV1":
        cell_count = 3 if system_type == "3qubit" else 4
        amplitudes = _state_values(state, "commanded_amplitudes", "amplitudes", cell_count)
        phases = _state_values(
            state, "commanded_phases_rad", "phases_rad", cell_count,
            degrees_key="phases_deg",
        )
        return cls(
            amplitudes,
            phases,
            tuple(float(value) for value in state.get("cumulative_amplitude_coordinates", (0.0,) * (cell_count - 1))),
            tuple(float(value) for value in state.get("cumulative_phase_coordinates", (0.0,) * (cell_count - 1))),
            state.get("parent_control_hash"),
            str(state.get("control_hash") or ""),
            system_type,
        )

    @classmethod
    def from_target(cls, target: TargetStateSpecV1) -> "ControlStateSpecV1":
        coordinate_count = len(target.desired_amplitudes) - 1
        return cls(
            target.desired_amplitudes, target.desired_phases_rad,
            (0.0,) * coordinate_count, (0.0,) * coordinate_count,
            None, "", target.system_type,
        )

    def evolve(
        self,
        *,
        commanded_amplitudes: Sequence[float],
        commanded_phases_rad: Sequence[float],
        cumulative_amplitude_coordinates: Sequence[float],
        cumulative_phase_coordinates: Sequence[float],
    ) -> "ControlStateSpecV1":
        return ControlStateSpecV1(
            tuple(commanded_amplitudes), tuple(commanded_phases_rad),
            tuple(cumulative_amplitude_coordinates), tuple(cumulative_phase_coordinates),
            self.control_hash, "", self.system_type,
        )

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": "ControlStateSpecV1",
            "system_type": self.system_type,
            "commanded_amplitudes": list(self.commanded_amplitudes),
            "commanded_phases_rad": list(self.commanded_phases_rad),
            "cumulative_amplitude_coordinates": list(self.cumulative_amplitude_coordinates),
            "cumulative_phase_coordinates": list(self.cumulative_phase_coordinates),
            "parent_control_hash": self.parent_control_hash,
            "control_hash": self.control_hash,
        }


@dataclass
class ExperimentRunConfig:
    """Configuration for one dataset-collection or feedback-optimization run."""

    system_type: str
    mode: str = "normal"
    workflow_mode: Optional[str] = None
    state_schedule: List[Dict[str, Any]] = field(default_factory=list)
    excitation_mode: int = 1
    feedback_interval_s: float = 1800.0
    normal_state_duration_s: float = DEFAULT_MEASUREMENT_DURATION_S
    collection_duration_s: float = DEFAULT_MEASUREMENT_DURATION_S
    target_fidelity: float = 0.95
    efficiency_drop_threshold: float = 0.02
    model_path: str = ""
    feedback_backend: str = "legacy"
    v6_checkpoint_path: Optional[str] = None
    v6_normalization_path: Optional[str] = None
    v6_config_path: Optional[str] = None
    v6_shadow_output_dir: Optional[str] = None
    v6_episode_enabled: bool = False
    v6_episode_output_dir: Optional[str] = None
    v6_measurement_window_s: float = 900.0
    v6_subwindow_s: float = 150.0
    v6_maximum_action_rounds: int = 5
    v6_maximum_accepted_actions: int = 5
    v6_operator_approval_required: bool = True
    v6_stop_after_serious_harmful: bool = True
    v7_checkpoint_path: Optional[str] = None
    v7_training_report_path: Optional[str] = None
    v7_shadow_output_dir: Optional[str] = None
    v7_measurement_window_s: float = 1200.0
    v7_subwindow_s: float = 150.0
    v7_maximum_action_rounds: int = 10
    v7_device: str = "auto"
    v8_checkpoint_path: Optional[str] = None
    v8_training_report_path: Optional[str] = None
    v8_frozen_candidate_path: Optional[str] = None
    v8_shadow_output_dir: Optional[str] = None
    v8_device: str = "auto"
    retrieval_map_path: Optional[str] = None
    phase_feedback_strategy: str = "w_subspace_phase"
    auto_apply: bool = True
    delta_step_schedule: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    model_confidence_threshold: float = 0.25
    min_heuristic_gain_proxy: Optional[float] = None
    min_predicted_fidelity_change: float = -0.005
    allow_heuristic_deltas: bool = False
    strict_hardware_mode: bool = False
    awg_ready_manual: bool = True
    output_dir: Optional[str] = None
    digital_delay_offset_ns: int = DEFAULT_DIGITAL_DELAY_OFFSET_NS
    serial_port: str = "COM6"
    use_mock_data: bool = True
    classifier_model_path: Optional[str] = None
    # Physics-transfer feedback model (A+C scheme: GNN + classifier physics_aux features
    # + two-stage training). Validated in ml_recon/sandbox/physics_transfer/ at -37%.
    feedback_model_kind: str = "auto"  # auto | linear | neural_mlp | hybrid_image_physics | physics_transfer_gnn
    deep_classifier_checkpoint_path: Optional[str] = None  # path to DeepPhysicsClassifier .pt
    ensemble_n_passes: int = 1  # 1 = single forward; >1 = MC Dropout ensemble inference
    uncertainty_threshold: Optional[float] = None  # if set, skip delta when ensemble_std exceeds it
    physics_transfer_pretrain_epochs: int = 15  # stage-1 synthetic pretrain epochs
    # Operation-point digital-twin controls.  Shadow is deliberately the safe
    # default: diagnostics are computed/persisted, but no matrix delta or AWG
    # regeneration may occur.
    noise_aware_mode: str = "shadow"  # disabled | shadow | gated
    noise_sampling_mode: str = "gaussian"  # gaussian | poisson | hybrid
    mc_samples_realtime: int = 32
    mc_samples_offline: int = 512
    pair_model_mode: str = "disabled"  # disabled | manual_topology | auto_suppressed_pair
    external_dlcz_gsi_reference: Optional[float] = None
    promotion_criteria: Dict[str, float] = field(default_factory=dict)
    promotion_ready: bool = False  # explicit human-set acknowledgement; never auto-set
    campaign_run_dir: Optional[str] = None
    campaign_id: Optional[str] = None
    enable_phase_probes: bool = True
    legacy_feedback_sample_export: bool = True

    def __post_init__(self) -> None:
        if self.system_type not in {"3qubit", "4qubit"}:
            raise ValueError("system_type must be '3qubit' or '4qubit'")
        if self.mode not in {"normal", "feedback"}:
            raise ValueError("mode must be 'normal' or 'feedback'")
        if self.workflow_mode is None:
            self.workflow_mode = "feedback_optimize" if self.mode == "feedback" else "collect_dataset"
        if self.workflow_mode not in {"collect_dataset", "feedback_optimize"}:
            raise ValueError("workflow_mode must be 'collect_dataset' or 'feedback_optimize'")
        if self.phase_feedback_strategy not in {
            "w_subspace_phase",
            "legacy_diff_guarded",
            "geometric_phase_experimental",
            "classifier_enhanced",
        }:
            raise ValueError(
                "phase_feedback_strategy must be w_subspace_phase, legacy_diff_guarded, "
                "geometric_phase_experimental, or classifier_enhanced"
            )
        self.mode = "feedback" if self.workflow_mode == "feedback_optimize" else "normal"
        if not self.model_path or self.model_path == LEGACY_FEEDBACK_MODEL_PATH:
            self.model_path = display_feedback_model_path(self.system_type)
        if (
            self.normal_state_duration_s != DEFAULT_MEASUREMENT_DURATION_S
            and self.collection_duration_s == DEFAULT_MEASUREMENT_DURATION_S
        ):
            self.collection_duration_s = self.normal_state_duration_s
        if self.min_heuristic_gain_proxy is None:
            self.min_heuristic_gain_proxy = float(self.min_predicted_fidelity_change)
        if self.feedback_model_kind not in {
            "auto", "linear", "neural_mlp", "hybrid_image_physics", "physics_transfer_gnn",
        }:
            raise ValueError(
                "feedback_model_kind must be auto, linear, neural_mlp, "
                "hybrid_image_physics, or physics_transfer_gnn"
            )
        if self.feedback_backend not in {
            "legacy", "v6_shadow", "v6_candidate", "v6_episode", "v7_shadow",
            "v8_shadow",
        }:
            raise ValueError(
                "feedback_backend must be legacy, v6_shadow, v6_candidate, v6_episode, "
                "v7_shadow, or v8_shadow"
            )
        if self.feedback_backend == "v6_episode" and not self.v6_episode_enabled:
            raise ValueError("v6_episode backend requires explicit v6_episode_enabled=true")
        if self.v6_measurement_window_s <= 0.0 or self.v6_subwindow_s <= 0.0:
            raise ValueError("V6 episode measurement durations must be positive")
        v6_window_ratio = self.v6_measurement_window_s / self.v6_subwindow_s
        if abs(v6_window_ratio - round(v6_window_ratio)) > 1e-6:
            raise ValueError("v6_measurement_window_s must be divisible by v6_subwindow_s")
        if self.v6_maximum_action_rounds < 1 or self.v6_maximum_action_rounds > 10:
            raise ValueError("v6_maximum_action_rounds must be in 1..10")
        if (
            self.v6_maximum_accepted_actions < 1
            or self.v6_maximum_accepted_actions > self.v6_maximum_action_rounds
        ):
            raise ValueError(
                "v6_maximum_accepted_actions must be in 1..v6_maximum_action_rounds"
            )
        if self.feedback_backend == "v6_episode" and not self.v6_operator_approval_required:
            raise ValueError("V6 episode cannot disable operator approval")
        if self.v7_measurement_window_s <= 0.0 or self.v7_subwindow_s <= 0.0:
            raise ValueError("V7 measurement durations must be positive")
        v7_window_ratio = self.v7_measurement_window_s / self.v7_subwindow_s
        if abs(v7_window_ratio - round(v7_window_ratio)) > 1e-6:
            raise ValueError("v7_measurement_window_s must be divisible by v7_subwindow_s")
        if self.v7_maximum_action_rounds < 1 or self.v7_maximum_action_rounds > 10:
            raise ValueError("v7_maximum_action_rounds must be in 1..10")
        if self.v7_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("v7_device must be auto, cpu, or cuda")
        if self.feedback_backend == "v7_shadow" and (
            not self.v7_checkpoint_path or not self.v7_training_report_path
        ):
            raise ValueError(
                "v7_shadow requires v7_checkpoint_path and v7_training_report_path"
            )
        if self.v8_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("v8_device must be auto, cpu, or cuda")
        if self.feedback_backend == "v8_shadow":
            has_single_member = bool(
                self.v8_checkpoint_path and self.v8_training_report_path
            )
            has_frozen_ensemble = bool(self.v8_frozen_candidate_path)
            if has_single_member == has_frozen_ensemble:
                raise ValueError(
                    "v8_shadow requires exactly one of v8_frozen_candidate_path "
                    "or the v8_checkpoint_path/v8_training_report_path pair"
                )
            if bool(self.v8_checkpoint_path) != bool(self.v8_training_report_path):
                raise ValueError(
                    "v8_checkpoint_path and v8_training_report_path must be paired"
                )
        if self.ensemble_n_passes < 1:
            raise ValueError("ensemble_n_passes must be >= 1")
        if self.uncertainty_threshold is not None and self.uncertainty_threshold <= 0:
            raise ValueError("uncertainty_threshold must be positive when set")
        if self.noise_aware_mode not in {"disabled", "shadow", "gated"}:
            raise ValueError("noise_aware_mode must be disabled, shadow, or gated")
        if self.noise_sampling_mode not in {"gaussian", "poisson", "hybrid"}:
            raise ValueError("noise_sampling_mode must be gaussian, poisson, or hybrid")
        if self.mc_samples_realtime < 0 or self.mc_samples_offline < 1:
            raise ValueError("mc_samples_realtime must be >= 0 and mc_samples_offline >= 1")
        if self.pair_model_mode not in {"disabled", "manual_topology", "auto_suppressed_pair"}:
            raise ValueError("pair_model_mode must be disabled, manual_topology, or auto_suppressed_pair")
        self.digital_delay_offset_ns = int(self.digital_delay_offset_ns)
        if not self.state_schedule:
            n = 3 if self.system_type == "3qubit" else 4
            self.state_schedule = [{
                "name": f"default_{self.system_type}_w_state",
                "amplitudes": [1.0] * n,
                "phases_deg": [0.0] * n,
            }]


@dataclass
class CalibrationDelta:
    """One additive matrix delta proposed by ML feedback."""

    target_channel: str
    matrix_type: str
    delta: np.ndarray
    limit: float
    accepted: bool = True
    reason: str = ""
    path: Optional[str] = None
    # V4 cycle-aware W_CH1 feedback.  The field is deliberately trailing and
    # optional so every historical positional constructor and JSON payload
    # retains legacy all-cycle replication semantics.
    cycle_index_zero_based: Optional[int] = None

    def clipped_delta(self) -> np.ndarray:
        return np.clip(self.delta, -self.limit, self.limit)


@dataclass
class MeasurementSnapshot:
    """Aggregated measurement data and reconstruction output for one window."""

    system_type: str
    signal_bins: np.ndarray
    idler_bins: np.ndarray
    coin_bins: np.ndarray
    segment_values: np.ndarray
    read_count: int
    write_count: int
    timestamp: float
    duration_s: float
    density_matrix: Optional[np.ndarray] = None
    fidelity: Optional[float] = None
    purity: Optional[float] = None
    retrieval_rate: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    deltas: List[CalibrationDelta] = field(default_factory=list)
    # Schema-v2 raw acquisition payload.  These are optional so historical W3
    # samples and existing callers remain loadable unchanged.
    schema_version: int = 2
    signal_bins_raw: Optional[np.ndarray] = None
    idler_bins_raw: Optional[np.ndarray] = None
    coin_bins_raw: Optional[np.ndarray] = None
    signal_bins_corrected: Optional[np.ndarray] = None
    idler_bins_corrected: Optional[np.ndarray] = None
    coin_bins_corrected: Optional[np.ndarray] = None
    signed_segment_values: Optional[np.ndarray] = None
    legacy_clipped_segment_values: Optional[np.ndarray] = None
    coin_idler_values: Optional[np.ndarray] = None
    signal_timing_histograms: Optional[np.ndarray] = None
    idler_timing_histograms: Optional[np.ndarray] = None
    multi2_count: Optional[int] = None
    multi3_count: Optional[int] = None
    multi4_count: Optional[int] = None
    multi2_pair_counts: Dict[str, int] = field(default_factory=dict)
    signal_total: Optional[int] = None
    idler_total: Optional[int] = None
    coin_total: Optional[int] = None
    operation_point_registry: Optional[List[Dict[str, Any]]] = None
    operation_point_registry_hash: Optional[str] = None
    calibration_snapshot: Optional[Dict[str, Any]] = None
    calibration_snapshot_hash: Optional[str] = None
    phase_sign: float = -1.0
    basis_phase_convention: str = "read_aod_negative_order"
    fpga_protocol_version: Optional[str] = None
    feature_schema_version: int = 1
    external_source_characterization: Dict[str, Any] = field(default_factory=dict)
    missing_value_masks: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        rho = self.density_matrix
        payload = {
            "system_type": self.system_type,
            "timestamp": self.timestamp,
            "duration_s": self.duration_s,
            "read_count": int(self.read_count),
            "write_count": int(self.write_count),
            "signal_sum": int(np.sum(self.signal_bins)),
            "idler_sum": int(np.sum(self.idler_bins)),
            "coin_sum": int(np.sum(self.coin_bins)),
            "fidelity": None if self.fidelity is None else float(self.fidelity),
            "purity": None if self.purity is None else float(self.purity),
            "retrieval_rate": None if self.retrieval_rate is None else float(self.retrieval_rate),
            "density_matrix_re": None if rho is None else np.real(rho).tolist(),
            "density_matrix_im": None if rho is None else np.imag(rho).tolist(),
            "signal_bins": self.signal_bins.tolist(),
            "idler_bins": self.idler_bins.tolist(),
            "coin_bins": self.coin_bins.tolist(),
            "segment_values": self.segment_values.tolist(),
            "schema_version": int(self.schema_version),
            "signed_segment_values": _optional_array_json(self.signed_segment_values, self.segment_values),
            "legacy_clipped_segment_values": _optional_array_json(self.legacy_clipped_segment_values),
            "coin_idler_values": _optional_array_json(self.coin_idler_values),
            "signal_bins_raw": _optional_array_json(self.signal_bins_raw),
            "idler_bins_raw": _optional_array_json(self.idler_bins_raw),
            "coin_bins_raw": _optional_array_json(self.coin_bins_raw),
            "signal_bins_corrected": _optional_array_json(self.signal_bins_corrected),
            "idler_bins_corrected": _optional_array_json(self.idler_bins_corrected),
            "coin_bins_corrected": _optional_array_json(self.coin_bins_corrected),
            "signal_timing_histograms": _optional_array_json(self.signal_timing_histograms),
            "idler_timing_histograms": _optional_array_json(self.idler_timing_histograms),
            "multi2_count": self.multi2_count,
            "multi3_count": self.multi3_count,
            "multi4_count": self.multi4_count,
            "multi2_pair_counts": self.multi2_pair_counts,
            "signal_total": self.signal_total,
            "idler_total": self.idler_total,
            "coin_total": self.coin_total,
            "operation_point_registry": self.operation_point_registry,
            "operation_point_registry_hash": self.operation_point_registry_hash,
            "calibration_snapshot": self.calibration_snapshot,
            "calibration_snapshot_hash": self.calibration_snapshot_hash,
            "phase_sign": float(self.phase_sign),
            "basis_phase_convention": self.basis_phase_convention,
            "fpga_protocol_version": self.fpga_protocol_version,
            "feature_schema_version": int(self.feature_schema_version),
            "external_source_characterization": self.external_source_characterization,
            "missing_value_masks": self.missing_value_masks,
            "diagnostics": self.diagnostics,
            "deltas": [
                {
                    "target_channel": delta.target_channel,
                    "matrix_type": delta.matrix_type,
                    "accepted": delta.accepted,
                    "reason": delta.reason,
                    "limit": delta.limit,
                    "path": delta.path,
                    "cycle_index_zero_based": delta.cycle_index_zero_based,
                }
                for delta in self.deltas
            ],
        }
        return payload


@dataclass
class FeedbackPrediction:
    """Inference-only feedback model output for one measurement window."""

    deltas: List[CalibrationDelta]
    safe_deltas: List[CalibrationDelta] = field(default_factory=list)
    confidence: float = 1.0
    heuristic_gain_proxy: Optional[float] = None
    predicted_fidelity_change: Optional[float] = None
    physics_diagnostics: Dict[str, Any] = field(default_factory=dict)
    guard_result: Dict[str, Any] = field(default_factory=dict)
    prediction_source: str = "unknown"

    def __post_init__(self) -> None:
        if self.heuristic_gain_proxy is None:
            self.heuristic_gain_proxy = (
                0.0 if self.predicted_fidelity_change is None else float(self.predicted_fidelity_change)
            )
        if self.predicted_fidelity_change is None:
            self.predicted_fidelity_change = float(self.heuristic_gain_proxy)

    def to_json_dict(self) -> Dict[str, Any]:
        gain_proxy = float(self.heuristic_gain_proxy or 0.0)
        return {
            "confidence": float(self.confidence),
            "heuristic_gain_proxy": gain_proxy,
            "predicted_fidelity_change": gain_proxy,
            "prediction_source": self.prediction_source,
            "physics_diagnostics": self.physics_diagnostics,
            "guard_result": self.guard_result,
            "safe_delta_count": len(self.safe_deltas),
            "deltas": [
                {
                    "target_channel": delta.target_channel,
                    "matrix_type": delta.matrix_type,
                    "accepted": bool(delta.accepted),
                    "reason": delta.reason,
                    "limit": float(delta.limit),
                    "path": delta.path,
                    "cycle_index_zero_based": delta.cycle_index_zero_based,
                    "max_abs_delta": float(np.max(np.abs(delta.delta))) if delta.delta.size else 0.0,
                }
                for delta in self.deltas
            ],
        }


@dataclass
class FeedbackSample:
    """One offline-training sample collected from a full experimental state."""

    sample_id: str
    system_type: str
    target_state_params: Dict[str, Any]
    rho_target: np.ndarray
    rho_measured: np.ndarray
    counts_snapshot: Dict[str, Any]
    retrieval_metrics: Dict[str, Any]
    basis_config: Dict[str, Any]
    matrix_snapshot: Dict[str, Any]
    accepted_delta: Optional[List[CalibrationDelta]] = None
    notes: str = ""
    timestamp: float = 0.0
    feature_schema_version: int = 1
    operation_point_registry_snapshot: Optional[List[Dict[str, Any]]] = None
    operation_point_registry_hash: Optional[str] = None
    calibration_snapshot: Optional[Dict[str, Any]] = None
    calibration_snapshot_hash: Optional[str] = None
    noise_context: Dict[str, Any] = field(default_factory=dict)
    external_source_characterization: Dict[str, Any] = field(default_factory=dict)
    missing_value_masks: Dict[str, Any] = field(default_factory=dict)
    # Explicit provenance replaces the historical notes-string heuristic.  Old
    # samples deserialize as ``unknown`` and are never silently promoted to
    # real transition labels by the physics-transfer trainer.
    data_provenance: str = "unknown"  # synthetic | real | unknown

    def __post_init__(self) -> None:
        if self.data_provenance not in {"synthetic", "real", "unknown"}:
            raise ValueError("data_provenance must be synthetic, real, or unknown")

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "system_type": self.system_type,
            "target_state_params": self.target_state_params,
            "rho_target": _complex_array_to_json(self.rho_target),
            "rho_measured": _complex_array_to_json(self.rho_measured),
            "counts_snapshot": self.counts_snapshot,
            "retrieval_metrics": self.retrieval_metrics,
            "basis_config": self.basis_config,
            "matrix_snapshot": self.matrix_snapshot,
            "accepted_delta": None if self.accepted_delta is None else [
                {
                    "target_channel": delta.target_channel,
                    "matrix_type": delta.matrix_type,
                    "limit": float(delta.limit),
                    "accepted": bool(delta.accepted),
                    "reason": delta.reason,
                    "delta": np.asarray(delta.delta, dtype=float).tolist(),
                    "cycle_index_zero_based": delta.cycle_index_zero_based,
                }
                for delta in self.accepted_delta
            ],
            "notes": self.notes,
            "timestamp": float(self.timestamp),
            "feature_schema_version": int(self.feature_schema_version),
            "operation_point_registry_snapshot": self.operation_point_registry_snapshot,
            "operation_point_registry_hash": self.operation_point_registry_hash,
            "calibration_snapshot": self.calibration_snapshot,
            "calibration_snapshot_hash": self.calibration_snapshot_hash,
            "noise_context": self.noise_context,
            "external_source_characterization": self.external_source_characterization,
            "missing_value_masks": self.missing_value_masks,
            "data_provenance": self.data_provenance,
        }

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "FeedbackSample":
        accepted_delta = payload.get("accepted_delta")
        deltas = None
        if accepted_delta:
            deltas = [
                CalibrationDelta(
                    target_channel=item["target_channel"],
                    matrix_type=item["matrix_type"],
                    delta=np.asarray(item["delta"], dtype=float),
                    limit=float(item["limit"]),
                    accepted=bool(item.get("accepted", True)),
                    reason=item.get("reason", ""),
                    cycle_index_zero_based=item.get("cycle_index_zero_based"),
                )
                for item in accepted_delta
            ]
        return cls(
            sample_id=payload["sample_id"],
            system_type=payload["system_type"],
            target_state_params=payload["target_state_params"],
            rho_target=_complex_array_from_json(payload["rho_target"]),
            rho_measured=_complex_array_from_json(payload["rho_measured"]),
            counts_snapshot=payload.get("counts_snapshot", {}),
            retrieval_metrics=payload.get("retrieval_metrics", {}),
            basis_config=payload.get("basis_config", {}),
            matrix_snapshot=payload.get("matrix_snapshot", {}),
            accepted_delta=deltas,
            notes=payload.get("notes", ""),
            timestamp=float(payload.get("timestamp", 0.0)),
            feature_schema_version=int(payload.get("feature_schema_version", 1)),
            operation_point_registry_snapshot=payload.get("operation_point_registry_snapshot"),
            operation_point_registry_hash=payload.get("operation_point_registry_hash"),
            calibration_snapshot=payload.get("calibration_snapshot"),
            calibration_snapshot_hash=payload.get("calibration_snapshot_hash"),
            noise_context=payload.get("noise_context", {}),
            external_source_characterization=payload.get("external_source_characterization", {}),
            missing_value_masks=payload.get("missing_value_masks", {}),
            data_provenance=str(payload.get("data_provenance", "unknown")),
        )


def _complex_array_to_json(values: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=complex)
    return {
        "real": np.real(arr).tolist(),
        "imag": np.imag(arr).tolist(),
        "shape": list(arr.shape),
    }


def _complex_array_from_json(payload: Dict[str, Any]) -> np.ndarray:
    real = np.asarray(payload.get("real", []), dtype=float)
    imag = np.asarray(payload.get("imag", []), dtype=float)
    arr = real + 1j * imag
    shape = payload.get("shape")
    if shape:
        arr = arr.reshape(tuple(shape))
    return arr


def _optional_array_json(values: Optional[np.ndarray], fallback: Optional[np.ndarray] = None):
    if values is None:
        values = fallback
    return None if values is None else np.asarray(values).tolist()
