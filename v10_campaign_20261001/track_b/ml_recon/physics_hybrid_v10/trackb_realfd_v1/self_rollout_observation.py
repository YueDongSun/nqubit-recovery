"""Complete immutable acquisitions for the versioned self-rollout diagnostic.

No V2.4 generator recipe is changed. Raw sampled counts are reconstructed by
replaying the private registered ``_sample_detected_setting`` with exactly the
uniforms used by ``phase_sensitive_density_measurement``. This deterministic
recomputation is not another acquisition. Ideal counts are expected counts,
never samples. Neither noise parameters nor latent states enter the record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml_recon.data_v2_4.hashing import sha256_array, sha256_json
from ml_recon.data_v2_4.sample_keys import V24SampleKey
from ml_recon.data_v2_4.synthetic.feedback_renderers import (
    TrackBRecedingHorizonForwardContextV1,
    _sample_detected_setting,
    phase_sensitive_density_measurement,
)
from ml_recon.physics_hybrid_v10.csm_observation_contract import (
    CARTESIAN_STATS_LAYOUT, CartesianFeedbackAccumulator,
)
from ml_recon.physics_hybrid_v10.measurement_model import born_probabilities
from ml_recon.physics_hybrid_v10.noise_interventions import registered_feedback_settings, w_density_matrix
from ml_recon.physics_hybrid_v10.state_backends import DenseDensityStateBackend


observation_stream = "trackb_self_rollout_observation_v2"
common_pairing = "common_random_variates_v1"
independent_pairing = "independent_branch_streams_v1"
pairing_kinds = (common_pairing, independent_pairing)
branch_kinds = ("model", "zero", "teacher")
Float_rows = tuple[tuple[float, ...], ...]
Int_rows = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Observation_schedule_v2:
    sample_key_sha: str
    replicate_index: int
    window_index: int
    pairing_kind: str
    branch_stream: str
    settings: tuple[tuple[str, ...], ...]
    shots_by_setting: tuple[int, ...]
    setting_substreams: tuple[str, ...]
    setting_uniform_shas: tuple[str, ...]
    panel_sha: str
    stream: str = observation_stream
    noise_component: str = "measurement_uniform_v1"
    schema: str = "TrackBSelfRolloutObservationScheduleV2"


@dataclass(frozen=True)
class Acquired_observation_v2:
    """Deeply immutable acquisition bytes/metadata; record_id covers every field."""

    schedule: Observation_schedule_v2
    shot_mode: str
    sampling_enabled: bool
    sampled_counts: Int_rows | None
    expected_counts: Float_rows | None
    outcome_order: Int_rows
    no_detection_column: int
    attempts_by_setting: tuple[int, ...]
    total_attempts: int
    detected_counts: tuple[int, ...] | None
    no_detection_counts: tuple[int, ...] | None
    expected_detected_counts: tuple[float, ...] | None
    expected_no_detection_counts: tuple[float, ...] | None
    detection_fraction: tuple[float, ...]
    measurement_tokens: Float_rows
    measurement_token_sha: str
    measurement_covariance: Float_rows
    cartesian_stats: tuple[float, ...]
    cartesian_covariance: Float_rows
    cartesian_statistic_valid_mask: tuple[bool, ...]
    cartesian_detection_fraction: tuple[float, ...]
    cartesian_covariance_estimator_kind: str
    cartesian_covariance_instance_sha: str
    cartesian_observation_contract_sha: str
    cartesian_observation_contract_kind: str
    cartesian_statistic_transform_sha: str
    cartesian_dirichlet_alpha: float
    cartesian_whitening_support_kind: str
    cartesian_geometry_source: str
    setting_valid_mask: tuple[bool, ...]
    raw_count_reconstruction: str
    counts_kind: str
    cartesian_stats_layout: str = CARTESIAN_STATS_LAYOUT
    measurement_token_dtype: str = "float32"
    token_uncertainty_kind: str = "POPULATION_VARIANCE_AND_STD_ONLY_NOT_PHASE_COVARIANCE"
    phase_uncertainty_status: str = "NOT_PROVIDED_BY_REGISTERED_PHASE_PROXY_RENDERER"
    postselection_kind: str = "NONE_NO_DETECTION_RETAINED_IN_ATTEMPT_DENOMINATOR"
    preparation_assumption: str = "STATIC_STATE_REPREPARED_FOR_EACH_SETTING_AND_ATTEMPT"
    backaction_assumption: str = "NO_DYNAMIC_MEASUREMENT_BACKACTION_MODELED"
    time_assumption: str = "TIME_AND_DRIFT_UNMODELED"
    calibration_status: str = "REGISTERED_SYNTHETIC_MEASUREMENT_RECIPE_NO_EXTERNAL_CALIBRATION_EVIDENCE"
    setting_duration_s: None = None
    timestamp: None = None
    acquisition_count: int = 1
    deterministic_recomputation_additional_attempts: int = 0
    schema: str = "TrackBSelfRolloutAcquiredObservationV2"
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        # asdict cannot read an unset init=False field, so exclude it explicitly.
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "record_id"}
        payload["schedule"] = asdict(self.schedule)
        object.__setattr__(self, "record_id", sha256_json(payload))


def build_observation_panel(
    key: V24SampleKey,
    settings: tuple[tuple[str, ...], ...],
    shots: NDArray[np.int64],
    *,
    replicate_index: int = 0,
    window_index: int,
    pairing_kind: str = common_pairing,
    branch_kind: str,
) -> tuple[NDArray[np.float64], Observation_schedule_v2]:
    """Key each setting by sample, replicate, acquisition window and component.

    Common pairing shares variates, not outcomes. Independent branch streams
    still share the latent episode and are not independent population samples.
    Padding to max(shots) is recorded but not counted as attempted measurements.
    """
    key.validate()
    if key.N not in (3, 4) or settings != registered_feedback_settings(key.N):
        raise ValueError("observation requires registered 3Q/4Q settings in exact order")
    if type(replicate_index) is not int or replicate_index < 0:
        raise ValueError("replicate_index must be a nonnegative integer")
    if type(window_index) is not int or not 0 <= window_index <= 10:
        raise ValueError("window_index must identify one of eleven acquisitions")
    if pairing_kind not in pairing_kinds or branch_kind not in branch_kinds:
        raise ValueError("unregistered pairing kind or branch")
    shots = np.asarray(shots)
    if shots.shape != (len(settings),) or shots.dtype.kind not in "iu" or np.any(shots <= 0):
        raise ValueError("shots must be positive registered per-setting integers")
    branch_stream = "common" if pairing_kind == common_pairing else branch_kind
    substreams = tuple(sha256_json({
        "sample_key_sha": key.sha256(), "replicate_index": replicate_index,
        "window_index": window_index, "setting_index": index, "setting": axes,
        "branch_stream": branch_stream, "pairing_kind": pairing_kind,
        "noise_component": "measurement_uniform_v1",
    }) for index, axes in enumerate(settings))
    panel = np.stack([key.rng(observation_stream, substream).random(int(shots.max()))
                      for substream in substreams])
    schedule = Observation_schedule_v2(
        sample_key_sha=key.sha256(), replicate_index=replicate_index, window_index=window_index,
        pairing_kind=pairing_kind, branch_stream=branch_stream, settings=settings,
        shots_by_setting=tuple(int(value) for value in shots), setting_substreams=substreams,
        setting_uniform_shas=tuple(sha256_array(panel[index, :int(count)]) for index, count in enumerate(shots)),
        panel_sha=sha256_array(panel),
    )
    return panel, schedule


def _float_rows(value: Any) -> Float_rows:
    return tuple(tuple(float(item) for item in row) for row in value)


def acquire_observation(
    context: TrackBRecedingHorizonForwardContextV1,
    key: V24SampleKey,
    shots: NDArray[np.int64],
    *,
    shot_mode: str,
    window_index: int,
    branch_kind: str,
    replicate_index: int = 0,
    pairing_kind: str = common_pairing,
) -> Acquired_observation_v2:
    """Acquire one full record; all latent physics stays inside this boundary."""
    context.validate()
    if (context.sample_id != key.sample_id or len(context.before_coefficients) != key.N
            or shot_mode not in ("S_setting", "S_total", "S_ideal")):
        raise ValueError("observation source identity or shot mode is invalid")
    shots = np.array(shots, copy=True)
    panel, schedule = build_observation_panel(
        key, context.settings, shots, window_index=window_index, branch_kind=branch_kind,
        replicate_index=replicate_index, pairing_kind=pairing_kind,
    )
    ideal = shot_mode == "S_ideal"
    rho = context.intervention.apply_state_channel(
        w_density_matrix(context.before_coefficients.copy()), context.physical_parameter)
    measured = phase_sensitive_density_measurement(
        rho.copy(), shots.copy(), panel.copy(), context.intervention,
        context.physical_parameter, ideal_probability=ideal,
    )
    tokens = np.asarray(measured["tokens"], dtype=np.float32)
    if tokens.shape != (key.N, 2) or not np.isfinite(tokens).all():
        raise ValueError("observation must contain finite registered [modes,2] tokens")
    backend = DenseDensityStateBackend(rho.copy())
    counts = []
    accumulator = CartesianFeedbackAccumulator(key.N, context.settings)
    outcome_order = None
    for index, axes in enumerate(context.settings):
        count = int(shots[index])
        if ideal:
            outcomes, probability = born_probabilities(backend, axes)
            probability = context.intervention.apply_apparatus_channel(
                probability, outcomes, context.physical_parameter)
            law = context.intervention.apply_detector_channel(probability, context.physical_parameter)
            setting_counts = law * count
            accumulator_counts = setting_counts
        else:
            sampled, detected, outcomes = _sample_detected_setting(
                backend, axes, panel[index, :count], context.intervention, context.physical_parameter)
            indices = ((sampled[detected] > 0).astype(np.int64)
                       @ (2 ** np.arange(key.N - 1, -1, -1)))
            setting_counts = np.concatenate((np.bincount(indices, minlength=2 ** key.N),
                                             [count - int(detected.sum())])).astype(np.int64)
            # Match the registered renderer's floating-point arithmetic exactly.
            accumulator_counts = (setting_counts.astype(np.float64) / count) * count
        order = tuple(tuple(int(value) for value in row) for row in outcomes)
        if outcome_order is not None and outcome_order != order:
            raise ValueError("registered outcome order differs across settings")
        outcome_order = order
        counts.append(setting_counts)
        accumulator.add_setting_counts(index, accumulator_counts, count, sampling_enabled=not ideal)
    reconstructed = accumulator.finalize()
    for name, measured_name in (("stats", "cartesian_stats"), ("covariance", "cartesian_covariance"),
                                ("valid_mask", "cartesian_statistic_valid_mask"),
                                ("detection_fraction", "cartesian_detection_fraction")):
        if not np.array_equal(reconstructed[name], measured[measured_name]):
            raise ValueError(f"reconstructed raw counts disagree with registered {name}")
    if reconstructed["covariance_instance_sha"] != measured["cartesian_covariance_instance_sha"]:
        raise ValueError("reconstructed uncertainty identity differs from registered observation")
    numeric_fields = ("covariance", "cartesian_stats", "cartesian_covariance", "detection_fraction",
                      "cartesian_detection_fraction", "cartesian_dirichlet_alpha")
    if any(not np.isfinite(np.asarray(measured[name])).all() for name in numeric_fields):
        raise ValueError("observation contains non-finite uncertainty or statistics")
    count_array = np.stack(counts)
    detected_counts = count_array[:, :-1].sum(axis=1)
    misses = count_array[:, -1]
    return Acquired_observation_v2(
        schedule=schedule, shot_mode=shot_mode, sampling_enabled=not ideal,
        sampled_counts=(None if ideal else tuple(tuple(int(item) for item in row) for row in count_array)),
        expected_counts=(_float_rows(count_array) if ideal else None),
        outcome_order=outcome_order, no_detection_column=2 ** key.N,
        attempts_by_setting=tuple(0 if ideal else int(value) for value in shots),
        total_attempts=0 if ideal else int(shots.sum()),
        detected_counts=None if ideal else tuple(int(value) for value in detected_counts),
        no_detection_counts=None if ideal else tuple(int(value) for value in misses),
        expected_detected_counts=tuple(float(value) for value in detected_counts) if ideal else None,
        expected_no_detection_counts=tuple(float(value) for value in misses) if ideal else None,
        detection_fraction=tuple(float(value) for value in measured["detection_fraction"]),
        measurement_tokens=_float_rows(tokens), measurement_token_sha=sha256_array(tokens),
        measurement_covariance=_float_rows(measured["covariance"]),
        cartesian_stats=tuple(float(value) for value in measured["cartesian_stats"]),
        cartesian_covariance=_float_rows(measured["cartesian_covariance"]),
        cartesian_statistic_valid_mask=tuple(bool(value) for value in measured["cartesian_statistic_valid_mask"]),
        cartesian_detection_fraction=tuple(float(value) for value in measured["cartesian_detection_fraction"]),
        cartesian_covariance_estimator_kind=str(measured["cartesian_covariance_estimator_kind"]),
        cartesian_covariance_instance_sha=str(measured["cartesian_covariance_instance_sha"]),
        cartesian_observation_contract_sha=str(measured["cartesian_observation_contract_sha"]),
        cartesian_observation_contract_kind=str(measured["cartesian_observation_contract_kind"]),
        cartesian_statistic_transform_sha=str(measured["cartesian_statistic_transform_sha"]),
        cartesian_dirichlet_alpha=float(measured["cartesian_dirichlet_alpha"]),
        cartesian_whitening_support_kind=str(measured["cartesian_whitening_support_kind"]),
        cartesian_geometry_source=str(measured["cartesian_geometry_source"]),
        setting_valid_mask=tuple(True for _ in context.settings),
        raw_count_reconstruction=("EXPECTED_COUNTS_FROM_REGISTERED_LAW_NO_SAMPLING" if ideal else
                                  "DETERMINISTIC_REPLAY_OF_REGISTERED_PRIVATE_SAMPLER_SAME_UNIFORMS"),
        counts_kind="EXPECTED_FLOAT64_COUNTS" if ideal else "SAMPLED_INTEGER_COUNTS",
    )


__all__ = ["Acquired_observation_v2", "Observation_schedule_v2", "acquire_observation",
           "build_observation_panel", "common_pairing", "independent_pairing", "pairing_kinds", "observation_stream"]
