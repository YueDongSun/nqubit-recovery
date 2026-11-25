"""Operation-bound DLCZ event renderer with deterministic CRN streams."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Sequence

import numpy as np

from ml_recon.physics_hybrid_v10.contracts import sha256_of

from ..contracts.dlcz_event_mc import (
    DlczAllocationReceipt,
    DlczCalibrationDraw,
    DlczDomainContract,
    DlczOperationParameters,
)
from ..physics.dlcz_readout_events import ReadoutEventLedger, render_readout_events
from ..physics.dlcz_kernel_adapter import kernel_provenance
from ..physics.dlcz_multi_cell_spinwave import (
    DlczMultiCellSpinwaveChannel,
    aod_analyzer_mode,
    calibrated_write_mode,
    unit_correlation_kernel,
)
from ..physics.dlcz_source_events import SourceEventLedger, sample_source_events
from ..physics.spinwave_event_forward import (
    SpinwaveEventState,
    propagate_spinwave_event,
)
from .aod_operation_schedule import Aod_operation_schedule
from .dlcz_observation_adapter import DlczObservation, adapt_readout_events


EVENT_RENDERER_SCHEMA = "V10DlczEventRendererV1"


def default_hypothesis_calibration_draw(n_mode: int) -> DlczCalibrationDraw:
    """Return an explicit zero-error placeholder for missing apparatus data."""
    draw = DlczCalibrationDraw(
        amplitude_error=(0.0,) * n_mode,
        phase_error=(0.0,) * n_mode,
        mode_overlap=(1.0,) * n_mode,
        crosstalk=(0.0,) * n_mode,
        status="HYPOTHESIS_UNBOUND",
    )
    draw.validate()
    return draw


def _calibration_changes_forward(draw: DlczCalibrationDraw) -> bool:
    """Report whether a draw changes any currently modelled forward factor."""

    return not (
        np.allclose(draw.amplitude_error, 0.0)
        and np.allclose(draw.phase_error, 0.0)
        and np.allclose(draw.mode_overlap, 1.0)
        and np.allclose(draw.crosstalk, 0.0)
    )


def _build_multi_cell_channel(
    *,
    schedule: Aod_operation_schedule,
    operation_index: int,
    mode_probability: np.ndarray,
    params: DlczOperationParameters,
    calibration_draw: DlczCalibrationDraw,
) -> DlczMultiCellSpinwaveChannel:
    """Construct one unbound multi-cell DLCZ read channel for an operation."""

    n_mode = int(mode_probability.size)
    write_mode = calibrated_write_mode(
        mode_probability,
        amplitude_error=calibration_draw.amplitude_error,
        phase_error=calibration_draw.phase_error,
        crosstalk=calibration_draw.crosstalk,
    )
    analyzer_mode = aod_analyzer_mode(
        schedule.amplitudes[operation_index], schedule.phases[operation_index]
    )
    if analyzer_mode.shape != (n_mode,):
        raise ValueError("AOD analyzer and multi-cell write mode dimensions differ")
    lifetime = max(float(params.storage_lifetime_s), np.finfo(float).tiny)
    base_survival = float(np.exp(-float(params.storage_wait_s) / lifetime))
    survival_by_cell = base_survival * np.asarray(
        calibration_draw.mode_overlap, dtype=np.float64
    )
    read_conversion_efficiency = (
        float(schedule.read_throughput[operation_index])
        * float(params.eta_retrieval)
        * float(params.w_overlap)
    )
    return DlczMultiCellSpinwaveChannel(
        write_mode=write_mode,
        analyzer_mode=analyzer_mode,
        survival_by_cell=survival_by_cell,
        storage_coherence_kernel=unit_correlation_kernel(n_mode),
        off_diagonal_visibility=float(params.spinwave_coherence_visibility),
        read_conversion_efficiency=read_conversion_efficiency,
        optical_path_efficiency=float(params.eta_optical_path),
    )


def _multi_cell_channel_sha(channel: DlczMultiCellSpinwaveChannel) -> str:
    """Hash channel arrays without serializing complex values into public JSON."""

    channel.validate()
    digest = hashlib.sha256()
    for value in (
        channel.normalized_write_mode(),
        channel.normalized_analyzer_mode(),
        channel.effective_storage_kernel(),
        np.asarray(channel.survival_by_cell, dtype=np.float64),
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(
        repr(
            (
                channel.off_diagonal_visibility,
                channel.read_conversion_efficiency,
                channel.optical_path_efficiency,
                channel.parameter_status,
            )
        ).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class DlczEventRenderResult:
    observations: tuple[DlczObservation, ...]
    source_ledgers: tuple[SourceEventLedger, ...]
    readout_ledgers: tuple[ReadoutEventLedger, ...]
    spinwave_states: tuple[SpinwaveEventState, ...]
    allocation: DlczAllocationReceipt
    metadata: dict

    def validate(self, *, n_mode: int) -> None:
        self.allocation.validate()
        if self.metadata.get("allocation_sha") != self.allocation.full_allocation_sha:
            raise ValueError("render metadata must bind the complete allocation hash")
        crn = self.metadata.get("crn_receipt")
        if not isinstance(crn, dict) or self.metadata.get(
            "crn_receipt_sha"
        ) != sha256_of(crn):
            raise ValueError("render metadata must bind a CRN receipt")
        required_crn_fields = {
            "sample_id",
            "global_seed",
            "split",
            "generator_sha",
            "group_id",
            "stream_namespace",
            "primitive_stream_names",
            "operation_parameter_sha",
            "calibration_sha",
            "domain_sha",
            "mode_probability_sha",
            "multi_cell_channel_sha",
            "kernel_provenance_sha",
            "stream_identity_sha",
        }
        if required_crn_fields - set(crn):
            raise ValueError("CRN receipt is missing primitive provenance")
        if crn["domain_sha"] != self.metadata.get("domain_sha"):
            raise ValueError("CRN receipt domain identity mismatch")
        if tuple(crn["operation_parameter_sha"]) != tuple(
            self.metadata.get("operation_parameter_sha", ())
        ):
            raise ValueError("render metadata operation parameters are not bound")
        if tuple(crn["calibration_sha"]) != tuple(
            self.metadata.get("calibration_sha", ())
        ):
            raise ValueError("render metadata calibration draws are not bound")
        if tuple(crn["multi_cell_channel_sha"]) != tuple(
            self.metadata.get("multi_cell_channel_sha", ())
        ):
            raise ValueError("render metadata multi-cell channel is not bound")
        if crn["kernel_provenance_sha"] != self.metadata.get("kernel_provenance_sha"):
            raise ValueError("render metadata kernel provenance is not bound")
        if len(self.observations) != len(self.allocation.operation_ids):
            raise ValueError("event render observations and allocation are misaligned")
        truth_ledger_retained = self.metadata.get("truth_ledger_retained")
        if type(truth_ledger_retained) is not bool:
            raise ValueError("render must declare truth-ledger retention")
        if truth_ledger_retained:
            if len(self.source_ledgers) != len(self.observations) or len(
                self.readout_ledgers
            ) != len(self.observations):
                raise ValueError("event render ledgers are misaligned")
        elif self.source_ledgers or self.readout_ledgers or self.spinwave_states:
            raise ValueError("stream-only render cannot retain truth ledgers or states")
        for index, observation in enumerate(self.observations):
            observation.validate(n_mode=n_mode)
            if observation.repeats != self.allocation.repeats_by_operation[index]:
                raise ValueError("observation repeats do not match allocation")


def build_dlcz_operation_parameters(
    schedule: Aod_operation_schedule,
    allocation: DlczAllocationReceipt,
    *,
    exposure_s: Sequence[float],
    lambda_pair: Sequence[float],
    write_chi: Sequence[float],
    background_signal_rate: Sequence[float],
    background_idler_rate: Sequence[float],
    eta_signal: Sequence[float],
    eta_retrieval: Sequence[float],
    eta_idler: Sequence[float],
    w_overlap: Sequence[float],
    dark_signal_rate: Sequence[float] | None = None,
    dark_idler_rate: Sequence[float] | None = None,
    classical_shared_rate: Sequence[float] | None = None,
    eta_optical_path: Sequence[float] | None = None,
    spinwave_coherence_visibility: Sequence[float] | None = None,
    source_law: str = "DLCZ_GEOMETRIC_PER_SHOT_V8",
    exposure_coordinate: str = "NORMALIZED_GATE_COORDINATE",
    parameter_status: str = "HYPOTHESIS_UNBOUND",
) -> tuple[DlczOperationParameters, ...]:
    """Bind operation-aligned rates to the registered schedule and budget."""

    operation_count = len(schedule.operation_ids)
    values = {
        "exposure_s": exposure_s,
        "lambda_pair": lambda_pair,
        "write_chi": write_chi,
        "background_signal_rate": background_signal_rate,
        "background_idler_rate": background_idler_rate,
        "eta_signal": eta_signal,
        "eta_retrieval": eta_retrieval,
        "eta_idler": eta_idler,
        "w_overlap": w_overlap,
    }
    defaults = {
        "dark_signal_rate": dark_signal_rate,
        "dark_idler_rate": dark_idler_rate,
        "classical_shared_rate": classical_shared_rate,
        "eta_optical_path": eta_optical_path,
        "spinwave_coherence_visibility": spinwave_coherence_visibility,
    }
    arrays: dict[str, np.ndarray] = {}
    for name, value in values.items():
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.shape != (operation_count,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite and operation-aligned")
        arrays[name] = array
    for name, value in defaults.items():
        default_value = (
            1.0
            if name in {"eta_optical_path", "spinwave_coherence_visibility"}
            else 0.0
        )
        arrays[name] = (
            np.full(operation_count, default_value, dtype=np.float64)
            if value is None
            else np.asarray(value, dtype=np.float64).reshape(-1)
        )
        if (
            arrays[name].shape != (operation_count,)
            or not np.isfinite(arrays[name]).all()
        ):
            raise ValueError(f"{name} must be finite and operation-aligned")
    if tuple(schedule.operation_ids) != allocation.operation_ids:
        raise ValueError("schedule and allocation operation identities differ")
    result = tuple(
        DlczOperationParameters(
            operation_id=operation_id,
            exposure_s=float(arrays["exposure_s"][index]),
            repeats=int(allocation.repeats_by_operation[index]),
            lambda_pair=float(arrays["lambda_pair"][index]),
            write_chi=float(arrays["write_chi"][index]),
            background_signal_rate=float(arrays["background_signal_rate"][index]),
            background_idler_rate=float(arrays["background_idler_rate"][index]),
            eta_signal=float(arrays["eta_signal"][index]),
            eta_retrieval=float(arrays["eta_retrieval"][index]),
            eta_idler=float(arrays["eta_idler"][index]),
            w_overlap=float(arrays["w_overlap"][index]),
            dark_signal_rate=float(arrays["dark_signal_rate"][index]),
            dark_idler_rate=float(arrays["dark_idler_rate"][index]),
            classical_shared_rate=float(arrays["classical_shared_rate"][index]),
            eta_optical_path=float(arrays["eta_optical_path"][index]),
            spinwave_coherence_visibility=float(
                arrays["spinwave_coherence_visibility"][index]
            ),
            source_law=source_law,
            exposure_coordinate=exposure_coordinate,
            parameter_status=parameter_status,
        )
        for index, operation_id in enumerate(schedule.operation_ids)
    )
    for item in result:
        item.validate()
    return result


def render_dlcz_event_row(
    *,
    schedule: Aod_operation_schedule,
    allocation: DlczAllocationReceipt,
    operation_parameters: Sequence[DlczOperationParameters],
    n_mode: int,
    mode_probability: np.ndarray,
    keyed_rng,
    group_id: str,
    domain: DlczDomainContract = DlczDomainContract(),
    arm: str = "PAIR",
    calibration_draws: Sequence[DlczCalibrationDraw] | None = None,
    rng_namespace: str | None = None,
    retain_truth_ledgers: bool = True,
) -> DlczEventRenderResult:
    """Render one latent row into source, readout and observation planes."""

    schedule.validate()
    if allocation.operation_registry_sha != schedule.sha256():
        raise ValueError("allocation must bind the exact operation registry SHA")
    if not allocation.full_allocation_sha:
        allocation = allocation.with_hash()
    allocation.validate()
    domain.validate()
    if tuple(item.operation_id for item in operation_parameters) != tuple(
        schedule.operation_ids
    ):
        raise ValueError(
            "operation parameters must follow the registered schedule order"
        )
    if len(operation_parameters) != len(schedule.operation_ids):
        raise ValueError("operation parameter count does not match schedule")
    if type(group_id) is not str or not group_id:
        raise ValueError("group_id is required")
    if type(retain_truth_ledgers) is not bool:
        raise ValueError("retain_truth_ledgers must be a strict Boolean")
    probability = np.asarray(mode_probability, dtype=np.float64).reshape(-1)
    if (
        probability.shape != (n_mode,)
        or not np.isfinite(probability).all()
        or np.any(probability < 0)
        or not np.isclose(probability.sum(), 1.0)
    ):
        raise ValueError("mode_probability must be normalized and n_mode-aligned")
    if not callable(getattr(keyed_rng, "stream", None)):
        raise ValueError("counter-based RNG is required")
    if calibration_draws is None:
        calibration = tuple(
            default_hypothesis_calibration_draw(n_mode) for _ in schedule.operation_ids
        )
    else:
        calibration = tuple(calibration_draws)
        if len(calibration) != len(schedule.operation_ids):
            raise ValueError(
                "calibration draws must align with the registered schedule"
            )
        for draw in calibration:
            draw.validate()
    observations: list[DlczObservation] = []
    source_ledgers: list[SourceEventLedger] = []
    readout_ledgers: list[ReadoutEventLedger] = []
    spin_states: list[SpinwaveEventState] = []
    source_ledger_shas: list[str] = []
    readout_ledger_shas: list[str] = []
    g2_domain_status: dict[str, str] = {}
    g2_reference_value: dict[str, float | None] = {}
    stream_namespace = arm if rng_namespace is None else str(rng_namespace)
    if not stream_namespace:
        raise ValueError("rng_namespace must be non-empty when supplied")
    calibration_effect_applied = False
    kernel_receipt = kernel_provenance()
    kernel_receipt_sha = sha256_of(kernel_receipt)
    operation_parameter_sha = tuple(item.sha256() for item in operation_parameters)
    calibration_sha = tuple(draw.sha256() for draw in calibration)
    mode_probability_sha = sha256_of(probability.tolist())
    multi_cell_channel_sha: list[str] = []
    selected_pre_splitter_probability: dict[str, float] = {}
    for index, params in enumerate(operation_parameters):
        params.validate()
        draw = calibration[index]
        if _calibration_changes_forward(draw):
            calibration_effect_applied = True
        # The event generator has a compound geometric source and finite time
        # gate, so the historical bivariate-Poisson intensity formula is not a
        # valid threshold-click reference for PAIR/CLASSICAL arms.  The fully
        # independent GSI generator has the exact threshold null g2=1; every
        # other reference remains unavailable until a registered forward is
        # supplied rather than being silently compared across domains.
        if arm == "GSI_NULL_GENERATIVE":
            g2_reference_value[params.operation_id] = 1.0
            g2_domain_status[params.operation_id] = "IN_DOMAIN"
        else:
            g2_reference_value[params.operation_id] = None
            g2_domain_status[params.operation_id] = (
                "NOT_AVAILABLE_COMPOUND_THRESHOLD_REFERENCE"
            )
        source_substream = f"{stream_namespace}:{params.operation_id}"
        source_rng = keyed_rng.stream("dlcz_source_shared", source_substream)
        independent_source_rng = keyed_rng.stream(
            "dlcz_source_independent_idler", source_substream
        )
        background_rng = keyed_rng.stream("dlcz_background", source_substream)
        source = sample_source_events(
            params=params,
            n_mode=n_mode,
            mode_probability=probability,
            rng=source_rng,
            opportunity_prefix=f"{group_id}:{params.operation_id}",
            arm=arm,
            calibration_draw=draw,
            independent_rng=independent_source_rng,
            background_rng=background_rng,
        )
        multi_cell_channel = _build_multi_cell_channel(
            schedule=schedule,
            operation_index=index,
            mode_probability=probability,
            params=params,
            calibration_draw=draw,
        )
        multi_cell_projection = multi_cell_channel.project_single_excitation()
        multi_cell_channel_sha.append(_multi_cell_channel_sha(multi_cell_channel))
        selected_pre_splitter_probability[params.operation_id] = float(
            multi_cell_projection.selected_pre_splitter_probability
        )
        spin_rng = keyed_rng.stream(
            "dlcz_spinwave", f"{stream_namespace}:{params.operation_id}"
        )
        states = tuple(
            propagate_spinwave_event(
                event,
                params,
                rng=spin_rng,
                multi_cell_channel=multi_cell_channel,
                multi_cell_projection=multi_cell_projection,
            )
            for event in source.events
            if event.channel in {"source", "idler"}
        )
        state_map = {state.event_id: state for state in states}
        readout = render_readout_events(
            source_events=source.events,
            spinwave_states=state_map,
            params=params,
            domain=domain,
            rng=keyed_rng.stream(
                "dlcz_readout", f"{stream_namespace}:{params.operation_id}"
            ),
        )
        observation = adapt_readout_events(readout, n_mode=n_mode)
        observations.append(observation)
        source_ledger_shas.append(source.sha256())
        readout_ledger_shas.append(readout.sha256())
        if retain_truth_ledgers:
            source_ledgers.append(source)
            readout_ledgers.append(readout)
            spin_states.extend(states)
    allocation_hash = (
        allocation.with_hash().full_allocation_sha
        if not allocation.full_allocation_sha
        else allocation.full_allocation_sha
    )
    global_multi_click_counts = tuple(
        int(
            sum(
                observation.global_multi_click_counts[index]
                for observation in observations
            )
        )
        for index in range(3)
    )
    global_multi2_pair_counts = tuple(
        int(
            sum(
                observation.global_multi2_pair_counts[index]
                for observation in observations
            )
        )
        for index in range(6)
    )
    if sum(global_multi2_pair_counts) != global_multi_click_counts[0]:
        raise RuntimeError("global exact-two SPAD pairs do not partition multi2")
    metadata = {
        "schema": EVENT_RENDERER_SCHEMA,
        "group_id": group_id,
        "arm": arm,
        "operation_ids": tuple(schedule.operation_ids),
        "operation_schedule_sha": schedule.sha256(),
        "allocation_sha": allocation_hash,
        "domain_sha": domain.sha256(),
        "source_event_ledger_sha": sha256_of(source_ledger_shas),
        "readout_event_ledger_sha": sha256_of(readout_ledger_shas),
        "truth_ledger_retained": retain_truth_ledgers,
        "student_latent_fields_forbidden": True,
        "research_only": True,
        "hardware_action_authorized": False,
        "parameter_status": tuple(
            item.parameter_status for item in operation_parameters
        ),
        "operation_parameter_sha": operation_parameter_sha,
        "calibration_status": tuple(draw.status for draw in calibration),
        "calibration_sha": calibration_sha,
        "calibration_effect_applied": calibration_effect_applied,
        "multi_cell_channel_sha": tuple(multi_cell_channel_sha),
        "selected_pre_splitter_probability": selected_pre_splitter_probability,
        "multi_cell_forward_status": "HYPOTHESIS_UNBOUND",
        "global_multi_click_counts": global_multi_click_counts,
        "global_multi2_pair_counts": global_multi2_pair_counts,
        "global_multi_scope": "NATIVE_CLOCK_GLOBAL_NOT_OPERATION_TARGET",
        "kernel_provenance": kernel_receipt,
        "kernel_provenance_sha": kernel_receipt_sha,
        "g2_reference_kind": {
            "PAIR": "NOT_AVAILABLE_GEOMETRIC_THRESHOLD_WITH_TIME_GATE",
            "GSI_NULL_GENERATIVE": "EXACT_INDEPENDENT_THRESHOLD_NULL",
            "CLASSICAL_SHARED_MATCHED": "NOT_AVAILABLE_GEOMETRIC_THRESHOLD_WITH_TIME_GATE",
        }[arm],
        "g2_reference_domain_status": g2_domain_status,
        "g2_reference_value": g2_reference_value,
        "g2_max": domain.g2_max,
        "out_of_scope_operations": tuple(
            operation_id
            for operation_id, status in g2_domain_status.items()
            if status.startswith("OUT_OF_SCOPE")
        ),
        "crn_receipt": {
            "schema": "V10DlczCrnReceiptV1",
            "sample_id": getattr(keyed_rng, "sample_id", None),
            "global_seed": getattr(keyed_rng, "global_seed", None),
            "split": getattr(keyed_rng, "split", None),
            "generator_sha": getattr(keyed_rng, "generator_sha", None),
            "group_id": group_id,
            "stream_namespace": stream_namespace,
            "primitive_stream_names": (
                "dlcz_source_shared",
                "dlcz_source_independent_idler",
                "dlcz_background",
                "dlcz_spinwave",
                "dlcz_readout",
            ),
            "operation_parameter_sha": operation_parameter_sha,
            "calibration_sha": calibration_sha,
            "domain_sha": domain.sha256(),
            "mode_probability_sha": mode_probability_sha,
            "multi_cell_channel_sha": tuple(multi_cell_channel_sha),
            "kernel_provenance_sha": kernel_receipt_sha,
            "stream_identity_sha": hashlib.sha256(
                f"{getattr(keyed_rng, 'sample_id', '')}|{getattr(keyed_rng, 'global_seed', '')}|{getattr(keyed_rng, 'split', '')}|{stream_namespace}|{schedule.sha256()}|{allocation_hash}".encode()
            ).hexdigest(),
        },
    }
    metadata["crn_receipt_sha"] = sha256_of(metadata["crn_receipt"])
    result = DlczEventRenderResult(
        tuple(observations),
        tuple(source_ledgers),
        tuple(readout_ledgers),
        tuple(spin_states),
        allocation,
        metadata,
    )
    result.validate(n_mode=n_mode)
    return result


def render_matched_dlcz_event_arms(
    *,
    schedule: Aod_operation_schedule,
    allocation: DlczAllocationReceipt,
    operation_parameters: Sequence[DlczOperationParameters],
    n_mode: int,
    mode_probability: np.ndarray,
    keyed_rng,
    group_id: str,
    domain: DlczDomainContract = DlczDomainContract(),
    retain_truth_ledgers: bool = True,
) -> dict[str, DlczEventRenderResult]:
    """Render pair, generative-null and classical shared matched arms."""
    pair = render_dlcz_event_row(
        schedule=schedule,
        allocation=allocation,
        operation_parameters=operation_parameters,
        n_mode=n_mode,
        mode_probability=mode_probability,
        keyed_rng=keyed_rng,
        group_id=group_id,
        domain=domain,
        arm="PAIR",
        rng_namespace="matched_common",
        retain_truth_ledgers=retain_truth_ledgers,
    )
    null = render_dlcz_event_row(
        schedule=schedule,
        allocation=allocation,
        operation_parameters=operation_parameters,
        n_mode=n_mode,
        mode_probability=mode_probability,
        keyed_rng=keyed_rng,
        group_id=group_id,
        domain=domain,
        arm="GSI_NULL_GENERATIVE",
        rng_namespace="matched_common",
        retain_truth_ledgers=retain_truth_ledgers,
    )
    classical = render_dlcz_event_row(
        schedule=schedule,
        allocation=allocation,
        operation_parameters=operation_parameters,
        n_mode=n_mode,
        mode_probability=mode_probability,
        keyed_rng=keyed_rng,
        group_id=group_id,
        domain=domain,
        arm="CLASSICAL_SHARED_MATCHED",
        rng_namespace="matched_common",
        retain_truth_ledgers=retain_truth_ledgers,
    )
    matched_receipt = {
        "schema": "V10DlczMatchedArmReceiptV1",
        "group_id": group_id,
        "arms": ("PAIR", "GSI_NULL_GENERATIVE", "CLASSICAL_SHARED_MATCHED"),
        "operation_schedule_sha": pair.metadata["operation_schedule_sha"],
        "allocation_sha": pair.metadata["allocation_sha"],
        "crn_receipt_sha": pair.metadata["crn_receipt_sha"],
        "operation_parameter_sha": pair.metadata["crn_receipt"][
            "operation_parameter_sha"
        ],
        "calibration_sha": pair.metadata["crn_receipt"]["calibration_sha"],
        "domain_sha": pair.metadata["crn_receipt"]["domain_sha"],
        "mode_probability_sha": pair.metadata["crn_receipt"]["mode_probability_sha"],
        "multi_cell_channel_sha": pair.metadata["crn_receipt"][
            "multi_cell_channel_sha"
        ],
        "kernel_provenance_sha": pair.metadata["crn_receipt"]["kernel_provenance_sha"],
        "marginal_rate_contract": "v8_geometric_per_shot_source_law_v1",
    }
    receipt_sha = sha256_of(matched_receipt)
    pair = replace(
        pair,
        metadata={
            **pair.metadata,
            "matched_arm_receipt": matched_receipt,
            "matched_arm_receipt_sha": receipt_sha,
        },
    )
    null = replace(
        null,
        metadata={
            **null.metadata,
            "matched_arm_receipt": matched_receipt,
            "matched_arm_receipt_sha": receipt_sha,
        },
    )
    classical = replace(
        classical,
        metadata={
            **classical.metadata,
            "matched_arm_receipt": matched_receipt,
            "matched_arm_receipt_sha": receipt_sha,
        },
    )
    if (
        pair.metadata["operation_schedule_sha"]
        != null.metadata["operation_schedule_sha"]
        or pair.metadata["allocation_sha"] != null.metadata["allocation_sha"]
    ):
        raise RuntimeError("matched arms lost schedule or allocation identity")
    if pair.metadata["crn_receipt_sha"] != null.metadata["crn_receipt_sha"]:
        raise RuntimeError("matched arms lost primitive-level CRN identity")
    if pair.metadata["crn_receipt_sha"] != classical.metadata["crn_receipt_sha"]:
        raise RuntimeError("matched arms lost classical primitive-level CRN identity")
    return {
        "PAIR": pair,
        "GSI_NULL_GENERATIVE": null,
        "CLASSICAL_SHARED_MATCHED": classical,
    }


__all__ = [
    "DlczEventRenderResult",
    "build_dlcz_operation_parameters",
    "default_hypothesis_calibration_draw",
    "render_matched_dlcz_event_arms",
    "render_dlcz_event_row",
]
