"""Operation-owned immutable spinwave snapshot; no global cache or RNG effects.

The existing public propagation function remains the reference implementation.
Only the renderer's already-validated, fixed per-operation channel is compiled.
Every event and every emitted state are still validated on the fast path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ..contracts.dlcz_event_mc import (
    DlczCalibrationDraw,
    DlczEvent,
    DlczOperationParameters,
)
from .dlcz_multi_cell_spinwave import DlczMultiCellSpinwaveChannel, SpinwaveProjection
from .spinwave_event_forward import (
    SAME_COLLECTIVE_MODE_FOCK_READOUT_HYPOTHESIS,
    TWO_PARTICLE_READOUT_NOT_APPLICABLE,
    SpinwaveEventState,
    _topology,
)


@dataclass(frozen=True, slots=True)
class _Complex_array_storage:
    data: bytes
    shape: tuple[int, ...]
    dtype: str

    @classmethod
    def capture(cls, value: np.ndarray) -> _Complex_array_storage:
        array = np.ascontiguousarray(value, dtype=np.complex128)
        return cls(array.tobytes(), tuple(array.shape), array.dtype.str)

    def view(self) -> np.ndarray:
        # A new header is essential: readonly buffers do not freeze ndarray
        # shape/dtype assignments on a previously returned view.
        return np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape)


@dataclass(frozen=True, slots=True)
class _Channel_storage:
    write_mode: _Complex_array_storage
    analyzer_mode: _Complex_array_storage
    survival_by_cell: tuple[float, ...]
    storage_coherence_kernel: _Complex_array_storage
    off_diagonal_visibility: float
    read_conversion_efficiency: float
    optical_path_efficiency: float
    parameter_status: str

    @classmethod
    def capture(cls, channel: DlczMultiCellSpinwaveChannel) -> _Channel_storage:
        return cls(
            _Complex_array_storage.capture(channel.write_mode),
            _Complex_array_storage.capture(channel.analyzer_mode),
            tuple(float(value) for value in channel.survival_by_cell),
            _Complex_array_storage.capture(channel.storage_coherence_kernel),
            float(channel.off_diagonal_visibility),
            float(channel.read_conversion_efficiency),
            float(channel.optical_path_efficiency),
            channel.parameter_status,
        )

    def value(self) -> DlczMultiCellSpinwaveChannel:
        return DlczMultiCellSpinwaveChannel(
            self.write_mode.view(),
            self.analyzer_mode.view(),
            self.survival_by_cell,
            self.storage_coherence_kernel.view(),
            self.off_diagonal_visibility,
            self.read_conversion_efficiency,
            self.optical_path_efficiency,
            self.parameter_status,
        )


@dataclass(frozen=True, slots=True)
class _Projection_storage:
    density_matrix: _Complex_array_storage
    storage_kernel: _Complex_array_storage
    selected_mode_probability: float
    selected_pre_splitter_probability: float
    status: str

    @classmethod
    def capture(cls, projection: SpinwaveProjection) -> _Projection_storage:
        return cls(
            _Complex_array_storage.capture(projection.density_matrix),
            _Complex_array_storage.capture(projection.storage_kernel),
            projection.selected_mode_probability,
            projection.selected_pre_splitter_probability,
            projection.status,
        )

    def value(self) -> SpinwaveProjection:
        return SpinwaveProjection(
            self.density_matrix.view(),
            self.storage_kernel.view(),
            self.selected_mode_probability,
            self.selected_pre_splitter_probability,
            self.status,
        )


def _complex_payload(value: np.ndarray) -> dict:
    return {"real": value.real.tolist(), "imag": value.imag.tolist()}


@dataclass(frozen=True, init=False)
class Prepared_spinwave_channel:
    """Validated numeric values, detached from mutable channel inputs.

    Construction performs the original parameter/channel/projection checks.
    Callers cannot pass a prevalidated flag, a free projection, or a state
    template: the only initializer derives them from the checked inputs.
    """

    n_cell: int
    operation_id: str
    opportunity_prefix: str
    repeats: int
    _channel_storage: _Channel_storage = field(repr=False)
    _projection_storage: _Projection_storage = field(repr=False)
    parameter_sha: str
    calibration_sha: str
    operation_registry_sha: str
    binding_sha: str
    state_template: SpinwaveEventState

    @property
    def channel(self) -> DlczMultiCellSpinwaveChannel:
        return self._channel_storage.value()

    @property
    def projection(self) -> SpinwaveProjection:
        return self._projection_storage.value()

    def contains_opportunity(self, opportunity_id: str) -> bool:
        if not opportunity_id.startswith(self.opportunity_prefix):
            return False
        suffix = opportunity_id[len(self.opportunity_prefix) :]
        if not suffix or not suffix.isascii() or not suffix.isdecimal():
            return False
        if (len(suffix) > 1 and suffix.startswith("0")) or len(suffix) > len(
            str(self.repeats)
        ):
            return False
        return int(suffix) < self.repeats

    def __init__(
        self,
        params: DlczOperationParameters,
        channel: DlczMultiCellSpinwaveChannel,
        *,
        calibration_draw: DlczCalibrationDraw,
        operation_registry_sha: str,
        opportunity_prefix: str,
    ) -> None:
        if (
            type(params) is not DlczOperationParameters
            or type(channel) is not DlczMultiCellSpinwaveChannel
            or type(calibration_draw) is not DlczCalibrationDraw
        ):
            raise ValueError(
                "Prepared propagation requires concrete parameter/channel contracts"
            )
        params.validate()
        channel.validate()
        calibration_draw.validate()
        if (
            not isinstance(operation_registry_sha, str)
            or len(operation_registry_sha) != 64
            or any(c not in "0123456789abcdef" for c in operation_registry_sha)
        ):
            raise ValueError(
                "Prepared channel requires its operation registry identity"
            )
        if not isinstance(opportunity_prefix, str) or not opportunity_prefix:
            raise ValueError(
                "Prepared channel requires its operation opportunity scope"
            )
        channel_storage = _Channel_storage.capture(channel)
        frozen_channel = channel_storage.value()
        projection = frozen_channel.project_single_excitation()
        projection_storage = _Projection_storage.capture(projection)
        frozen_projection = projection_storage.value()
        n_cell = len(frozen_channel.write_mode)
        if len(calibration_draw.amplitude_error) != n_cell:
            raise ValueError("Calibration draw differs from prepared channel dimension")
        survival_by_cell = tuple(
            float(value.real) for value in np.diag(frozen_projection.storage_kernel)
        )
        visibility = frozen_channel.off_diagonal_visibility
        template = SpinwaveEventState(
            event_id="prepared_template",
            occupancy=(0,) * n_cell,
            topology="NONE",
            phase_rad=0.0,
            wavevector=(),
            storage_wait_s=float(params.storage_wait_s),
            survival_probability=(
                float(np.mean(survival_by_cell)) if survival_by_cell else 1.0
            ),
            dephasing_factor=visibility,
            retrieval_overlap=frozen_projection.selected_mode_probability,
            parameter_status="HYPOTHESIS_UNBOUND",
            survival_by_cell=survival_by_cell,
            dephasing_by_cell=(visibility,) * n_cell,
            evolution_kernel="MULTI_CELL_PSD_AOD_UNBOUND",
            selected_pre_splitter_probability=frozen_projection.selected_pre_splitter_probability,
            coherent_sector_kind="NONE",
            multi_particle_readout_status=TWO_PARTICLE_READOUT_NOT_APPLICABLE,
        )
        template.validate()
        parameter_sha, calibration_sha = params.sha256(), calibration_draw.sha256()
        binding_sha = sha256_of(
            {
                "schema": "PreparedSpinwaveChannelV1",
                "operation_id": params.operation_id,
                "opportunity_prefix": opportunity_prefix,
                "repeats": params.repeats,
                "n_cell": n_cell,
                "operation_registry_sha": operation_registry_sha,
                "parameter_sha": parameter_sha,
                "calibration_sha": calibration_sha,
                "write_mode": _complex_payload(frozen_channel.write_mode),
                "analyzer_mode": _complex_payload(frozen_channel.analyzer_mode),
                "survival_by_cell": frozen_channel.survival_by_cell,
                "storage_coherence_kernel": _complex_payload(
                    frozen_channel.storage_coherence_kernel
                ),
                "off_diagonal_visibility": visibility,
                "read_conversion_efficiency": frozen_channel.read_conversion_efficiency,
                "optical_path_efficiency": frozen_channel.optical_path_efficiency,
                "density_matrix": _complex_payload(frozen_projection.density_matrix),
                "storage_kernel": _complex_payload(frozen_projection.storage_kernel),
                "selected_mode_probability": frozen_projection.selected_mode_probability,
                "selected_pre_splitter_probability": frozen_projection.selected_pre_splitter_probability,
            }
        )
        object.__setattr__(self, "n_cell", n_cell)
        object.__setattr__(self, "operation_id", params.operation_id)
        object.__setattr__(self, "opportunity_prefix", opportunity_prefix)
        object.__setattr__(self, "repeats", params.repeats)
        object.__setattr__(self, "_channel_storage", channel_storage)
        object.__setattr__(self, "_projection_storage", projection_storage)
        object.__setattr__(self, "parameter_sha", parameter_sha)
        object.__setattr__(self, "calibration_sha", calibration_sha)
        object.__setattr__(self, "operation_registry_sha", operation_registry_sha)
        object.__setattr__(self, "binding_sha", binding_sha)
        object.__setattr__(self, "state_template", template)


def propagate_prepared_spinwave_event(
    event: DlczEvent,
    prepared: Prepared_spinwave_channel,
    *,
    rng: np.random.Generator,
) -> SpinwaveEventState:
    """Reuse one checked snapshot while preserving event order and RNG calls."""
    if type(prepared) is not Prepared_spinwave_channel:
        raise ValueError("A prepared spinwave snapshot is required")
    event.validate(n_mode=prepared.n_cell)
    if (
        event.operation_id != prepared.operation_id
        or not prepared.contains_opportunity(event.opportunity_id)
    ):
        raise ValueError(
            "Event belongs to a different prepared operation/opportunity scope"
        )
    topology = _topology(event)
    # Even vacuum/zero-width phase draws consume the exact legacy RNG call.
    phase = float(event.phase_rad + rng.normal(scale=0.0))
    state = replace(
        prepared.state_template,
        event_id=event.event_id,
        occupancy=event.occupancy,
        topology=topology,
        phase_rad=phase,
        wavevector=event.wavevector,
        coherent_sector_kind=(
            "NONE" if topology == "NONE" else "SAME_COLLECTIVE_MODE_FOCK_HYPOTHESIS"
        ),
        multi_particle_readout_status=(
            SAME_COLLECTIVE_MODE_FOCK_READOUT_HYPOTHESIS
            if sum(event.occupancy) >= 2
            else TWO_PARTICLE_READOUT_NOT_APPLICABLE
        ),
    )
    state.validate()
    return state


__all__ = ["Prepared_spinwave_channel", "propagate_prepared_spinwave_event"]
