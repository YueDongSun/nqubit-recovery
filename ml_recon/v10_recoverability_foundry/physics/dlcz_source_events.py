"""Event-level DLCZ source and background sampling.

This module samples actual opportunities and source marks.  It intentionally
does not expose latent marks to the observation adapter or student inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Sequence

import numpy as np

from ..contracts.dlcz_event_mc import (
    DlczCalibrationDraw,
    DlczEvent,
    DlczOperationParameters,
    source_event_mark_for_occupancy,
)
from .dlcz_kernel_adapter import (
    sample_geometric_count,
    sample_single_cell,
    sample_two_boson_occupancy,
)
from .dlcz_multi_cell_spinwave import calibrated_write_mode


SOURCE_EVENT_SCHEMA = "V10DlczSourceEventLedgerV1"
SOURCE_ARMS = ("PAIR", "GSI_NULL_GENERATIVE", "CLASSICAL_SHARED_MATCHED")


def _event_id(opportunity_id: str, index: int, mark: str) -> str:
    digest = hashlib.sha256(f"{opportunity_id}|{index}|{mark}".encode()).hexdigest()[
        :16
    ]
    return f"{opportunity_id}:event:{index}:{digest}"


def _empty_occupancy(n_mode: int) -> tuple[int, ...]:
    return (0,) * n_mode


def _sample_occupancy(
    rng: np.random.Generator,
    *,
    n_mode: int,
    mode_probability: np.ndarray,
    excitation_count: int,
    local_double_fraction: float = 0.0,
    cross_double_fraction: float = 0.0,
) -> tuple[int, ...]:
    if excitation_count == 1:
        occupancy = np.zeros(n_mode, dtype=np.int64)
        occupancy[sample_single_cell(rng, mode_probability)] = 1
        return tuple(int(value) for value in occupancy)
    if excitation_count == 2:
        if local_double_fraction or cross_double_fraction:
            if not np.isclose(local_double_fraction + cross_double_fraction, 1.0):
                raise ValueError("local/cross double fractions must sum to one")
            if rng.random() < local_double_fraction:
                weights = np.square(mode_probability)
                cell = int(rng.choice(n_mode, p=weights / weights.sum()))
                occupancy = np.zeros(n_mode, dtype=np.int64)
                occupancy[cell] = 2
                return tuple(int(value) for value in occupancy)
            pairs = [
                (left, right)
                for left in range(n_mode)
                for right in range(left + 1, n_mode)
            ]
            weights = np.asarray(
                [
                    2.0 * mode_probability[left] * mode_probability[right]
                    for left, right in pairs
                ],
                dtype=np.float64,
            )
            left, right = pairs[int(rng.choice(len(pairs), p=weights / weights.sum()))]
            occupancy = np.zeros(n_mode, dtype=np.int64)
            occupancy[[left, right]] = 1
            return tuple(int(value) for value in occupancy)
        return sample_two_boson_occupancy(rng, mode_probability)
    draws = rng.choice(n_mode, size=excitation_count, p=mode_probability)
    occupancy = np.bincount(draws, minlength=n_mode).astype(np.int64)
    return tuple(int(value) for value in occupancy)


def materialize_source_event(
    *,
    opportunity_id: str,
    operation_id: str,
    event_index: int,
    mark: str,
    occupancy: tuple[int, ...],
    rng: np.random.Generator,
    channel: str = "source",
    phase_offset: float = 0.0,
) -> DlczEvent:
    """Construct canonical source/background metadata from an explicit mark."""
    event = DlczEvent(
        event_id=_event_id(opportunity_id, event_index, mark),
        opportunity_id=opportunity_id,
        operation_id=operation_id,
        mark=mark,
        channel=channel,
        cell_indices=tuple(index for index, value in enumerate(occupancy) if value),
        occupancy=occupancy,
        time_s=float(rng.random()),
        phase_rad=float(rng.uniform(-math.pi, math.pi) + phase_offset),
        wavevector=tuple(float(value) for value in rng.normal(size=3)),
        detected=False,
    )
    event.validate(n_mode=len(occupancy))
    return event


# Preserve the historical private name for existing diagnostic callers.
_source_event = materialize_source_event


@dataclass(frozen=True)
class SourceEventLedger:
    events: tuple[DlczEvent, ...]
    opportunity_count: int
    operation_id: str
    arm: str
    n_mode: int

    def validate(self) -> None:
        if type(self.opportunity_count) is not int or self.opportunity_count < 0:
            raise ValueError("opportunity_count must be a non-negative integer")
        if self.arm not in SOURCE_ARMS:
            raise ValueError("unregistered source arm")
        if type(self.n_mode) is not int or self.n_mode < 3:
            raise ValueError("n_mode must be at least three")
        seen: set[str] = set()
        for event in self.events:
            event.validate(n_mode=self.n_mode)
            if event.event_id in seen:
                raise ValueError("source event identities must be unique")
            seen.add(event.event_id)

    @property
    def source_event_count(self) -> int:
        return sum(event.channel == "source" for event in self.events)

    def sha256(self) -> str:
        self.validate()
        payload = {
            "schema": SOURCE_EVENT_SCHEMA,
            "operation_id": self.operation_id,
            "arm": self.arm,
            "n_mode": self.n_mode,
            "opportunity_count": self.opportunity_count,
            "events": [event.to_dict() for event in self.events],
        }
        return hashlib.sha256(
            repr(sorted(payload.items(), key=lambda item: item[0])).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class Prepared_dlcz_source:
    """Validated, immutable deterministic setup for repeated source draws."""

    params: DlczOperationParameters
    n_mode: int
    arm: str
    calibration_draw: DlczCalibrationDraw
    mode_probability: tuple[float, ...]
    calibrated_probability: tuple[float, ...]
    phase_offsets: tuple[float, ...]
    background_specs: tuple[tuple[float, str, str], ...]
    repeats: int

    def __init__(
        self,
        params: DlczOperationParameters,
        n_mode: int,
        mode_probability: Sequence[float],
        arm: str = "PAIR",
        calibration_draw: DlczCalibrationDraw | None = None,
    ) -> None:
        params.validate()
        if arm not in SOURCE_ARMS:
            raise ValueError(f"unregistered source arm: {arm}")
        if type(n_mode) is not int or n_mode < 3:
            raise ValueError("n_mode must be at least three")
        probability = np.asarray(mode_probability, dtype=np.float64).reshape(-1)
        if (
            probability.shape != (n_mode,)
            or not np.isfinite(probability).all()
            or np.any(probability < 0)
            or not np.isclose(probability.sum(), 1.0)
        ):
            raise ValueError("mode_probability must be a finite normalized vector")
        if calibration_draw is None:
            calibration_draw = DlczCalibrationDraw(
                amplitude_error=(0.0,) * n_mode,
                phase_error=(0.0,) * n_mode,
                mode_overlap=(1.0,) * n_mode,
                crosstalk=(0.0,) * n_mode,
            )
        calibration_draw.validate()
        if len(calibration_draw.amplitude_error) != n_mode:
            raise ValueError("calibration draw must match n_mode")
        calibration_draw = replace(
            calibration_draw,
            amplitude_error=tuple(float(value) for value in calibration_draw.amplitude_error),
            phase_error=tuple(float(value) for value in calibration_draw.phase_error),
            mode_overlap=tuple(float(value) for value in calibration_draw.mode_overlap),
            crosstalk=tuple(float(value) for value in calibration_draw.crosstalk),
        )
        write_mode = calibrated_write_mode(
            probability,
            amplitude_error=calibration_draw.amplitude_error,
            phase_error=calibration_draw.phase_error,
            crosstalk=calibration_draw.crosstalk,
        )
        calibrated_probability = np.square(np.abs(write_mode))
        phase_offsets = np.angle(write_mode)
        if arm == "CLASSICAL_SHARED_MATCHED" and params.repeats > 0:
            params.effective_classical_shared_rate
        background_specs = (
            (
                params.background_signal_mean_per_shot,
                "UNCORRELATED_SIGNAL_BACKGROUND",
                "signal",
            ),
            (
                params.background_idler_mean_per_shot,
                "UNCORRELATED_IDLER_BACKGROUND",
                "idler",
            ),
            (params.dark_signal_mean_per_shot, "SIGNAL_DARK", "signal"),
            (params.dark_idler_mean_per_shot, "IDLER_DARK", "idler"),
        )
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "n_mode", n_mode)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "calibration_draw", calibration_draw)
        object.__setattr__(
            self, "mode_probability", tuple(float(value) for value in probability)
        )
        object.__setattr__(
            self,
            "calibrated_probability",
            tuple(float(value) for value in calibrated_probability),
        )
        object.__setattr__(
            self, "phase_offsets", tuple(float(value) for value in phase_offsets)
        )
        object.__setattr__(self, "background_specs", background_specs)
        object.__setattr__(self, "repeats", int(params.repeats))

    @property
    def operation_id(self) -> str:
        return self.params.operation_id


def materialize_positive_source_event(
    prepared_source: Prepared_dlcz_source, *, excitation_count: int,
    opportunity_id: str, event_index: int, rng: np.random.Generator,
    channel: str = "source", classical_mark: bool = False,
) -> DlczEvent:
    """Realize the canonical mark and metadata after a positive sector draw."""
    if type(prepared_source) is not Prepared_dlcz_source:
        raise ValueError("A prepared DLCZ source snapshot is required")
    if type(excitation_count) is not int or excitation_count < 1:
        raise ValueError("A positive integer source sector is required")
    params = prepared_source.params
    occupancy = _sample_occupancy(
        rng, n_mode=prepared_source.n_mode,
        mode_probability=np.asarray(prepared_source.calibrated_probability),
        excitation_count=excitation_count,
        local_double_fraction=params.local_double_fraction,
        cross_double_fraction=params.cross_double_fraction,
    )
    mark = "CLASSICAL_SHARED_SHOCK" if classical_mark else source_event_mark_for_occupancy(occupancy)
    phase_offset = float(np.average(
        np.asarray(prepared_source.phase_offsets), weights=np.asarray(occupancy, dtype=np.float64)))
    return materialize_source_event(
        opportunity_id=opportunity_id, operation_id=params.operation_id,
        event_index=event_index, mark=mark, occupancy=occupancy, rng=rng,
        channel=channel, phase_offset=phase_offset,
    )


def sample_prepared_source_events(
    prepared_source: Prepared_dlcz_source,
    *,
    rng: np.random.Generator,
    opportunity_prefix: str,
    independent_rng: np.random.Generator | None = None,
    background_rng: np.random.Generator | None = None,
) -> SourceEventLedger:
    """Sample the canonical per-shot law from checked deterministic setup.

    PAIR samples one V8 geometric DLCZ Fock sector per write opportunity,
    without a Poisson cluster layer. GSI_NULL_GENERATIVE uses independent
    copies for the signal and idler source arms. The classical control keeps
    that marginal law and changes only a truth-private mark. Every scalar RNG
    call, vacuum event and independent background remains in the same order.
    """

    if type(prepared_source) is not Prepared_dlcz_source:
        raise ValueError("A prepared DLCZ source snapshot is required")
    if type(opportunity_prefix) is not str or not opportunity_prefix:
        raise ValueError("opportunity_prefix is required")
    params = prepared_source.params
    n_mode = prepared_source.n_mode
    arm = prepared_source.arm
    repeats = prepared_source.repeats
    events: list[DlczEvent] = []
    event_index = 0
    independent_source_rng = rng if independent_rng is None else independent_rng
    optical_background_rng = rng if background_rng is None else background_rng

    def append_source(
        opportunity_id: str,
        mark: str,
        occupancy: tuple[int, ...],
        *,
        channel: str = "source",
        phase_offset: float = 0.0,
        event_rng: np.random.Generator | None = None,
    ) -> None:
        nonlocal event_index
        events.append(
            materialize_source_event(
                opportunity_id=opportunity_id,
                operation_id=params.operation_id,
                event_index=event_index,
                mark=mark,
                occupancy=occupancy,
                rng=rng if event_rng is None else event_rng,
                channel=channel,
                phase_offset=phase_offset,
            )
        )
        event_index += 1

    def append_geometric_source_event(
        opportunity_id: str,
        *,
        channel: str,
        sample_rng: np.random.Generator,
        classical_mark: bool,
    ) -> bool:
        """Sample one geometric DLCZ sector for one physical write shot."""

        excitation_count = sample_geometric_count(sample_rng, params.write_chi)
        if excitation_count == 0:
            return False
        nonlocal event_index
        events.append(materialize_positive_source_event(
            prepared_source, excitation_count=excitation_count,
            opportunity_id=opportunity_id, event_index=event_index, rng=sample_rng,
            channel=channel, classical_mark=classical_mark))
        event_index += 1
        return True

    for opportunity in range(repeats):
        opportunity_id = f"{opportunity_prefix}:opportunity:{opportunity}"
        source_added = False
        if arm == "GSI_NULL_GENERATIVE":
            signal_added = append_geometric_source_event(
                opportunity_id,
                channel="signal",
                sample_rng=rng,
                classical_mark=False,
            )
            if not signal_added:
                # Preserve the shared signal primitive stream even when the
                # independent idler branch happens to emit a sector.
                append_source(
                    opportunity_id,
                    "NO_SOURCE_EVENT",
                    _empty_occupancy(n_mode),
                    channel="signal",
                    event_rng=rng,
                )
            idler_added = append_geometric_source_event(
                opportunity_id,
                channel="idler",
                sample_rng=independent_source_rng,
                classical_mark=False,
            )
            if not idler_added:
                append_source(
                    opportunity_id,
                    "NO_SOURCE_EVENT",
                    _empty_occupancy(n_mode),
                    channel="idler",
                    event_rng=independent_source_rng,
                )
            source_added = signal_added or idler_added
        else:
            source_added = append_geometric_source_event(
                opportunity_id,
                channel="source",
                sample_rng=rng,
                classical_mark=arm == "CLASSICAL_SHARED_MATCHED",
            )

        if not source_added and arm != "GSI_NULL_GENERATIVE":
            append_source(opportunity_id, "NO_SOURCE_EVENT", _empty_occupancy(n_mode))

        # Independent optical backgrounds and dark events are separate marks;
        # they can coexist with source events in the same opportunity.
        for mean_count, mark, channel in prepared_source.background_specs:
            count = int(optical_background_rng.poisson(mean_count))
            for _ in range(count):
                append_source(
                    opportunity_id,
                    mark,
                    _empty_occupancy(n_mode),
                    channel=channel,
                    event_rng=optical_background_rng,
                )

    ledger = SourceEventLedger(tuple(events), repeats, params.operation_id, arm, n_mode)
    ledger.validate()
    return ledger


def sample_source_events(
    *,
    params: DlczOperationParameters,
    n_mode: int,
    mode_probability: Sequence[float],
    rng: np.random.Generator,
    opportunity_prefix: str,
    arm: str = "PAIR",
    calibration_draw: DlczCalibrationDraw | None = None,
    independent_rng: np.random.Generator | None = None,
    background_rng: np.random.Generator | None = None,
) -> SourceEventLedger:
    """Validate a source request and enter the canonical prepared source body."""
    prepared_source = Prepared_dlcz_source(
        params,
        n_mode,
        mode_probability,
        arm=arm,
        calibration_draw=calibration_draw,
    )
    return sample_prepared_source_events(
        prepared_source,
        rng=rng,
        opportunity_prefix=opportunity_prefix,
        independent_rng=independent_rng,
        background_rng=background_rng,
    )


def events_for_opportunity(
    ledger: SourceEventLedger, opportunity_id: str
) -> tuple[DlczEvent, ...]:
    ledger.validate()
    if not opportunity_id:
        raise ValueError("opportunity_id is required")
    return tuple(
        event for event in ledger.events if event.opportunity_id == opportunity_id
    )


__all__ = [
    "Prepared_dlcz_source",
    "SOURCE_ARMS",
    "SourceEventLedger",
    "events_for_opportunity",
    "materialize_positive_source_event",
    "materialize_source_event",
    "sample_prepared_source_events",
    "sample_source_events",
]
