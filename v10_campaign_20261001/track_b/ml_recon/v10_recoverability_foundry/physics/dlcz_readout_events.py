"""Readout, loss and detector event realization for the DLCZ diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np

from ..contracts.dlcz_event_mc import (
    DlczDomainContract,
    DlczEvent,
    DlczOperationParameters,
)
from .spinwave_event_forward import (
    TWO_PARTICLE_READOUT_DEFERRED,
    SpinwaveEventState,
)


READOUT_SCHEMA = "V10DlczReadoutEventLedgerV1"


READOUT_PARAMETER_STATUS = "HYPOTHESIS_UNBOUND"
SAME_COLLECTIVE_MODE_FOCK_READOUT = "SAME_COLLECTIVE_MODE_LINEAR_OPTICS_FOCK_HYPOTHESIS"


@dataclass(frozen=True)
class ReadoutEventLedger:
    candidates: tuple[DlczEvent, ...]
    detected_events: tuple[DlczEvent, ...]
    operation_id: str
    opportunity_count: int
    gate_window_s: float
    dead_time_s: float
    dead_time_mode: str
    opportunity_ids: tuple[str, ...] = ()
    gate_rejected_count: int = 0
    dead_time_rejected_count: int = 0
    read_clock_period_s: float = 10.0e-9
    read_clock_count: int = 19
    deferred_multi_particle_readout_count: int = 0
    parameter_status: str = READOUT_PARAMETER_STATUS

    def validate(self, *, n_mode: int | None = None) -> None:
        if type(self.opportunity_count) is not int or self.opportunity_count < 0:
            raise ValueError("opportunity_count must be a non-negative integer")
        if not math.isfinite(float(self.gate_window_s)) or self.gate_window_s < 0:
            raise ValueError("gate_window_s must be finite and non-negative")
        if not math.isfinite(float(self.dead_time_s)) or self.dead_time_s < 0:
            raise ValueError("dead_time_s must be finite and non-negative")
        if self.opportunity_ids:
            if (
                len(self.opportunity_ids) != self.opportunity_count
                or len(set(self.opportunity_ids)) != len(self.opportunity_ids)
                or any(
                    not isinstance(value, str) or not value
                    for value in self.opportunity_ids
                )
            ):
                raise ValueError(
                    "opportunity_ids must be unique and complete when present"
                )
        if type(self.gate_rejected_count) is not int or self.gate_rejected_count < 0:
            raise ValueError("gate_rejected_count must be a non-negative integer")
        if (
            type(self.dead_time_rejected_count) is not int
            or self.dead_time_rejected_count < 0
        ):
            raise ValueError("dead_time_rejected_count must be a non-negative integer")
        if (
            not math.isfinite(float(self.read_clock_period_s))
            or self.read_clock_period_s <= 0.0
            or type(self.read_clock_count) is not int
            or self.read_clock_count < 1
        ):
            raise ValueError("read clock contract must be finite and positive")
        if (
            type(self.deferred_multi_particle_readout_count) is not int
            or self.deferred_multi_particle_readout_count < 0
        ):
            raise ValueError(
                "deferred_multi_particle_readout_count must be a non-negative integer"
            )
        if self.parameter_status != READOUT_PARAMETER_STATUS:
            raise ValueError("readout parameters must remain HYPOTHESIS_UNBOUND")
        seen: set[str] = set()
        for event in self.candidates + self.detected_events:
            event.validate(n_mode=n_mode)
            if event.event_id in seen and event not in self.detected_events:
                raise ValueError("readout event identities must be unique")
            seen.add(event.event_id)
        candidate_ids = {event.event_id for event in self.candidates}
        if any(event.event_id not in candidate_ids for event in self.detected_events):
            raise ValueError("detected events must originate from candidates")
        if any(not event.detected for event in self.detected_events):
            raise ValueError("detected event ledger contains an undetected event")
        if self.opportunity_ids and any(
            event.opportunity_id not in self.opportunity_ids
            for event in self.candidates
        ):
            raise ValueError(
                "candidate event lies outside the registered opportunities"
            )

    def sha256(self) -> str:
        self.validate()
        payload = {
            "schema": READOUT_SCHEMA,
            "operation_id": self.operation_id,
            "opportunity_count": self.opportunity_count,
            "gate_window_s": self.gate_window_s,
            "dead_time_s": self.dead_time_s,
            "dead_time_mode": self.dead_time_mode,
            "opportunity_ids": self.opportunity_ids,
            "gate_rejected_count": self.gate_rejected_count,
            "dead_time_rejected_count": self.dead_time_rejected_count,
            "read_clock_period_s": self.read_clock_period_s,
            "read_clock_count": self.read_clock_count,
            "deferred_multi_particle_readout_count": self.deferred_multi_particle_readout_count,
            "parameter_status": self.parameter_status,
            "candidates": [event.to_dict() for event in self.candidates],
            "detected_events": [event.to_dict() for event in self.detected_events],
        }
        return hashlib.sha256(repr(payload).encode()).hexdigest()


def _candidate(
    source: DlczEvent,
    *,
    channel: str,
    mark: str,
    time_s: float,
    detector_id: str,
    suffix: str = "0",
) -> DlczEvent:
    return DlczEvent(
        event_id=f"{source.event_id}:{channel}:{suffix}",
        opportunity_id=source.opportunity_id,
        operation_id=source.operation_id,
        mark=mark,
        channel=channel,
        cell_indices=source.cell_indices,
        occupancy=source.occupancy,
        time_s=time_s,
        phase_rad=source.phase_rad,
        wavevector=source.wavevector,
        detected=False,
        detector_id=detector_id,
    )


def _apply_dead_time(
    events: list[DlczEvent], *, dead_time_s: float, mode: str
) -> list[DlczEvent]:
    if mode == "none" or dead_time_s <= 0.0:
        return events
    accepted: list[DlczEvent] = []
    busy_until = -math.inf
    for event in sorted(events, key=lambda item: (item.time_s, item.event_id)):
        if event.time_s < busy_until:
            if mode == "paralyzable":
                busy_until = event.time_s + dead_time_s
            continue
        accepted.append(event)
        busy_until = event.time_s + dead_time_s
    return accepted


def _apply_dead_time_by_opportunity(
    events: list[DlczEvent], *, dead_time_s: float, mode: str
) -> tuple[list[DlczEvent], int]:
    """Apply detector dead-time independently inside each opportunity gate."""
    grouped: dict[str, list[DlczEvent]] = {}
    for event in events:
        grouped.setdefault(event.opportunity_id, []).append(event)
    accepted: list[DlczEvent] = []
    rejected = 0
    for opportunity_events in grouped.values():
        selected = _apply_dead_time(
            opportunity_events, dead_time_s=dead_time_s, mode=mode
        )
        accepted.extend(selected)
        rejected += len(opportunity_events) - len(selected)
    return accepted, rejected


def _apply_detector_dead_time_by_opportunity(
    events: list[DlczEvent], *, dead_time_s: float, mode: str
) -> tuple[list[DlczEvent], int]:
    """Apply dead time per physical detector within every FPGA gate.

    Read SPADs are distinct on/off devices after the fixed splitter.  Applying
    one shared idler dead time would incorrectly erase valid same-clock
    multi-clicks from different SPADs.
    """

    grouped: dict[tuple[str, str, str], list[DlczEvent]] = {}
    for event in events:
        detector_id = event.detector_id or event.channel
        grouped.setdefault(
            (event.opportunity_id, event.channel, detector_id), []
        ).append(event)
    accepted: list[DlczEvent] = []
    rejected = 0
    for detector_events in grouped.values():
        selected = _apply_dead_time(detector_events, dead_time_s=dead_time_s, mode=mode)
        accepted.extend(selected)
        rejected += len(detector_events) - len(selected)
    return accepted, rejected


def render_readout_events(
    *,
    source_events: tuple[DlczEvent, ...],
    spinwave_states: Mapping[str, SpinwaveEventState],
    params: DlczOperationParameters,
    domain: DlczDomainContract,
    rng: np.random.Generator,
) -> ReadoutEventLedger:
    """Generate candidate and detected signal/idler events from source marks."""

    params.validate()
    domain.validate()
    candidates: list[DlczEvent] = []
    gate_rejected = 0
    deferred_multi_particle_readout = 0
    opportunity_ids = tuple(
        dict.fromkeys(event.opportunity_id for event in source_events)
    )
    if len(opportunity_ids) != params.repeats:
        raise ValueError("source events must bind every fixed-repeat opportunity")

    def append_signal_candidate(
        source: DlczEvent, *, channel: str, probability: float, suffix: str
    ) -> None:
        nonlocal gate_rejected
        if not 0.0 <= probability <= 1.0 or not math.isfinite(probability):
            raise ValueError("readout probability must lie in [0,1]")
        if rng.random() >= probability:
            return
        # Arrival time is sampled around the source event time.  A nonzero
        # jitter can place an event outside the finite detector gate; such
        # events are counted and excluded rather than silently clamped.
        jitter = rng.normal(scale=0.2 * domain.external_association_gate_s)
        arrival_time = float(
            source.time_s * domain.external_association_gate_s + jitter
        )
        if not 0.0 <= arrival_time < domain.external_association_gate_s:
            gate_rejected += 1
            return
        candidates.append(
            _candidate(
                source,
                channel=channel,
                mark=source.mark,
                time_s=arrival_time,
                detector_id=channel,
                suffix=suffix,
            )
        )

    def append_idler_candidate(
        source: DlczEvent,
        *,
        arrival_probability: float,
        suffix: str,
        detector_dark: bool = False,
        clock_index: int | None = None,
    ) -> None:
        """Route a read photon through the declared fixed four-SPAD network."""

        nonlocal gate_rejected
        if not 0.0 <= arrival_probability <= 1.0 or not math.isfinite(
            arrival_probability
        ):
            raise ValueError("readout probability must lie in [0,1]")
        if not detector_dark and rng.random() >= arrival_probability:
            return
        detector_index = (
            int(rng.integers(4))
            if detector_dark
            else int(rng.choice(4, p=np.asarray(domain.splitter_probability_by_spad)))
        )
        detector_id = f"read_spad_{detector_index + 1}"
        # ``spad_efficiency_by_detector`` is a per-arriving-photon, post-splitter
        # threshold-click hypothesis.  A detector dark candidate is already a
        # detector-side click and therefore does not traverse this efficiency.
        detector_efficiency = domain.spad_efficiencies(params.eta_idler)[detector_index]
        if not detector_dark and rng.random() >= detector_efficiency:
            return
        if clock_index is None:
            clock_index = int(rng.integers(domain.read_clock_count))
        if not 0 <= clock_index < domain.read_clock_count:
            raise ValueError("read clock index lies outside the accepted FPGA window")
        arrival_time = float((clock_index + rng.random()) * domain.read_clock_period_s)
        if not 0.0 <= arrival_time < domain.read_clock_support_s:
            gate_rejected += 1
            return
        candidates.append(
            _candidate(
                source,
                channel="idler",
                mark=source.mark,
                time_s=arrival_time,
                detector_id=detector_id,
                suffix=f"{suffix}:clock{clock_index}:{detector_id}",
            )
        )

    def selected_fock_photon_count(
        *, photon_count: int, selected_pre_splitter_probability: float
    ) -> int:
        """Sample a passive linear-optical same-mode Fock selection law.

        The channel is declared only for a same collective spatial mode.
        Local/cross source marks are therefore not read as separate optical
        branches.  For an n-Fock input, a selected one-photon probability q
        gives K~Binomial(n,q) before the fixed splitter/SPAD threshold model.
        """

        if type(photon_count) is not int or photon_count < 0:
            raise ValueError("photon_count must be a non-negative integer")
        if (
            not math.isfinite(selected_pre_splitter_probability)
            or not 0.0 <= selected_pre_splitter_probability <= 1.0
        ):
            raise ValueError("selected pre-splitter probability must lie in [0,1]")
        return int(rng.binomial(photon_count, selected_pre_splitter_probability))

    for source in source_events:
        if source.operation_id != params.operation_id:
            raise ValueError("source event operation does not match readout parameters")
        if source.mark == "NO_SOURCE_EVENT":
            continue
        state = spinwave_states.get(source.event_id)
        selected_pre_splitter_probability = None
        if (
            state is not None
            and getattr(state, "evolution_kernel", None) == "MULTI_CELL_PSD_AOD_UNBOUND"
        ):
            selected_pre_splitter_probability = getattr(
                state, "selected_pre_splitter_probability", None
            )
        # ``selected_read_probability`` was used by an early parallel draft.
        # Retain it as an internal compatibility alias while the public channel
        # contract settles on the explicit pre-splitter name.
        if (
            selected_pre_splitter_probability is None
            and state is not None
            and getattr(state, "evolution_kernel", None) == "MULTI_CELL_PSD_AOD_UNBOUND"
        ):
            selected_pre_splitter_probability = getattr(
                state, "selected_read_probability", None
            )
        if selected_pre_splitter_probability is not None and (
            not math.isfinite(float(selected_pre_splitter_probability))
            or not 0.0 <= float(selected_pre_splitter_probability) <= 1.0
        ):
            raise ValueError("selected pre-splitter probability must lie in [0,1]")
        survival_by_cell = (
            state.survival_by_cell
            if state is not None and state.survival_by_cell
            else (1.0,) * (len(source.occupancy) if source.occupancy else 1)
        )
        if source.channel in {"source", "signal"} and source.mark not in {
            "UNCORRELATED_IDLER_BACKGROUND",
            "IDLER_DARK",
        }:
            photon_count = int(sum(source.occupancy))
            signal_probability = (
                1.0
                if source.mark == "SIGNAL_DARK"
                else (
                    1.0 - math.pow(1.0 - params.eta_signal, photon_count)
                    if photon_count > 0
                    else params.eta_signal
                )
            )
            append_signal_candidate(
                source,
                channel="signal",
                probability=signal_probability,
                suffix="threshold",
            )
        if source.channel in {"source", "idler"} and source.mark not in {
            "UNCORRELATED_SIGNAL_BACKGROUND",
            "SIGNAL_DARK",
        }:
            if source.mark == "IDLER_DARK":
                append_idler_candidate(
                    source,
                    arrival_probability=1.0,
                    suffix="dark",
                    detector_dark=True,
                )
            elif source.mark == "UNCORRELATED_IDLER_BACKGROUND":
                # W-cancel suppresses correlated retrieval only.
                append_idler_candidate(
                    source,
                    arrival_probability=params.eta_optical_path,
                    suffix="background",
                )
            elif source.occupancy:
                photon_count = int(sum(source.occupancy))
                if (
                    state is not None
                    and state.multi_particle_readout_status
                    == TWO_PARTICLE_READOUT_DEFERRED
                ):
                    # A one-body selected-mode probability cannot supply a
                    # topology-specific two-particle measurement law.  Do not
                    # silently replace that missing law with independent
                    # Bernoulli trials; retain the omitted photon number as a
                    # diagnostic fact and wait for a registered forward map.
                    deferred_multi_particle_readout += photon_count
                    continue
                same_collective_fock_readout = (
                    state is not None
                    and state.multi_particle_readout_status
                    == SAME_COLLECTIVE_MODE_FOCK_READOUT
                )
                if same_collective_fock_readout:
                    if selected_pre_splitter_probability is None:
                        raise ValueError(
                            "same-mode Fock readout requires a selected pre-splitter probability"
                        )
                    photon_count = selected_fock_photon_count(
                        photon_count=photon_count,
                        selected_pre_splitter_probability=float(
                            selected_pre_splitter_probability
                        ),
                    )
                read_clock_indices: list[int | None]
                if photon_count == 2:
                    first_clock = int(rng.integers(domain.read_clock_count))
                    if (
                        domain.read_clock_count == 1
                        or rng.random() < domain.double_source_same_clock_probability
                    ):
                        read_clock_indices = [first_clock, first_clock]
                    else:
                        remaining = np.delete(
                            np.arange(domain.read_clock_count), first_clock
                        )
                        read_clock_indices = [
                            first_clock,
                            int(rng.choice(remaining)),
                        ]
                else:
                    read_clock_indices = [None] * photon_count
                for excitation_index in range(photon_count):
                    if same_collective_fock_readout:
                        # Selection was sampled jointly from the n-Fock sector
                        # above.  Only splitter routing and SPAD efficiency
                        # remain for each selected optical photon.
                        arrival_probability = 1.0
                    elif selected_pre_splitter_probability is not None:
                        arrival_probability = float(selected_pre_splitter_probability)
                    else:
                        occupied_survival = [
                            survival_by_cell[cell_index]
                            for cell_index, occupancy in enumerate(source.occupancy)
                            for _ in range(occupancy)
                        ]
                        survival = (
                            occupied_survival[excitation_index]
                            if excitation_index < len(occupied_survival)
                            else 1.0
                        )
                        arrival_probability = (
                            params.eta_optical_path
                            * params.eta_retrieval
                            * params.w_overlap
                            * survival
                        )
                    append_idler_candidate(
                        source,
                        arrival_probability=arrival_probability,
                        suffix=f"excitation:{excitation_index}",
                        clock_index=read_clock_indices[excitation_index],
                    )
            else:
                survival = survival_by_cell[0] if survival_by_cell else 1.0
                legacy_arrival_probability = (
                    params.eta_retrieval * params.w_overlap * survival
                )
                append_idler_candidate(
                    source,
                    arrival_probability=(
                        float(selected_pre_splitter_probability)
                        if selected_pre_splitter_probability is not None
                        else params.eta_optical_path * legacy_arrival_probability
                    ),
                    suffix="single",
                )

    accepted, dead_time_rejected = _apply_detector_dead_time_by_opportunity(
        candidates, dead_time_s=domain.dead_time_s, mode=domain.dead_time_mode
    )
    accepted_ids = {event.event_id for event in accepted}
    detected = tuple(
        DlczEvent(**{**event.to_dict(), "detected": True})
        for event in sorted(accepted, key=lambda item: (item.time_s, item.event_id))
    )
    result = ReadoutEventLedger(
        candidates=tuple(candidates),
        detected_events=detected,
        operation_id=params.operation_id,
        opportunity_count=params.repeats,
        # The signal--read association is the nominal external 200 ns gate;
        # read-SPAD aggregation below remains restricted to the 19 accepted
        # 10 ns FPGA clocks (190 ns).
        gate_window_s=domain.external_association_gate_s,
        dead_time_s=domain.dead_time_s,
        dead_time_mode=domain.dead_time_mode,
        opportunity_ids=opportunity_ids,
        gate_rejected_count=gate_rejected,
        dead_time_rejected_count=dead_time_rejected,
        read_clock_period_s=domain.read_clock_period_s,
        read_clock_count=domain.read_clock_count,
        deferred_multi_particle_readout_count=deferred_multi_particle_readout,
    )
    result.validate(
        n_mode=(
            len(source_events[0].occupancy)
            if source_events and source_events[0].occupancy
            else None
        )
    )
    if not accepted_ids.issubset({event.event_id for event in candidates}):
        raise RuntimeError("dead-time detector emitted an unknown event")
    return result


def conditional_randomize_idler_events(
    ledger: ReadoutEventLedger, *, rng: np.random.Generator
) -> ReadoutEventLedger:
    """Build an exact-marginal GSI_NULL control and reapply detector dead time."""

    ledger.validate()
    idler_candidates = [
        event for event in ledger.candidates if event.channel == "idler"
    ]
    signal_candidates = [
        event for event in ledger.candidates if event.channel == "signal"
    ]
    if not idler_candidates:
        return ledger
    if not ledger.opportunity_ids:
        raise ValueError(
            "conditional randomization requires complete opportunity identities"
        )
    idler_by_opportunity: dict[str, list[DlczEvent]] = {
        opportunity_id: [] for opportunity_id in ledger.opportunity_ids
    }
    for event in idler_candidates:
        try:
            idler_by_opportunity[event.opportunity_id].append(event)
        except KeyError as exc:
            raise ValueError(
                "idler candidate lies outside the registered opportunities"
            ) from exc
    shuffled = tuple(rng.permutation(ledger.opportunity_ids))
    remapped_candidates: list[DlczEvent] = []
    for source_index, target_opportunity_id in enumerate(shuffled):
        source_opportunity_id = ledger.opportunity_ids[source_index]
        for event_index, event in enumerate(
            idler_by_opportunity[source_opportunity_id]
        ):
            remapped_candidates.append(
                DlczEvent(
                    **{
                        **event.to_dict(),
                        "event_id": f"{event.event_id}:conditional_null:{source_index}:{event_index}",
                        "opportunity_id": target_opportunity_id,
                    }
                )
            )
    remapped_idler_candidates, dead_time_rejected = (
        _apply_detector_dead_time_by_opportunity(
            remapped_candidates,
            dead_time_s=ledger.dead_time_s,
            mode=ledger.dead_time_mode,
        )
    )
    # The signal candidate set is unchanged; re-running its per-opportunity
    # dead-time filter gives the same accepted set and guards against a stale
    # pre-randomization ledger.
    signal_candidates_accepted, signal_rejected = (
        _apply_detector_dead_time_by_opportunity(
            signal_candidates,
            dead_time_s=ledger.dead_time_s,
            mode=ledger.dead_time_mode,
        )
    )
    remapped_detected = [
        DlczEvent(**{**event.to_dict(), "detected": True})
        for event in signal_candidates_accepted + remapped_idler_candidates
    ]
    result = ReadoutEventLedger(
        candidates=tuple(signal_candidates + remapped_candidates),
        detected_events=tuple(remapped_detected),
        operation_id=ledger.operation_id,
        opportunity_count=ledger.opportunity_count,
        gate_window_s=ledger.gate_window_s,
        dead_time_s=ledger.dead_time_s,
        dead_time_mode=ledger.dead_time_mode,
        opportunity_ids=ledger.opportunity_ids,
        gate_rejected_count=ledger.gate_rejected_count,
        dead_time_rejected_count=dead_time_rejected + signal_rejected,
        read_clock_period_s=ledger.read_clock_period_s,
        read_clock_count=ledger.read_clock_count,
        deferred_multi_particle_readout_count=ledger.deferred_multi_particle_readout_count,
        parameter_status=ledger.parameter_status,
    )
    result.validate()
    return result


__all__ = [
    "READOUT_SCHEMA",
    "ReadoutEventLedger",
    "conditional_randomize_idler_events",
    "render_readout_events",
]
