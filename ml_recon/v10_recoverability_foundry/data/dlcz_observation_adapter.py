"""Convert detected event ledgers into the public observation plane."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections import defaultdict

import numpy as np

from ..contracts.dlcz_event_mc import DlczEvent
from ..physics.dlcz_readout_events import ReadoutEventLedger


OBSERVATION_ADAPTER_SCHEMA = "V10DlczObservationAdapterV1"


@dataclass(frozen=True)
class DlczObservation:
    operation_id: str
    repeats: int
    four_cell_counts: np.ndarray
    signal_clicks: np.ndarray
    idler_clicks: np.ndarray
    coincidence_clicks: np.ndarray
    cell_resolved_idler_counts: np.ndarray | None
    time_tags: tuple[tuple[str, str, str, float], ...]
    observation_mask: np.ndarray
    observation_adapter_sha: str
    global_multi_click_counts: tuple[int, int, int] = (0, 0, 0)
    global_multi2_pair_counts: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)

    def validate(self, *, n_mode: int | None = None) -> None:
        if type(self.repeats) is not int or self.repeats < 0:
            raise ValueError("repeats must be a non-negative integer")
        counts = np.asarray(self.four_cell_counts)
        if counts.shape != (4,) or counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise ValueError("four_cell_counts must be four non-negative integers")
        if int(counts.sum()) != self.repeats:
            raise ValueError("four-cell counts must conserve the fixed repeats")
        for value, name in (
            (self.signal_clicks, "signal_clicks"),
            (self.idler_clicks, "idler_clicks"),
            (self.coincidence_clicks, "coincidence_clicks"),
        ):
            if type(value) is not int or value < 0 or value > self.repeats:
                raise ValueError(f"{name} must be a bounded integer")
        if self.coincidence_clicks > min(self.signal_clicks, self.idler_clicks):
            raise ValueError("coincidences cannot exceed either marginal")
        if self.cell_resolved_idler_counts is not None:
            cell = np.asarray(self.cell_resolved_idler_counts)
            if (
                cell.ndim != 1
                or (n_mode is not None and cell.shape != (n_mode,))
                or cell.dtype.kind not in "iu"
                or np.any(cell < 0)
            ):
                raise ValueError("cell-resolved idler counts have invalid shape")
        mask = np.asarray(self.observation_mask)
        if mask.dtype != np.bool_ or mask.shape != (4,):
            raise ValueError("observation_mask must be strict Boolean [4]")
        if not isinstance(self.time_tags, tuple):
            raise ValueError("time_tags must be an immutable tuple")
        for tag in self.time_tags:
            if not isinstance(tag, tuple) or len(tag) != 4:
                raise ValueError(
                    "time tags must be (opportunity, channel, detector, time)"
                )
            if not all(isinstance(value, str) and value for value in tag[:3]):
                raise ValueError("time-tag identities must be non-empty strings")
            if not np.isfinite(float(tag[3])) or float(tag[3]) < 0.0:
                raise ValueError("time-tag time must be finite and non-negative")
        if (
            not isinstance(self.observation_adapter_sha, str)
            or len(self.observation_adapter_sha) != 64
        ):
            raise ValueError("observation adapter SHA is required")
        if len(self.global_multi_click_counts) != 3 or any(
            type(value) is not int or value < 0
            for value in self.global_multi_click_counts
        ):
            raise ValueError(
                "global multi-click counts must be three non-negative integers"
            )
        if (
            len(self.global_multi2_pair_counts) != 6
            or any(
                type(value) is not int or value < 0
                for value in self.global_multi2_pair_counts
            )
            or sum(self.global_multi2_pair_counts) != self.global_multi_click_counts[0]
        ):
            raise ValueError("exact-two pair counts must partition global multi2")


def _observation_sha(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _read_clock_key(event: DlczEvent, ledger: ReadoutEventLedger) -> int | str:
    """Return the accepted FPGA clock, with a legacy fallback for old ledgers."""

    clock_index = int(float(event.time_s) // ledger.read_clock_period_s)
    if 0 <= clock_index < ledger.read_clock_count:
        return clock_index
    # Old callers supplied aggregate ``idler`` events without FPGA-clock
    # coordinates.  Preserve their threshold semantics while never pretending
    # they identify a physical multi-click clock pattern.
    return f"legacy:{event.time_s:.17g}"


def _global_readout_diagnostics(
    idler_events: list[DlczEvent], ledger: ReadoutEventLedger
) -> tuple[
    set[tuple[str, int | str]],
    tuple[int, int, int],
    tuple[int, int, int, int, int, int],
]:
    """Aggregate native clock-level four-SPAD summaries without cell routing."""

    detector_sets: dict[tuple[str, int | str], set[str]] = defaultdict(set)
    for event in idler_events:
        key = (event.opportunity_id, _read_clock_key(event, ledger))
        detector_sets[key].add(event.detector_id or "legacy_idler")
    singleton_keys: set[tuple[str, int | str]] = set()
    multi = [0, 0, 0]
    pair_counts = [0] * 6
    pair_order = {
        ("read_spad_1", "read_spad_2"): 0,
        ("read_spad_1", "read_spad_3"): 1,
        ("read_spad_1", "read_spad_4"): 2,
        ("read_spad_2", "read_spad_3"): 3,
        ("read_spad_2", "read_spad_4"): 4,
        ("read_spad_3", "read_spad_4"): 5,
    }
    for key, detectors in detector_sets.items():
        multiplicity = len(detectors)
        if multiplicity == 1:
            singleton_keys.add(key)
        elif 2 <= multiplicity <= 4:
            multi[multiplicity - 2] += 1
            if multiplicity == 2:
                pair = tuple(sorted(detectors))
                pair_index = pair_order.get(pair)
                if pair_index is not None:
                    pair_counts[pair_index] += 1
    return singleton_keys, tuple(multi), tuple(pair_counts)


def adapt_readout_events(
    ledger: ReadoutEventLedger,
    *,
    n_mode: int,
    operation_mask: bool = True,
) -> DlczObservation:
    """Aggregate threshold clicks while hiding source attribution."""

    ledger.validate(n_mode=n_mode)
    if type(n_mode) is not int or n_mode < 3:
        raise ValueError("n_mode must be at least three")
    by_opportunity: dict[str, dict[str, list[DlczEvent]]] = defaultdict(
        lambda: {"signal": [], "idler": []}
    )
    idler_events: list[DlczEvent] = []
    for event in ledger.detected_events:
        by_opportunity[event.opportunity_id][event.channel].append(event)
        if event.channel == "idler":
            idler_events.append(event)
    singleton_keys, global_multi, global_multi2_pairs = _global_readout_diagnostics(
        idler_events, ledger
    )
    counts = np.zeros(4, dtype=np.int64)
    signal_total = idler_total = coincidence_total = 0
    tags: list[tuple[str, str, str, float]] = []
    if not ledger.opportunity_ids:
        raise ValueError("observation adapter requires complete opportunity identities")
    if len(ledger.opportunity_ids) != ledger.opportunity_count:
        raise ValueError("registered opportunities do not match the repeat count")
    for opportunity_id in by_opportunity:
        if opportunity_id not in ledger.opportunity_ids:
            raise ValueError(
                "detected event lies outside the registered opportunity set"
            )
    for opportunity_index, opportunity_id in enumerate(ledger.opportunity_ids):
        grouped = by_opportunity.get(opportunity_id, {"signal": [], "idler": []})
        signal_events = grouped["signal"]
        opportunity_idler_events = grouped["idler"]
        signal = bool(signal_events)
        # FPGA ``getidler`` is an OR over accepted clocks with K_t == 1.  A
        # gate can have this sticky singleton bit and a global multi counter.
        idler = any(
            (opportunity_id, _read_clock_key(event, ledger)) in singleton_keys
            for event in opportunity_idler_events
        )
        signal_total += int(signal)
        idler_total += int(idler)
        coincidence_total += int(signal and idler)
        counts[2 * int(signal) + int(idler)] += 1
        for event in signal_events + opportunity_idler_events:
            tags.append(
                (
                    # Expose only a row-local opaque gate ordinal.  The raw
                    # opportunity identity contains caller-controlled group
                    # text and must remain in the private truth ledger.
                    f"gate:{opportunity_index}",
                    event.channel,
                    event.detector_id or event.channel,
                    float(event.time_s),
                )
            )
    if int(counts.sum()) != ledger.opportunity_count:
        raise RuntimeError("observation adapter violated fixed-repeat conservation")
    tags.sort(key=lambda value: (value[0], value[3], value[1]))
    payload = {
        "schema": OBSERVATION_ADAPTER_SCHEMA,
        "operation_id": ledger.operation_id,
        "repeats": ledger.opportunity_count,
        "four_cell_counts": counts.tolist(),
        "signal_clicks": signal_total,
        "idler_clicks": idler_total,
        "coincidence_clicks": coincidence_total,
        "cell_resolved_idler_counts": None,
        "time_tags": tags,
        "observation_mask": [bool(operation_mask)] * 4,
        "global_multi_click_counts": list(global_multi),
        "global_multi2_pair_counts": list(global_multi2_pairs),
    }
    result = DlczObservation(
        operation_id=ledger.operation_id,
        repeats=ledger.opportunity_count,
        four_cell_counts=counts,
        signal_clicks=signal_total,
        idler_clicks=idler_total,
        coincidence_clicks=coincidence_total,
        cell_resolved_idler_counts=None,
        time_tags=tuple(tags),
        observation_mask=np.full(4, bool(operation_mask), dtype=bool),
        observation_adapter_sha=_observation_sha(payload),
        global_multi_click_counts=global_multi,
        global_multi2_pair_counts=global_multi2_pairs,
    )
    result.validate(n_mode=n_mode)
    return result


def aggregate_four_cell_counts(observations: tuple[DlczObservation, ...]) -> np.ndarray:
    if not observations:
        raise ValueError("at least one observation is required")
    for observation in observations:
        observation.validate()
    return np.stack(
        [
            np.asarray(observation.four_cell_counts, dtype=np.int64)
            for observation in observations
        ]
    )


__all__ = [
    "DlczObservation",
    "OBSERVATION_ADAPTER_SCHEMA",
    "adapt_readout_events",
    "aggregate_four_cell_counts",
]
