"""Helpers for automated multi-state experiment schedules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


MAX_AUTO_STATES = 1000
V4_4Q_COVERAGE_STATES = 940
V4_4Q_ZERO_PROBE_ANCHORS = 12
V4_4Q_SIGNED_PROBE_ANCHORS = 48
V4_4Q_SIGNED_PROBE_PAIRS = 24
_PHASE_ALPHAS = (
    0.6180339887498949,  # golden-ratio conjugate
    math.sqrt(2.0) - 1.0,
    math.sqrt(3.0) - 1.0,
)
_AMP_ALPHAS = (
    math.sqrt(5.0) - 2.0,
    math.sqrt(7.0) - 2.0,
    math.sqrt(11.0) - 3.0,
)


def generate_coverage_schedule(
    system_type: str,
    base_state: Dict[str, Any],
    count: int,
) -> List[Dict[str, Any]]:
    """Generate deterministic, non-repeating states that fill the target space.

    The sequence uses irrational rotations instead of a fixed phase step. That
    makes the phase spacing look slightly irregular while remaining fully
    reproducible, so a 1000-state plan can be resumed by index. Amplitudes are
    spread over a softened simplex and scaled by the current UI state's RMS
    amplitude, which keeps the generated W-state targets inside the same
    practical amplitude range while avoiding duplicate phase-ramp states.
    """

    n = _num_qubits(system_type)
    count = max(1, min(MAX_AUTO_STATES, int(count)))
    base_amplitudes = _values(base_state.get("amplitudes"), n, 1.0)
    base_phases = _values(base_state.get("phases_deg"), n, 0.0)
    amp_scale = math.sqrt(sum(value * value for value in base_amplitudes) / n) or 1.0
    name_width = max(3, len(str(count)))

    schedule: List[Dict[str, Any]] = []
    seen = set()
    source_index = 1
    for state_index in range(count):
        while True:
            amplitudes = _coverage_amplitudes(source_index, n, amp_scale)
            phases = _coverage_phases(source_index, n, base_phases)
            key = _state_key(amplitudes, phases)
            if key not in seen:
                seen.add(key)
                schedule.append(
                    {
                        "name": f"auto_state_{state_index + 1:0{name_width}d}",
                        "amplitudes": amplitudes,
                        "phases_deg": phases,
                        "coverage_index": source_index,
                    }
                )
                source_index += 1
                break
            source_index += 1
    return schedule


def generate_drift_aware_coverage_schedule(
    system_type: str,
    base_state: Dict[str, Any],
    count: int = MAX_AUTO_STATES,
    *,
    anchor_count: int = 60,
    plan_id: str = "deterministic_low_discrepancy_coverage_v2",
) -> List[Dict[str, Any]]:
    """Interleave low-discrepancy 4Q coverage with repeated phase anchors."""

    if system_type != "4qubit":
        raise ValueError("drift-aware coverage is currently defined only for 4qubit")
    count = max(1, min(MAX_AUTO_STATES, int(count)))
    anchor_count = int(anchor_count)
    if anchor_count < 4 or anchor_count >= count:
        raise ValueError("anchor_count must be >= 4 and smaller than count")

    coverage_count = count - anchor_count
    coverage = generate_coverage_schedule(system_type, base_state, coverage_count)
    base_amplitudes = _values(base_state.get("amplitudes"), 4, 1.0)
    base_phases = _values(base_state.get("phases_deg"), 4, 0.0)
    quotient, remainder = divmod(coverage_count, anchor_count)

    schedule: List[Dict[str, Any]] = []
    coverage_cursor = 0
    for block_index in range(anchor_count):
        block_size = quotient + (1 if block_index < remainder else 0)
        for _ in range(block_size):
            state = dict(coverage[coverage_cursor])
            state.update(
                state_role="coverage",
                block_id=block_index + 1,
                design_version=2,
                plan_id=str(plan_id),
            )
            schedule.append(state)
            coverage_cursor += 1

        # Q1 is the fixed global-phase reference.  The other anchors span the
        # three observable relative-phase directions without changing power.
        anchor_kind = block_index % 4
        phases = list(base_phases)
        anchor_label = "reference"
        if anchor_kind:
            phases[anchor_kind] = _wrap_degrees(phases[anchor_kind] + 90.0)
            anchor_label = f"q{anchor_kind + 1}_plus90"
        repeat_index = block_index // 4 + 1
        anchor_state = {
                "name": f"anchor_{block_index + 1:03d}_{anchor_label}",
                "amplitudes": list(base_amplitudes),
                "phases_deg": [round(value, 6) for value in phases],
                "state_role": "anchor",
                "anchor_id": anchor_label,
                "anchor_repeat_index": repeat_index,
                "block_id": block_index + 1,
                "design_version": 2,
                "plan_id": str(plan_id),
            }
        # Imported lazily to keep the production schedule module usable when
        # optional ML dependencies are absent.
        from ml_recon.physics_hybrid_v4.probe_design import phase_probe_schedule_fields

        anchor_state["phase_probe"] = phase_probe_schedule_fields(plan_id, anchor_label, repeat_index)
        schedule.append(anchor_state)

    for schedule_index, state in enumerate(schedule, start=1):
        state["schedule_index"] = schedule_index
    return schedule


def validate_v4_4qubit_campaign_schedule(schedule: Iterable[Dict[str, Any]]) -> None:
    """Validate the fixed 940 coverage + 60 probe-anchor campaign contract."""

    states = list(schedule)
    if len(states) != MAX_AUTO_STATES:
        raise ValueError(f"V4 4qubit campaign must contain exactly {MAX_AUTO_STATES} states")

    coverage = [state for state in states if state.get("state_role") == "coverage"]
    anchors = [state for state in states if state.get("state_role") == "anchor"]
    if any(state.get("phase_probe") for state in coverage):
        raise ValueError("V4 4qubit coverage states must not contain phase probes")

    zero = []
    signed = []
    for state in anchors:
        probe = dict(state.get("phase_probe") or {})
        if probe.get("schema") != "OperationPhaseProbeScheduleV1":
            raise ValueError("every V4 4qubit anchor must contain a phase-probe record")
        sign = int(probe.get("pair_sign", 0))
        if sign == 0:
            zero.append(state)
        elif sign in {-1, 1}:
            signed.append(state)
        else:
            raise ValueError("V4 4qubit probe pair_sign must be -1, 0, or +1")

    counts = (len(coverage), len(zero), len(signed))
    expected = (
        V4_4Q_COVERAGE_STATES,
        V4_4Q_ZERO_PROBE_ANCHORS,
        V4_4Q_SIGNED_PROBE_ANCHORS,
    )
    if counts != expected:
        raise ValueError(
            "V4 4qubit schedule must contain exactly 940 coverage, 12 zero-probe, "
            "and 48 signed-probe states"
        )

    pair_signs: Dict[str, List[int]] = {}
    for state in signed:
        probe = dict(state["phase_probe"])
        pair_id = str(probe.get("code_pair_id") or "")
        if not pair_id:
            raise ValueError("V4 4qubit signed probes must contain code_pair_id")
        pair_signs.setdefault(pair_id, []).append(int(probe["pair_sign"]))
    if len(pair_signs) != V4_4Q_SIGNED_PROBE_PAIRS or any(
        sorted(signs) != [-1, 1] for signs in pair_signs.values()
    ):
        raise ValueError("V4 4qubit signed probes must form exactly 24 complete +/- pairs")


def generate_phase_ramp_schedule(
    system_type: str,
    base_state: Dict[str, Any],
    count: int,
    phase_step_deg: float,
) -> List[Dict[str, Any]]:
    """Generate W-state targets by ramping relative phases.

    The first entry is the current UI state. Later entries add
    ``index * phase_step_deg * qubit_index`` to the phase vector, which changes
    relative phases while keeping amplitudes unchanged.
    """

    n = _num_qubits(system_type)
    count = max(1, int(count))
    amplitudes = _values(base_state.get("amplitudes"), n, 1.0)
    phases = _values(base_state.get("phases_deg"), n, 0.0)

    schedule = []
    for state_index in range(count):
        ramped_phases = [
            _wrap_degrees(phases[qubit_index] + state_index * phase_step_deg * qubit_index)
            for qubit_index in range(n)
        ]
        schedule.append(
            {
                "name": f"auto_state_{state_index + 1:03d}",
                "amplitudes": list(amplitudes),
                "phases_deg": ramped_phases,
            }
        )
    return schedule


def parse_state_schedule(text: str, system_type: str) -> List[Dict[str, Any]]:
    """Parse a JSON state schedule from the UI text box."""

    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("states", [])
    if not isinstance(payload, list) or not payload:
        raise ValueError("schedule JSON must be a non-empty list or {'states': [...]}")

    n = _num_qubits(system_type)
    schedule = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"state {index} must be an object")
        amplitudes = _values(item.get("amplitudes"), n, 1.0)
        phases = _values(item.get("phases_deg"), n, 0.0)
        state = {
            key: value
            for key, value in item.items()
            if key not in {"name", "amplitudes", "phases_deg", "phases_rad"}
        }
        state.update(
            {
                "name": str(item.get("name") or f"auto_state_{index:03d}"),
                "amplitudes": amplitudes,
                "phases_deg": phases,
            }
        )
        schedule.append(state)
    return schedule


def load_state_schedule_file(path: str | Path, system_type: str) -> List[Dict[str, Any]]:
    """Load and normalize an automation schedule JSON file."""

    schedule_path = Path(path)
    return parse_state_schedule(schedule_path.read_text(encoding="utf-8"), system_type)


def schedule_to_json(schedule: Iterable[Dict[str, Any]]) -> str:
    return json.dumps(list(schedule), indent=2)


def _num_qubits(system_type: str) -> int:
    if system_type == "3qubit":
        return 3
    if system_type == "4qubit":
        return 4
    raise ValueError("system_type must be '3qubit' or '4qubit'")


def _values(values, length: int, default: float) -> List[float]:
    if values is None:
        return [float(default)] * length
    if len(values) != length:
        raise ValueError(f"expected {length} values, got {len(values)}")
    return [float(value) for value in values]


def _coverage_amplitudes(source_index: int, n: int, amp_scale: float) -> List[float]:
    if n <= 1:
        return [round(float(amp_scale), 6)]
    breaks = [0.0, 1.0]
    for dim in range(n - 1):
        alpha = _AMP_ALPHAS[dim % len(_AMP_ALPHAS)]
        offset = (dim + 1) / (n + 2)
        breaks.append(_frac(source_index * alpha + offset))
    breaks.sort()
    raw_weights = [breaks[index + 1] - breaks[index] for index in range(n)]

    # Blend with the equal W-state point so coverage avoids exact zero-amplitude
    # corners, which are usually poor hardware targets.
    spread = 0.72
    weights = [((1.0 - spread) / n) + spread * weight for weight in raw_weights]
    return [
        round(min(5.0, max(0.0, amp_scale * math.sqrt(max(weight, 0.0) * n))), 6)
        for weight in weights
    ]


def _coverage_phases(source_index: int, n: int, base_phases: List[float]) -> List[float]:
    phases = [float(base_phases[0])]
    for qubit_index in range(1, n):
        alpha = _PHASE_ALPHAS[(qubit_index - 1) % len(_PHASE_ALPHAS)]
        offset = qubit_index / (n + 1)
        phases.append(_wrap_degrees(base_phases[qubit_index] + 360.0 * _frac(source_index * alpha + offset)))
    return [round(value, 6) for value in phases]


def _frac(value: float) -> float:
    return float(value) - math.floor(float(value))


def _state_key(amplitudes: List[float], phases: List[float]) -> tuple:
    return tuple(round(value, 6) for value in amplitudes + phases)


def _wrap_degrees(value: float) -> float:
    wrapped = ((float(value) + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and value > 0:
        return 180.0
    return wrapped
