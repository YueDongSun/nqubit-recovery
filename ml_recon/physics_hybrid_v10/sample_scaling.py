"""Pre-registered 10k -> 50k -> 200k -> 2M data-rung gates."""

from __future__ import annotations

from typing import Dict, Iterable


SAMPLE_RUNGS = (10_000, 50_000, 200_000, 2_000_000)


def next_sample_rung(current: int) -> int:
    if current not in SAMPLE_RUNGS:
        raise ValueError(f"current sample count must be one of {SAMPLE_RUNGS}")
    index = SAMPLE_RUNGS.index(current)
    if index == len(SAMPLE_RUNGS) - 1:
        raise ValueError("2M is the final pre-registered V10 rung")
    return SAMPLE_RUNGS[index + 1]


def validate_rung_sequence(completed_rungs: Iterable[int], requested_rung: int) -> None:
    if requested_rung not in SAMPLE_RUNGS:
        raise ValueError(f"requested sample rung must be one of {SAMPLE_RUNGS}")
    completed = tuple(int(value) for value in completed_rungs)
    prerequisites = SAMPLE_RUNGS[:SAMPLE_RUNGS.index(requested_rung)]
    if tuple(completed[:len(prerequisites)]) != prerequisites:
        raise ValueError(
            f"sample rungs must execute in order; {requested_rung} requires {prerequisites}")


def validate_rung_authorization(
    completed_rungs: Iterable[int],
    requested_rung: int,
    *,
    training_authorized: bool,
    phase_1_to_7_gate_sha: str,
    phase_8_gate_sha: str = "",
) -> None:
    """Fail before dataset generation unless an ordered rung is gate-bound."""
    from ml_recon.data_v2_4.gate_artifact import validate_gate_artifact

    validate_rung_sequence(completed_rungs, requested_rung)
    if not training_authorized:
        raise ValueError(
            "registered sample rungs are fail-closed until Phase 1-7 authorization is bound")
    phase_1_gate = validate_gate_artifact(
        phase_1_to_7_gate_sha, requested_rung=requested_rung,
        expected_kind="phase_1_to_7")
    if requested_rung == SAMPLE_RUNGS[-1] and not phase_8_gate_sha:
        raise ValueError(
            "the 2M sample rung requires a Phase 8 pilot/negative-control gate artifact")
    if requested_rung == SAMPLE_RUNGS[-1]:
        phase_8_gate = validate_gate_artifact(
            phase_8_gate_sha, requested_rung=requested_rung,
            expected_kind="phase_8_pilot_negative_control")
        if phase_8_gate.contract_sha != phase_1_gate.contract_sha \
                or phase_8_gate.schema_sha != phase_1_gate.schema_sha \
                or phase_8_gate.source_tree != phase_1_gate.source_tree:
            raise ValueError("Phase 1-7 and Phase 8 gate dependencies disagree")


def should_expand(
    history: Iterable[Dict[str, float]],
    relative_threshold: float = 0.005,
    predicted_gain_threshold: float = 0.0,
) -> Dict[str, object]:
    rows = list(history)
    if not rows:
        return {"expand": False, "reason": "no_validation_evidence"}
    completed_rungs = tuple(int(row["sample_count"]) for row in rows)
    if completed_rungs != SAMPLE_RUNGS[:len(completed_rungs)]:
        return {"expand": False, "reason": "rung_sequence_invalid"}
    latest = rows[-1]
    if latest.get("finite_audit_passed", 0.0) <= 0.5:
        return {"expand": False, "reason": "finite_audit_failed"}
    if latest.get("capacity_underfit", 0.0) > 0.5:
        return {"expand": False, "reason": "model_limited"}
    if latest.get("measurement_contract_error", 0.0) > 0.5:
        return {"expand": False, "reason": "measurement_limited"}
    if latest.get("predicted_next_gain", 0.0) <= predicted_gain_threshold:
        return {"expand": False, "reason": "scaling_prediction_nonpositive"}
    if len(rows) >= 3:
        gains = [rows[index]["primary_utility"] - rows[index - 1]["primary_utility"]
                 for index in range(len(rows) - 2, len(rows))]
        relatives = [gain / max(abs(rows[index - 1]["primary_utility"]), 1e-12)
                     for gain, index in zip(gains, range(len(rows) - 2, len(rows)))]
        if all(gain <= 0.0 or relative < relative_threshold
               for gain, relative in zip(gains, relatives)):
            return {"expand": False, "reason": "two_rung_plateau"}
    if completed_rungs[-1] == 200_000:
        required_gates = (
            "phase_1_to_7_passed", "five_arm_match_passed",
            "resume_audit_passed", "negative_controls_passed")
        missing = [key for key in required_gates if latest.get(key, 0.0) <= 0.5]
        if missing:
            return {"expand": False, "reason": "2m_prerequisite_gate_failed",
                    "missing_gates": tuple(missing)}
    if completed_rungs[-1] == 2_000_000:
        return {"expand": False, "reason": "final_registered_rung_complete"}
    return {"expand": True, "reason": "positive_registered_scaling_signal",
            "next_rung": next_sample_rung(completed_rungs[-1])}
