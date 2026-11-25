"""Canonical feedback gate feature composition contract.

Both ephemeral (range) and persistent (fixture) loaders MUST use
``compose_feedback_gate_features()`` to build the 24-dim gate input.
This guarantees persistent/ephemeral parity for all feedback arms.

Gate layout:
    [0:8]  — observed base gate statistics (from current measurement)
    [8:23] — oracle context: mechanism one-hot (15) + severity (1)
             (zeros for A0–A3, truth for A4/teacher)
"""

from __future__ import annotations

import numpy as np

FEEDBACK_GATE_BASE_DIM = 8
FEEDBACK_GATE_ORACLE_DIM = 16
FEEDBACK_GATE_DIM = 24


def compose_feedback_gate_features(
    base_gate: np.ndarray,
    *,
    arm: str,
    true_mechanism: np.ndarray | None = None,
    true_severity: float | None = None,
) -> np.ndarray:
    """Compose 24-dim gate features for any feedback arm.

    Parameters
    ----------
    base_gate : ndarray of shape ``(8,)``
        Observed base gate statistics derived from the current measurement.
        Must be the **observation-derived** gate for all arms (A0–A4).
    arm : str
        Feedback arm identifier.  ``A4_ORACLE`` and ``feedback_teacher``
        require oracle context; all others pad with zeros.
    true_mechanism : ndarray of shape ``(15,)``, optional
        One-hot mechanism truth.  Required for A4/teacher.
    true_severity : float, optional
        Severity scalar.  Required for A4/teacher.

    Returns
    -------
    np.ndarray of shape ``(24,)``
    """
    base = np.asarray(base_gate, dtype=np.float32).reshape(-1)
    if base.shape != (FEEDBACK_GATE_BASE_DIM,):
        raise ValueError(
            f"base feedback gate must have shape "
            f"({FEEDBACK_GATE_BASE_DIM},), got {base.shape}")

    if arm in ("A4_ORACLE", "feedback_teacher"):
        if true_mechanism is None or true_severity is None:
            raise ValueError("A4/teacher requires mechanism and severity truth")
        oracle = np.concatenate((
            np.asarray(true_mechanism, dtype=np.float32).reshape(-1),
            np.asarray([float(true_severity)], dtype=np.float32),
        ))
        if oracle.shape != (FEEDBACK_GATE_ORACLE_DIM,):
            raise ValueError(
                f"oracle gate context must have shape "
                f"({FEEDBACK_GATE_ORACLE_DIM},), got {oracle.shape}")
    else:
        oracle = np.zeros(FEEDBACK_GATE_ORACLE_DIM, dtype=np.float32)

    result = np.concatenate((base, oracle))
    if result.shape != (FEEDBACK_GATE_DIM,):
        raise ValueError(
            f"feedback gate must have shape "
            f"({FEEDBACK_GATE_DIM},), got {result.shape}")
    return result
