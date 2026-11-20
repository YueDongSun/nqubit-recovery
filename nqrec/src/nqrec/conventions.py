"""Cell order, phase sign, gauge coordinates, and sigma1 / R1 naming.

Physics code uses these names exactly. Nothing here supplies a default phase_sign,
exposure, or amplitude vector: callers pass every value.
"""

from __future__ import annotations

import numpy as np

# Track B cell order. First bit is the signal click, second bit is the idler singleton.
CELL_LABELS = ("00", "10", "01", "11")
# Legacy Track A order. Converted only by track_a_to_track_b / track_b_to_track_a.
TRACK_A_LABELS = ("00", "01", "10", "11")

# Read-AOD analyzer phases are stored from the negative diffraction order.
# This constant documents the lab convention. It is not a default for loaders.
READ_AOD_PHASE_SIGN = -1


def require_phase_sign(value) -> int:
    """Return +1 or -1. Reject bools and anything else. Never invent a sign."""
    if isinstance(value, bool) or value not in (-1, 1, -1.0, 1.0):
        raise ValueError(f"phase_sign must be -1 or +1, got {value!r}")
    return int(value)


def apply_phase_sign(theta_rad, phase_sign: int):
    """Stored analyzer phases enter the amplitude as phase_sign * theta."""
    sign = require_phase_sign(phase_sign)
    theta = np.asarray(theta_rad, dtype=np.float64)
    return sign * theta


def track_a_to_track_b(counts):
    """Reorder Track A cells [00, 01, 10, 11] into Track B [00, 10, 01, 11]."""
    counts = np.asarray(counts)
    if counts.shape[-1] != 4:
        raise ValueError(f"expected 4 cells, got shape {counts.shape}")
    index = [0, 2, 1, 3]
    return counts[..., index]


def track_b_to_track_a(counts):
    """Inverse of track_a_to_track_b."""
    counts = np.asarray(counts)
    if counts.shape[-1] != 4:
        raise ValueError(f"expected 4 cells, got shape {counts.shape}")
    index = [0, 2, 1, 3]
    return counts[..., index]


def counts_to_cells(read, signal, idler, coin):
    """Map (READ, S, I, C) into Track B cells [00, 10, 01, 11].

    n = (READ - S - I + C, S - C, I - C, C). Every cell is a non-negative
    integer and the four cells sum to READ.
    """
    read = np.asarray(read)
    signal = np.asarray(signal)
    idler = np.asarray(idler)
    coin = np.asarray(coin)
    for name, arr in ("READ", read), ("S", signal), ("I", idler), ("C", coin):
        if np.any(arr < 0):
            raise ValueError(f"{name} has a negative count")
    n00 = read - signal - idler + coin
    n10 = signal - coin
    n01 = idler - coin
    n11 = np.asarray(coin)
    cells = np.stack([n00, n10, n01, n11], axis=-1)
    if np.any(cells < 0):
        raise ValueError(
            "cell counts are negative; need READ >= S, READ >= I, S >= C, I >= C, "
            f"and READ - S - I + C >= 0. Example cells {cells.reshape(-1, 4)[int(np.argmin(cells))]}."
        )
    if not np.all(cells.sum(axis=-1) == read):
        raise AssertionError("cells do not sum to READ")
    return cells


def excitation_weights(log_a, mu_tot):
    """Per-cell mean Stokes counts.

    Parameters are log-amplitudes. Excitation rates scale as a^2, so the
    weights use exp(2 log a). The old materializer stored log(a^2); the factor
    2 is applied here, at the boundary, and nowhere else.

    Relative weights are normalized and then scaled by mu_tot = sum_j w_j.
    A global shift of log_a is gauge: it cancels in the normalization.
    """
    log_a = np.asarray(log_a, dtype=np.float64)
    mu = np.asarray(mu_tot, dtype=np.float64)
    if log_a.ndim == 1:
        log_a = log_a.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    if np.any(mu <= 0):
        raise ValueError("mu_tot must be positive")
    log_w = 2.0 * log_a
    log_w = log_w - log_w.max(axis=-1, keepdims=True)
    unnorm = np.exp(log_w)
    rel = unnorm / unnorm.sum(axis=-1, keepdims=True)
    mu = np.reshape(mu, (-1, 1))
    if mu.shape[0] == 1 and rel.shape[0] != 1:
        mu = np.broadcast_to(mu, (rel.shape[0], 1))
    if mu.shape[0] != rel.shape[0]:
        raise ValueError(f"mu_tot batch {mu.shape[0]} != log_a batch {rel.shape[0]}")
    w = rel * mu
    return w[0] if squeeze else w


def coherent_amplitudes_numpy(w, phi):
    """c_j = sqrt(w_j / sum w) exp(i phi_j), so sum |c_j|^2 = 1."""
    w = np.asarray(w, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("Stokes means must be non-negative")
    total = w.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("total Stokes mean must be positive")
    c_mod = np.sqrt(w / total)
    c = c_mod * np.exp(1j * phi)
    norm = np.sum(np.abs(c) ** 2, axis=-1)
    if not np.allclose(norm, 1.0, atol=1e-8):
        raise AssertionError(f"sum |c|^2 = {norm}, expected 1")
    return c


def helmert_matrix(n: int) -> np.ndarray:
    """Orthonormal (n-1, n) contrast matrix, rows orthogonal to the all-ones vector."""
    if n < 2:
        raise ValueError(f"Helmert matrix needs n >= 2, got {n}")
    rows = []
    for k in range(1, n):
        scale = 1.0 / np.sqrt(k * (k + 1))
        row = np.zeros(n, dtype=np.float64)
        row[:k] = scale
        row[k] = -k * scale
        rows.append(row)
    H = np.stack(rows, axis=0)
    return H


def helmert_reduce(x):
    """Gauge-reduced coordinates. A constant shift of x is annihilated."""
    x = np.asarray(x, dtype=np.float64)
    H = helmert_matrix(x.shape[-1])
    return H @ x if x.ndim == 1 else x @ H.T


def lift_helmert(coords, n: int) -> np.ndarray:
    """Lift Helmert coordinates to a length-n vector with entries summing to 0.

    No per-coordinate 2π wrap. Phase branches are chosen by this linear lift.
    """
    coords = np.asarray(coords, dtype=np.float64)
    H = helmert_matrix(n)
    if coords.shape[-1] != n - 1:
        raise ValueError(f"expected {n - 1} coordinates, got {coords.shape[-1]}")
    return H.T @ coords if coords.ndim == 1 else coords @ H


def gauge_fix_phases(phi):
    """Subtract phi[..., 0]. Do not wrap each phase into (-π, π)."""
    phi = np.asarray(phi, dtype=np.float64)
    return phi - phi[..., :1]


def assert_sigma1(rho, atol: float = 1e-8) -> None:
    """sigma1 is an N x N Hermitian PSD state with trace 1."""
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise AssertionError(f"sigma1 must be square, got {rho.shape}")
    if not np.allclose(rho, rho.conj().T, atol=atol):
        raise AssertionError("sigma1 is not Hermitian")
    trace = np.trace(rho).real
    if abs(trace - 1.0) > atol:
        raise AssertionError(f"Tr(sigma1) = {trace}, expected 1")
    evals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    if evals.min() < -1e-10:
        raise AssertionError(f"sigma1 is not PSD, min eig {evals.min()}")


def compose_r1(p1: float, sigma1: np.ndarray) -> np.ndarray:
    """R1 = p1 * sigma1, trace p1. p1 is the single-excitation weight."""
    if p1 <= 0:
        raise ValueError(f"p1 must be positive, got {p1}")
    assert_sigma1(sigma1)
    return p1 * np.asarray(sigma1)


def split_r1(r1: np.ndarray):
    """Return (p1, sigma1) with sigma1 = R1 / Tr(R1)."""
    r1 = np.asarray(r1)
    p1 = float(np.trace(r1).real)
    if p1 <= 0:
        raise ValueError(f"Tr(R1) must be positive, got {p1}")
    sigma1 = r1 / p1
    assert_sigma1(sigma1)
    return p1, sigma1
