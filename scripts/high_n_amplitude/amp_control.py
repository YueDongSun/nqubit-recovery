"""Bounded relative-amplitude preparation for a phase-aligned conditional W family.

The map multiplies amplitudes by exp(u), renormalizes the conditional state,
and holds excitation mass and visibility fixed. The ideal per-mode controls
here carry no RF calibration or passive-filter survival factor.
"""
from __future__ import annotations

import numpy as np


POPULATION_SUM_TOLERANCE = 1e-6  # Allows populations from a float32 softmax.


def _population_matrix(population):
    if np.iscomplexobj(population):
        raise ValueError("Populations must be real.")
    p = np.asarray(population, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("Populations must have shape [rows, modes] with at least two modes.")
    if not np.isfinite(p).all() or np.any(p < 0.):
        raise ValueError("Populations must be finite and nonnegative.")
    if not np.allclose(p.sum(axis=1), 1., atol=POPULATION_SUM_TOLERANCE, rtol=0.):
        raise ValueError("Each population row must sum to one.")
    return p


def centered_action(p_hat, bound):
    """Return real, zero-mean per-mode log-amplitude commands using the estimate.

    On each positive-support mode the unconstrained command is -log(p)/2
    minus its support mean. A single factor in [0,1] scales that vector
    to fit the per-mode box. A zero estimate gets zero command. Thus the
    vector remains centered even when the estimated support is incomplete.
    Using truth in place of the estimate gives a feasible matching reference.
    For an active box this construction makes no optimality claim.
    """
    p = _population_matrix(p_hat)
    if np.ndim(bound) != 0 or np.iscomplexobj(bound):
        raise ValueError("The per-mode bound must be a real scalar.")
    bound = float(bound)
    if not np.isfinite(bound) or bound < 0.:
        raise ValueError("The per-mode bound must be finite and nonnegative.")
    support = p > 0.
    desired = -0.5 * np.log(p, out=np.zeros_like(p), where=support)
    support_mean = desired.sum(axis=1, keepdims=True) / support.sum(axis=1, keepdims=True)
    desired = np.where(support, desired - support_mean, 0.)
    peak = np.max(np.abs(desired), axis=1, keepdims=True)
    scale = np.divide(bound, peak, out=np.ones_like(peak), where=peak > bound)
    return desired * scale


def conditional_fidelity(p_true, actions, visibility=1.0):
    """Score each action against the aligned W target, conditional on K=1.

    For normalized positive output amplitudes a, F1 = [V*(sum a)^2+1-V]/N.
    Negative/zero gain is retained. Visibility can be scalar or one value
    per row. Natural-log arithmetic preserves weak and exactly zero modes.
    """
    p = _population_matrix(p_true)
    if np.iscomplexobj(actions) or np.iscomplexobj(visibility):
        raise ValueError("Amplitude-only actions and visibility must be real.")
    u = np.asarray(actions, dtype=np.float64)
    if u.shape != p.shape or not np.isfinite(u).all():
        raise ValueError("Each mode needs one finite log-amplitude command.")
    v = np.asarray(visibility, dtype=np.float64)
    if v.shape not in ((), (len(p),)) or not np.isfinite(v).all() or np.any((v < 0.) | (v > 1.)):
        raise ValueError("Supply fixed visibility in [0,1], scalar or one per row.")
    support = p > 0.
    log_mag = 0.5 * np.log(p, out=np.full_like(p, -np.inf), where=support)
    u = np.where(support, u, -np.inf)
    # A common log gain cancels in normalized preparation.
    with np.errstate(over="ignore"):
        log_mag += u - np.max(u, axis=1, keepdims=True)
    weights = np.exp(log_mag - np.max(log_mag, axis=1, keepdims=True))
    coherent_sum_sq = weights.sum(axis=1)**2 / np.einsum("ij,ij->i", weights, weights)
    return (v * coherent_sum_sq + (1. - v)) / p.shape[1]
