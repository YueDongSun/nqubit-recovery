"""Entanglement and metrology physics audit suite (V5 physics-math upgrade).

Read-only diagnostics for reconstructed density matrices: tight W-state
witness margins, PPT/negativity profiles, symmetric-subspace weight, and
quantum Fisher information (QFI).  Everything is a pure function of ``rho``,
so frozen campaign snapshots can be audited retroactively at analysis time
without modifying their manifests, and new snapshots can embed the same
fields as additive metadata.

This module is part of the research-only V5 upgrade programme documented in
``docs/ml/architecture/v5/design/V5_PHYSICS_MATH_UPGRADES.md``.  It has no
hardware-write authority.

References
----------
* Peres, Phys. Rev. Lett. 77, 1413 (1996); Horodecki^3, Phys. Lett. A 223,
  1 (1996) — PPT criterion.
* Vidal & Werner, Phys. Rev. A 65, 032314 (2002) — computable negativity.
* Guhne & Toth, Phys. Rep. 474, 1 (2009) — fidelity witnesses; the
  biseparable overlap bound ``(N-1)/N`` for |W_N>.
* Wei & Goldbart, Phys. Rev. A 68, 042307 (2003) — fully-separable overlap
  bound ``((N-1)/N)^(N-1)`` for |W_N>.
* Bourennane et al., Phys. Rev. Lett. 92, 087902 (2004) — experimental W
  witnesses.
* Toth et al., Phys. Rev. Lett. 105, 250403 (2010) — permutationally
  invariant (symmetric-subspace) tomography for W/Dicke experiments.
* Pezze & Smerzi, Phys. Rev. Lett. 102, 100401 (2009) — entanglement
  certification via (quantum) Fisher information.
* Hyllus et al., Phys. Rev. A 85, 022321 (2012); Toth, Phys. Rev. A 85,
  022322 (2012) — QFI and multipartite entanglement depth.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# basic helpers (NumPy-only, kept local so physics_models stays lightweight)
# ---------------------------------------------------------------------------

def physicalize_density(rho: np.ndarray) -> np.ndarray:
    """Symmetrize, clip negative eigenvalues, and renormalize to trace one."""

    matrix = np.asarray(rho, dtype=complex)
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    values = np.clip(np.real(values), 0.0, None)
    if float(np.sum(values)) <= EPS:
        values = np.ones_like(values)
    projected = (vectors * values) @ vectors.conj().T
    return projected / np.trace(projected)


def unique_bipartition_masks(num_qubits: int) -> List[int]:
    """Canonical bipartition masks (one representative per cut)."""

    full = (1 << int(num_qubits)) - 1
    return [mask for mask in range(1, full) if mask < (full ^ mask)]


def mask_label(mask: int, num_qubits: int) -> str:
    bits = ["1" if int(mask) & (1 << q) else "0" for q in range(int(num_qubits))]
    return "q" + "".join(bits)


def partial_transpose(rho: np.ndarray, num_qubits: int, mask: int) -> np.ndarray:
    """Partial transpose over the qubits selected by ``mask`` (full space)."""

    n = int(num_qubits)
    tensor = np.asarray(rho, dtype=complex).reshape([2] * (2 * n))
    axes = list(range(2 * n))
    for q in range(n):
        if int(mask) & (1 << q):
            axes[q], axes[n + q] = axes[n + q], axes[q]
    return np.transpose(tensor, axes=axes).reshape(2**n, 2**n)


def w_state_vector(num_qubits: int) -> np.ndarray:
    """Uniform |W_N> embedded in the full 2^N Hilbert space."""

    vector = np.zeros(2 ** int(num_qubits), dtype=complex)
    for q in range(int(num_qubits)):
        vector[1 << q] = 1.0 / np.sqrt(float(num_qubits))
    return vector


# ---------------------------------------------------------------------------
# PPT / negativity
# ---------------------------------------------------------------------------

def negativity_profile(rho: np.ndarray, num_qubits: int) -> Dict[str, Any]:
    """Full-space PPT negativity for every canonical bipartition.

    Returns per-cut negativity, most negative partial-transpose eigenvalue,
    and log-negativity, plus the worst cut summary.  For an ideal W state
    every cut is entangled, so a near-zero profile under high counts is a
    reconstruction-failure (or genuine full decoherence) signal.
    """

    state = physicalize_density(rho)
    cuts: List[Dict[str, Any]] = []
    for mask in unique_bipartition_masks(num_qubits):
        transposed = partial_transpose(state, num_qubits, mask)
        eigenvalues = np.real(np.linalg.eigvalsh(0.5 * (transposed + transposed.conj().T)))
        trace_norm = float(np.sum(np.abs(eigenvalues)))
        negativity = 0.5 * max(0.0, trace_norm - 1.0)
        cuts.append({
            "cut": mask_label(mask, num_qubits),
            "mask": int(mask),
            "negativity": negativity,
            "min_eigenvalue": float(np.min(eigenvalues)),
            "log_negativity": float(np.log2(max(trace_norm, EPS))),
        })
    worst = min(cuts, key=lambda entry: entry["negativity"]) if cuts else None
    return {
        "cuts": cuts,
        "min_negativity": float(worst["negativity"]) if worst else 0.0,
        "min_negativity_cut": str(worst["cut"]) if worst else "",
        "max_negativity": float(max(entry["negativity"] for entry in cuts)) if cuts else 0.0,
    }


def w_state_reference_negativities(num_qubits: int) -> Dict[str, float]:
    """Ideal |W_N> negativity per canonical cut, ``sqrt(p_T (1 - p_T))``."""

    n = int(num_qubits)
    references: Dict[str, float] = {}
    for mask in unique_bipartition_masks(n):
        size = bin(mask).count("1")
        p_t = size / float(n)
        references[mask_label(mask, n)] = float(np.sqrt(p_t * (1.0 - p_t)))
    return references


# ---------------------------------------------------------------------------
# W-state fidelity witness with tight margins
# ---------------------------------------------------------------------------

def single_excitation_target_vector(
    num_qubits: int,
    amplitudes: Optional[Sequence[float]] = None,
    phases: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Target W-like state ``sum_i a_i e^{i phi_i} |i>`` in the full space."""

    n = int(num_qubits)
    amps = np.ones(n, dtype=float) if amplitudes is None else np.asarray(amplitudes, dtype=float).reshape(-1)
    if amps.size != n:
        raise ValueError(f"amplitudes must have {n} entries, got {amps.size}")
    phs = np.zeros(n, dtype=float) if phases is None else np.asarray(phases, dtype=float).reshape(-1)
    if phs.size != n:
        raise ValueError(f"phases must have {n} entries, got {phs.size}")
    core = amps * np.exp(1j * phs)
    norm = np.linalg.norm(core)
    if norm <= EPS:
        raise ValueError("target amplitudes must not all vanish")
    core = core / norm
    vector = np.zeros(2**n, dtype=complex)
    for q in range(n):
        vector[1 << q] = core[q]
    return vector


def biseparable_bound_single_excitation(amplitudes: Sequence[float]) -> float:
    """Maximum biseparable overlap of a single-excitation pure state.

    A single-excitation state ``sum_i a_i |i>`` is, across cut T|B, an
    effective two-level state with Schmidt coefficients squared
    ``p_T = sum_{i in T} |a_i|^2`` and ``1 - p_T``; the largest biseparable
    overlap is the maximum over cuts of the larger Schmidt coefficient.
    Reduces to ``(N-1)/N`` for the uniform W state.
    """

    amps = np.asarray(amplitudes, dtype=float).reshape(-1)
    pops = amps**2
    total = float(np.sum(pops))
    if total <= EPS:
        raise ValueError("amplitudes must not all vanish")
    pops = pops / total
    n = pops.size
    full = (1 << n) - 1
    best = 0.0
    for mask in range(1, full):
        p_t = float(sum(pops[q] for q in range(n) if mask & (1 << q)))
        best = max(best, p_t, 1.0 - p_t)
    return best


def fully_separable_bound_single_excitation(
    amplitudes: Sequence[float],
    *,
    restarts: int = 8,
    sweeps: int = 25,
    seed: int = 20260722,
) -> float:
    """Geometric-measure overlap of a single-excitation state with product states.

    Maximize ``|<phi|psi>|`` over product states ``|phi> = tensor_i (alpha_i
    |0> + beta_i |1>)``; ``<phi|psi> = sum_i a_i beta_i prod_{j != i} alpha_j``
    for single-excitation ``|psi>``.  Alternating least squares over qubits
    with seeded restarts.  Reduces to ``((N-1)/N)^((N-1)/2)`` (amplitude) for
    the uniform W state, i.e. overlap-squared ``((N-1)/N)^(N-1)``.
    """

    amps = np.asarray(amplitudes, dtype=complex).reshape(-1)
    norm = np.linalg.norm(amps)
    if norm <= EPS:
        raise ValueError("amplitudes must not all vanish")
    # Product-state phases can absorb the target phases (beta_i -> beta_i
    # e^{-i arg a_i}), so the maximization is over real non-negative
    # (alpha_i, beta_i) with amplitude magnitudes only.
    amps = np.abs(amps) / norm
    n = amps.size
    rng = np.random.default_rng(seed)

    def _overlap(params: np.ndarray) -> float:
        alpha = params[:, 0]
        beta = params[:, 1]
        total = 0.0
        for i in range(n):
            total += amps[i] * beta[i] * np.prod(np.delete(alpha, i))
        return float(np.abs(total))

    def _normalize(row: np.ndarray) -> np.ndarray:
        norm_row = np.linalg.norm(row)
        return row / norm_row if norm_row > EPS else np.array([1.0, 0.0])

    best = 0.0
    for restart in range(max(1, int(restarts))):
        if restart == 0:
            params = np.tile(np.array([np.sqrt((n - 1.0) / n), np.sqrt(1.0 / n)]), (n, 1))
        else:
            params = rng.normal(size=(n, 2))
            params = np.array([_normalize(row) for row in params])
        for _ in range(max(1, int(sweeps))):
            for i in range(n):
                others = np.delete(np.arange(n), i)
                a_coeff = 0.0
                for k in others:
                    a_coeff += amps[k] * params[k, 1] * np.prod(params[others[others != k], 0])
                b_coeff = amps[i] * np.prod(params[others, 0])
                combined = np.array([a_coeff, b_coeff])
                norm_c = np.linalg.norm(combined)
                if norm_c > EPS:
                    params[i] = combined / norm_c
            best = max(best, _overlap(params))
    return float(best**2)


def w_witness_audit(
    rho: np.ndarray,
    num_qubits: int,
    *,
    target_amplitudes: Optional[Sequence[float]] = None,
    target_phases: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """W-state fidelity-witness evaluation with tight separability margins.

    ``witness_value = genuine_bound - <target|rho|target>`` is the expectation
    of ``W = genuine_bound * I - |target><target>``; negative values certify
    genuine multipartite entanglement.  For the uniform W target the bounds
    are analytic: ``(N-1)/N`` (biseparable, Guhne & Toth 2009) and
    ``((N-1)/N)^(N-1)`` (fully separable, Wei & Goldbart 2003).  For weighted
    targets the biseparable bound is the exact Schmidt maximum and the
    fully-separable bound is ALS-estimated (marked as a lower estimate).
    """

    n = int(num_qubits)
    state = physicalize_density(rho)
    target = single_excitation_target_vector(n, target_amplitudes, target_phases)
    overlap = float(np.clip(np.real(np.vdot(target, state @ target)), 0.0, 1.0))
    uniform = target_amplitudes is None and target_phases is None
    core_amplitudes = np.abs(target[[1 << q for q in range(n)]])
    genuine_bound = (
        (n - 1.0) / n if uniform else biseparable_bound_single_excitation(core_amplitudes)
    )
    if uniform:
        separable_bound = genuine_bound ** (n - 1.0)
        separable_bound_kind = "analytic"
    else:
        separable_bound = fully_separable_bound_single_excitation(core_amplitudes)
        separable_bound_kind = "als_lower_estimate"
    return {
        "w_overlap": overlap,
        "genuine_bound": float(genuine_bound),
        "separable_bound": float(separable_bound),
        "separable_bound_kind": separable_bound_kind,
        "witness_value": float(genuine_bound - overlap),
        "genuine_margin": float(overlap - genuine_bound),
        "separable_margin": float(overlap - separable_bound),
        "certified_genuine_multipartite": bool(overlap > genuine_bound),
        "certified_entangled": bool(overlap > separable_bound),
    }


# ---------------------------------------------------------------------------
# symmetric (Dicke) subspace
# ---------------------------------------------------------------------------

def dicke_states(num_qubits: int) -> List[Tuple[int, np.ndarray]]:
    """All ``N + 1`` Dicke states ``|D_k^N>`` spanning the symmetric subspace."""

    n = int(num_qubits)
    states: List[Tuple[int, np.ndarray]] = []
    for excitations in range(n + 1):
        vector = np.zeros(2**n, dtype=complex)
        members = list(combinations(range(n), excitations))
        for member in members:
            index = sum(1 << q for q in member)
            vector[index] = 1.0 / np.sqrt(float(len(members)))
        states.append((excitations, vector))
    return states


def symmetric_subspace_weight(rho: np.ndarray, num_qubits: int) -> Dict[str, Any]:
    """``Tr(Pi_sym rho)``: population inside the fully symmetric (Dicke) subspace.

    Balanced-W experiments target the symmetric subspace; the weight is a
    physically motivated denoising meter, and the per-excitation breakdown
    separates single-excitation W content from multi-excitation leakage
    (Toth et al., PRL 105, 250403, 2010).
    """

    state = physicalize_density(rho)
    per_excitation: Dict[str, float] = {}
    total = 0.0
    for excitations, vector in dicke_states(num_qubits):
        weight = float(np.real(np.vdot(vector, state @ vector)))
        per_excitation[f"dicke_{excitations}"] = weight
        total += weight
    return {
        "symmetric_weight": float(np.clip(total, 0.0, 1.0)),
        "per_excitation": per_excitation,
        "single_excitation_symmetric_weight": per_excitation.get("dicke_1", 0.0),
    }


# ---------------------------------------------------------------------------
# quantum Fisher information
# ---------------------------------------------------------------------------

def single_excitation_phase_generator(
    num_qubits: int,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Collective local-phase generator ``J = sum_i g_i n_i`` (full space).

    ``n_i`` is the excitation-number operator of qubit ``i``.  Default weights
    are the zero-mean centered indices, i.e. the generator of differential
    phase gradients across modes (the common mode is unobservable and carries
    no QFI).
    """

    n = int(num_qubits)
    g = (
        np.arange(n, dtype=float) - 0.5 * (n - 1.0)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    if g.size != n:
        raise ValueError(f"generator weights must have {n} entries, got {g.size}")
    generator = np.zeros((2**n, 2**n), dtype=complex)
    for q in range(n):
        generator[1 << q, 1 << q] = g[q]
    return generator


def quantum_fisher_information(rho: np.ndarray, generator: np.ndarray) -> float:
    """QFI ``F_Q[rho, G] = 2 sum_{ij} (l_i - l_j)^2 / (l_i + l_j) |G_ij|^2``.

    Pure-state limit gives ``4 Var(G)``; the maximally mixed state gives 0.
    """

    state = physicalize_density(rho)
    values, vectors = np.linalg.eigh(state)
    g_eigen = vectors.conj().T @ np.asarray(generator, dtype=complex) @ vectors
    lam = np.real(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        coefficient = (lam[:, None] - lam[None, :]) ** 2 / (lam[:, None] + lam[None, :])
    coefficient = np.where(lam[:, None] + lam[None, :] > EPS, coefficient, 0.0)
    return float(2.0 * np.sum(coefficient * np.abs(g_eigen) ** 2))


def qfi_shot_noise_bound(weights: Sequence[float]) -> float:
    """Separable-state QFI ceiling ``sum_i g_i^2`` for ``J = sum_i g_i n_i``.

    Each local generator ``g_i n_i`` has variance at most ``g_i^2 / 4`` on any
    qubit state, so product states obey ``F_Q <= 4 sum_i Var(g_i n_i) <=
    sum_i g_i^2`` (Pezze & Smerzi, PRL 102, 100401, 2009).  Exceeding the
    bound certifies entanglement participating in the phase estimation.
    """

    g = np.asarray(weights, dtype=float).reshape(-1)
    return float(np.sum(g**2))


# ---------------------------------------------------------------------------
# combined audit
# ---------------------------------------------------------------------------

def physics_audit(
    rho: np.ndarray,
    num_qubits: int,
    *,
    target_amplitudes: Optional[Sequence[float]] = None,
    target_phases: Optional[Sequence[float]] = None,
    generator_weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Combined read-only physics audit of a reconstructed density matrix.

    Pure function of ``rho`` — safe to apply to frozen campaign snapshots at
    analysis time, and suitable as additive metadata for new snapshots.
    """

    state = physicalize_density(rho)
    purity = float(np.real(np.trace(state @ state)))
    witness = w_witness_audit(
        state,
        num_qubits,
        target_amplitudes=target_amplitudes,
        target_phases=target_phases,
    )
    negativity = negativity_profile(state, num_qubits)
    reference = w_state_reference_negativities(num_qubits)
    symmetric = symmetric_subspace_weight(state, num_qubits)
    generator = single_excitation_phase_generator(num_qubits, generator_weights)
    f_q = quantum_fisher_information(state, generator)
    weights = (
        (np.arange(int(num_qubits), dtype=float) - 0.5 * (int(num_qubits) - 1.0))
        if generator_weights is None
        else np.asarray(generator_weights, dtype=float)
    )
    shot_noise = qfi_shot_noise_bound(weights)
    return {
        "schema": "physics_audit_v1",
        "v5_upgrade": True,
        "num_qubits": int(num_qubits),
        "purity": purity,
        "witness": witness,
        "negativity": negativity,
        "w_state_reference_negativity": reference,
        "symmetric_subspace": symmetric,
        "qfi": {
            "value": f_q,
            "generator": "single_excitation_phase",
            "generator_weights": [float(w) for w in weights],
            "shot_noise_bound": shot_noise,
            "qfi_per_qubit": f_q / max(int(num_qubits), 1),
            "certified_entangled_metrology": bool(f_q > shot_noise),
        },
    }


# ---------------------------------------------------------------------------
# core / leakage decomposition (connects the N x N training core to the full
# 2^N reconstruction; quantifies how multi-excitation leakage impacts the core)
# ---------------------------------------------------------------------------

def single_excitation_indices(num_qubits: int) -> List[int]:
    """Computational-basis indices of the ``N`` single-excitation states."""

    n = int(num_qubits)
    return [1 << q for q in range(n)]


def extract_core_block(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    """The ``N x N`` single-excitation core block of a full ``2^N`` density."""

    state = physicalize_density(rho)
    indices = single_excitation_indices(num_qubits)
    return state[np.ix_(indices, indices)].copy()


def excitation_number_weights(rho: np.ndarray, num_qubits: int) -> Dict[str, float]:
    """Population in each excitation-number sector ``|D_k^N>`` (k = 0..N)."""

    state = physicalize_density(rho)
    n = int(num_qubits)
    diagonal = np.real(np.diag(state))
    weights: Dict[str, float] = {}
    for k in range(n + 1):
        sector = [m for m in range(2 ** n) if bin(m).count("1") == k]
        weights[f"excitation_{k}"] = float(np.sum(diagonal[sector])) if sector else 0.0
    return weights


def leakage_decomposition(rho: np.ndarray, num_qubits: int) -> Dict[str, Any]:
    """Split a full ``2^N`` state into single-excitation core + leakage (G audit).

    The W physics lives in the ``N x N`` single-excitation core (the training
    object); everything else - vacuum, double/multi-excitation - is *leakage*
    that biases core-based conclusions.  This decomposition reports the core
    weight, the per-excitation-number breakdown, and the leakage weight so a
    high-leakage snapshot is flagged before its core is trusted for feedback.
    """

    n = int(num_qubits)
    state = physicalize_density(rho)
    core = extract_core_block(state, n)
    core_weight = float(np.real(np.trace(core)))
    per_excitation = excitation_number_weights(state, n)
    leakage_weight = 1.0 - core_weight
    # Ideal W: excitation_1 == 1, everything else 0, core weight == 1.
    return {
        "schema": "leakage_decomposition_v1",
        "v5_upgrade": True,
        "num_qubits": n,
        "core_weight": float(np.clip(core_weight, 0.0, 1.0)),
        "leakage_weight": float(np.clip(leakage_weight, 0.0, 1.0)),
        "vacuum_weight": per_excitation["excitation_0"],
        "single_excitation_weight": per_excitation["excitation_1"],
        "double_excitation_weight": per_excitation["excitation_2"],
        "multi_excitation_weight": sum(
            per_excitation[f"excitation_{k}"] for k in range(3, n + 1)
        ),
        "per_excitation": per_excitation,
        "core_block": core,
        "high_leakage": bool(leakage_weight > 0.1),
    }


# ---------------------------------------------------------------------------
# Husimi Q on the full 2^N space (product coherent states; captures leakage)
# ---------------------------------------------------------------------------

def husimi_q_full(
    rho: np.ndarray,
    num_qubits: int,
    *,
    theta_grid: int = 16,
    phi_grid: int = 32,
) -> Dict[str, Any]:
    """Spin-coherent-state Husimi Q on the Bloch sphere from the full ``2^N`` state.

    ``Q(theta, phi) = <theta, phi| rho |theta, phi>`` with the product coherent
    state ``|theta, phi> = otimes_i (c|0> + e^{i phi} s |1>)``.  Unlike the
    single-excitation ``CP^{N-1}`` core Q (phase-blind to leakage), this captures
    multi-excitation content, so it is the audit-side image that reveals how
    double-excitation leakage distorts the W distribution.  Returns the grid
    plus summary statistics (max, visibility, integrated weight).
    """

    n = int(num_qubits)
    state = physicalize_density(rho)
    diagonal = np.real(np.diag(state))
    populations = np.array([bin(m).count("1") for m in range(2 ** n)], dtype=float)
    thetas = np.linspace(0.0, np.pi, int(theta_grid))
    phis = np.linspace(0.0, 2.0 * np.pi, int(phi_grid), endpoint=False)
    theta_grid_vals, phi_grid_vals = np.meshgrid(thetas, phis, indexing="ij")
    c = np.cos(theta_grid_vals / 2.0)
    s = np.sin(theta_grid_vals / 2.0)
    # Q(theta, phi) = sum_mn rho_mn c^{2N-k_m-k_n} s^{k_m+k_n} e^{i phi (k_m - k_n)}.
    q_values = np.zeros_like(theta_grid_vals, dtype=float)
    for m in range(2 ** n):
        km = populations[m]
        for nn in range(2 ** n):
            value = state[m, nn]
            if abs(value) < 1e-15:
                continue
            kn = populations[nn]
            q_values = q_values + np.real(
                value
                * c ** (2 * n - km - kn)
                * s ** (km + kn)
                * np.exp(1j * phi_grid_vals * (km - kn))
            )
    return {
        "schema": "husimi_q_full_v1",
        "v5_upgrade": True,
        "num_qubits": n,
        "theta_grid": thetas.tolist(),
        "phi_grid": phis.tolist(),
        "q_values": q_values.tolist(),
        "q_max": float(np.max(q_values)),
        "q_mean": float(np.mean(q_values)),
        "visibility": float((np.max(q_values) - np.min(q_values)) / max(np.mean(q_values), EPS)),
    }


# ---------------------------------------------------------------------------
# entanglement entropy profile (full space, all canonical cuts; audit)
# ---------------------------------------------------------------------------

def reduced_density_matrix(rho: np.ndarray, num_qubits: int, mask: int) -> np.ndarray:
    """Partial trace over the qubits selected by ``mask`` (full space).

    The traced-out qubits' row and column axes must share an index label so they
    are *contracted* (sum over the diagonal), not summed independently.
    """

    n = int(num_qubits)
    state = physicalize_density(rho)
    tensor = state.reshape([2] * (2 * n))
    row_chars = [chr(ord("a") + q) for q in range(n)]
    col_chars = [chr(ord("a") + n + q) for q in range(n)]
    keep = [q for q in range(n) if not (int(mask) & (1 << q))]
    for q in range(n):
        if int(mask) & (1 << q):
            col_chars[q] = row_chars[q]  # contract traced-out qubit's row/col
    input_sub = "".join(row_chars + col_chars)
    output_sub = "".join([row_chars[q] for q in keep] + [col_chars[q] for q in keep])
    traced = np.einsum(input_sub + "->" + output_sub, tensor)
    dim = 2 ** len(keep)
    return traced.reshape(dim, dim)


def entanglement_entropy_profile(rho: np.ndarray, num_qubits: int) -> Dict[str, Any]:
    """Von Neumann entropy of every reduced state, all canonical bipartitions.

    For a pure W state across cut ``T|B`` the entropy is the binary entropy
    ``H(p_T)`` with ``p_T = |T|/N``; the profile therefore diagnoses the
    entanglement structure complementing the negativity profile.  All canonical
    cuts (audit-side; exponential in N, intended for the small-N real rho).
    """

    n = int(num_qubits)
    state = physicalize_density(rho)
    cuts: List[Dict[str, Any]] = []
    for mask in unique_bipartition_masks(n):
        reduced = reduced_density_matrix(state, n, mask)
        evals = np.clip(np.real(np.linalg.eigvalsh(reduced)), EPS, 1.0)
        evals = evals / np.sum(evals)
        entropy = float(-np.sum(evals * np.log(evals)))
        cuts.append({
            "cut": mask_label(mask, n),
            "mask": int(mask),
            "entanglement_entropy": entropy,
            "max_entropy": float(np.log2(2 ** min(bin(mask).count("1"), n - bin(mask).count("1")))),
        })
    return {
        "schema": "entanglement_entropy_profile_v1",
        "v5_upgrade": True,
        "cuts": cuts,
        "min_entropy": float(min(c["entanglement_entropy"] for c in cuts)) if cuts else 0.0,
        "max_entropy": float(max(c["entanglement_entropy"] for c in cuts)) if cuts else 0.0,
    }

