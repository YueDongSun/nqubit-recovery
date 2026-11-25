"""Differentiable physics primitives for the V5 physics-math architecture upgrade.

Torch, batched, autograd-safe companions to the NumPy physics in
``ml_recon.physics_models`` and ``ml_recon.quantum_representation``.
Everything here operates either on the single-excitation "core" density block
(dimension = qubit count, the convention of ``PhysicalDensityHead`` and the
``rho_image`` core block) or on Helmert gauge coordinates / materialized phase
offsets (the feedback action surfaces).

This module is part of the research-only V5 upgrade programme documented in
``docs/ml/architecture/v5/design/V5_PHYSICS_MATH_UPGRADES.md``.  It has no
hardware-write authority: it never touches ``ExperimentRunner``,
``matrix/ml_delta``, AWG sequences, or FPGA state.

Key formulas (single-excitation subspace):

* Partial-transpose negativity of a single-excitation state across cut T|B is
  the Frobenius norm of the cross block, ``N(T|B) = ||rho_{T,B}||_F``.  The
  full-space partial transpose decomposes into PSD same-side blocks plus one
  arrowhead block through the vacuum state whose only negative eigenvalue is
  ``-||rho_{T,B}||_F``.  This makes the negativity profile exactly
  differentiable without an eigensolver.
* W-state fidelity witness (Guhne & Toth, Phys. Rep. 474, 2009):
  ``W = ((N-1)/N) I - |W><W|`` certifies genuine multipartite entanglement
  when ``Tr(W rho) < 0``; the full-separability bound is
  ``((N-1)/N)^(N-1)`` (Wei & Goldbart, Phys. Rev. A 68, 042307, 2003).
* Circular phase residuals use ``1 - cos(delta)`` and the von Mises NLL
  ``-kappa cos(delta) + log I0(kappa)`` (Mardia & Jupp, Directional
  Statistics; Prokudin et al., NeurIPS 2018).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional ML dependency
    torch = None
    F = None
    TORCH_AVAILABLE = False


def unique_bipartition_cuts(num_qubits: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Canonical bipartitions (one representative per cut, mirror of
    ``tomography_features._unique_bipartition_masks``)."""

    n = int(num_qubits)
    full = (1 << n) - 1
    cuts: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for mask in range(1, full):
        if mask < (full ^ mask):
            rows = tuple(q for q in range(n) if mask & (1 << q))
            cols = tuple(q for q in range(n) if not mask & (1 << q))
            cuts.append((rows, cols))
    return cuts


def w_state_separability_bounds(num_qubits: int) -> Tuple[float, float]:
    """Return (genuine_multipartite_bound, full_separability_bound) for |W_N>.

    * Biseparable states overlap at most ``(N-1)/N`` with |W_N> (Guhne & Toth
      2009): fidelity above it certifies genuine N-partite entanglement.
    * Fully separable states overlap at most ``((N-1)/N)^(N-1)`` (Wei &
      Goldbart 2003): fidelity above it certifies entanglement.
    """

    n = int(num_qubits)
    genuine = (n - 1.0) / n
    separable = genuine ** (n - 1.0)
    return genuine, separable


if TORCH_AVAILABLE:

    def circular_phase_error(delta_rad: "torch.Tensor") -> "torch.Tensor":
        """Wrap-invariant phase error ``1 - cos(delta)``, elementwise.

        Smooth at +/- pi unlike Euclidean residuals; identical quadratic
        behaviour to 0.5*delta^2 near zero.
        """

        return 1.0 - torch.cos(delta_rad.float())

    def stable_log_i0(kappa: "torch.Tensor") -> "torch.Tensor":
        """``log I0(kappa)`` with an overflow-safe asymptotic branch."""

        kappa = torch.clamp(kappa.float(), min=1e-8)
        small = torch.clamp(kappa, max=50.0)
        exact = torch.log(torch.i0(small))
        asymptotic = (
            kappa
            - 0.5 * torch.log(2.0 * math.pi * kappa)
            + 1.0 / (8.0 * kappa)
        )
        return torch.where(kappa <= 50.0, exact, asymptotic)

    def von_mises_nll(
        delta_rad: "torch.Tensor",
        log_std: "torch.Tensor",
    ) -> "torch.Tensor":
        """Von Mises negative log-likelihood of an angular residual.

        Concentration ``kappa = exp(-2 * log_std)`` matches the Gaussian
        precision of the existing heteroscedastic head at small sigma, so the
        head parameterization is unchanged; the constant ``log(2*pi)`` is
        dropped to stay comparable with the Gaussian NLL it replaces.
        """

        kappa = torch.exp(-2.0 * log_std.float()).clamp(min=1e-4, max=1e8)
        return -kappa * torch.cos(delta_rad.float()) + stable_log_i0(kappa)

    def helmert_matrix_torch(cell_count: int, *, device=None, dtype=None) -> "torch.Tensor":
        """Torch copy of ``phase_geometry.helmert_matrix`` (orthonormal,
        common-mode-free): maps gauge coordinates to materialized phases."""

        from .phase_geometry import helmert_matrix

        matrix = helmert_matrix(int(cell_count))
        return torch.as_tensor(
            np.asarray(matrix, dtype=np.float32),
            device=device,
            dtype=dtype or torch.float32,
        )

    def gauge_centered_delta(
        predicted_offsets: "torch.Tensor",
        target_offsets: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Per-operation gauge-centering of a materialized phase difference.

        Subtracts the masked mean of ``predicted - target`` along the last
        (cell) axis so the unobservable common-mode phase never contributes to
        a circular residual.
        """

        delta = predicted_offsets.float() - target_offsets.float()
        weight = mask.float()
        mean = (delta * weight).sum(dim=-1, keepdim=True) / torch.clamp(
            weight.sum(dim=-1, keepdim=True), min=1.0
        )
        return delta - mean

    def uhlmann_fidelity_torch(
        rho: "torch.Tensor",
        sigma: "torch.Tensor",
    ) -> "torch.Tensor":
        """Batched squared Uhlmann fidelity ``F(rho, sigma)`` (Jozsa convention).

        Autograd-safe: uses two Hermitian eigendecompositions in complex64.
        Both inputs are assumed physical (Hermitian, PSD, trace one); inputs
        are Hermitized defensively before the decomposition.
        """

        rho64 = rho.to(torch.complex64)
        sigma64 = sigma.to(torch.complex64)
        rho64 = 0.5 * (rho64 + rho64.mH)
        sigma64 = 0.5 * (sigma64 + sigma64.mH)
        evals, evecs = torch.linalg.eigh(rho64)
        root = torch.sqrt(torch.clamp(evals.real, min=0.0)).to(torch.complex64)
        sqrt_rho = (evecs * root.unsqueeze(-2)) @ evecs.mH
        product = sqrt_rho @ sigma64 @ sqrt_rho
        product = 0.5 * (product + product.mH)
        product_evals = torch.linalg.eigvalsh(product).real
        root_sum = torch.sum(torch.sqrt(torch.clamp(product_evals, min=0.0)), dim=-1)
        return torch.clamp(root_sum * root_sum, min=0.0, max=1.0)

    def cross_cut_negativity_torch(
        rho_core: "torch.Tensor",
        num_qubits: int,
    ) -> "torch.Tensor":
        """Negativity profile of a single-excitation state, closed form.

        ``N(T|B) = ||rho_{T,B}||_F`` for every canonical bipartition; returns
        ``[batch, n_cuts]`` (or ``[n_cuts]`` for an unbatched ``[N, N]`` input),
        ordered like :func:`unique_bipartition_cuts`.  Exact for states
        supported on the single-excitation subspace (the ``PhysicalDensityHead``
        convention); differentiable without an eigensolver.
        """

        squeezed = rho_core.ndim == 2
        rho = rho_core.unsqueeze(0) if squeezed else rho_core
        values = []
        for rows, cols in unique_bipartition_cuts(num_qubits):
            block = rho[:, rows][:, :, cols]
            values.append(
                torch.sqrt(torch.sum(torch.abs(block) ** 2, dim=(-2, -1)) + 1e-20)
            )
        result = torch.stack(values, dim=-1)
        return result.squeeze(0) if squeezed else result

    def w_witness_overlap_torch(rho_core: "torch.Tensor") -> "torch.Tensor":
        """``<W_N|rho|W_N>`` for the uniform single-excitation W state.

        ``|W_N> = (1/sqrt(N)) sum_i |i>`` on the core block, so the overlap is
        the plain mean of all density-matrix entries (real by Hermiticity).
        """

        dimension = rho_core.shape[-1]
        total = torch.sum(rho_core, dim=(-2, -1))
        return torch.real(total) / float(dimension)

    def w_witness_margins_torch(
        rho_core: "torch.Tensor",
        num_qubits: int,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return (overlap, genuine_margin, separable_margin) for the W witness.

        Positive ``genuine_margin`` certifies genuine N-partite entanglement;
        positive ``separable_margin`` certifies entanglement.
        """

        genuine, separable = w_state_separability_bounds(num_qubits)
        overlap = w_witness_overlap_torch(rho_core)
        return overlap, overlap - float(genuine), overlap - float(separable)

    # ------------------------------------------------------------------
    # V5 upgrade stage 2 (A-G): the additions below extend the core (N x N,
    # single-excitation, arbitrary-N) differentiable physics.
    # ------------------------------------------------------------------

    def phase_gram_matrix_torch(rho_core: "torch.Tensor") -> "torch.Tensor":
        """Complex phase-only coherence matrix ``G_ij = exp(i arg rho_ij)`` (B).

        Gauge-invariant (a common phase cancels in every difference
        ``arg rho_ij - arg rho_kl``) and permutation-equivariant.  Magnitude is
        discarded, so ``G`` isolates the pure phase structure of the W
        coherence - the angular-synchronization observable.  Vanishing
        off-diagonals are mapped to zero phase (no contribution).
        """

        phase = torch.angle(rho_core)
        magnitude = torch.abs(rho_core)
        return torch.where(
            magnitude > 1e-8, torch.exp(1j * phase), torch.zeros_like(phase, dtype=torch.complex64)
        )

    def one_vs_rest_cuts(num_qubits: int) -> List[Tuple[int, Tuple[int, ...]]]:
        """The ``N`` physically relevant W bipartitions: ``{i}`` vs the rest."""

        n = int(num_qubits)
        return [(i, (i,)) for i in range(n)]

    def cross_cut_negativity_1vsrest_torch(
        rho_core: "torch.Tensor",
        num_qubits: int,
    ) -> "torch.Tensor":
        """1-vs-rest negativity profile on the core, ``[batch, N]`` (D, training).

        For cut ``{i}|rest`` the cross block is row ``i`` / column block of the
        rest; ``N({i}|rest) = ||rho[i, rest]||_F``.  Exact and differentiable
        for single-excitation states, ``O(N)`` cuts (vs the exponential
        all-cuts audit profile).
        """

        n = int(num_qubits)
        values = []
        rest_index = [torch.arange(n, device=rho_core.device) != i for i in range(n)]
        for i in range(n):
            block = rho_core[:, i, :][:, rest_index[i]]
            values.append(
                torch.sqrt(torch.sum(torch.abs(block) ** 2, dim=(-2, -1)) + 1e-20)
            )
        return torch.stack(values, dim=-1)

    def entanglement_entropy_1vsrest_torch(
        rho_core: "torch.Tensor",
        num_qubits: int,
    ) -> "torch.Tensor":
        """1-vs-rest von Neumann entanglement entropy on the core (E, training).

        For a single-excitation state the reduced state of mode ``i`` across
        cut ``{i}|rest`` is exactly ``diag(1 - rho_ii, rho_ii)``: the
        off-diagonal vanishes because the rest-space states ``|vac>`` (mode i
        excited) and ``|chi_i>`` (rest excited) are orthogonal, so no coherence
        couples them.  The entropy is therefore the binary entropy
        ``H(rho_ii)`` - for a pure W state it equals ``H(1/N)``.  Exact,
        differentiable, and ``O(N)``.
        """

        diag = torch.diagonal(rho_core, dim1=-2, dim2=-1).real
        p = torch.clamp(diag, 1e-12, 1.0 - 1e-12)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        return entropy

    def quantum_fisher_information_core_torch(
        rho_core: "torch.Tensor",
        generator_weights: "torch.Tensor",
    ) -> "torch.Tensor":
        """QFI on the core for the local phase generator ``J = sum_i g_i n_i`` (A).

        ``n_i`` is diagonal in the single-excitation basis, so ``J`` is the
        diagonal matrix ``diag(g_1..g_N)`` on the core - block-diagonal, hence
        the single-excitation QFI is exact on the N x N core without any full
        2^N space.  Returns ``[batch]``; pure-state limit ``F_Q = 4 Var(J)``.
        """

        rho64 = 0.5 * (rho_core.to(torch.complex64) + rho_core.to(torch.complex64).mH)
        evals, evecs = torch.linalg.eigh(rho64)
        generator = torch.diag(generator_weights.to(rho_core.device)).to(torch.complex64)
        g_eigen = evecs.mH @ generator @ evecs
        lam = torch.clamp(evals.real, 0.0, None)
        lam_i = lam.unsqueeze(-1)
        lam_j = lam.unsqueeze(-2)
        denom = lam_i + lam_j
        # Safe division: add a tiny epsilon, then mask degenerate (lam_i+lam_j~0) terms.
        coefficient = (lam_i - lam_j) ** 2 / (denom + 1e-30)
        coefficient = torch.where(denom > 1e-12, coefficient, torch.zeros_like(coefficient))
        return torch.clamp(2.0 * torch.sum(coefficient * torch.abs(g_eigen) ** 2, dim=(-2, -1)), min=0.0)

    def qfi_separable_bound(generator_weights: "torch.Tensor") -> "torch.Tensor":
        """Separable shot-noise ceiling ``sum_i g_i^2`` for ``J = sum_i g_i n_i``.

        Each local ``g_i n_i`` has variance at most ``g_i^2 / 4`` on any product
        state, so ``F_Q^sep <= sum_i g_i^2`` (Pezze & Smerzi, PRL 102, 100401).
        Exceeding it certifies metrological entanglement.
        """

        return torch.sum(generator_weights.float() ** 2)

    def default_phase_generator_weights(num_qubits: int, *, device=None) -> "torch.Tensor":
        """Zero-mean centered mode indices ``g_i = i - (N-1)/2`` (AOD-matched).

        The common-mode weight vanishes, so ``J`` generates only observable
        relative phases (the gauge-free direction), matching the AOD action
        surface.
        """

        n = int(num_qubits)
        weights = torch.arange(n, dtype=torch.float32, device=device) - 0.5 * (n - 1.0)
        return weights

    def mixture_von_mises_nll(
        delta_rad: "torch.Tensor",
        log_kappas: "torch.Tensor",
        log_weights: "torch.Tensor",
        component_means_rad: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """K-component mixture von Mises NLL for bimodal phase posteriors (D).

        ``log_kappas`` has shape ``(..., K)`` (one concentration per
        component), ``log_weights`` contains mixture logits and
        ``component_means_rad`` contains the circular component locations. If
        means are omitted all components are centered at zero, preserving the
        historical single-mode interpretation without pretending that
        duplicated concentrations form a mixture. ``delta_rad`` is broadcast
        over the component axis. Returns the *negative* log likelihood with
        the constant ``log(2*pi)`` omitted, matching :func:`von_mises_nll`.
        """

        kappas = torch.exp(log_kappas.float()).clamp(min=1e-4, max=1e8)
        if component_means_rad is None:
            means = torch.zeros_like(kappas)
        else:
            means = torch.broadcast_to(component_means_rad.float(), kappas.shape)
        component_nll = (
            -kappas * torch.cos(delta_rad.float().unsqueeze(-1) - means)
            + stable_log_i0(kappas)
        )
        log_weights_normalized = torch.log_softmax(log_weights.float(), dim=-1)
        return -torch.logsumexp(log_weights_normalized - component_nll, dim=-1)

    def dephasing_channel_diagonal_torch(
        rho_core: "torch.Tensor",
        log_rates: "torch.Tensor",
    ) -> "torch.Tensor":
        """Mode-dependent dephasing ``rho' = Gamma . rho`` with diagonal Gamma (F).

        ``Gamma = diag(exp(-gamma_1..gamma_N))`` (per-mode dephasing rates);
        automatically PSD with unit diagonal, hence CP and trace-preserving.
        Populations are unchanged, off-diagonal coherences decay by
        ``exp(-(gamma_i+gamma_j)/2)``.
        """

        gamma = F.softplus(log_rates.float())
        decay = torch.exp(-0.5 * (gamma.unsqueeze(-2) + gamma.unsqueeze(-1)))
        mask = 1.0 - torch.eye(gamma.shape[-1], device=rho_core.device, dtype=decay.dtype)
        gamma_matrix = decay * mask + torch.diag_embed(torch.ones_like(gamma))
        return gamma_matrix.to(rho_core.dtype) * rho_core

    def dephasing_channel_full_torch(
        rho_core: "torch.Tensor",
        lower_triangular: "torch.Tensor",
    ) -> "torch.Tensor":
        """Full PSD correlation dephasing ``rho' = (LL† / diag(LL†)) . rho`` (F).

        ``lower_triangular`` is the unconstrained Cholesky factor of ``Gamma``
        (shape ``(..., N, N)``); ``Gamma = LL†`` is normalized to unit diagonal so
        the map is CP and trace-preserving and captures cross-mode dephasing
        correlations (THEORY_BASIS section 4.2).  Off by default (opt-in).
        Works batched ``[B, N, N]`` or unbatched ``[N, N]``.
        """

        full = lower_triangular @ lower_triangular.transpose(-2, -1)
        diagonal = torch.clamp(torch.diagonal(full, dim1=-2, dim2=-1).real, min=1e-8)
        # Normalize Gamma = D^{-1/2} (LL†) D^{-1/2} so the diagonal is unit
        # (CP, trace-preserving). d[..., None] * d[..., None, :] broadcasts
        # correctly for both batched [B,N,N] and unbatched [N,N] inputs.
        d = torch.sqrt(diagonal)
        gamma = full / (d.unsqueeze(-1) * d.unsqueeze(-2))
        return gamma.to(rho_core.dtype) * rho_core

    def husimi_q_core_torch(
        rho_core: "torch.Tensor",
        test_amplitudes: "torch.Tensor",
    ) -> "torch.Tensor":
        """Husimi Q on ``CP^{N-1}`` from the single-excitation core (G).

        ``Q(a) = <a|rho|a>`` with ``|a> = sum_i a_i |i>`` a normalized
        single-excitation test state (``a in C^N``, ``|a|=1``).  Phase-sensitive
        over the full core (unlike the symmetric-subspace spin-coherent Q which
        is theta-only for single-excitation states).  ``test_amplitudes`` has
        shape ``(K, N)`` (normalized); returns ``[batch, K]``.
        """

        normalized = F.normalize(test_amplitudes.to(torch.complex64), dim=-1)
        # Q_k = sum_{ij} a_k_i* rho_ij a_k_j  =  Tr(rho |a_k><a_k|)
        overlaps = torch.einsum("ki,bij,kj->bk", normalized.conj(), rho_core.to(torch.complex64), normalized)
        return torch.real(overlaps)

    def coherent_state_grid(num_qubits: int, grid_size: int, *, device=None) -> "torch.Tensor":
        """A permutation-equivariant ``CP^{N-1}`` test-state grid for Husimi Q.

        Built from the N cardinal directions plus structured superpositions on
        a phase ring, so the grid is invariant (up to reordering) under mode
        permutation - matching the W exchange symmetry.
        """

        n = int(num_qubits)
        states = [torch.eye(n, dtype=torch.complex64, device=device)]
        for k in range(grid_size):
            phase = 2.0 * math.pi * k / float(max(grid_size, 1))
            vec = torch.zeros(n, dtype=torch.complex64, device=device)
            vec[0] = math.cos(phase / 2.0)
            if n > 1:
                vec[1] = math.sin(phase / 2.0) * complex(math.cos(phase / 2.0), math.sin(phase / 2.0))
            states.append(vec.unsqueeze(0))
        grid = torch.cat(states, dim=0)
        return F.normalize(grid, dim=-1)

else:  # pragma: no cover - optional ML dependency

    def circular_phase_error(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def stable_log_i0(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def von_mises_nll(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def helmert_matrix_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def gauge_centered_delta(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def uhlmann_fidelity_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def cross_cut_negativity_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def w_witness_overlap_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def w_witness_margins_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def phase_gram_matrix_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def one_vs_rest_cuts(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def cross_cut_negativity_1vsrest_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def entanglement_entropy_1vsrest_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def quantum_fisher_information_core_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def qfi_separable_bound(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def default_phase_generator_weights(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def mixture_von_mises_nll(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def dephasing_channel_diagonal_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def dephasing_channel_full_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def husimi_q_core_torch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")

    def coherent_state_grid(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for differentiable physics")
