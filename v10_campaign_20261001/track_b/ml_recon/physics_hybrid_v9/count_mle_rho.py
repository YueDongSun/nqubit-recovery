"""V9 count-MLE rho estimator (charter section 11).

Implements a true maximum-likelihood rho estimate from the authoritative raw counts:

- ``rho_W = L L^dag / Tr(L L^dag)`` with a complex Cholesky parameterization (guaranteed
  PSD and unit trace by construction);
- joint estimation of W-subspace mass, brightness/scale, dark, background, accidental and
  overdispersion nuisance from the same count batch;
- the primary FPGA-joint beta-binomial likelihood (reused from V8
  ``fpga_joint_observation.nested_beta_binomial_log_likelihood``) used exactly once;
- monotonic-likelihood refinement (accept only non-decreasing iterates) — the positivity
  and monotonicity requirements from Hradil (PRA 55, R1561, 1997), Rehacek et al.
  (PRA 75, 042108, 2007), Blume-Kohout (NJP 12, 043034, 2010);
- parametric-bootstrap / deep-ensemble uncertainty intervals, boundary-eigenvalue
  warnings, and explicit non-identifiability diagnostics (Silva et al. PRA 95, 022107,
  2017; Sim et al. PRA 100, 022333, 2019).

The ratio views (``cs`` / ``cs_minus_ir``) are NOT treated as Poisson raw counts; they
only provide physical init candidates. The refined output is a posterior draw / mean, not
a deterministic corrected real label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .rho8_maps import RETRIEVAL_SHAPES, build_rho8_maps
from .rho8_reconstruction import _load_calibrated_reconstruction_module

_EPS = 1e-8
DARK_MAX = 0.25
ACCIDENTAL_MAX = 0.05
BACKGROUND_MAX = 0.10
SIGNAL_SCALE_MAX = 2.0
CORE_DIM = {"3qubit": 3, "4qubit": 4}


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for V9CountLikelihoodRhoEstimator")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class V9CountMleState:
    """A candidate state: PSD unit-trace rho_W plus bounded nuisance parameters."""

    system_type: str
    rho: np.ndarray  # N x N complex, PSD, unit trace
    w_mass: float = 1.0
    signal_scale: float = 1.0
    dark_prob: float = 0.0
    background_singles_prob: float = 0.0
    accidental_rate: float = 0.0
    concentration: float = 100.0
    boundary_eigenvalue_warning: bool = False

    def validate(self) -> None:
        n = CORE_DIM[self.system_type]
        if self.rho.shape != (n, n):
            raise ValueError(f"rho shape must be {(n, n)}, got {self.rho.shape}")
        if not np.all(np.isfinite(self.rho)):
            raise ValueError("rho is non-finite")
        if not np.allclose(self.rho, self.rho.conj().T, atol=1e-6):
            raise ValueError("rho is not Hermitian")
        if not np.isclose(np.trace(self.rho).real, 1.0, atol=1e-4):
            raise ValueError("rho is not unit trace")
        for name, lo, hi in (
            ("w_mass", 0.0, 1.0), ("signal_scale", 0.0, SIGNAL_SCALE_MAX),
            ("dark_prob", 0.0, DARK_MAX), ("background_singles_prob", 0.0, BACKGROUND_MAX),
            ("accidental_rate", 0.0, ACCIDENTAL_MAX), ("concentration", 1e-6, 1e9),
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < lo or value > hi:
                raise ValueError(f"{name}={value} out of bounds [{lo},{hi}]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_type": self.system_type,
            "w_mass": self.w_mass,
            "signal_scale": self.signal_scale,
            "dark_prob": self.dark_prob,
            "background_singles_prob": self.background_singles_prob,
            "accidental_rate": self.accidental_rate,
            "concentration": self.concentration,
            "boundary_eigenvalue_warning": self.boundary_eigenvalue_warning,
        }


# ---------------------------------------------------------------------------
# Parameterization helpers
# ---------------------------------------------------------------------------


def cholesky_vector_length(n: int) -> int:
    """Number of complex entries in the lower triangle (diagonal real)."""
    return n * (n + 1) // 2


def rho_from_cholesky_vector(v: np.ndarray, n: int) -> np.ndarray:
    """Build rho = L L^dag / Tr(L L^dag) from a complex Cholesky vector.

    ``v`` has length 2*cholesky_vector_length(n): interleaved real/imag, with the
    imaginary parts of diagonal entries forced to zero.
    """
    length = cholesky_vector_length(n)
    if v.size != 2 * length:
        raise ValueError(f"vector length {v.size} != {2 * length}")
    tri = np.zeros((n, n), dtype=np.complex128)
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            re = v[2 * idx]
            im = v[2 * idx + 1]
            tri[i, j] = re + 1j * (im if i != j else 0.0)
            idx += 1
    rho = tri @ tri.conj().T
    trace = np.trace(rho).real
    if trace <= 1e-12:
        raise ValueError("Cholesky matrix is degenerate")
    return rho / trace


def rho_to_cholesky_vector(rho: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rho_from_cholesky_vector` (used to seed refinement).

    Uses a lightly regularized direct Cholesky (``rho + eps*I``) so rank-deficient
    inputs stay stable; the trace normalization in ``rho_from_cholesky_vector`` makes
    the reconstruction match ``rho`` up to ``O(eps)``.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    n = rho.shape[0]
    lower = np.linalg.cholesky(rho + 1e-12 * np.eye(n))
    vec = []
    for i in range(n):
        for j in range(i + 1):
            vec.append(lower[i, j].real)
            vec.append(lower[i, j].imag if i != j else 0.0)
    return np.asarray(vec, dtype=np.float64)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V9CountMleRefinerConfig:
    """Fixed-iteration monotonic refinement settings (frozen before real test)."""

    steps: int = 200
    learning_rate: float = 5e-2
    accept_tol: float = 1e-6
    backtrack_factor: float = 0.5
    boundary_eigenvalue_threshold: float = 1e-3
    seed: int = 0
    n_starts: int = 3  # multiple starts -> evidence-weighted posterior distinct from MLE
    n_bootstrap: int = 4  # parametric bootstrap draws for the data-resampling posterior
    bootstrap_steps: int = 12
    n_laplace: int = 8  # Laplace-approximation posterior draws (curvature-based)


@dataclass
class V9CountMleResult:
    """Output of one estimation run (charter section 11.3).

    ``rho_count_mle`` is the best-likelihood point; ``rho_count_posterior_mean`` is the
    evidence-weighted mixture over the multiple starts (distinct from the MLE point,
    closes V9-R2 failure #13).
    """

    rho_count_mle: np.ndarray
    rho_count_posterior_mean: np.ndarray
    nuisance_posterior: Dict[str, Any]
    likelihood_before: float
    likelihood_after: float
    convergence: bool
    boundary_eigenvalue_warning: bool
    iterations: int
    likelihood_trace: List[float] = field(default_factory=list)
    identifiability: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "likelihood_before": self.likelihood_before,
            "likelihood_after": self.likelihood_after,
            "convergence": self.convergence,
            "boundary_eigenvalue_warning": self.boundary_eigenvalue_warning,
            "iterations": self.iterations,
            "likelihood_monotonic": _monotone(self.likelihood_trace),
            "posterior_distinct_from_mle": not np.allclose(
                self.rho_count_mle, self.rho_count_posterior_mean, atol=1e-8),
        }
        if self.identifiability is not None:
            d["identifiability"] = self.identifiability
        return d


def _monotone(values: List[float], tol: float = 1e-6) -> bool:
    return all(b - a >= -tol for a, b in zip(values, values[1:]))


class V9CountLikelihoodRhoEstimator:
    """Forward count model + likelihood + monotonic refinement for rho."""

    def __init__(
        self,
        system_type: str,
        *,
        signal_counts: np.ndarray,
        idler_counts: np.ndarray,
        coin_counts: np.ndarray,
        read_count: np.ndarray | float,
        retrieval_maps: Optional[Dict[str, Any]] = None,
        basis_config_path: Optional[str] = None,
        refiner_config: Optional[V9CountMleRefinerConfig] = None,
        torch_device: str = "cpu",
    ):
        _require_torch()
        if system_type not in RETRIEVAL_SHAPES:
            raise ValueError(f"system_type must be one of {list(RETRIEVAL_SHAPES)}")
        self.system_type = system_type
        self.n = CORE_DIM[system_type]
        self.device = torch_device
        self.refiner_config = refiner_config or V9CountMleRefinerConfig()

        self.signal = np.asarray(signal_counts, dtype=np.float64).reshape(-1)
        self.idler = np.asarray(idler_counts, dtype=np.float64).reshape(-1)
        self.coin = np.asarray(coin_counts, dtype=np.float64).reshape(-1)
        read = np.asarray(read_count, dtype=np.float64).reshape(-1)
        if read.size == 1:
            read = np.full(self.signal.size, float(read[0]))
        self.read = read
        n_ops = self.signal.size
        if not (self.idler.size == self.coin.size == self.read.size == n_ops):
            raise ValueError("count arrays must have equal operation counts")
        if np.any(self.signal < 0) or np.any(self.idler < 0) or np.any(self.coin < 0) or np.any(read <= 0):
            raise ValueError("counts must be nonnegative and read_count positive")

        self._load_modes(basis_config_path)
        self._load_retrieval(retrieval_maps)
        self._build_count_tensors()

    # -- setup -----------------------------------------------------------
    def _load_modes(self, basis_config_path: Optional[str]) -> None:
        module = _load_calibrated_reconstruction_module()
        import os

        basis_path = basis_config_path or str(
            __import__("pathlib").Path(
                "system_control/qubit_recon_files/matrix"
            ) / f"basis_config_{self.system_type}.json"
        )
        self._calib = module
        entries = module.raw_reconstruction.load_basis_entries(self.system_type, basis_path)
        modes = []
        for entry in entries:
            amp = np.asarray(entry["amp"], dtype=np.float64)
            phase = np.asarray(entry.get("phase", np.zeros_like(amp)), dtype=np.float64)
            if amp.size != self.n:
                raise ValueError(
                    f"basis entry amp length {amp.size} does not match core dim {self.n}"
                )
            # The entry amp vector is already ordered over the W-core qubits (N entries);
            # operation_cells_for_entry cells are grid indices for retrieval maps, not
            # qubit indices, and must not be used to place amplitudes here.
            mode = amp * np.exp(1j * phase)
            modes.append(mode.astype(np.complex128))
        self.modes = modes  # list of complex (N,)
        self.mode_norms = np.array([np.vdot(m, m).real for m in modes], dtype=np.float64)
        self.mode_norms[self.mode_norms <= 0.0] = 1.0

    def _load_retrieval(self, retrieval_maps: Optional[Dict[str, Any]]) -> None:
        if retrieval_maps is None:
            maps = build_rho8_maps(self.system_type)
            retrieval_maps = maps
        retrieval_matrix = np.asarray(retrieval_maps["retrieval_raw"][0], dtype=np.float64)
        calib = self._calib
        rate_vectors = calib.rate_vectors_for_entries(
            self.system_type, retrieval_matrix, len(self.modes)
        )
        self.retrieval_o = rate_vectors.mean(axis=1)

    def _build_count_tensors(self) -> None:
        from ml_recon.physics_hybrid_v8.fpga_joint_observation import (
            FPGAJointFourCellCounts,
        )

        cells = torch.from_numpy(np.stack(
            [
                self.read - self.signal - self.idler + self.coin,
                self.signal - self.coin,
                self.idler - self.coin,
                self.coin,
            ], axis=-1,
        )).to(torch.float64).to(self.device)
        # V8 likelihood expects counts of shape [..., 4]; add a leading batch dim so
        # concentration broadcasting is well-defined.
        cells = cells[None, ...]
        self.four_cell = FPGAJointFourCellCounts(counts=cells, valid_mask=torch.ones(
            cells.shape[0], dtype=torch.bool, device=self.device))
        self.mode_t = torch.from_numpy(np.stack(self.modes)).to(torch.complex128).to(self.device)
        self.mode_norm_t = torch.from_numpy(self.mode_norms).to(torch.float64).to(self.device)
        self.retrieval_t = torch.from_numpy(self.retrieval_o).to(torch.float64).to(self.device)

    # -- forward ---------------------------------------------------------
    def expected_probabilities(self, state: V9CountMleState) -> Dict[str, np.ndarray]:
        """Per-operation probability triple from a numpy state (for diagnostics)."""
        b = self._born_weights(state.rho)
        b_norm = b / (float(np.max(b)) if np.max(b) > 0 else 1.0)
        p_s = np.clip(state.signal_scale * state.w_mass * b_norm + state.dark_prob, _EPS, 1 - _EPS)
        p_coin = np.clip(self.retrieval_o * b_norm + state.accidental_rate, _EPS, 1 - _EPS)
        p_no_signal = np.full(
            self.signal.size,
            np.clip(state.dark_prob + state.background_singles_prob, _EPS, 1 - _EPS),
        )
        return {"signal": p_s, "singleton_given_signal": p_coin, "singleton_given_no_signal": p_no_signal}

    def _born_weights(self, rho: np.ndarray) -> np.ndarray:
        weights = []
        for mode, norm in zip(self.modes, self.mode_norms):
            weights.append((np.vdot(mode, rho @ mode).real) / norm)
        return np.asarray(weights, dtype=np.float64)

    def log_likelihood(self, state: V9CountMleState) -> float:
        """Primary FPGA-joint log-likelihood of a state (used exactly once)."""
        from ml_recon.physics_hybrid_v8.fpga_joint_observation import (
            nested_beta_binomial_log_likelihood,
        )

        prob = self.expected_probabilities(state)
        signal = torch.from_numpy(prob["signal"]).to(torch.float64).to(self.device).unsqueeze(0)
        coin = torch.from_numpy(prob["singleton_given_signal"]).to(torch.float64).to(self.device).unsqueeze(0)
        no_signal = torch.from_numpy(prob["singleton_given_no_signal"]).to(torch.float64).to(self.device).unsqueeze(0)
        concentration = torch.full_like(signal, state.concentration)
        ll = nested_beta_binomial_log_likelihood(
            self.four_cell, signal, coin, no_signal, concentration
        )
        return float(ll.sum().item())

    # -- refinement ------------------------------------------------------
    def _refine_single(self, init: V9CountMleState, start_index: int = 0) -> V9CountMleResult:
        """Monotonic likelihood refinement from ONE start (Adam + backtracking)."""
        cfg = self.refiner_config
        torch.manual_seed(cfg.seed + start_index)
        n = self.n
        chol = torch.from_numpy(rho_to_cholesky_vector(init.rho)).to(torch.float64).to(self.device)
        nuisance = torch.tensor(
            [
                _inverse_sigmoid(init.w_mass),
                _inverse_sigmoid(init.signal_scale / SIGNAL_SCALE_MAX),
                _inverse_sigmoid(init.dark_prob / DARK_MAX),
                _inverse_sigmoid(init.background_singles_prob / BACKGROUND_MAX),
                _inverse_sigmoid(init.accidental_rate / ACCIDENTAL_MAX),
                math.log(max(init.concentration - 1e-6, 1e-6)),
            ],
            dtype=torch.float64, device=self.device,
        )
        params = [chol, nuisance]
        for p in params:
            p.requires_grad_(True)

        likelihood_before = self.log_likelihood(init)
        best_ll = likelihood_before
        best_params = [p.detach().clone() for p in params]
        trace = [likelihood_before]

        opt_state = {"m": [torch.zeros_like(p) for p in params],
                     "v": [torch.zeros_like(p) for p in params],
                     "t": 0}
        for step in range(cfg.steps):
            opt_state["t"] += 1
            grads = self._gradient(params)
            step_size = cfg.learning_rate
            accepted = False
            for _bt in range(8):
                proposed = []
                for i, p in enumerate(params):
                    m = opt_state["m"][i] = 0.9 * opt_state["m"][i] + 0.1 * grads[i]
                    v = opt_state["v"][i] = 0.999 * opt_state["v"][i] + 0.001 * grads[i].square()
                    mhat = m / (1 - 0.9 ** opt_state["t"])
                    vhat = v / (1 - 0.999 ** opt_state["t"])
                    new_p = (p.detach() + step_size * mhat / (torch.sqrt(vhat) + 1e-8))
                    proposed.append(new_p.clone().detach().requires_grad_(True))
                ll = self._log_likelihood_from_params(proposed)
                if math.isfinite(float(ll.detach())) and float(ll.detach()) >= best_ll - cfg.accept_tol:
                    best_params = [p.detach().clone() for p in proposed]
                    best_ll = float(ll.detach())
                    params = [p.detach().clone().requires_grad_(True) for p in proposed]
                    accepted = True
                    break
                step_size *= cfg.backtrack_factor
            trace.append(best_ll)
        final = self._state_from_params(best_params)
        ll_after = self.log_likelihood(final)
        converged = ll_after >= likelihood_before - cfg.accept_tol and math.isfinite(ll_after)
        return V9CountMleResult(
            rho_count_mle=final.rho,
            rho_count_posterior_mean=final.rho,
            nuisance_posterior=final.to_dict(),
            likelihood_before=likelihood_before,
            likelihood_after=ll_after,
            convergence=converged,
            boundary_eigenvalue_warning=final.boundary_eigenvalue_warning,
            iterations=cfg.steps,
            likelihood_trace=trace,
        )

    def _perturbed_init(self, init: V9CountMleState, start_index: int,
                        n_starts: int) -> V9CountMleState:
        """Small deterministic perturbation of the init for a multiple-start run."""
        if n_starts <= 1 or start_index == 0:
            return init
        rng = np.random.default_rng(self.refiner_config.seed + 1000 * start_index)
        noise = rng.normal(size=init.rho.shape) + 1j * rng.normal(size=init.rho.shape)
        H = 0.5 * (noise + noise.conj().T)
        w, v = np.linalg.eigh(H)
        U = (v * np.exp(1j * 0.05 * w)) @ v.conj().T
        rho_p = 0.5 * (U @ init.rho @ U.conj().T + (U @ init.rho @ U.conj().T).conj().T)
        rho_p = rho_p / np.trace(rho_p).real

        def _clip(x, lo, hi, factor=0.2):
            span = hi - lo
            return float(np.clip(x + factor * span * rng.uniform(-1, 1), lo, hi))

        return V9CountMleState(
            system_type=init.system_type, rho=rho_p,
            w_mass=_clip(init.w_mass, 0.0, 1.0),
            signal_scale=_clip(init.signal_scale, 0.0, SIGNAL_SCALE_MAX),
            dark_prob=_clip(init.dark_prob, 0.0, DARK_MAX),
            background_singles_prob=_clip(init.background_singles_prob, 0.0, BACKGROUND_MAX),
            accidental_rate=_clip(init.accidental_rate, 0.0, ACCIDENTAL_MAX),
            concentration=max(init.concentration * float(np.exp(rng.uniform(-0.5, 0.5))), 1e-6),
        )

    @staticmethod
    def _evidence_weights(results: Sequence[V9CountMleResult]) -> np.ndarray:
        logits = np.asarray([r.likelihood_after for r in results], dtype=np.float64)
        logits = logits - logits.max()
        w = np.exp(np.clip(logits, -30, 0))
        return w / w.sum()

    @staticmethod
    def _psd_unit_trace_mixture(rhos: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
        mix = np.zeros_like(np.asarray(rhos[0], dtype=np.complex128))
        for rho, w in zip(rhos, weights):
            mix = mix + w * np.asarray(rho, dtype=np.complex128)
        mix = 0.5 * (mix + mix.conj().T)
        return mix / np.trace(mix).real

    def _bootstrap_posterior(self, best: V9CountMleResult,
                             cfg: V9CountMleRefinerConfig):
        """Parametric-bootstrap posterior samples from the best start's fitted model."""
        state = V9CountMleState(system_type=self.system_type, rho=best.rho_count_mle,
                                **{k: best.nuisance_posterior[k] for k in (
                                    "w_mass", "signal_scale", "dark_prob",
                                    "background_singles_prob", "accidental_rate",
                                    "concentration")})
        prob = self.expected_probabilities(state)
        p_s, p_c, p_n = prob["signal"], prob["singleton_given_signal"], prob["singleton_given_no_signal"]
        rng = np.random.default_rng(cfg.seed + 777)
        rhos: List[np.ndarray] = []
        nuisance: List[Dict[str, float]] = []
        bcfg = V9CountMleRefinerConfig(
            steps=max(1, int(getattr(cfg, "bootstrap_steps", 12))),
            learning_rate=cfg.learning_rate, seed=cfg.seed + 42)
        for b in range(max(0, int(getattr(cfg, "n_bootstrap", 0)))):
            r_sim = self.read.astype(int)
            s_sim = rng.binomial(r_sim, p_s)
            c_sim = rng.binomial(s_sim, p_c)
            i_sim = rng.binomial(r_sim - s_sim, p_n) + c_sim
            try:
                est = V9CountLikelihoodRhoEstimator(
                    self.system_type, signal_counts=s_sim, idler_counts=i_sim,
                    coin_counts=c_sim, read_count=r_sim, refiner_config=bcfg,
                    torch_device=self.device)
                res = est._refine_single(state, start_index=1000 + b)
                rhos.append(res.rho_count_mle)
                nuisance.append(dict(res.nuisance_posterior))
            except (ValueError, np.linalg.LinAlgError):
                continue
        return rhos, nuisance

    def _laplace_posterior(self, best: V9CountMleResult,
                           cfg: V9CountMleRefinerConfig):
        """Laplace-approximation posterior draws (samples of the likelihood curvature).

        Draws ``z ~ N(param_mode, H^{-1})`` where H is the Hessian of the negative
        log-likelihood at the mode, then maps each draw through the (nonlinear)
        Cholesky->rho map. The mean of these draws in rho space is a genuinely distinct
        posterior mean from the MLE point (finite-data curvature), unlike a point
        estimate.
        """
        state = V9CountMleState(
            system_type=self.system_type, rho=best.rho_count_mle,
            **{k: best.nuisance_posterior[k] for k in (
                "w_mass", "signal_scale", "dark_prob", "background_singles_prob",
                "accidental_rate", "concentration")})
        chol_vec = rho_to_cholesky_vector(state.rho)
        nu_vec = np.asarray([
            _inverse_sigmoid(state.w_mass),
            _inverse_sigmoid(state.signal_scale / SIGNAL_SCALE_MAX),
            _inverse_sigmoid(state.dark_prob / DARK_MAX),
            _inverse_sigmoid(state.background_singles_prob / BACKGROUND_MAX),
            _inverse_sigmoid(state.accidental_rate / ACCIDENTAL_MAX),
            math.log(max(state.concentration - 1e-6, 1e-6)),
        ], dtype=np.float64)
        chol_t = torch.from_numpy(chol_vec).to(torch.float64).to(self.device).requires_grad_(True)
        nu_t = torch.from_numpy(nu_vec).to(torch.float64).to(self.device).requires_grad_(True)
        params = [chol_t, nu_t]
        ll = self._log_likelihood_from_params(params)
        g = torch.autograd.grad(ll, params, create_graph=True, allow_unused=True)
        n1 = chol_t.numel()
        n2 = nu_t.numel()
        g0 = g[0].reshape(-1)
        g1 = g[1].reshape(-1)
        H = torch.zeros(n1 + n2, n1 + n2, dtype=torch.float64, device=self.device)
        gf = torch.cat([g0, g1])
        zeros1 = torch.zeros(n1, dtype=torch.float64, device=self.device)
        zeros2 = torch.zeros(n2, dtype=torch.float64, device=self.device)
        for i in range(n1 + n2):
            h = torch.autograd.grad(gf[i], params, retain_graph=(i < n1 + n2 - 1),
                                    allow_unused=True)
            h0 = h[0].reshape(-1) if h[0] is not None else zeros1
            h1 = h[1].reshape(-1) if h[1] is not None else zeros2
            H[i] = torch.cat([h0, h1])
        cov = torch.inverse(H + 1e-4 * torch.eye(
            n1 + n2, dtype=torch.float64, device=self.device))
        mode = torch.cat([chol_t.detach().reshape(-1), nu_t.detach().reshape(-1)]).cpu().numpy()
        w_eig, v_eig = np.linalg.eigh(cov.cpu().numpy())
        w_eig = np.clip(w_eig, 1e-6, None)  # non-PD-safe symmetric square root
        L = (v_eig * np.sqrt(w_eig)) @ v_eig.T
        rng = np.random.default_rng(cfg.seed + 999)
        n_draw = max(1, int(getattr(cfg, "n_laplace", 8)))
        rhos: List[np.ndarray] = []
        for d in range(n_draw):
            z = mode + L @ rng.normal(size=n1 + n2)
            draw = [torch.from_numpy(z[:n1]).to(self.device),
                    torch.from_numpy(z[n1:]).to(self.device)]
            try:
                st = self._state_from_params(draw)
                rhos.append(st.rho)
            except ValueError:
                continue
        return rhos

    def _partial_identification(self, nuisance_samples: Sequence[Dict[str, float]],
                                init: V9CountMleState) -> Dict[str, Any]:
        """Partial-identification diagnostics over the posterior nuisance samples.

        Reports identified combinations (``signal_scale*w_mass``, ``dark+background``),
        prior-dominated parameters (posterior spread tiny vs prior width), profile
        intervals, and boundary warnings.
        """
        if not nuisance_samples:
            return {"n_samples": 0, "identifiability_rank": 0,
                    "identified_combinations": [], "prior_dominated_parameters": [],
                    "profile_intervals": {}, "boundary_warnings": []}
        names = ("w_mass", "signal_scale", "dark_prob", "background_singles_prob", "accidental_rate")
        priors = {"w_mass": (0.0, 1.0), "signal_scale": (0.0, SIGNAL_SCALE_MAX),
                  "dark_prob": (0.0, DARK_MAX), "background_singles_prob": (0.0, BACKGROUND_MAX),
                  "accidental_rate": (0.0, ACCIDENTAL_MAX)}
        vals = {k: np.asarray([s[k] for s in nuisance_samples], dtype=np.float64)
                for k in names}
        spans = {k: float(np.ptp(v)) for k, v in vals.items()}
        prior_widths = {k: hi - lo for k, (lo, hi) in priors.items()}
        prior_dominated = [
            k for k in names
            if prior_widths[k] > 1e-9 and spans[k] / prior_widths[k] < 0.02]
        boundary = sorted({
            k for s in nuisance_samples for k, (lo, hi) in priors.items()
            if abs(s[k] - lo) < 1e-6 or abs(s[k] - hi) < 1e-6
        })
        identified_combos = []
        if float(np.ptp(vals["signal_scale"] * vals["w_mass"])) > 1e-6:
            identified_combos.append("signal_scale*w_mass")
        if float(np.ptp(vals["dark_prob"] + vals["background_singles_prob"])) > 1e-6:
            identified_combos.append("dark+background")
        profile = {k: {"q05": float(np.quantile(v, 0.05)),
                       "q50": float(np.quantile(v, 0.50)),
                       "q95": float(np.quantile(v, 0.95))} for k, v in vals.items()}
        return {
            "n_samples": len(nuisance_samples),
            "identifiability_rank": int(sum(
                1 for k in names if spans[k] / (prior_widths[k] + 1e-12) > 0.05)),
            "identified_combinations": identified_combos,
            "prior_dominated_parameters": prior_dominated,
            "profile_intervals": profile,
            "boundary_warnings": boundary,
            "posterior_spans": {k: round(v, 6) for k, v in spans.items()},
        }

    def refine_from_state(self, init: V9CountMleState) -> V9CountMleResult:
        """Multiple-start + bootstrap refinement with a data-resampling posterior.

        ``rho_count_mle`` = best-likelihood start; ``rho_count_posterior_mean`` = mean
        over the parametric-bootstrap posterior samples (distinct from the MLE point).
        """
        cfg = self.refiner_config
        n_starts = max(1, int(getattr(cfg, "n_starts", 3)))
        results = [
            self._refine_single(self._perturbed_init(init, s, n_starts), s)
            for s in range(n_starts)
        ]
        best = max(results, key=lambda r: r.likelihood_after)
        bootstrap_rhos, bootstrap_nuisance = self._bootstrap_posterior(best, cfg)
        laplace_rhos = self._laplace_posterior(best, cfg)
        all_rhos = [best.rho_count_mle] + bootstrap_rhos + laplace_rhos
        if len(all_rhos) <= 1:
            # Fallback: evidence-weighted start mixture (still multi-point).
            weights = self._evidence_weights(results)
            rho_pm = self._psd_unit_trace_mixture(
                [r.rho_count_mle for r in results], weights)
        else:
            w = np.ones(len(all_rhos), dtype=np.float64) / len(all_rhos)
            rho_pm = self._psd_unit_trace_mixture(all_rhos, w)
        nuisance_samples = [dict(r.nuisance_posterior) for r in results] + bootstrap_nuisance
        return V9CountMleResult(
            rho_count_mle=best.rho_count_mle,
            rho_count_posterior_mean=rho_pm,
            nuisance_posterior=best.nuisance_posterior,
            likelihood_before=best.likelihood_before,
            likelihood_after=best.likelihood_after,
            convergence=best.convergence,
            boundary_eigenvalue_warning=best.boundary_eigenvalue_warning,
            iterations=sum(r.iterations for r in results)
            + max(1, int(getattr(cfg, "n_bootstrap", 0)))
            * max(1, int(getattr(cfg, "bootstrap_steps", 12))),
            likelihood_trace=best.likelihood_trace,
            identifiability=self._partial_identification(nuisance_samples, init),
        )

    def _gradient(self, params: List[torch.Tensor]) -> List[torch.Tensor]:
        ll = self._log_likelihood_from_params(params)
        if not math.isfinite(float(ll.detach())):
            return [torch.zeros_like(p) for p in params]
        grads = torch.autograd.grad(ll, params, retain_graph=False, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]

    def _rho_torch_from_vector(self, chol_vec: torch.Tensor, n: int) -> torch.Tensor:
        """Build rho = L L^dag / Tr(L L^dag) in torch (differentiable)."""
        length = cholesky_vector_length(n)
        if chol_vec.numel() != 2 * length:
            raise ValueError(f"chol_vec length {chol_vec.numel()} != {2 * length}")
        tri = torch.zeros((n, n), dtype=torch.complex128, device=self.device)
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                re = chol_vec[2 * idx]
                im = chol_vec[2 * idx + 1] if i != j else chol_vec[2 * idx + 1] * 0.0
                tri[i, j] = re + 1j * im
                idx += 1
        rho = tri @ tri.conj().T
        return rho / torch.real(torch.trace(rho)).clamp_min(1e-12)

    def _log_likelihood_from_params(self, params: List[torch.Tensor]) -> torch.Tensor:
        from ml_recon.physics_hybrid_v8.fpga_joint_observation import (
            nested_beta_binomial_log_likelihood,
        )

        rho_t = self._rho_torch_from_vector(params[0], self.n)
        # b_o = (mode_o^dag rho mode_o) / |mode_o|^2  (differentiable in rho).
        rho_mode = self.mode_t @ rho_t  # (n_ops, N)
        b = torch.real((self.mode_t.conj() * rho_mode).sum(dim=1)) / self.mode_norm_t
        b_norm = b / b.clamp_min(1e-12).max()
        w_mass = torch.sigmoid(params[1][0])
        signal_scale = torch.sigmoid(params[1][1]) * SIGNAL_SCALE_MAX
        dark = torch.sigmoid(params[1][2]) * DARK_MAX
        background = torch.sigmoid(params[1][3]) * BACKGROUND_MAX
        accidental = torch.sigmoid(params[1][4]) * ACCIDENTAL_MAX
        concentration = torch.nn.functional.softplus(params[1][5]) + 1e-6
        b_norm = b_norm.unsqueeze(0)  # [1, ops]
        signal = (signal_scale * w_mass * b_norm + dark).clamp(_EPS, 1 - _EPS)
        coin = (self.retrieval_t.unsqueeze(0) * b_norm + accidental).clamp(_EPS, 1 - _EPS)
        no_signal = (dark + background).clamp(_EPS, 1 - _EPS).expand_as(signal)
        concentration_b = concentration.expand_as(signal)
        ll = nested_beta_binomial_log_likelihood(
            self.four_cell, signal, coin, no_signal, concentration_b
        )
        return ll.sum()

    def _state_from_params(self, params: List[torch.Tensor]) -> V9CountMleState:
        chol_vec = params[0].detach().cpu().numpy()
        rho = rho_from_cholesky_vector(chol_vec, self.n)
        nu = params[1].detach().cpu().numpy()
        state = V9CountMleState(
            system_type=self.system_type,
            rho=rho,
            w_mass=float(torch.sigmoid(torch.tensor(nu[0]))),
            signal_scale=float(torch.sigmoid(torch.tensor(nu[1])) * SIGNAL_SCALE_MAX),
            dark_prob=float(torch.sigmoid(torch.tensor(nu[2])) * DARK_MAX),
            background_singles_prob=float(torch.sigmoid(torch.tensor(nu[3])) * BACKGROUND_MAX),
            accidental_rate=float(torch.sigmoid(torch.tensor(nu[4])) * ACCIDENTAL_MAX),
            concentration=float(torch.nn.functional.softplus(torch.tensor(nu[5])) + 1e-6),
        )
        eigenvalues = np.real(np.linalg.eigvalsh(state.rho))
        state.boundary_eigenvalue_warning = (
            float(np.min(eigenvalues)) < self.refiner_config.boundary_eigenvalue_threshold
        )
        state.validate()
        return state

    # -- scoring and mixture ---------------------------------------------
    def score_rho_candidates(self, rhos: Sequence[np.ndarray]) -> Dict[str, Any]:
        """Evidence-weighted scoring of init candidates (step 3 of section 11.3)."""
        scores = []
        for rho in rhos:
            state = V9CountMleState(system_type=self.system_type, rho=np.asarray(rho, dtype=np.complex128))
            try:
                state.validate()
                scores.append(self.log_likelihood(state))
            except ValueError:
                scores.append(-np.inf)
        best = float(np.max(scores)) if scores else -np.inf
        weights = np.zeros(len(scores), dtype=np.float64)
        finite = np.isfinite(scores)
        if finite.any():
            logits = np.asarray(scores)[finite]
            logits = logits - logits.max()
            exp_w = np.exp(np.clip(logits, -30, 0))
            weights[finite] = exp_w / exp_w.sum()
        return {
            "per_candidate_log_likelihood": list(scores),
            "best_log_likelihood": best,
            "weights": weights.tolist(),
        }

    def mixture_from_candidates(self, rhos: Sequence[np.ndarray]) -> np.ndarray:
        """Evidence-weighted convex mixture of candidate rhos (step 4)."""
        result = self.score_rho_candidates(rhos)
        weights = np.asarray(result["weights"])
        mixture = np.zeros_like(np.asarray(rhos[0], dtype=np.complex128))
        for rho, w in zip(rhos, weights):
            mixture = mixture + w * np.asarray(rho, dtype=np.complex128)
        return mixture

    # -- bootstrap -------------------------------------------------------
    def bootstrap_intervals(
        self,
        mle_state: V9CountMleState,
        n_samples: int = 100,
        seed: int = 0,
        refinement_steps: int = 20,
    ) -> Dict[str, Any]:
        """Parametric bootstrap intervals (finite-sample bias, section 11.4)."""
        rng = np.random.default_rng(seed)
        prob = self.expected_probabilities(mle_state)
        p_s, p_c, p_n = prob["signal"], prob["singleton_given_signal"], prob["singleton_given_no_signal"]
        fidelities, purities, masses, leakages, eig_min = [], [], [], [], []
        likelihoods = []
        for _ in range(n_samples):
            r_sim = self.read.astype(int).copy()
            s_sim = rng.binomial(r_sim, p_s)
            c_sim = rng.binomial(s_sim, p_c)
            i_sim = rng.binomial(r_sim - s_sim, p_n) + c_sim
            try:
                est = V9CountLikelihoodRhoEstimator(
                    self.system_type,
                    signal_counts=s_sim, idler_counts=i_sim, coin_counts=c_sim,
                    read_count=r_sim,
                    refiner_config=V9CountMleRefinerConfig(
                        steps=refinement_steps, seed=seed),
                )
                res = est._refine_single(mle_state, start_index=0)
                rho_b = res.rho_count_mle
                fidelities.append(float(np.real(np.trace(mle_state.rho @ rho_b))))
                purities.append(float(np.real(np.trace(rho_b @ rho_b))))
                masses.append(res.nuisance_posterior["w_mass"])
                leakages.append(max(0.0, 1.0 - res.nuisance_posterior["w_mass"]))
                eig_min.append(float(np.min(np.real(np.linalg.eigvalsh(rho_b)))))
                likelihoods.append(res.likelihood_after)
            except (ValueError, np.linalg.LinAlgError):
                continue
        arrays = {
            "fidelity": fidelities, "purity": purities, "w_mass": masses,
            "leakage": leakages, "eigenvalue_min": eig_min, "likelihood": likelihoods,
        }
        return {name: _interval_stats(vals) for name, vals in arrays.items()}


class V9CountMleRefiner:
    """Public refiner wrapper (fixed-iteration monotonic refinement)."""

    def __init__(self, config: Optional[V9CountMleRefinerConfig] = None):
        self.config = config or V9CountMleRefinerConfig()

    def refine(self, estimator: V9CountLikelihoodRhoEstimator, init: V9CountMleState) -> V9CountMleResult:
        return estimator.refine_from_state(init)


class V9CountRhoPosterior:
    """Posterior over rho from bootstrap/ensemble samples."""

    def __init__(self, samples: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None):
        self.samples = [np.asarray(s, dtype=np.complex128) for s in samples]
        self.weights = np.asarray(weights if weights is not None else np.ones(len(samples)), dtype=np.float64)
        self.weights = self.weights / self.weights.sum()

    def posterior_mean(self) -> np.ndarray:
        mean = np.zeros_like(self.samples[0])
        for s, w in zip(self.samples, self.weights):
            mean = mean + w * s
        return mean

    def target_fidelity_interval(self, target: np.ndarray, quantiles=(0.05, 0.5, 0.95)) -> Dict[str, float]:
        fids = np.array([np.real(np.trace(target @ s)) for s in self.samples])
        return {f"q{int(q * 100)}": float(np.quantile(fids, q)) for q in quantiles}

    def purity_interval(self, quantiles=(0.05, 0.5, 0.95)) -> Dict[str, float]:
        pur = np.array([np.real(np.trace(s @ s)) for s in self.samples])
        return {f"q{int(q * 100)}": float(np.quantile(pur, q)) for q in quantiles}


def _inverse_sigmoid(x: float) -> float:
    x = float(np.clip(x, 1e-6, 1 - 1e-6))
    return float(np.log(x / (1.0 - x)))


def _interval_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "q05": float("nan"),
                "q50": float("nan"), "q95": float("nan"), "std": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "std": float(np.std(arr)),
    }
