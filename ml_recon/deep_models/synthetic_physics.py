"""Physics-constrained synthetic data for the deep W-state models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch is an optional runtime dependency.
    torch = None
    Dataset = object  # type: ignore[misc,assignment]

from .configs import SystemPhysicsConfig, get_system_config, normalize_system
from .taxonomy import (
    COARSE_ERROR_LABELS,
    ENTANGLEMENT_TIER_LABELS,
    EXCITATION_MODE_LABELS,
    FINE_ERROR_LABELS,
    FINE_TO_COARSE,
    PHYSICS_AUX_TARGET_NAMES,
    STATE_FAMILY_LABELS,
    coarse_index,
    correctable_fraction_for_errors,
    entanglement_index,
    excitation_index,
    family_index,
    fine_index,
)


MAX_RHO_SIZE = 16
RHO_CHANNELS = 12


@dataclass
class SyntheticSample:
    system: str
    target_rho: np.ndarray
    measured_rho: np.ndarray
    image: np.ndarray
    physics: np.ndarray
    phase_map_input: np.ndarray
    phase_map_delta: np.ndarray
    scalar_delta: np.ndarray
    freq_delta_map: np.ndarray
    crosstalk_map: np.ndarray
    family_idx: int
    entanglement_idx: int
    excitation_idx: int
    coarse_idx: int
    primary_idx: int
    fine_multi: np.ndarray
    severity: float
    witness: np.ndarray
    physics_targets: np.ndarray
    fidelity: float
    purity: float
    correctable_fraction: float
    expected_gain: float
    system_id: int
    phase_mask: np.ndarray

    def as_torch(self) -> Dict[str, "torch.Tensor"]:
        if torch is None:
            raise RuntimeError("PyTorch is required for tensor conversion")
        return {
            "image": torch.tensor(self.image, dtype=torch.float32),
            "physics": torch.tensor(self.physics, dtype=torch.float32),
            "phase_map_input": torch.tensor(self.phase_map_input[None, :, :], dtype=torch.float32),
            "phase_map_delta": torch.tensor(self.phase_map_delta, dtype=torch.float32),
            "phase_mask": torch.tensor(self.phase_mask, dtype=torch.float32),
            "scalar_delta": torch.tensor(self.scalar_delta, dtype=torch.float32),
            "family": torch.tensor(self.family_idx, dtype=torch.long),
            "entanglement": torch.tensor(self.entanglement_idx, dtype=torch.long),
            "excitation": torch.tensor(self.excitation_idx, dtype=torch.long),
            "coarse": torch.tensor(self.coarse_idx, dtype=torch.long),
            "primary": torch.tensor(self.primary_idx, dtype=torch.long),
            "fine_multi": torch.tensor(self.fine_multi, dtype=torch.float32),
            "severity": torch.tensor([self.severity], dtype=torch.float32),
            "witness": torch.tensor(self.witness, dtype=torch.float32),
            "physics_targets": torch.tensor(self.physics_targets, dtype=torch.float32),
            "fidelity": torch.tensor([self.fidelity], dtype=torch.float32),
            "purity": torch.tensor([self.purity], dtype=torch.float32),
            "correctable": torch.tensor([self.correctable_fraction], dtype=torch.float32),
            "expected_gain": torch.tensor([self.expected_gain], dtype=torch.float32),
            "system_id": torch.tensor(self.system_id, dtype=torch.long),
        }


def basis_indices_with_weight(num_qubits: int, weight: int) -> List[int]:
    return [idx for idx in range(2**num_qubits) if int(idx).bit_count() == weight]


def pure_density(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.complex128)
    norm = np.linalg.norm(vec)
    if norm <= 0:
        vec = vec.copy()
        vec[0] = 1.0
        norm = 1.0
    vec = vec / norm
    return np.outer(vec, np.conjugate(vec))


def w_state_vector(num_qubits: int, rng: np.random.Generator, phase_scale: float = 0.0) -> np.ndarray:
    dim = 2**num_qubits
    vec = np.zeros(dim, dtype=np.complex128)
    phases = rng.normal(0.0, phase_scale, size=num_qubits)
    for q, idx in enumerate(basis_indices_with_weight(num_qubits, 1)):
        vec[idx] = np.exp(1j * phases[q])
    return vec / np.sqrt(num_qubits)


def canonical_w_vector(num_qubits: int) -> np.ndarray:
    dim = 2**num_qubits
    vec = np.zeros(dim, dtype=np.complex128)
    for idx in basis_indices_with_weight(num_qubits, 1):
        vec[idx] = 1.0
    return vec / np.sqrt(num_qubits)


def canonical_ghz_vector(num_qubits: int) -> np.ndarray:
    dim = 2**num_qubits
    vec = np.zeros(dim, dtype=np.complex128)
    vec[0] = 1.0
    vec[-1] = 1.0
    return vec / np.sqrt(2.0)


def family_density(config: SystemPhysicsConfig, family: str, rng: np.random.Generator) -> np.ndarray:
    dim = config.hilbert_dim
    n = config.num_qubits
    if family == "w_state":
        return pure_density(w_state_vector(n, rng, 0.04))
    if family == "ghz_state":
        vec = np.zeros(dim, dtype=np.complex128)
        vec[0] = 1.0
        vec[-1] = np.exp(1j * rng.normal(0.0, 0.08))
        return pure_density(vec)
    if family == "single_excitation":
        vec = np.zeros(dim, dtype=np.complex128)
        vec[rng.choice(basis_indices_with_weight(n, 1))] = 1.0
        return pure_density(vec)
    if family == "pair_entangled":
        pair = basis_indices_with_weight(n, min(2, n))
        vec = np.zeros(dim, dtype=np.complex128)
        vec[0] = 1.0
        vec[rng.choice(pair)] = np.exp(1j * rng.normal(0.0, 0.10))
        return pure_density(vec)
    if family == "dicke_like":
        weight = 2 if n >= 4 else 1
        ids = basis_indices_with_weight(n, weight)
        vec = np.zeros(dim, dtype=np.complex128)
        phases = rng.normal(0.0, 0.15, size=len(ids))
        for phase, idx in zip(phases, ids):
            vec[idx] = np.exp(1j * phase)
        return pure_density(vec)
    diag = rng.dirichlet(np.ones(dim))
    return np.diag(diag.astype(np.complex128))


def physicalize_density(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(vals.real, 0.0, None)
    if float(vals.sum()) <= 1e-12:
        vals[0] = 1.0
    rho = (vecs * vals) @ vecs.conj().T
    rho = rho / np.trace(rho)
    return 0.5 * (rho + rho.conj().T)


def density_fidelity_proxy(target: np.ndarray, measured: np.ndarray) -> float:
    # A stable Hilbert-Schmidt proxy; enough for synthetic supervision and fast tests.
    overlap = float(np.real(np.trace(target.conj().T @ measured)))
    denom = float(np.sqrt(np.real(np.trace(target @ target)) * np.real(np.trace(measured @ measured))))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(overlap / denom, 0.0, 1.0))


def pure_state_overlap(rho: np.ndarray, vec: np.ndarray) -> float:
    vec = np.asarray(vec, dtype=np.complex128)
    return float(np.clip(np.real(np.vdot(vec, rho @ vec)), 0.0, 1.0))


def entanglement_witness_features(config: SystemPhysicsConfig, rho: np.ndarray) -> Tuple[np.ndarray, str]:
    """Return witness-like overlaps and a coarse entanglement tier.

    The W-state margin uses the standard fidelity-witness threshold
    F_W > (N-1)/N for a genuine W-like candidate. The GHZ margin uses a
    conservative F_GHZ > 1/2 candidate threshold. These are auxiliary labels
    for synthetic diagnosis, not a replacement for a full experiment-specific
    entanglement proof.
    """
    w_overlap = pure_state_overlap(rho, canonical_w_vector(config.num_qubits))
    ghz_overlap = pure_state_overlap(rho, canonical_ghz_vector(config.num_qubits))
    w_margin = w_overlap - ((config.num_qubits - 1.0) / config.num_qubits)
    ghz_margin = ghz_overlap - 0.5
    diag = np.real(np.diag(rho))
    single_pop = float(np.sum(diag[basis_indices_with_weight(config.num_qubits, 1)]))
    offdiag = ~np.eye(config.hilbert_dim, dtype=bool)
    coherence = float(np.mean(np.abs(rho[offdiag])))
    purity = float(np.real(np.trace(rho @ rho)))
    if w_margin > 0.0 and single_pop > 0.60:
        tier = "w_genuine_candidate"
    elif ghz_margin > 0.0:
        tier = "ghz_genuine_candidate"
    elif coherence > 0.055 and purity > 0.35:
        tier = "biseparable_pair_like"
    elif purity > 0.55:
        tier = "separable_like"
    else:
        tier = "mixed_or_inconclusive"
    features = np.asarray([w_overlap, ghz_overlap, w_margin, ghz_margin], dtype=np.float32)
    return features, tier


def phase_map_tv_norm(phase_map: np.ndarray, clamp: float) -> float:
    if phase_map.size == 0:
        return 0.0
    diffs = []
    if phase_map.shape[0] > 1:
        diffs.append(np.abs(np.diff(phase_map, axis=0)).ravel())
    if phase_map.shape[1] > 1:
        diffs.append(np.abs(np.diff(phase_map, axis=1)).ravel())
    if not diffs:
        return 0.0
    return float(np.mean(np.concatenate(diffs)) / max(float(clamp), 1e-12))


def physical_hierarchy_targets(
    config: SystemPhysicsConfig,
    target: np.ndarray,
    measured: np.ndarray,
    phase_map_delta: np.ndarray,
    freq_delta: np.ndarray,
    crosstalk: np.ndarray,
    severity_hint: float,
) -> np.ndarray:
    """Physics/mathematics hierarchy used as auxiliary supervision.

    These values cover entanglement witnesses, W-subspace populations,
    leakage, coherence, correctable residual fractions, and scan-control
    diagnostics. They are deliberately compact so the model learns an
    interpretable physical coordinate system while still using the full
    density-matrix image.
    """
    residual = measured - target
    diag_m = np.real(np.diag(measured))
    single = basis_indices_with_weight(config.num_qubits, 1)
    double = basis_indices_with_weight(config.num_qubits, min(2, config.num_qubits))
    offdiag = ~np.eye(config.hilbert_dim, dtype=bool)
    witness, _ = entanglement_witness_features(config, measured)
    single_pop = float(np.sum(diag_m[single]))
    double_pop = float(np.sum(diag_m[double])) if double else 0.0
    off_w = float(np.clip(1.0 - single_pop, 0.0, 1.0))
    single_imbalance = float(np.std(diag_m[single]) / max(np.mean(diag_m[single]), 1e-6)) if single else 0.0
    single_imbalance = float(np.clip(single_imbalance, 0.0, 5.0))
    single_diag = diag_m[single] if single else np.asarray([], dtype=float)
    if single_diag.size:
        centered_idx = np.linspace(-1.0, 1.0, single_diag.size)
        single_slope = float(np.polyfit(centered_idx, single_diag, 1)[0])
        sorted_power = np.sort(single_diag)
        power_compression_signature = float((sorted_power[-1] - sorted_power[0]) / max(np.mean(sorted_power), 1e-6))
        random_amp_signature = float(np.mean(np.abs(np.diff(single_diag))) / max(np.mean(single_diag), 1e-6)) if single_diag.size > 1 else 0.0
    else:
        single_slope = 0.0
        power_compression_signature = 0.0
        random_amp_signature = 0.0
    offdiag_coh = float(np.mean(np.abs(measured[offdiag])))
    coh_res = float(np.mean(np.abs(residual[offdiag])))
    s1 = measured[np.ix_(single, single)]
    s1_target = target[np.ix_(single, single)]
    s1_residual = residual[np.ix_(single, single)]
    phase_values = np.angle(s1 + 1e-12)
    phase_spread = float(np.std(phase_values))
    total_norm = float(np.linalg.norm(residual, "fro"))
    s1_norm = float(np.linalg.norm(s1_residual, "fro"))
    w_subspace_fraction = float(np.clip(s1_norm / max(total_norm, 1e-12), 0.0, 1.0))
    phase_res = np.angle(s1 + 1e-12) - np.angle(s1_target + 1e-12)
    coherent_phase_fraction = float(np.clip(np.linalg.norm(np.sin(phase_res)) / max(np.sqrt(phase_res.size), 1.0), 0.0, 1.0))
    phase_map_rms = float(np.sqrt(np.mean(np.square(phase_map_delta))) / max(config.phase_map_clamp_rad, 1e-12))
    freq_rms = float(np.sqrt(np.mean(np.square(freq_delta))) / max(config.max_freq_delta_mhz, 1e-12))
    crosstalk_rms = float(np.sqrt(np.mean(np.square(crosstalk))))
    scan_control_norm = phase_map_rms + freq_rms + crosstalk_rms
    phase_residual_vs_scan = float(np.clip(coherent_phase_fraction / (0.05 + scan_control_norm), 0.0, 8.0))
    scan_vs_phase_residual = float(np.clip(scan_control_norm / (0.05 + coherent_phase_fraction), 0.0, 8.0))
    amp_slope_vs_random = float(np.clip(abs(single_slope) / (1e-4 + random_amp_signature), 0.0, 8.0))
    values = np.asarray(
        [
            density_fidelity_proxy(target, measured),
            float(np.real(np.trace(measured @ measured))),
            float(witness[0]),
            float(witness[1]),
            float(witness[2]),
            float(witness[3]),
            single_pop,
            double_pop,
            off_w,
            single_imbalance,
            offdiag_coh,
            coh_res,
            phase_spread / np.pi,
            w_subspace_fraction,
            coherent_phase_fraction,
            phase_map_rms,
            phase_map_tv_norm(phase_map_delta, config.phase_map_clamp_rad),
            freq_rms,
            crosstalk_rms,
            float(severity_hint),
            single_slope,
            power_compression_signature,
            random_amp_signature,
            phase_residual_vs_scan,
            scan_vs_phase_residual,
            amp_slope_vs_random,
        ],
        dtype=np.float32,
    )
    if values.shape[0] != len(PHYSICS_AUX_TARGET_NAMES):
        raise RuntimeError("physics hierarchy target length mismatch")
    return np.nan_to_num(values, copy=False)


def smooth_map(shape: Tuple[int, int], rng: np.random.Generator, scale: float) -> np.ndarray:
    rows, cols = shape
    coarse = rng.normal(0.0, scale, size=(4, 4))
    x_old = np.linspace(0.0, 1.0, coarse.shape[0])
    y_old = np.linspace(0.0, 1.0, coarse.shape[1])
    x_new = np.linspace(0.0, 1.0, rows)
    y_new = np.linspace(0.0, 1.0, cols)
    tmp = np.vstack([np.interp(y_new, y_old, row) for row in coarse])
    return np.vstack([np.interp(x_new, x_old, tmp[:, col]) for col in range(cols)]).T


def phase_map_to_component_phases(phase_map: np.ndarray, num_qubits: int) -> np.ndarray:
    rows, cols = phase_map.shape
    anchors = np.linspace(0, rows - 1, num_qubits).astype(int)
    phases = []
    for i, row in enumerate(anchors):
        col = anchors[-(i + 1)]
        phases.append(float(phase_map[row, col]))
    return np.asarray(phases, dtype=float)


def apply_local_phase(rho: np.ndarray, num_qubits: int, phases: np.ndarray) -> np.ndarray:
    dim = 2**num_qubits
    diag = np.ones(dim, dtype=np.complex128)
    single = basis_indices_with_weight(num_qubits, 1)
    for phase, idx in zip(phases, single):
        diag[idx] = np.exp(1j * phase)
    unitary = np.diag(diag)
    return unitary @ rho @ unitary.conj().T


def apply_amplitude_weights(rho: np.ndarray, num_qubits: int, weights: np.ndarray) -> np.ndarray:
    dim = 2**num_qubits
    diag = np.ones(dim, dtype=np.complex128)
    single = basis_indices_with_weight(num_qubits, 1)
    for weight, idx in zip(weights, single):
        diag[idx] = float(max(weight, 0.03))
    op = np.diag(diag)
    return physicalize_density(op @ rho @ op.conj().T)


def mix_density(rho: np.ndarray, other: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 0.95))
    return physicalize_density((1.0 - amount) * rho + amount * other)


def _error_labels(rng: np.random.Generator, split: str) -> List[str]:
    primary = str(rng.choice(FINE_ERROR_LABELS))
    labels = [primary]
    if split in {"train", "val"}:
        p_secondary = 0.0
    elif split == "mixture":
        p_secondary = 0.65
    else:
        p_secondary = 0.22
    if rng.random() < p_secondary:
        candidates = [label for label in FINE_ERROR_LABELS if label != primary]
        labels.append(str(rng.choice(candidates)))
    return labels


def _severity(rng: np.random.Generator, split: str, tier: str) -> float:
    if split == "ood":
        base = rng.uniform(0.72, 1.0)
    elif split == "mixture":
        base = rng.uniform(0.35, 0.95)
    else:
        base = rng.beta(2.0, 3.0)
    mult = {"low": 0.65, "medium": 0.85, "high": 1.0, "extreme": 1.15}.get(tier, 0.85)
    return float(np.clip(base * mult, 0.05, 1.0))


def generate_sample(
    config: SystemPhysicsConfig,
    rng: np.random.Generator,
    split: str = "train",
    noise_tier: str = "medium",
    family_mode: str = "broad",
) -> SyntheticSample:
    if family_mode == "w_feedback":
        family = str(rng.choice(STATE_FAMILY_LABELS, p=[0.82, 0.02, 0.10, 0.02, 0.02, 0.02]))
    elif family_mode == "classifier_diagnostic":
        family = str(rng.choice(STATE_FAMILY_LABELS, p=[0.74, 0.04, 0.10, 0.04, 0.04, 0.04]))
    else:
        family = str(rng.choice(STATE_FAMILY_LABELS, p=[0.50, 0.10, 0.12, 0.10, 0.08, 0.10]))
    target = family_density(config, family, rng)
    measured = target.copy()
    labels = _error_labels(rng, split)
    primary = labels[0]
    severity = _severity(rng, split, noise_tier)
    diagnostic_boost = family_mode == "classifier_diagnostic"
    if diagnostic_boost and split in {"train", "val"}:
        severity = float(max(severity, 0.30))
    signal_boost = 1.40 if diagnostic_boost else 1.0

    phase_shape = config.phase_map_shape
    phase_map_input = smooth_map(phase_shape, rng, 0.003 * (1.0 + severity))
    injected_phase_map = np.zeros(phase_shape, dtype=float)
    freq_delta = smooth_map(config.freq_grid_shape, rng, config.max_freq_delta_mhz * 0.08)
    crosstalk = smooth_map(config.crosstalk_shape, rng, 0.015)
    scalar_delta = np.zeros(config.scalar_delta_dim, dtype=np.float32)

    for label_idx, label in enumerate(labels):
        if label_idx == 0:
            local_sev = severity * rng.uniform(1.00, 1.35)
        else:
            local_sev = severity * rng.uniform(0.18, 0.42)
        if label == "phase_drift":
            phases = rng.normal(0.0, 0.55 * signal_boost * local_sev, size=config.num_qubits)
            measured = apply_local_phase(measured, config.num_qubits, phases)
            scalar_delta[0] += float(-np.mean(phases))
        elif label == "wch1_phase_map_error":
            injected_phase_map += smooth_map(phase_shape, rng, config.phase_map_clamp_rad * 1.4 * local_sev)
            measured = apply_local_phase(measured, config.num_qubits, phase_map_to_component_phases(injected_phase_map, config.num_qubits))
        elif label == "phase_crosstalk":
            base = rng.normal(0.0, 0.45 * local_sev)
            phases = base * np.linspace(-1.0, 1.0, config.num_qubits)
            measured = apply_local_phase(measured, config.num_qubits, phases)
            crosstalk += smooth_map(config.crosstalk_shape, rng, 0.12 * local_sev)
        elif label == "amp_imbalance":
            noise = rng.normal(0.0, 0.55 * signal_boost * local_sev, size=config.num_qubits)
            if diagnostic_boost and config.num_qubits > 1:
                axis = np.linspace(-1.0, 1.0, config.num_qubits)
                noise = noise - axis * (float(np.dot(noise, axis)) / max(float(np.dot(axis, axis)), 1e-12))
                noise_std = float(np.std(noise))
                if noise_std > 1e-9:
                    noise *= (0.55 * signal_boost * local_sev) / noise_std
            weights = 1.0 + noise
            measured = apply_amplitude_weights(measured, config.num_qubits, weights)
            scalar_delta[1] += float(-np.std(weights))
        elif label == "rf_power_compression":
            rf_boost = 1.12 if diagnostic_boost else 1.0
            weights = np.linspace(1.0, 1.0 - 0.62 * signal_boost * rf_boost * local_sev, config.num_qubits)
            measured = apply_amplitude_weights(measured, config.num_qubits, weights)
            scalar_delta[2] += float(0.35 * local_sev)
        elif label == "dephasing":
            dephase_boost = 1.18 if diagnostic_boost else 1.0
            damp = np.exp(-1.65 * signal_boost * dephase_boost * local_sev)
            measured = measured.copy()
            offdiag = ~np.eye(measured.shape[0], dtype=bool)
            measured[offdiag] *= damp
            measured = physicalize_density(measured)
        elif label == "depolarizing":
            identity = np.eye(config.hilbert_dim, dtype=np.complex128) / config.hilbert_dim
            measured = mix_density(measured, identity, 0.46 * local_sev)
        elif label == "retrieval_imbalance":
            weights = 1.0 + rng.normal(0.0, 0.38 * local_sev, size=config.hilbert_dim)
            op = np.diag(np.clip(weights, 0.20, 2.0))
            measured = physicalize_density(op @ measured @ op.conj().T)
            measured = mix_density(measured, np.diag(np.clip(weights, 0.0, None)).astype(np.complex128), 0.08 * local_sev)
            scalar_delta[3] += float(-np.std(weights))
        elif label == "shot_dark_noise":
            diag_noise = np.diag(rng.dirichlet(np.ones(config.hilbert_dim))).astype(np.complex128)
            measured = mix_density(measured, diag_noise, 0.30 * local_sev)
        elif label == "double_excitation":
            ids = basis_indices_with_weight(config.num_qubits, min(2, config.num_qubits))
            leak_vec = np.zeros(config.hilbert_dim, dtype=np.complex128)
            leak_vec[rng.choice(ids)] = 1.0
            measured = mix_density(measured, pure_density(leak_vec), 0.48 * local_sev)
        elif label == "mode_contamination":
            other = family_density(config, str(rng.choice(["ghz_state", "dicke_like", "mixed"])), rng)
            measured = mix_density(measured, other, 0.45 * local_sev)
        elif label == "frequency_nonlinearity":
            freq_delta += smooth_map(config.freq_grid_shape, rng, config.max_freq_delta_mhz * 0.55 * local_sev)
            phases = np.linspace(-1, 1, config.num_qubits) * np.mean(freq_delta) * 0.10
            measured = apply_local_phase(measured, config.num_qubits, phases)
            scalar_delta[4] += float(-np.mean(freq_delta) / config.max_freq_delta_mhz)
        elif label == "thermal_drift":
            drift = smooth_map(config.freq_grid_shape, rng, config.max_freq_delta_mhz * 0.30 * local_sev)
            freq_delta += drift
            measured = mix_density(measured, target, min(0.08 * local_sev, 0.20))

    measured = physicalize_density(measured)
    phase_map_delta = np.clip(-injected_phase_map, -config.phase_map_clamp_rad, config.phase_map_clamp_rad)
    # In feedback pretraining this input represents a coarse observed W_CH1
    # correction surface, when available. It should be correlated with true
    # pointwise phase-map errors, but still noisy and imperfect.
    phase_map_input = phase_map_input + injected_phase_map + smooth_map(phase_shape, rng, 0.010)
    freq_delta = np.clip(freq_delta, -config.max_freq_delta_mhz, config.max_freq_delta_mhz)
    fine_multi = np.zeros(len(FINE_ERROR_LABELS), dtype=np.float32)
    for label in labels:
        fine_multi[fine_index(label)] = 1.0

    fidelity = density_fidelity_proxy(target, measured)
    purity = float(np.real(np.trace(measured @ measured)))
    witness, entanglement_tier = entanglement_witness_features(config, measured)
    physics_targets = physical_hierarchy_targets(
        config,
        target,
        measured,
        phase_map_delta,
        freq_delta,
        crosstalk,
        severity,
    )
    correctable = correctable_fraction_for_errors(labels, severity)
    expected_gain = float(np.clip((1.0 - fidelity) * correctable, 0.0, 1.0))
    excitation = _excitation_mode(config, measured, labels)
    image = rho_pair_image(target, measured, config)
    phase_mask = np.zeros((MAX_RHO_SIZE, MAX_RHO_SIZE), dtype=np.float32)
    phase_mask[: phase_shape[0], : phase_shape[1]] = 1.0
    phase_input_padded = np.zeros((MAX_RHO_SIZE, MAX_RHO_SIZE), dtype=np.float32)
    phase_delta_padded = np.zeros((MAX_RHO_SIZE, MAX_RHO_SIZE), dtype=np.float32)
    phase_input_padded[: phase_shape[0], : phase_shape[1]] = phase_map_input.astype(np.float32)
    phase_delta_padded[: phase_shape[0], : phase_shape[1]] = phase_map_delta.astype(np.float32)
    physics = physics_features(
        config,
        target,
        measured,
        phase_map_input,
        freq_delta,
        crosstalk,
        severity_hint=severity,
    )

    return SyntheticSample(
        system=config.system,
        target_rho=target,
        measured_rho=measured,
        image=image,
        physics=physics,
        phase_map_input=phase_input_padded,
        phase_map_delta=phase_delta_padded,
        scalar_delta=scalar_delta,
        freq_delta_map=freq_delta,
        crosstalk_map=crosstalk,
        family_idx=family_index(family),
        entanglement_idx=entanglement_index(entanglement_tier),
        excitation_idx=excitation_index(excitation),
        coarse_idx=coarse_index(FINE_TO_COARSE[primary]),
        primary_idx=fine_index(primary),
        fine_multi=fine_multi,
        severity=severity,
        witness=witness,
        physics_targets=physics_targets,
        fidelity=fidelity,
        purity=purity,
        correctable_fraction=correctable,
        expected_gain=expected_gain,
        system_id=0 if config.system == "3qubit" else 1,
        phase_mask=phase_mask,
    )


def _excitation_mode(config: SystemPhysicsConfig, rho: np.ndarray, labels: Iterable[str]) -> str:
    labels = set(labels)
    diag = np.real(np.diag(rho))
    single_pop = float(np.sum(diag[basis_indices_with_weight(config.num_qubits, 1)]))
    double_pop = float(np.sum(diag[basis_indices_with_weight(config.num_qubits, min(2, config.num_qubits))]))
    if "double_excitation" in labels or double_pop > 0.18:
        return "over_excited"
    if "mode_contamination" in labels:
        return "mode_contaminated"
    if single_pop < 0.48:
        return "under_excited"
    if float(np.real(np.trace(rho @ rho))) < 0.55:
        return "incoherent_mixed"
    return "ideal_single_excitation"


def rho_pair_image(target: np.ndarray, measured: np.ndarray, config: SystemPhysicsConfig) -> np.ndarray:
    size = config.rho_image_size
    d = config.hilbert_dim
    image = np.zeros((RHO_CHANNELS, size, size), dtype=np.float32)
    residual = measured - target
    mats = [
        target.real,
        target.imag,
        measured.real,
        measured.imag,
        residual.real,
        residual.imag,
        np.abs(target),
        np.abs(measured),
        np.abs(residual),
    ]
    for idx, mat in enumerate(mats):
        image[idx, :d, :d] = mat.astype(np.float32)
    image[9, :d, :d] = np.diag(np.real(target))[:, None]
    image[10, :d, :d] = np.diag(np.real(measured))[None, :]
    image[11, :d, :d] = np.eye(d, dtype=np.float32)
    return image


def physics_features(
    config: SystemPhysicsConfig,
    target: np.ndarray,
    measured: np.ndarray,
    phase_map: np.ndarray,
    freq_delta: np.ndarray,
    crosstalk: np.ndarray,
    severity_hint: float = 0.0,
) -> np.ndarray:
    residual = measured - target
    diag_t = np.real(np.diag(target))
    diag_m = np.real(np.diag(measured))
    single = basis_indices_with_weight(config.num_qubits, 1)
    double = basis_indices_with_weight(config.num_qubits, min(2, config.num_qubits))
    offdiag = ~np.eye(config.hilbert_dim, dtype=bool)
    phase_vals = np.angle(measured[np.ix_(single, single)] + 1e-12)
    hierarchy = physical_hierarchy_targets(
        config,
        target,
        measured,
        phase_map,
        freq_delta,
        crosstalk,
        severity_hint,
    )
    s1 = measured[np.ix_(single, single)]
    s1_target = target[np.ix_(single, single)]
    s1_residual = s1 - s1_target
    diag_single_res = (diag_m[single] - diag_t[single]).tolist()
    phase_residual = np.angle(s1 + 1e-12) - np.angle(s1_target + 1e-12)
    phase_res_summary = [
        float(np.mean(np.sin(phase_residual))),
        float(np.std(np.sin(phase_residual))),
        float(np.mean(np.cos(phase_residual))),
        float(np.std(np.cos(phase_residual))),
    ]
    scalars = [
        *hierarchy.astype(float).tolist(),
        float(np.linalg.norm(residual.real)),
        float(np.linalg.norm(residual.imag)),
        float(np.mean(np.abs(residual))),
        float(np.linalg.norm(s1_residual.real)),
        float(np.linalg.norm(s1_residual.imag)),
        float(np.sum(diag_m[single])),
        float(np.sum(diag_m[double])) if double else 0.0,
        float(np.std(diag_m[single])) if single else 0.0,
        float(np.mean(np.abs(measured[offdiag]))),
        float(np.mean(np.abs(target[offdiag] - measured[offdiag]))),
        float(np.std(phase_vals)),
        float(np.mean(phase_map)),
        float(np.std(phase_map)),
        float(np.max(np.abs(phase_map))) if phase_map.size else 0.0,
        float(np.mean(freq_delta) / config.max_freq_delta_mhz),
        float(np.std(freq_delta) / config.max_freq_delta_mhz),
        float(np.max(np.abs(freq_delta)) / config.max_freq_delta_mhz),
        float(np.mean(crosstalk)),
        float(np.std(crosstalk)),
        float(severity_hint),
        float(config.num_qubits / 4.0),
        float(config.freq_grid_shape[0] / 16.0),
        float(config.phase_map_shape[0] / 16.0),
    ]
    diag_res = (diag_m - diag_t).tolist()
    single_diag = diag_m[single].tolist()
    values = np.asarray(scalars + phase_res_summary + diag_single_res + diag_res + single_diag, dtype=np.float32)
    if values.size > int(config.physics_feature_dim):
        raise ValueError(
            f"physics feature schema requires {values.size} values, "
            f"but config.physics_feature_dim={config.physics_feature_dim}; truncation is forbidden"
        )
    out = np.zeros(config.physics_feature_dim, dtype=np.float32)
    out[: values.size] = values
    return np.nan_to_num(out, copy=False)


class SyntheticDeepDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        system: str = "both",
        num_samples: int = 512,
        split: str = "train",
        seed: int = 0,
        noise_tier: str = "medium",
        family_mode: str = "broad",
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for SyntheticDeepDataset")
        self.system = normalize_system(system)
        self.num_samples = int(num_samples)
        self.split = str(split)
        self.seed = int(seed)
        self.noise_tier = str(noise_tier)
        self.family_mode = str(family_mode)
        rng = np.random.default_rng(self.seed)
        self.samples: List[SyntheticSample] = []
        for _ in range(self.num_samples):
            if self.system == "both":
                cfg = get_system_config("4qubit" if rng.random() < 0.55 else "3qubit")
            else:
                cfg = get_system_config(self.system)
            self.samples.append(
                generate_sample(cfg, rng, split=self.split, noise_tier=self.noise_tier, family_mode=self.family_mode)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        return self.samples[idx].as_torch()
