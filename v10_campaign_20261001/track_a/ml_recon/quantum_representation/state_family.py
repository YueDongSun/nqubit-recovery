"""Full-state family classification with CNN and covariance features."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import re

import json
import numpy as np

from .covariance_model import PAULI_AXES, pauli_covariance_matrix
from .features import density_fidelity_pure_target, rho_single_features
from .synthetic import ERROR_SOURCES
from .torch_model import (
    IMAGE_RECOGNITION_ARCHITECTURES,
    _select_training_device,
    _training_accelerator,
    _training_device_name,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    random_split = None
    TORCH_AVAILABLE = False


LEGACY_STATE_FAMILIES = ["w", "ghz", "bell_pair", "product", "mixed"]
COARSE_STATE_FAMILIES = [
    "fully_separable",
    "biseparable",
    "pair_entangled",
    "two_pair_entangled",
    "single_excitation",
    "w",
    "ghz",
    "dicke",
    "cluster",
    "singlet",
    "mixed",
]
FINE_STATE_FAMILIES_BY_QUBITS = {
    3: [
        "fully_separable",
        "biseparable_12_3",
        "biseparable_13_2",
        "biseparable_23_1",
        "single_excitation_e1",
        "single_excitation_e2",
        "single_excitation_e3",
        "w_balanced",
        "w_e1_dominant",
        "w_e2_dominant",
        "w_e3_dominant",
        "ghz",
        "dicke_2",
        "mixed",
    ],
    4: [
        "fully_separable",
        "bell_12_3_4",
        "bell_13_2_4",
        "bell_14_2_3",
        "bell_23_1_4",
        "bell_24_1_3",
        "bell_34_1_2",
        "two_bell_12_34",
        "two_bell_13_24",
        "two_bell_14_23",
        "single_excitation_e1",
        "single_excitation_e2",
        "single_excitation_e3",
        "single_excitation_e4",
        "w_balanced",
        "w_e1_dominant",
        "w_e2_dominant",
        "w_e3_dominant",
        "w_e4_dominant",
        "ghz",
        "dicke_2",
        "linear_cluster",
        "four_qubit_singlet",
        "mixed",
    ],
}
STATE_FAMILIES = sorted({
    *LEGACY_STATE_FAMILIES,
    *FINE_STATE_FAMILIES_BY_QUBITS[3],
    *FINE_STATE_FAMILIES_BY_QUBITS[4],
})
DEFAULT_ERROR_LABELS = list(ERROR_SOURCES)


@dataclass
class SyntheticStateFamilySample:
    rho_ideal: np.ndarray
    rho_measured: np.ndarray
    state_family: str
    error_source: str
    fidelity: float
    purity: float
    state_group: str = ""


@dataclass
class StateClassifierTrainingConfig:
    num_qubits: int = 4
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    latent_dim: int = 48
    covariance_dim: int = 96
    hidden_dim: int = 160
    seed: int = 1234
    validation_fraction: float = 0.2
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.10
    use_ema: bool = True
    ema_decay: float = 0.995
    architecture: str = "residual_attention"
    base_channels: int = 32
    dropout: float = 0.08
    device: str = "auto"
    early_stopping_patience: int = 0
    min_delta: float = 0.0
    lr_scheduler_patience: int = 0
    lr_scheduler_factor: float = 0.5


def generate_synthetic_state_family_dataset(
    num_samples: int,
    num_qubits: int = 4,
    seed: int = 1234,
    state_families: Optional[Sequence[str]] = None,
    error_labels: Optional[Sequence[str]] = None,
) -> List[SyntheticStateFamilySample]:
    """Generate labelled 3Q/4Q state-family density matrices.

    The default labels are experiment-oriented fine labels. They preserve the
    old W/GHZ/Bell/product/mixed aliases but add E1..En single-excitation
    labels, W imbalance variants, partition-specific Bell/biseparable states,
    and 4Q Dicke/cluster/singlet representatives.
    """

    if num_qubits not in {3, 4}:
        raise ValueError("num_qubits must be 3 or 4")
    families = list(state_families or default_state_family_labels(num_qubits, scheme="fine"))
    labels = list(error_labels or DEFAULT_ERROR_LABELS)
    unknown = [
        name
        for name in families
        if not _is_supported_state_family(num_qubits, name)
    ]
    if unknown:
        raise ValueError(f"Unsupported state families: {unknown}")

    rng = np.random.default_rng(int(seed))
    samples: List[SyntheticStateFamilySample] = []
    for sample_index in range(int(num_samples)):
        if sample_index < len(families):
            family = families[sample_index]
        else:
            family = families[int(rng.integers(0, len(families)))]
        compatible = _compatible_error_labels(family, labels)
        error = compatible[int(rng.integers(0, len(compatible)))]
        rho_ideal = ideal_state_family_density(num_qubits, family, rng)
        rho_measured = _apply_family_error(rho_ideal, num_qubits, error, rng)
        purity = float(np.real(np.trace(rho_measured @ rho_measured)))
        fidelity = density_fidelity_pure_target(rho_ideal, rho_measured)
        samples.append(SyntheticStateFamilySample(
            rho_ideal=rho_ideal,
            rho_measured=rho_measured,
            state_family=family,
            error_source=error,
            fidelity=fidelity,
            purity=purity,
            state_group=state_family_group(family),
        ))
    return samples


def default_state_family_labels(num_qubits: int, scheme: str = "fine") -> List[str]:
    if scheme == "legacy":
        return list(LEGACY_STATE_FAMILIES)
    if scheme == "coarse":
        if int(num_qubits) == 3:
            return ["fully_separable", "biseparable", "single_excitation", "w", "ghz", "dicke", "mixed"]
        if int(num_qubits) == 4:
            return list(COARSE_STATE_FAMILIES)
    if scheme == "fine" and int(num_qubits) in FINE_STATE_FAMILIES_BY_QUBITS:
        return list(FINE_STATE_FAMILIES_BY_QUBITS[int(num_qubits)])
    raise ValueError("scheme must be fine, coarse, or legacy; num_qubits must be 3 or 4")


def state_family_group(label: str) -> str:
    name = label.lower()
    if name in {"product", "fully_separable"}:
        return "fully_separable"
    if name == "biseparable" or name.startswith("biseparable_"):
        return "biseparable"
    if name in {"bell_pair", "pair_entangled"} or name.startswith("bell_"):
        return "pair_entangled"
    if name == "two_pair_entangled" or name.startswith("two_bell_"):
        return "two_pair_entangled"
    if name == "single_excitation" or name.startswith("single_excitation_"):
        return "single_excitation"
    if name in {"w", "w_balanced"} or name.startswith("w_e"):
        return "w"
    if name == "ghz":
        return "ghz"
    if name == "dicke" or name.startswith("dicke_"):
        return "dicke"
    if name in {"cluster", "linear_cluster"}:
        return "cluster"
    if name in {"singlet", "four_qubit_singlet"}:
        return "singlet"
    if name == "mixed":
        return "mixed"
    return "unknown"


def ideal_state_family_density(num_qubits: int, family: str, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    dim = 2 ** int(num_qubits)
    family = family.lower()
    if family in {"w", "w_balanced"}:
        return _w_density(num_qubits, rng, dominant=None, balanced=(family == "w_balanced"))

    match = re.fullmatch(r"w_e([1-9]\d*)_dominant", family)
    if match:
        return _w_density(num_qubits, rng, dominant=int(match.group(1)) - 1, balanced=False)

    match = re.fullmatch(r"single_excitation_e([1-9]\d*)", family)
    if match:
        return _single_excitation_density(num_qubits, int(match.group(1)) - 1)
    if family == "single_excitation":
        return _single_excitation_density(num_qubits, int(rng.integers(0, num_qubits)))

    if family == "ghz":
        psi = np.zeros(dim, dtype=complex)
        phase = float(rng.uniform(-np.pi, np.pi))
        imbalance = float(rng.uniform(0.85, 1.15))
        psi[0] = 1.0
        psi[-1] = imbalance * np.exp(1j * phase)
        return _pure_density(psi)

    if family in {"bell_pair", "pair_entangled", "biseparable"}:
        pair = tuple(rng.choice(num_qubits, size=2, replace=False))
        return _bell_pair_density(num_qubits, pair, rng)

    match = re.fullmatch(r"biseparable_([1-9]\d*)([1-9]\d*)_([1-9]\d*)", family)
    if match:
        pair = (int(match.group(1)) - 1, int(match.group(2)) - 1)
        return _bell_pair_density(num_qubits, pair, rng)

    match = re.fullmatch(r"bell_([1-9]\d*)([1-9]\d*)_(.+)", family)
    if match:
        pair = (int(match.group(1)) - 1, int(match.group(2)) - 1)
        return _bell_pair_density(num_qubits, pair, rng)

    match = re.fullmatch(r"two_bell_([1-9]\d*)([1-9]\d*)_([1-9]\d*)([1-9]\d*)", family)
    if match:
        pair_a = (int(match.group(1)) - 1, int(match.group(2)) - 1)
        pair_b = (int(match.group(3)) - 1, int(match.group(4)) - 1)
        return _two_bell_density(num_qubits, pair_a, pair_b, rng)
    if family == "two_pair_entangled":
        pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        pair_a, pair_b = pairings[int(rng.integers(0, len(pairings)))]
        return _two_bell_density(num_qubits, pair_a, pair_b, rng)

    if family in {"product", "fully_separable"}:
        psi = np.asarray([1.0 + 0.0j])
        for _ in range(num_qubits):
            theta = float(rng.uniform(0.0, np.pi))
            phi = float(rng.uniform(-np.pi, np.pi))
            single = np.asarray([np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)], dtype=complex)
            psi = np.kron(psi, single)
        return _pure_density(psi)

    match = re.fullmatch(r"dicke_([0-9]+)", family)
    if family == "dicke":
        return _dicke_density(num_qubits, max(1, num_qubits // 2), rng)
    if match:
        return _dicke_density(num_qubits, int(match.group(1)), rng)

    if family in {"cluster", "linear_cluster"}:
        if num_qubits != 4:
            raise ValueError("linear_cluster is only supported for 4 qubits")
        return _linear_cluster_4q_density()

    if family in {"singlet", "four_qubit_singlet"}:
        if num_qubits != 4:
            raise ValueError("four_qubit_singlet is only supported for 4 qubits")
        return _four_qubit_singlet_density()

    if family == "mixed":
        weights = rng.dirichlet(np.ones(dim) * 0.7)
        rho = np.diag(weights.astype(float)).astype(complex)
        coherent = ideal_state_family_density(num_qubits, "w_balanced", rng)
        return _physicalize(0.65 * rho + 0.35 * coherent)

    raise ValueError(f"Unsupported state family: {family}")


def _is_supported_state_family(num_qubits: int, family: str) -> bool:
    try:
        ideal_state_family_density(int(num_qubits), family, np.random.default_rng(0))
        return True
    except Exception:
        return False


def _single_excitation_density(num_qubits: int, excitation_index: int) -> np.ndarray:
    if excitation_index < 0 or excitation_index >= int(num_qubits):
        raise ValueError(f"single excitation index out of range for {num_qubits} qubits")
    dim = 2 ** int(num_qubits)
    psi = np.zeros(dim, dtype=complex)
    psi[2 ** (int(num_qubits) - 1 - excitation_index)] = 1.0
    return _pure_density(psi)


def _w_density(
    num_qubits: int,
    rng: np.random.Generator,
    dominant: Optional[int] = None,
    balanced: bool = False,
) -> np.ndarray:
    if dominant is not None and (dominant < 0 or dominant >= int(num_qubits)):
        raise ValueError(f"W dominant index out of range for {num_qubits} qubits")
    dim = 2 ** int(num_qubits)
    psi = np.zeros(dim, dtype=complex)
    if balanced:
        amps = np.ones(int(num_qubits))
        phases = rng.uniform(-0.08, 0.08, size=int(num_qubits))
    else:
        amps = rng.uniform(0.8, 1.2, size=int(num_qubits))
        phases = rng.uniform(-np.pi, np.pi, size=int(num_qubits))
    if dominant is not None:
        amps[dominant] *= float(rng.uniform(1.8, 2.6))
    for idx in range(int(num_qubits)):
        basis_index = 2 ** (int(num_qubits) - 1 - idx)
        psi[basis_index] = amps[idx] * np.exp(1j * phases[idx])
    return _pure_density(psi)


def _bell_pair_density(num_qubits: int, pair: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    pair = tuple(int(value) for value in pair)
    if len(set(pair)) != 2 or any(value < 0 or value >= int(num_qubits) for value in pair):
        raise ValueError(f"Invalid Bell pair {pair} for {num_qubits} qubits")
    dim = 2 ** int(num_qubits)
    psi = np.zeros(dim, dtype=complex)
    rest_bits = rng.integers(0, 2, size=int(num_qubits))
    bits_a = rest_bits.copy()
    bits_b = rest_bits.copy()
    bits_a[pair[0]] = 0
    bits_a[pair[1]] = 0
    bits_b[pair[0]] = 1
    bits_b[pair[1]] = 1
    psi[_bits_to_index(bits_a)] = 1.0
    psi[_bits_to_index(bits_b)] = np.exp(1j * float(rng.uniform(-np.pi, np.pi)))
    return _pure_density(psi)


def _two_bell_density(
    num_qubits: int,
    pair_a: Sequence[int],
    pair_b: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    pair_a = tuple(int(value) for value in pair_a)
    pair_b = tuple(int(value) for value in pair_b)
    all_indices = list(pair_a) + list(pair_b)
    if int(num_qubits) != 4 or len(set(all_indices)) != 4 or any(value < 0 or value >= 4 for value in all_indices):
        raise ValueError("two_bell labels require two disjoint pairs covering four qubits")
    psi = np.zeros(2 ** int(num_qubits), dtype=complex)
    phase_a = np.exp(1j * float(rng.uniform(-np.pi, np.pi)))
    phase_b = np.exp(1j * float(rng.uniform(-np.pi, np.pi)))
    for bit_a in (0, 1):
        for bit_b in (0, 1):
            bits = np.zeros(int(num_qubits), dtype=int)
            bits[pair_a[0]] = bit_a
            bits[pair_a[1]] = bit_a
            bits[pair_b[0]] = bit_b
            bits[pair_b[1]] = bit_b
            amp = (phase_a if bit_a else 1.0) * (phase_b if bit_b else 1.0)
            psi[_bits_to_index(bits)] = amp
    return _pure_density(psi)


def _dicke_density(num_qubits: int, excitations: int, rng: np.random.Generator) -> np.ndarray:
    num_qubits = int(num_qubits)
    excitations = int(excitations)
    if excitations < 0 or excitations > num_qubits:
        raise ValueError(f"Invalid Dicke excitation count {excitations} for {num_qubits} qubits")
    psi = np.zeros(2 ** num_qubits, dtype=complex)
    for ones in combinations(range(num_qubits), excitations):
        bits = np.zeros(num_qubits, dtype=int)
        bits[list(ones)] = 1
        psi[_bits_to_index(bits)] = np.exp(1j * float(rng.uniform(-0.08, 0.08)))
    return _pure_density(psi)


def _linear_cluster_4q_density() -> np.ndarray:
    psi = np.zeros(16, dtype=complex)
    for bits, amp in {
        "0000": 1.0,
        "0011": 1.0,
        "1100": 1.0,
        "1111": -1.0,
    }.items():
        psi[_bits_to_index(int(bit) for bit in bits)] = amp
    return _pure_density(psi)


def _four_qubit_singlet_density() -> np.ndarray:
    psi = np.zeros(16, dtype=complex)
    for bits, amp in {
        "0011": 2.0,
        "1100": 2.0,
        "0101": -1.0,
        "1010": -1.0,
        "0110": -1.0,
        "1001": -1.0,
    }.items():
        psi[_bits_to_index(int(bit) for bit in bits)] = amp
    return _pure_density(psi)


def rho_state_tensor(rho: np.ndarray) -> np.ndarray:
    mat = np.asarray(rho, dtype=complex)
    return np.asarray([
        np.real(mat),
        np.imag(mat),
        np.abs(mat),
        np.angle(mat),
    ], dtype=np.float32)


def state_covariance_features(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    cov = pauli_covariance_matrix(rho, num_qubits)
    upper = cov[np.triu_indices(cov.shape[0])]
    return np.concatenate([upper, rho_single_features(rho)]).astype(np.float32)


def error_sensitive_features(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    """Compute physics features specifically sensitive to error-source discrimination.

    Returns a 15-dim vector capturing diagonal/off-diagonal statistics,
    phase coherence, and subspace projections that distinguish the five
    synthetic error sources: phase_drift, amp_imbalance, dephasing,
    retrieval_imbalance, double_excitation.
    """
    dim = rho.shape[0]
    diag = np.abs(np.diag(rho))
    off_mask = ~np.eye(dim, dtype=bool)
    off_vals = rho[off_mask]
    off_mag = np.abs(off_vals)
    off_phase = np.angle(off_vals)

    feat = np.zeros(15, dtype=np.float32)
    idx = 0
    feat[idx] = np.mean(diag); idx += 1
    feat[idx] = np.std(diag); idx += 1
    feat[idx] = np.max(diag) / (np.min(diag) + 1e-8); idx += 1
    feat[idx] = float(np.linalg.norm(diag, 1) - np.linalg.norm(diag, float('inf'))); idx += 1
    feat[idx] = np.mean(off_mag); idx += 1
    feat[idx] = np.std(off_mag); idx += 1
    feat[idx] = np.mean(np.abs(off_phase)); idx += 1
    feat[idx] = np.std(off_phase); idx += 1
    row_off_std = np.std(np.abs(rho) * off_mask.astype(float), axis=1)
    feat[idx] = np.mean(row_off_std); idx += 1
    feat[idx] = np.std(row_off_std); idx += 1
    single_idx = [2 ** (int(num_qubits) - 1 - i) for i in range(int(num_qubits))]
    double_idx = []
    for i in range(int(num_qubits)):
        for j in range(i + 1, int(num_qubits)):
            didx = 2 ** (int(num_qubits) - 1 - i) + 2 ** (int(num_qubits) - 1 - j)
            if didx < dim:
                double_idx.append(didx)
    single_power = float(sum(np.abs(rho[si, si]) for si in single_idx))
    feat[idx] = single_power; idx += 1
    if double_idx:
        double_power = float(sum(np.abs(rho[di, di]) for di in double_idx))
        feat[idx] = double_power; idx += 1
        feat[idx] = double_power / (single_power + 1e-8); idx += 1
    else:
        feat[idx] = 0.0; idx += 1
        feat[idx] = 0.0; idx += 1
    feat[idx] = float(np.sum(off_mag) / (np.sum(diag) + 1e-8)); idx += 1
    feat[idx] = float(np.max(off_mag) / (np.mean(off_mag) + 1e-8)); idx += 1
    return feat.astype(np.float32)


if TORCH_AVAILABLE:
    from .torch_model import ImageResidualEncoder

    class QuantumStateFamilyCNN(nn.Module):
        """CNN classifier with an explicit covariance-feature side branch."""

        def __init__(
            self,
            num_qubits: int = 4,
            covariance_feature_dim: int = 0,
            num_families: int = len(STATE_FAMILIES),
            num_errors: int = len(DEFAULT_ERROR_LABELS),
            num_groups: int = 0,
            latent_dim: int = 48,
            covariance_dim: int = 96,
            hidden_dim: int = 160,
            architecture: str = "residual_attention",
            base_channels: int = 32,
            dropout: float = 0.08,
        ):
            super().__init__()
            architecture = str(architecture or "residual_attention").lower()
            if architecture not in IMAGE_RECOGNITION_ARCHITECTURES:
                raise ValueError(f"architecture must be one of {sorted(IMAGE_RECOGNITION_ARCHITECTURES)}")
            self.num_qubits = int(num_qubits)
            self.covariance_feature_dim = int(covariance_feature_dim)
            self.num_groups = int(num_groups)
            self.architecture = architecture
            self.base_channels = int(base_channels)
            self.dropout = float(dropout)
            if architecture == "legacy_cnn":
                self.encoder = nn.Sequential(
                    nn.Conv2d(4, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(4, 32),
                    nn.SiLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.SiLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 128),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                )
                encoder_dim = 128
            else:
                self.encoder = ImageResidualEncoder(4, base_channels=base_channels, dropout=dropout)
                encoder_dim = int(self.encoder.output_dim)
            self.image_proj = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.SiLU(),
            )
            self.cov_proj = nn.Sequential(
                nn.Linear(max(1, self.covariance_feature_dim), covariance_dim),
                nn.LayerNorm(covariance_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            )
            fused_dim = latent_dim + covariance_dim
            if architecture == "legacy_cnn":
                self.fusion = nn.Identity()
                head_dim = fused_dim
            else:
                self.fusion = nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                )
                head_dim = hidden_dim
            self.family_head = nn.Linear(head_dim, num_families)
            self.error_head = nn.Linear(head_dim, num_errors)
            self.group_head = nn.Linear(head_dim, self.num_groups) if self.num_groups > 0 else None
            self.fidelity_head = nn.Linear(head_dim, 1)
            self.purity_head = nn.Linear(head_dim, 1)
            self.physics_head = nn.Linear(head_dim, 15)

        def forward(self, x, covariance):
            image_latent = self.image_proj(self.encoder(x))
            cov_latent = self.cov_proj(covariance)
            latent = self.fusion(torch.cat([image_latent, cov_latent], dim=1))
            out = {
                "latent": latent,
                "family_logits": self.family_head(latent),
                "error_logits": self.error_head(latent),
                "fidelity": torch.sigmoid(self.fidelity_head(latent)).squeeze(-1),
                "purity": torch.sigmoid(self.purity_head(latent)).squeeze(-1),
                "physics_pred": self.physics_head(latent),
            }
            if self.group_head is not None:
                out["group_logits"] = self.group_head(latent)
            return out

    class ErrorPrototypeCrossAttention(nn.Module):
        """Cross-attention between encoded features and learnable error prototypes."""

        def __init__(self, feature_dim: int, num_prototypes: int, num_heads: int = 4):
            super().__init__()
            self.num_prototypes = int(num_prototypes)
            self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
            self.num_heads = int(num_heads)
            self.head_dim = feature_dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.q_proj = nn.Linear(feature_dim, feature_dim)
            self.k_proj = nn.Linear(feature_dim, feature_dim)
            self.v_proj = nn.Linear(feature_dim, feature_dim)
            self.out_proj = nn.Linear(feature_dim, feature_dim)

        def forward(self, x):
            B = x.shape[0]
            prototypes = self.prototypes.unsqueeze(0).expand(B, -1, -1)
            q = self.q_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(prototypes).view(B, self.num_prototypes, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(prototypes).view(B, self.num_prototypes, self.num_heads, self.head_dim).transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, -1)
            return self.out_proj(out)

    class ErrorAwareStateFamilyCNN(nn.Module):
        """Enhanced classifier with error-prototype cross-attention, physics error features,
        and supervised contrastive loss for better error-source discrimination."""

        def __init__(
            self,
            num_qubits: int = 4,
            covariance_feature_dim: int = 0,
            num_families: int = len(STATE_FAMILIES),
            num_errors: int = len(DEFAULT_ERROR_LABELS),
            num_groups: int = 0,
            latent_dim: int = 48,
            covariance_dim: int = 96,
            hidden_dim: int = 160,
            architecture: str = "error_aware_attention",
            base_channels: int = 32,
            dropout: float = 0.08,
            error_feature_dim: int = 15,
        ):
            super().__init__()
            self.num_qubits = int(num_qubits)
            self.covariance_feature_dim = int(covariance_feature_dim)
            self.num_groups = int(num_groups)
            self.architecture = architecture
            self.base_channels = int(base_channels)
            self.dropout = float(dropout)

            self.encoder = ImageResidualEncoder(4, base_channels=base_channels, dropout=dropout)
            encoder_dim = int(self.encoder.output_dim)

            self.image_proj = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.SiLU(),
            )
            self.cov_proj = nn.Sequential(
                nn.Linear(max(1, self.covariance_feature_dim), covariance_dim),
                nn.LayerNorm(covariance_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            )
            self.error_feat_proj = nn.Sequential(
                nn.Linear(error_feature_dim, covariance_dim),
                nn.LayerNorm(covariance_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            )

            fused_dim = latent_dim + covariance_dim * 2
            self.error_cross_attn = ErrorPrototypeCrossAttention(
                feature_dim=fused_dim, num_prototypes=num_errors, num_heads=4,
            )

            self.fusion = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            head_dim = hidden_dim

            self.family_head = nn.Linear(head_dim, num_families)
            self.error_head = nn.Linear(head_dim, num_errors)
            self.group_head = nn.Linear(head_dim, self.num_groups) if self.num_groups > 0 else None
            self.fidelity_head = nn.Linear(head_dim, 1)
            self.purity_head = nn.Linear(head_dim, 1)
            self.physics_head = nn.Linear(head_dim, error_feature_dim)
            self.contrastive_proj = nn.Sequential(
                nn.Linear(head_dim, head_dim),
                nn.LayerNorm(head_dim),
                nn.SiLU(),
                nn.Linear(head_dim, latent_dim),
            )

        def forward(self, x, covariance, error_features=None):
            image_latent = self.image_proj(self.encoder(x))
            cov_latent = self.cov_proj(covariance)
            if error_features is not None:
                error_latent = self.error_feat_proj(error_features)
            else:
                error_latent = torch.zeros(
                    (x.shape[0], self.error_feat_proj[0].out_features),
                    dtype=x.dtype, device=x.device,
                )
            shared = torch.cat([image_latent, cov_latent, error_latent], dim=1)
            error_context = self.error_cross_attn(shared)
            fused = self.fusion(shared + error_context)
            contrastive_latent = self.contrastive_proj(fused)
            out = {
                "latent": fused,
                "family_logits": self.family_head(fused),
                "error_logits": self.error_head(fused),
                "fidelity": torch.sigmoid(self.fidelity_head(fused)).squeeze(-1),
                "purity": torch.sigmoid(self.purity_head(fused)).squeeze(-1),
                "contrastive_latent": contrastive_latent,
                "physics_pred": self.physics_head(fused),
            }
            if self.group_head is not None:
                out["group_logits"] = self.group_head(fused)
            return out

else:

    class QuantumStateFamilyCNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available")

    class ErrorAwareStateFamilyCNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available")


class StateFamilyDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SyntheticStateFamilySample],
        family_labels: Sequence[str],
        error_labels: Sequence[str],
        group_labels: Sequence[str],
        num_qubits: int,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
        self.samples = list(samples)
        self.family_labels = list(family_labels)
        self.error_labels = list(error_labels)
        self.group_labels = list(group_labels)
        self.num_qubits = int(num_qubits)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(rho_state_tensor(sample.rho_measured).tolist(), dtype=torch.float32),
            torch.tensor(state_covariance_features(sample.rho_measured, self.num_qubits).tolist(), dtype=torch.float32),
            torch.tensor(error_sensitive_features(sample.rho_measured, self.num_qubits).tolist(), dtype=torch.float32),
            torch.tensor(self.family_labels.index(sample.state_family), dtype=torch.long),
            torch.tensor(self.group_labels.index(state_family_group(sample.state_family)), dtype=torch.long),
            torch.tensor(self.error_labels.index(sample.error_source), dtype=torch.long),
            torch.tensor([sample.fidelity, sample.purity], dtype=torch.float32),
        )


def train_state_family_classifier(
    samples: Sequence[SyntheticStateFamilySample],
    output: str,
    config: Optional[StateClassifierTrainingConfig] = None,
    family_labels: Optional[Sequence[str]] = None,
    error_labels: Optional[Sequence[str]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for the state-family classifier")
    if not samples:
        raise ValueError("At least one sample is required")

    cfg = config or StateClassifierTrainingConfig()
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = _select_training_device(cfg.device)
    if family_labels is None:
        default_labels = default_state_family_labels(cfg.num_qubits, scheme="fine")
        sample_labels = {sample.state_family for sample in samples}
        family_labels = default_labels + sorted(sample_labels.difference(default_labels))
    else:
        family_labels = list(family_labels)
    error_labels = list(error_labels or DEFAULT_ERROR_LABELS)
    group_labels = [
        group
        for group in COARSE_STATE_FAMILIES
        if group in {state_family_group(label) for label in family_labels}
    ]
    dataset = StateFamilyDataset(samples, family_labels, error_labels, group_labels, cfg.num_qubits)
    cov_dim = int(dataset[0][1].numel())

    val_count = max(1, int(round(len(dataset) * float(cfg.validation_fraction)))) if len(dataset) > 4 else 1
    train_count = max(1, len(dataset) - val_count)
    if train_count + val_count > len(dataset):
        val_count = len(dataset) - train_count
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_ds, val_ds = random_split(dataset, [train_count, val_count], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False)

    train_error_features = torch.stack([dataset[index][2] for index in train_ds.indices]).float()
    error_feature_mean = train_error_features.mean(dim=0)
    error_feature_std = train_error_features.std(dim=0).clamp(min=1e-6)

    is_error_aware = str(cfg.architecture).lower() == "error_aware_attention"
    model_cls = ErrorAwareStateFamilyCNN if is_error_aware else QuantumStateFamilyCNN

    model = model_cls(
        num_qubits=cfg.num_qubits,
        covariance_feature_dim=cov_dim,
        num_families=len(family_labels),
        num_errors=len(error_labels),
        num_groups=len(group_labels),
        latent_dim=cfg.latent_dim,
        covariance_dim=cfg.covariance_dim,
        hidden_dim=cfg.hidden_dim,
        architecture=cfg.architecture,
        base_channels=cfg.base_channels,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=1e-4)

    steps_per_epoch = max(1, train_count // int(cfg.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(5, int(cfg.epochs) // 3), T_mult=2, eta_min=float(cfg.learning_rate) * 0.01,
    )

    ema_state = _initial_ema_state(model) if bool(cfg.use_ema) else None

    contrastive_temperature = 0.07
    contrastive_error_weight = 0.5 if is_error_aware else 0.0
    error_loss_weight = 1.5
    use_focal_loss = True
    physics_loss_weight = 0.30

    error_feature_mean_device = error_feature_mean.to(device)
    error_feature_std_device = error_feature_std.to(device)

    def _normalize_err_feat(ef):
        return (ef - error_feature_mean_device.unsqueeze(0)) / error_feature_std_device.unsqueeze(0)

    history = []
    best_score = None
    patience_score = None
    stale_epochs = 0
    stopped_early = False
    best_state_dict = None
    best_metrics = None
    for epoch in range(int(cfg.epochs)):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            x, cov, err_feat, family, group, error, metrics = batch
            x = x.to(device, non_blocking=True)
            cov = cov.to(device, non_blocking=True)
            err_feat = err_feat.to(device, non_blocking=True)
            family = family.to(device, non_blocking=True)
            group = group.to(device, non_blocking=True)
            error = error.to(device, non_blocking=True)
            metrics = metrics.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if is_error_aware:
                x_train, cov_train, err_train, family_a, family_b, group_a, group_b, error_a, error_b, metrics_train, mix_lam = _mixup_state_batch_7(
                    x.float(), cov.float(), err_feat.float(), family, group, error, metrics, float(cfg.mixup_alpha),
                )
                out = model(x_train, cov_train, err_train)
                loss = (
                    _mixed_ce(out["family_logits"], family_a, family_b, mix_lam, float(cfg.label_smoothing))
                    + error_loss_weight * _mixed_focal_loss(out["error_logits"], error_a, error_b, mix_lam)
                    + F.mse_loss(out["fidelity"], metrics_train[:, 0])
                    + F.mse_loss(out["purity"], metrics_train[:, 1])
                )
                if "physics_pred" in out:
                    loss = loss + physics_loss_weight * F.mse_loss(
                        out["physics_pred"], _normalize_err_feat(err_train),
                    )
            else:
                x_train, cov_train, err_train, family_a, family_b, group_a, group_b, error_a, error_b, metrics_train, mix_lam = _mixup_state_batch_7(
                    x.float(), cov.float(), err_feat.float(), family, group, error, metrics, float(cfg.mixup_alpha),
                )
                out = model(x_train, cov_train)
                loss = (
                    _mixed_ce(out["family_logits"], family_a, family_b, mix_lam, float(cfg.label_smoothing))
                    + error_loss_weight * _mixed_focal_loss(out["error_logits"], error_a, error_b, mix_lam)
                    + F.mse_loss(out["fidelity"], metrics_train[:, 0])
                    + F.mse_loss(out["purity"], metrics_train[:, 1])
                    + physics_loss_weight * F.mse_loss(
                        out["physics_pred"], _normalize_err_feat(err_train),
                    )
                )
            if "group_logits" in out:
                loss = loss + 0.45 * _mixed_ce(
                    out["group_logits"],
                    group_a,
                    group_b,
                    mix_lam,
                    float(cfg.label_smoothing),
                )
            if is_error_aware and "contrastive_latent" in out:
                contrastive_loss = _supervised_contrastive_loss(
                    out["contrastive_latent"], error, contrastive_temperature,
                )
                loss = loss + contrastive_error_weight * contrastive_loss
            loss.backward()
            optimizer.step()
            if ema_state is not None:
                _update_ema_state(ema_state, model, float(cfg.ema_decay))
            running += float(loss.item()) * x.shape[0]
            seen += int(x.shape[0])
        metrics_out = _evaluate_with_optional_ema(model, val_loader, device, ema_state, is_error_aware)
        metrics_out["epoch"] = epoch + 1
        metrics_out["train_loss"] = running / max(seen, 1)
        metrics_out["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history.append(metrics_out)
        if progress_callback is not None:
            progress_callback(epoch + 1, dict(metrics_out))
        score = _selection_score(metrics_out)
        scheduler.step()
        if best_score is None or score > best_score:
            best_score = score
            best_metrics = dict(metrics_out)
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in _merged_ema_state(model, ema_state).items()
            }
        if patience_score is None or score > patience_score + float(cfg.min_delta):
            patience_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1
        if int(cfg.early_stopping_patience) > 0 and stale_epochs >= int(cfg.early_stopping_patience):
            stopped_early = True
            break

    training_metrics = {
        "num_samples": len(dataset),
        "train_samples": train_count,
        "validation_samples": val_count,
        "epochs": int(cfg.epochs),
        "epochs_completed": len(history),
        "device": str(device),
        "requested_device": str(cfg.device),
        "accelerator": _training_accelerator(device),
        "accelerator_device_name": _training_device_name(device),
        "cuda_device_name": _training_device_name(torch.device("cuda")) if torch.cuda.is_available() else None,
        "final": history[-1] if history else {},
        "best": best_metrics or (history[-1] if history else {}),
        "checkpoint_selection": "best_validation_score",
        "regularization": {
            "label_smoothing": float(cfg.label_smoothing),
            "mixup_alpha": float(cfg.mixup_alpha),
            "use_ema": bool(cfg.use_ema),
            "ema_decay": float(cfg.ema_decay),
        },
        "convergence": {
            "stopped_early": bool(stopped_early),
            "stop_epoch": int(history[-1]["epoch"]) if history else 0,
            "early_stopping_patience": int(cfg.early_stopping_patience),
            "min_delta": float(cfg.min_delta),
            "lr_scheduler_patience": int(cfg.lr_scheduler_patience),
            "lr_scheduler_factor": float(cfg.lr_scheduler_factor),
            "best_score": float(best_score) if best_score is not None else None,
            "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        },
        "architecture": {
            "name": str(cfg.architecture),
            "base_channels": int(cfg.base_channels),
            "hidden_dim": int(cfg.hidden_dim),
            "latent_dim": int(cfg.latent_dim),
            "covariance_dim": int(cfg.covariance_dim),
            "dropout": float(cfg.dropout),
            "image_blocks": "multiscale_stem+5_residual_attention_blocks+error_cross_attn" if is_error_aware else "multiscale_stem+5_residual_attention_blocks",
            "fusion_layers": 2 if str(cfg.architecture).lower() != "legacy_cnn" else 0,
            "error_aware": is_error_aware,
        },
        "val_std": _history_metric_std(history),
        "history": history,
    }
    checkpoint = {
        "model_state_dict": best_state_dict or {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "model_type": "error_aware_state_family" if is_error_aware else "state_family_cnn_covariance",
        "num_qubits": int(cfg.num_qubits),
        "family_labels": family_labels,
        "family_groups": {label: state_family_group(label) for label in family_labels},
        "group_labels": group_labels,
        "error_labels": error_labels,
        "feature_config": {
            "input": "rho_measured",
            "channels_per_matrix": ["real", "imag", "abs", "phase"],
            "num_input_channels": 4,
            "covariance_feature_dim": cov_dim,
            "covariance_axes": list(PAULI_AXES),
            "latent_dim": int(cfg.latent_dim),
            "covariance_dim": int(cfg.covariance_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "architecture": str(cfg.architecture),
            "base_channels": int(cfg.base_channels),
            "dropout": float(cfg.dropout),
            "image_blocks": (
                "multiscale_stem+5_residual_attention_blocks+error_cross_attn"
                if is_error_aware else "multiscale_stem+5_residual_attention_blocks"
            ),
            "fusion_layers": 2 if str(cfg.architecture).lower() != "legacy_cnn" else 0,
            "auxiliary_heads": ["state_group", "contrastive_projection"] if is_error_aware else ["state_group"],
            "error_feature_mean": error_feature_mean.astype(float).tolist()
            if isinstance(error_feature_mean, np.ndarray)
            else error_feature_mean.cpu().numpy().astype(float).tolist(),
            "error_feature_std": error_feature_std.astype(float).tolist()
            if isinstance(error_feature_std, np.ndarray)
            else error_feature_std.cpu().numpy().astype(float).tolist(),
        },
        "training_metrics": training_metrics,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    metrics = {
        "output": str(output_path),
        "backend": "torch",
        "num_qubits": int(cfg.num_qubits),
        "family_labels": family_labels,
        "family_groups": {label: state_family_group(label) for label in family_labels},
        "group_labels": group_labels,
        "error_labels": error_labels,
        **training_metrics,
    }
    metrics_path = str(output_path.with_suffix(".metrics.json"))
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    metrics["metrics_path"] = metrics_path
    return metrics


def evaluate_state_family_classifier(model, data_loader, device=None, is_error_aware: bool = False) -> Dict[str, float]:
    device = device or next(model.parameters()).device
    model.eval()
    total = 0
    family_correct = 0
    group_correct = 0
    group_total = 0
    error_correct = 0
    fidelity_abs = 0.0
    purity_abs = 0.0
    with torch.no_grad():
        for batch in data_loader:
            x, cov, err_feat, family, group, error, metrics = batch
            x = x.to(device, non_blocking=True)
            cov = cov.to(device, non_blocking=True)
            err_feat = err_feat.to(device, non_blocking=True)
            family = family.to(device, non_blocking=True)
            group = group.to(device, non_blocking=True)
            error = error.to(device, non_blocking=True)
            metrics = metrics.to(device, non_blocking=True)
            if is_error_aware:
                out = model(x.float(), cov.float(), err_feat.float())
            else:
                out = model(x.float(), cov.float())
            family_pred = torch.argmax(out["family_logits"], dim=1)
            error_pred = torch.argmax(out["error_logits"], dim=1)
            total += int(x.shape[0])
            family_correct += int(torch.sum(family_pred == family).item())
            if "group_logits" in out:
                group_pred = torch.argmax(out["group_logits"], dim=1)
                group_correct += int(torch.sum(group_pred == group).item())
                group_total += int(x.shape[0])
            error_correct += int(torch.sum(error_pred == error).item())
            fidelity_abs += float(torch.sum(torch.abs(out["fidelity"] - metrics[:, 0])).item())
            purity_abs += float(torch.sum(torch.abs(out["purity"] - metrics[:, 1])).item())
    metrics_out = {
        "state_family_accuracy": family_correct / max(total, 1),
        "error_source_accuracy": error_correct / max(total, 1),
        "mean_abs_fidelity_error": fidelity_abs / max(total, 1),
        "mean_abs_purity_error": purity_abs / max(total, 1),
    }
    if group_total:
        metrics_out["state_group_accuracy"] = group_correct / max(group_total, 1)
    return metrics_out


def _mixup_state_batch(x, cov, family, group, error, metrics, alpha: float):
    if alpha <= 0.0 or int(x.shape[0]) < 2:
        return x, cov, family, family, group, group, error, error, metrics, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(int(x.shape[0]), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    mixed_cov = lam * cov + (1.0 - lam) * cov[perm]
    mixed_metrics = lam * metrics + (1.0 - lam) * metrics[perm]
    return mixed_x, mixed_cov, family, family[perm], group, group[perm], error, error[perm], mixed_metrics, lam


def _mixup_state_batch_7(x, cov, err_feat, family, group, error, metrics, alpha: float):
    if alpha <= 0.0 or int(x.shape[0]) < 2:
        return x, cov, err_feat, family, family, group, group, error, error, metrics, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(int(x.shape[0]), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    mixed_cov = lam * cov + (1.0 - lam) * cov[perm]
    mixed_err = lam * err_feat + (1.0 - lam) * err_feat[perm]
    mixed_metrics = lam * metrics + (1.0 - lam) * metrics[perm]
    return mixed_x, mixed_cov, mixed_err, family, family[perm], group, group[perm], error, error[perm], mixed_metrics, lam


def _supervised_contrastive_loss(features, labels, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss: pull same-label features together, push different apart."""
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)
    features = F.normalize(features, dim=1)
    sim = features @ features.T / temperature
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    mask.fill_diagonal_(0)
    exp_sim = torch.exp(sim)
    pos_numer = (exp_sim * mask).sum(dim=1)
    neg_denom = (exp_sim * (1.0 - torch.eye(B, device=features.device, dtype=features.dtype))).sum(dim=1)
    neg_denom = torch.clamp(neg_denom, min=1e-8)
    loss = -torch.log(pos_numer / neg_denom + 1e-8)
    valid = (mask.sum(dim=1) > 0).float()
    if valid.sum() < 1:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)
    return (loss * valid).sum() / valid.sum()


def _mixed_ce(logits, target_a, target_b, lam: float, smoothing: float):
    loss_a = F.cross_entropy(logits, target_a, label_smoothing=float(smoothing))
    if lam >= 0.999:
        return loss_a
    loss_b = F.cross_entropy(logits, target_b, label_smoothing=float(smoothing))
    return float(lam) * loss_a + (1.0 - float(lam)) * loss_b


def _mixed_focal_loss(logits, target_a, target_b, lam: float, gamma: float = 2.0, alpha: float = 0.25):
    """MixUp-aware focal loss for imbalanced classification."""
    ce_a = F.cross_entropy(logits, target_a, reduction='none')
    pt_a = torch.exp(-ce_a)
    focal_a = alpha * (1.0 - pt_a) ** gamma * ce_a
    if lam >= 0.999:
        return focal_a.mean()
    ce_b = F.cross_entropy(logits, target_b, reduction='none')
    pt_b = torch.exp(-ce_b)
    focal_b = alpha * (1.0 - pt_b) ** gamma * ce_b
    return (float(lam) * focal_a + (1.0 - float(lam)) * focal_b).mean()


def _initial_ema_state(model) -> Dict[str, Any]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if torch.is_floating_point(value)
    }


def _update_ema_state(ema_state: Dict[str, Any], model, decay: float) -> None:
    state = model.state_dict()
    for key, value in state.items():
        if key not in ema_state or not torch.is_floating_point(value):
            continue
        ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def _evaluate_with_optional_ema(model, data_loader, device, ema_state, is_error_aware: bool = False):
    if ema_state is None:
        return evaluate_state_family_classifier(model, data_loader, device=device, is_error_aware=is_error_aware)
    current = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    merged = dict(current)
    merged.update(ema_state)
    model.load_state_dict(merged)
    try:
        return evaluate_state_family_classifier(model, data_loader, device=device, is_error_aware=is_error_aware)
    finally:
        model.load_state_dict(current)


def _merged_ema_state(model, ema_state):
    state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    if ema_state:
        state.update(ema_state)
    return state


def _history_metric_std(history: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = [
        "state_family_accuracy",
        "state_group_accuracy",
        "error_source_accuracy",
        "mean_abs_fidelity_error",
        "mean_abs_purity_error",
    ]
    out: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in history], dtype=float)
        values = values[np.isfinite(values)]
        out[key] = float(np.std(values)) if values.size else float("nan")
    return out


class QuantumStateFamilyTorchClassifier:
    """Loaded state-family classifier with a simple predict API."""

    def __init__(self, model, checkpoint: Dict[str, Any]):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
        self.model = model
        self.checkpoint = checkpoint
        self.device = next(model.parameters()).device
        self.num_qubits = int(checkpoint["num_qubits"])
        self.family_labels = list(checkpoint.get("family_labels") or STATE_FAMILIES)
        self.family_groups = dict(checkpoint.get("family_groups") or {
            label: state_family_group(label)
            for label in self.family_labels
        })
        self.group_labels = list(checkpoint.get("group_labels") or [])
        self.error_labels = list(checkpoint.get("error_labels") or DEFAULT_ERROR_LABELS)
        self.feature_config = dict(checkpoint.get("feature_config") or {})
        self.training_metrics = dict(checkpoint.get("training_metrics") or {})
        self.is_error_aware = (
            checkpoint.get("model_type") == "error_aware_state_family"
            or str(self.feature_config.get("architecture", "")).lower() == "error_aware_attention"
        )
        self.backend = "torch_state_family"

    @classmethod
    def load(cls, path: str) -> "QuantumStateFamilyTorchClassifier":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
        checkpoint = _torch_load_checkpoint(path)
        feature_config = dict(checkpoint.get("feature_config") or {})
        family_labels = list(checkpoint.get("family_labels") or STATE_FAMILIES)
        error_labels = list(checkpoint.get("error_labels") or DEFAULT_ERROR_LABELS)
        group_labels = list(checkpoint.get("group_labels") or [])
        architecture = str(feature_config.get("architecture", "legacy_cnn"))
        is_error_aware = (
            checkpoint.get("model_type") == "error_aware_state_family"
            or architecture.lower() == "error_aware_attention"
        )
        model_cls = ErrorAwareStateFamilyCNN if is_error_aware else QuantumStateFamilyCNN
        model = model_cls(
            num_qubits=int(checkpoint["num_qubits"]),
            covariance_feature_dim=int(feature_config.get("covariance_feature_dim", 0)),
            num_families=len(family_labels),
            num_errors=len(error_labels),
            num_groups=len(group_labels),
            latent_dim=int(feature_config.get("latent_dim", 48)),
            covariance_dim=int(feature_config.get("covariance_dim", 96)),
            hidden_dim=int(feature_config.get("hidden_dim", 160)),
            architecture=architecture,
            base_channels=int(feature_config.get("base_channels", 32)),
            dropout=float(feature_config.get("dropout", 0.05)),
        )
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # Historical checkpoints predate the auxiliary physics head.  It is
        # permitted to be absent, but no other schema mismatch is accepted.
        allowed_missing = {"physics_head.weight", "physics_head.bias"}
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "incompatible state-family checkpoint: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return cls(model, checkpoint)

    def predict(self, rho: np.ndarray) -> Dict[str, Any]:
        x = torch.tensor(rho_state_tensor(rho).tolist(), dtype=torch.float32, device=self.device).unsqueeze(0)
        cov = torch.tensor(
            state_covariance_features(rho, self.num_qubits).tolist(),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        error_features = torch.tensor(
            error_sensitive_features(rho, self.num_qubits).tolist(),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x, cov, error_features) if self.is_error_aware else self.model(x, cov)
            family_probs = torch.softmax(out["family_logits"], dim=1).cpu()[0].tolist()
            error_probs = torch.softmax(out["error_logits"], dim=1).cpu()[0].tolist()
            learned_group_probs = (
                torch.softmax(out["group_logits"], dim=1).cpu()[0].tolist()
                if "group_logits" in out
                else []
            )
            latent = out["latent"].cpu()[0].tolist()
        aggregated_group_probs = _aggregate_family_group_probabilities(
            self.family_labels,
            family_probs,
        )
        return {
            "available": True,
            "backend": self.backend,
            "state_family_probabilities": {
                label: float(prob)
                for label, prob in zip(self.family_labels, family_probs)
            },
            "state_family_groups": dict(self.family_groups),
            "state_family_group_probabilities": aggregated_group_probs,
            "learned_state_group_probabilities": {
                label: float(prob)
                for label, prob in zip(self.group_labels, learned_group_probs)
            } if learned_group_probs else aggregated_group_probs,
            "error_source_probabilities": {
                label: float(prob)
                for label, prob in zip(self.error_labels, error_probs)
            },
            "fidelity_estimate": float(out["fidelity"].cpu()[0].item()),
            "purity_estimate": float(out["purity"].cpu()[0].item()),
            "latent_vector": [float(value) for value in latent],
            "covariance_feature_vector": state_covariance_features(rho, self.num_qubits).astype(float).tolist(),
            "error_sensitive_feature_vector": error_features.cpu()[0].tolist(),
        }


def _apply_family_error(rho: np.ndarray, num_qubits: int, error_source: str, rng: np.random.Generator) -> np.ndarray:
    dim = rho.shape[0]
    measured = np.array(rho, dtype=complex, copy=True)
    if error_source == "phase_drift":
        # Correlated local-Z drift is unitary: it preserves purity and
        # coherence magnitudes.  This is intentionally distinct from the
        # incoherent CPTP dephasing channel below.
        local_phase = rng.normal(0.0, 0.25, size=num_qubits)
        basis_phase = np.zeros(dim, dtype=float)
        for basis_index in range(dim):
            basis_phase[basis_index] = sum(
                local_phase[qubit]
                for qubit in range(num_qubits)
                if basis_index & (1 << (num_qubits - 1 - qubit))
            )
        phases = np.exp(1j * basis_phase)
        op = np.diag(phases)
        measured = op @ measured @ op.conj().T
    elif error_source == "amp_imbalance":
        local_scale = rng.uniform(0.7, 1.3, size=num_qubits)
        scale = np.ones(dim, dtype=float)
        for basis_index in range(dim):
            scale[basis_index] = np.prod([
                local_scale[qubit]
                for qubit in range(num_qubits)
                if basis_index & (1 << (num_qubits - 1 - qubit))
            ])
        op = np.diag(scale)
        measured = op @ measured @ op.conj().T
    elif error_source == "dephasing":
        strength = rng.uniform(0.10, 0.50)
        measured = (1.0 - strength) * measured + strength * np.diag(np.diag(measured))
    elif error_source == "retrieval_imbalance":
        # This is an apparent-state effect of a measurement-efficiency filter,
        # not a latent-state preparation channel.  Count-level training uses
        # CountNoiseProfile instead of this legacy density-only approximation.
        scale = np.ones(dim)
        single = [2 ** (num_qubits - 1 - i) for i in range(num_qubits)]
        for idx in single:
            scale[idx] = rng.uniform(0.45, 1.0)
        measured = np.diag(scale) @ measured @ np.diag(scale)
    elif error_source == "double_excitation":
        leak = rng.uniform(0.01, 0.08)
        leak_state = np.zeros(dim, dtype=complex)
        candidates = []
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                candidates.append(2 ** (num_qubits - 1 - i) + 2 ** (num_qubits - 1 - j))
        leak_state[candidates[int(rng.integers(0, len(candidates)))]] = 1.0
        measured = (1.0 - leak) * measured + leak * np.outer(leak_state, leak_state.conj())
    depol = rng.uniform(0.0, 0.06)
    measured = (1.0 - depol) * measured + depol * np.eye(dim, dtype=complex) / dim
    return _physicalize(measured)


def _compatible_error_labels(family: str, labels: Sequence[str]) -> List[str]:
    """Exclude family/error pairs that are invisible to density-only input."""

    group = state_family_group(family)
    requested = list(labels)
    if group in {"fully_separable", "single_excitation"}:
        compatible = [label for label in requested if label == "double_excitation"]
    else:
        compatible = requested
    return compatible or requested


def _pure_density(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    norm = np.linalg.norm(psi)
    if norm <= 0:
        psi = np.zeros_like(psi)
        psi[0] = 1.0
        norm = 1.0
    psi = psi / norm
    return np.outer(psi, psi.conj())


def _physicalize(rho: np.ndarray) -> np.ndarray:
    herm = 0.5 * (rho + rho.conj().T)
    eigvals, eigvecs = np.linalg.eigh(herm)
    eigvals = np.clip(np.real(eigvals), 0.0, None)
    if float(np.sum(eigvals)) <= 0:
        return np.eye(rho.shape[0], dtype=complex) / rho.shape[0]
    out = eigvecs @ np.diag(eigvals / np.sum(eigvals)) @ eigvecs.conj().T
    return 0.5 * (out + out.conj().T)


def _bits_to_index(bits: Iterable[int]) -> int:
    idx = 0
    for bit in bits:
        idx = (idx << 1) | int(bit)
    return idx


def _aggregate_family_group_probabilities(
    family_labels: Sequence[str],
    family_probabilities: Sequence[float],
) -> Dict[str, float]:
    grouped: Dict[str, float] = {}
    for label, probability in zip(family_labels, family_probabilities):
        group = state_family_group(label)
        grouped[group] = grouped.get(group, 0.0) + float(probability)
    return {
        group: grouped[group]
        for group in COARSE_STATE_FAMILIES
        if group in grouped
    }


def _selection_score(metrics: Dict[str, float]) -> float:
    return (
        float(metrics.get("state_family_accuracy", 0.0))
        + 0.10 * float(metrics.get("state_group_accuracy", 0.0))
        + 0.20 * float(metrics.get("error_source_accuracy", 0.0))
        - 0.10 * float(metrics.get("mean_abs_fidelity_error", 1.0))
        - 0.10 * float(metrics.get("mean_abs_purity_error", 1.0))
    )


def _torch_load_checkpoint(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
