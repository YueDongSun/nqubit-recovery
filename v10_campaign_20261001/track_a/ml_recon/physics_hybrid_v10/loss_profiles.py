"""Registered classifier loss-weight profiles for bounded V10 experiments."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


_DEFAULT_WEIGHTS = {
    "measurement": 1.0,
    "denoise": 0.5,
    "small_n_family": 0.4,
    "pair_map": 0.4,
    "hyperedge": 0.2,
    "partition": 0.3,
    "depth": 0.3,
    "intactness": 0.3,
    "stretchability": 0.2,
    "coarse": 0.3,
    "w_r1": 0.8,
    "w_coherence": 0.6,
    "w_sector": 0.2,
    "operational_noise": 0.2,
    "psd": 0.1,
    "rho8_weak": 0.0,
    "ood": 0.1,
    "calibration": 0.1,
    "setting_alignment": 0.2,
}

LOSS_WEIGHT_PROFILES: Mapping[str, Mapping[str, float]] = MappingProxyType({
    "default": MappingProxyType(dict(_DEFAULT_WEIGHTS)),
    # Candidate profile motivated by the ddc4 diagnostic: noisy-cell pair
    # utility regressed while setting-alignment controls remained weak.
    "pair_alignment_v1": MappingProxyType({
        **_DEFAULT_WEIGHTS,
        "pair_map": 0.8,
        "setting_alignment": 0.5,
    }),
    # Gradient-balanced candidate: fixed-batch diagnostics showed pair and
    # partition gradients dominating small_n_family in noisy cells.
    "balanced_structure_v1": MappingProxyType({
        **_DEFAULT_WEIGHTS,
        "pair_map": 0.2,
        "partition": 0.15,
        "setting_alignment": 0.3,
    }),
    # Controlled alignment candidate: retain the balanced structure weights
    # while making the train-only descriptor/value margin measurable.
    "alignment_focus_v1": MappingProxyType({
        **_DEFAULT_WEIGHTS,
        "pair_map": 0.2,
        "partition": 0.15,
        "setting_alignment": 1.5,
    }),
    # Noisy-cell focus candidate (campaign 2026-10-01 exploration phase):
    # n4_noisy family loss dominated the remaining validation error; upweight
    # the primary family objective and downweight the already-converged
    # measurement NLL, retaining the balanced_structure_v1 structure weights.
    "noisy_focus_v1": MappingProxyType({
        **_DEFAULT_WEIGHTS,
        "small_n_family": 0.8,
        "measurement": 0.6,
        "pair_map": 0.2,
        "partition": 0.15,
        "setting_alignment": 0.3,
    }),
})
LOSS_WEIGHT_PROFILE_NAMES = tuple(LOSS_WEIGHT_PROFILES)


def resolve_loss_weights(profile: str) -> Mapping[str, float]:
    """Return a registered immutable profile or fail before model creation."""
    if type(profile) is not str or profile not in LOSS_WEIGHT_PROFILES:
        raise ValueError(
            f"loss_weight_profile must be one of {LOSS_WEIGHT_PROFILE_NAMES}")
    return LOSS_WEIGHT_PROFILES[profile]
