"""Validate and rebuild the explicit classifier constructor in a checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .classifier_n_full import CLASSIFIER_PROFILES, V10ClassifierNFull


CONSTRUCTOR_DIMENSIONS = ("descriptor_dim", "mode_feature_dim", "certificate_dim")
CONSTRUCTOR_FIELDS = frozenset(("architecture_class", "profile", *CONSTRUCTOR_DIMENSIONS))


def classifier_from_build_spec(build_spec: Mapping[str, Any]) -> V10ClassifierNFull:
    """Reject malformed explicit specs instead of guessing a different model."""
    if not isinstance(build_spec, Mapping) or set(build_spec) != CONSTRUCTOR_FIELDS:
        raise ValueError("classifier checkpoint requires the complete constructor specification")
    if (not isinstance(build_spec["architecture_class"], str)
            or build_spec["architecture_class"] != "V10ClassifierNFull"):
        raise ValueError("unsupported classifier checkpoint architecture")
    profile = build_spec["profile"]
    if not isinstance(profile, str) or profile not in CLASSIFIER_PROFILES:
        raise ValueError("classifier checkpoint requires a registered profile")
    if any(type(build_spec[name]) is not int or build_spec[name] <= 0
           for name in CONSTRUCTOR_DIMENSIONS):
        raise ValueError("classifier constructor dimensions must be positive integers")
    return V10ClassifierNFull(profile=profile, **{
        name: build_spec[name] for name in CONSTRUCTOR_DIMENSIONS})
