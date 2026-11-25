"""Deep physics-informed training package for W-state calibration.

This package is intentionally self-contained so the existing lightweight
models and older deep entry points remain untouched.
"""

from .configs import DeepTrainingConfig, SystemPhysicsConfig, get_system_config
from .taxonomy import (
    COARSE_ERROR_LABELS,
    ENTANGLEMENT_TIER_LABELS,
    FINE_ERROR_LABELS,
    PHYSICS_AUX_TARGET_NAMES,
    STATE_FAMILY_LABELS,
)

__all__ = [
    "DeepTrainingConfig",
    "SystemPhysicsConfig",
    "get_system_config",
    "COARSE_ERROR_LABELS",
    "ENTANGLEMENT_TIER_LABELS",
    "FINE_ERROR_LABELS",
    "PHYSICS_AUX_TARGET_NAMES",
    "STATE_FAMILY_LABELS",
]
