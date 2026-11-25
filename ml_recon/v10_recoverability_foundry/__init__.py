"""Track A recoverability-first research foundry.

The package is an isolated, research-only project boundary over the verified
``physics_hybrid_v10`` primitives.  It never imports experiment-control or
hardware-write modules.
"""

from .config import TrackAConfig, load_track_a_config
from .contracts import (
    MainlineBundleV1,
    RecoverabilityBundleV1,
    TrackAAuthority,
    TrackASample,
    TrackASampleKey,
)

__all__ = [
    "MainlineBundleV1",
    "RecoverabilityBundleV1",
    "TrackAAuthority",
    "TrackAConfig",
    "TrackASample",
    "TrackASampleKey",
    "load_track_a_config",
]

__version__ = "0.1.0.dev0"
