"""Track A losses, gates, and engineering smoke."""

from .gates import TrackAGateLedger, verify_external_gate_artifact
from .gate_specs import gate_spec, validate_gate_evidence
from .losses import (
    TrackALossWeights,
    balanced_multitask_loss,
    masked_mean,
    published_head_losses,
)
from .runtime import TrackARuntimeAdapter
from .server_acceptance import SourceAcceptanceError, bind_source_provenance
from .smoke import run_track_a_smoke

__all__ = [
    "TrackAGateLedger",
    "TrackALossWeights",
    "TrackARuntimeAdapter",
    "SourceAcceptanceError",
    "balanced_multitask_loss",
    "bind_source_provenance",
    "gate_spec",
    "masked_mean",
    "published_head_losses",
    "run_track_a_smoke",
    "validate_gate_evidence",
    "verify_external_gate_artifact",
]
