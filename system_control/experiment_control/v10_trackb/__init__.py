"""Dry-run-only Track B proposal-consumer boundary.

Nothing in this namespace loads model weights, performs model inference,
generates a sequence, contacts a device, or enables repeated execution.
"""

from __future__ import annotations

from .execution_authority import ExternalRFExecutionAuthorityV1, OneShotNonceRegistry
from .materialization_preview import RFMaterializationPreviewV1, preview_rf_materialization
from .proposal_reader import (
    ProposalBindingsV1,
    ProposalValidationError,
    ResearchProposalDocumentV1,
    parse_research_proposal_bytes,
    read_research_proposal,
)
from .rf_inverse import (
    NominalRFInverseContractV1,
    RFInverseResultV1,
    from_actuator_transfer_payload,
    solve_nominal_rf_inverse,
)
from .support_audit import audit_recoverable_support, shrink_action_ray_until_supported
from .transition_logger import TrackBTransitionLogger, TrackBTransitionManifestV1
from .awg_reproduction import (
    TrackBAWGReproductionInputV1,
    TrackBAWGReproductionPairReceiptV1,
    TrackBAWGReproductionReceiptV1,
    preview_trackb_awg_reproduction,
    verify_trackb_awg_reproduction_pair,
    write_trackb_awg_reproduction,
)

__all__ = [
    "ExternalRFExecutionAuthorityV1",
    "NominalRFInverseContractV1",
    "OneShotNonceRegistry",
    "ProposalBindingsV1",
    "ProposalValidationError",
    "RFInverseResultV1",
    "RFMaterializationPreviewV1",
    "ResearchProposalDocumentV1",
    "TrackBTransitionLogger",
    "TrackBTransitionManifestV1",
    "TrackBAWGReproductionInputV1",
    "TrackBAWGReproductionPairReceiptV1",
    "TrackBAWGReproductionReceiptV1",
    "audit_recoverable_support",
    "from_actuator_transfer_payload",
    "parse_research_proposal_bytes",
    "preview_rf_materialization",
    "read_research_proposal",
    "shrink_action_ray_until_supported",
    "solve_nominal_rf_inverse",
    "preview_trackb_awg_reproduction",
    "verify_trackb_awg_reproduction_pair",
    "write_trackb_awg_reproduction",
]
