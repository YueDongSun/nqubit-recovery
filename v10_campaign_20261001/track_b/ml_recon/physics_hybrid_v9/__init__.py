"""Physics Hybrid V9: controllability-aware, count-native, multi-view-rho-assisted
recoverability learning for spatially encoded W-state quantum networks.

Research-only successor track to V8. V9 has NO hardware-write authority: it does not
import ExperimentRunner, write matrix/ml_delta, upload AWG sequences, program the
FPGA, or request collection. It inherits Data Contract V2.3 and the count/detector
semantics from V8 and does not reclassify rows or create real nuisance/action labels.
"""

from .contracts import (
    V9ModelCell,
    ModelCellStatus,
    V9RhoEightViewContractV1,
    V9RhoEightViewBundleV1,
    V9RhoViewProvenanceV1,
    V9CountContractV1,
    V9FourCellCountsV1,
    V9GlobalMultiV1,
    V9OperationPointActionProposalV1,
    V9OperationPointActionContractV1,
    V9PreparedOperationActionArtifactV1,
    V9ResearchAuthorityV1,
)

__all__ = [
    "V9ModelCell",
    "ModelCellStatus",
    "V9RhoEightViewContractV1",
    "V9RhoEightViewBundleV1",
    "V9RhoViewProvenanceV1",
    "V9CountContractV1",
    "V9FourCellCountsV1",
    "V9GlobalMultiV1",
    "V9OperationPointActionProposalV1",
    "V9OperationPointActionContractV1",
    "V9PreparedOperationActionArtifactV1",
    "V9ResearchAuthorityV1",
]
