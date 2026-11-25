"""Physics Hybrid V4 six-model research and training stack.

The package is deliberately independent from the production AWG path.  It
provides versioned data contracts, gauge-free operation-phase geometry,
synthetic physical curricula, scalable PyTorch models, and a unified training
orchestrator.  Hardware promotion remains the responsibility of the existing
candidate/guard/rollback workflow.
"""

from .config import MODEL_IDS, PhysicsHybridV4Config, load_v4_config
from .architecture import (
    ArchitectureSelection,
    ModelBuildSpec,
    model_architecture_ladder,
    resolve_architecture_selection,
)
from .campaign_analysis import analyze_campaign
from .contracts import CrosstalkScanV1, OperationPhaseActionV1, OperationPhaseProbeV1
from .drift import DriftFitResult, fit_drift_candidates
from .phase_geometry import (
    action_tensor_to_cycle_matrices,
    cycle_matrices_to_action_tensor,
    helmert_matrix,
    materialize_gauge_coordinates,
    project_phase_offsets,
)
from .probe_design import phase_probe_tensor_to_cycle_matrices

__all__ = [
    "MODEL_IDS",
    "ArchitectureSelection",
    "CrosstalkScanV1",
    "DriftFitResult",
    "OperationPhaseActionV1",
    "OperationPhaseProbeV1",
    "ModelBuildSpec",
    "PhysicsHybridV4Config",
    "action_tensor_to_cycle_matrices",
    "analyze_campaign",
    "cycle_matrices_to_action_tensor",
    "fit_drift_candidates",
    "helmert_matrix",
    "load_v4_config",
    "materialize_gauge_coordinates",
    "model_architecture_ladder",
    "phase_probe_tensor_to_cycle_matrices",
    "project_phase_offsets",
    "resolve_architecture_selection",
]
