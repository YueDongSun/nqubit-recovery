"""Research-only V7 operation-response, hierarchical-posterior, and global frequency-feedback models.

This package now exposes four V7 research tracks:

* :mod:`.model` — closed-form local-nonlinear + global frequency-feedback model.
* :mod:`.joint_nb2` — joint NB2 coherent/accidental mixture fit on measured snapshots.
* :mod:`.scalable_model` — Perceiver-style physics-constrained count model (v1).
* :mod:`.scalable_model_v2` — V2 with AOD positive response, retrieval field
  amplitude, cycle random effects, and multi-excitation mixture.
* :mod:`.hierarchical_posterior` — closed-form coherent/accidental/multi-excitation
  Bayesian posterior with per-cycle random effects and a positive AOD frequency
  response.
* :mod:`.posterior_calibration` — posterior predictive checks and simulation-based
  calibration for the hierarchical posterior.
* :mod:`.extended_smoke` — three-profile (local / mid / server) + five-seed
  training runner that combines the v2 model with the closed-form posterior.
* :mod:`.vram_sweep` — RTX 4070 Ti memory pressure sweep; one profile per
  (width, layers, batch_size) with peak reserved bytes.
* :mod:`.learning_rate_ab` — paired baseline / v3 trainer comparison on a
  matched synthetic holdout.
"""

from .geometry import OperationFrequencyGeometryV1, build_operation_frequency_geometry
from .model import (
    V7GlobalFeedbackModel,
    V7PretrainingPriorV1,
    V7ResponseArrays,
    fit_v7_global_feedback,
)
from .program_commands import ProgramCommandV1, resolve_program_commands
from .noise_subtraction import V7GsiNoiseModelV1, fit_v7_gsi_noise_model
from .physical_ceiling import (
    ConformalBestFidelityModelV1,
    PhysicsResponseSurfaceV1,
    fit_physics_response_surface,
    load_retrieval_rate_vectors,
)
from .joint_nb2 import V7JointNB2Arrays, V7JointNB2Model, fit_v7_joint_nb2
from .scalable_model import (
    V7ScalableModelConfig,
    V7ScalablePhysicsModel,
    v7_scalable_multitask_loss,
)
from .scalable_model_v2 import (
    V7ScalableModelV2Config,
    V7ScalablePhysicsModelV2,
    build_cycle_index,
)
from .hierarchical_posterior import (
    HierarchicalPosteriorArrays,
    HierarchicalPosteriorConfig,
    HierarchicalPosteriorResult,
    build_command_design,
    fit_hierarchical_posterior,
    mechanism_attribution,
    predict_hierarchical_posterior,
)
from .posterior_calibration import (
    PosteriorPredictiveDiagnostics,
    five_seed_aggregator,
    posterior_predictive_check,
    simulation_based_calibration,
)
from .extended_smoke import run_extended_smoke
from .vram_sweep import run_vram_sweep
from .learning_rate_ab import run_learning_rate_ab
from .v3_weight_sweep import run_v3_weight_sweep
from .v4_ab import run_v4_ab
from .v4_measured_ab import run_v4_measured_ab
from .feedback_data import (
    FeedbackMeasuredArraysV1,
    FeedbackNuisanceProfileV1,
    fit_feedback_nuisance_profile,
    load_feedback_measured_arrays,
)
from .feedback_models import V7FeedbackModelBundle, V7FeedbackModelConfig
from .feedback_synthetic import (
    FeedbackSyntheticConfig,
    evaluate_counterfactual_action,
    generate_feedback_transition_batch,
)

__all__ = [
    "ConformalBestFidelityModelV1",
    "FeedbackMeasuredArraysV1",
    "FeedbackNuisanceProfileV1",
    "FeedbackSyntheticConfig",
    "HierarchicalPosteriorArrays",
    "HierarchicalPosteriorConfig",
    "HierarchicalPosteriorResult",
    "OperationFrequencyGeometryV1",
    "PhysicsResponseSurfaceV1",
    "PosteriorPredictiveDiagnostics",
    "ProgramCommandV1",
    "V7GlobalFeedbackModel",
    "V7FeedbackModelBundle",
    "V7FeedbackModelConfig",
    "V7GsiNoiseModelV1",
    "V7JointNB2Arrays",
    "V7JointNB2Model",
    "V7PretrainingPriorV1",
    "V7ResponseArrays",
    "V7ScalableModelConfig",
    "V7ScalableModelV2Config",
    "V7ScalablePhysicsModel",
    "V7ScalablePhysicsModelV2",
    "build_command_design",
    "build_cycle_index",
    "build_operation_frequency_geometry",
    "five_seed_aggregator",
    "fit_hierarchical_posterior",
    "fit_feedback_nuisance_profile",
    "fit_physics_response_surface",
    "fit_v7_global_feedback",
    "fit_v7_gsi_noise_model",
    "fit_v7_joint_nb2",
    "load_retrieval_rate_vectors",
    "load_feedback_measured_arrays",
    "mechanism_attribution",
    "posterior_predictive_check",
    "predict_hierarchical_posterior",
    "resolve_program_commands",
    "evaluate_counterfactual_action",
    "generate_feedback_transition_batch",
    "run_extended_smoke",
    "run_learning_rate_ab",
    "run_v3_weight_sweep",
    "run_v4_ab",
    "run_v4_measured_ab",
    "run_vram_sweep",
    "simulation_based_calibration",
    "v7_scalable_multitask_loss",
]
