"""Strict conversion of complete classifier and feedback exports for the hierarchy."""

from __future__ import annotations

from .classifier_n_full import V10ClassifierNFull

from .training_ideas_data import PROJECTOR_READOUT_SCOPE

CLASSIFIER_SCHEMA = "v10_projector_core_v2"


def classifier_configuration(payload):
    if payload.get("schema") == CLASSIFIER_SCHEMA:
        config = dict(payload["classifier_config"])
        state = payload["classifier_state_dict"]
    elif payload.get("model_id") == "C_N_FULL_V10_TRAINING_IDEAS_PROJECTOR_V1":
        exp = payload["experiment"]
        if (
            exp["readout_domain"] != "single_excitation_projector"
            or exp["experiment_version"] != "v10_training_ideas_projector_v1"
        ):
            raise ValueError("unrecognized archived projector constructor")
        config = dict(
            profile=exp["model_profile"],
            training_ideas=dict(exp["training_ideas"]),
            readout_domain=exp["readout_domain"],
            observation_law=PROJECTOR_READOUT_SCOPE,
        )
        state = payload["model_state_dict"]
    else:
        raise ValueError("a full identified projector classifier export is required")
    return config, state


def make_backbone_envelope(classifier_export, feedback_export, *, hierarchy_config=None):
    """Validate both complete states before returning the PR43 six-key envelope."""
    from .feedback_n_student_csm import V10FeedbackNStudentCSM
    from .trackb_inherited_hierarchy import TrackBInheritedHierarchyConfig

    config, classifier_state = classifier_configuration(classifier_export)
    classifier = V10ClassifierNFull(**config)
    classifier.load_state_dict(classifier_state, strict=True)
    feedback_config = dict(feedback_export["feedback_config"])
    feedback_state = feedback_export["feedback_state_dict"]
    feedback = V10FeedbackNStudentCSM(**feedback_config)
    feedback.load_state_dict(feedback_state, strict=True)
    hierarchy = TrackBInheritedHierarchyConfig(**(hierarchy_config or {}))
    return dict(
        schema="TrackBSharedBackboneV1",
        classifier_config=config,
        classifier_state_dict=classifier_state,
        feedback_config=feedback_config,
        feedback_state_dict=feedback_state,
        hierarchy_config=hierarchy.to_dict(),
    )
