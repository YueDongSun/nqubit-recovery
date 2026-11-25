"""Compose the verified V10 classifier and feedback models as Track A mainline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch

from ml_recon.data_v2_4.feedback_gate_contract import FEEDBACK_GATE_DIM
from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.physics_hybrid_v10.track_a_n8.classifier_n_full import V10ClassifierNFull
from ml_recon.physics_hybrid_v10.feedback_n_student import V10FeedbackNStudent
from ml_recon.physics_hybrid_v10.track_a_n8.feedback_n_student_csm import V10FeedbackNStudentCSM
from ml_recon.physics_hybrid_v10.published_output_registry import (
    require_calibrated_feedback_outputs,
)

from ..config import TrackAConfig
from ..contracts import MainlineBundleV1, TrackAAuthority
from ..contracts.core import validate_hex_digest


def state_dict_sha(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validated_source_acceptance_sha(
    report: Mapping[str, Any],
    *,
    expected_config_sha: str,
    expected_source_commit: str,
    expected_source_tree_sha: str,
) -> str:
    """Validate the source-binding report before a model bundle can claim it."""

    if not isinstance(report, Mapping):
        raise ValueError("source acceptance report must be a mapping")
    value = dict(report)
    if (
        value.get("schema") != "V10TrackAEngineeringSmokeV1"
        or value.get("status") != "PASS_ENGINEERING_SMOKE"
        or value.get("source_bound") is not True
        or value.get("config_sha") != expected_config_sha
        or value.get("authority") != TrackAAuthority().__dict__
    ):
        raise ValueError("mainline source acceptance report identity is invalid")
    checks = value.get("checks")
    gate_ledger = value.get("gate_ledger")
    runtime = value.get("runtime")
    platform_identity = value.get("platform")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(type(result) is not bool for result in checks.values())
        or not all(checks.values())
        or not isinstance(gate_ledger, dict)
        or gate_ledger.get("authority") != TrackAAuthority().__dict__
        or not isinstance(runtime, dict)
        or not isinstance(runtime.get("device"), str)
        or not isinstance(runtime.get("torch_version"), str)
        or type(runtime.get("cuda_available")) is not bool
        or not isinstance(runtime.get("torch_cuda_version"), str)
        or not isinstance(runtime.get("cuda_device_name"), str)
        or not isinstance(platform_identity, dict)
        or not isinstance(platform_identity.get("python_version"), str)
    ):
        raise ValueError("mainline source acceptance evidence blocks are incomplete")
    from ..training.server_acceptance import (
        SourceAcceptanceError,
        validate_empty_smoke_ledger,
    )

    try:
        validate_empty_smoke_ledger(gate_ledger)
    except SourceAcceptanceError as exc:
        raise ValueError("mainline source acceptance gate ledger is invalid") from exc
    source = value.get("source")
    if (
        not isinstance(source, dict)
        or source.get("revision") != expected_source_commit
        or source.get("tree_sha") != expected_source_tree_sha
        or source.get("worktree_clean") is not True
        or not isinstance(source.get("repository"), str)
        or not source.get("repository")
        or not isinstance(source.get("job_id"), str)
        or not source.get("job_id")
    ):
        raise ValueError("mainline source acceptance Git identity is invalid")
    acceptance = value.get("source_acceptance")
    required_checks = (
        "config_identity_verified",
        "authority_verified",
        "runtime_platform_verified",
        "trusted_repository_verified",
    )
    if (
        not isinstance(acceptance, dict)
        or acceptance.get("schema") != "V10TrackASourceAcceptanceV2"
        or any(acceptance.get(field_name) is not True for field_name in required_checks)
    ):
        raise ValueError("mainline source acceptance checks are incomplete")
    declared_sha = value.get("report_sha")
    validate_hex_digest(declared_sha, field_name="source acceptance report_sha")
    unhashed = dict(value)
    unhashed.pop("report_sha")
    if declared_sha != sha256_of(unhashed):
        raise ValueError("mainline source acceptance report SHA is stale")
    return declared_sha


@dataclass
class TrackAModelSuite:
    classifier: V10ClassifierNFull
    feedback: V10FeedbackNStudent | V10FeedbackNStudentCSM
    config: TrackAConfig
    feedback_input_dim: int
    gate_input_dim: int

    @classmethod
    def build(
        cls,
        config: TrackAConfig,
        *,
        feedback_input_dim: int = 6,
        gate_input_dim: int = FEEDBACK_GATE_DIM,
    ) -> "TrackAModelSuite":
        config.validate()
        if gate_input_dim != FEEDBACK_GATE_DIM:
            raise ValueError(
                "Track A requires the canonical 24-dimensional V2.4 feedback gate ABI"
            )
        classifier = V10ClassifierNFull(profile=config.model.classifier_profile)
        feedback = V10FeedbackNStudent(
            input_dim=feedback_input_dim,
            gate_input_dim=gate_input_dim,
            width=config.model.feedback_width,
            heads=config.model.feedback_heads,
            blocks=config.model.feedback_blocks,
        )
        return cls(classifier, feedback, config, feedback_input_dim, gate_input_dim)

    @classmethod
    def build_mainline(
        cls,
        config: TrackAConfig,
        *,
        feedback_input_dim: int = 6,
    ) -> "TrackAModelSuite":
        """Build the deployable-observation mainline with the formal V2.4 gate ABI."""

        suite = cls.build(
            config,
            feedback_input_dim=feedback_input_dim,
            gate_input_dim=FEEDBACK_GATE_DIM,
        )
        suite.feedback = V10FeedbackNStudentCSM(
            input_dim=feedback_input_dim,
            gate_input_dim=FEEDBACK_GATE_DIM,
            width=config.model.feedback_width,
            heads=config.model.feedback_heads,
            blocks=config.model.feedback_blocks,
        )
        return suite

    def parameter_counts(self) -> Dict[str, int]:
        return {
            "classifier": sum(
                parameter.numel() for parameter in self.classifier.parameters()
            ),
            "feedback": sum(
                parameter.numel() for parameter in self.feedback.parameters()
            ),
        }


def build_mainline_bundle(
    suite: TrackAModelSuite,
    *,
    source_commit: str,
    source_tree_sha: str,
    frozen: bool = False,
    source_acceptance_report: Mapping[str, Any] | None = None,
    calibration_admission_path: str | None = None,
    calibration_expected_bindings: Mapping[str, str] | None = None,
) -> MainlineBundleV1:
    if frozen and calibration_expected_bindings is None:
        require_calibrated_feedback_outputs()
    classifier_state_sha = state_dict_sha(suite.classifier.state_dict())
    if frozen:
        bindings = dict(calibration_expected_bindings or {})
        actual_bindings = {
            "source_commit": source_commit,
            "source_tree_sha": source_tree_sha,
            "config_sha": suite.config.sha256(),
            "model_state_sha": classifier_state_sha,
        }
        for name, value in actual_bindings.items():
            if name in bindings and bindings[name] != value:
                raise ValueError(
                    f"frozen calibration binding does not match loaded {name}"
                )
        bindings.update(actual_bindings)
        calibration_admission = require_calibrated_feedback_outputs(
            calibration_admission_path, bindings
        )
    else:
        calibration_admission = None
    source_acceptance_sha = ""
    if source_acceptance_report is not None:
        source_acceptance_sha = _validated_source_acceptance_sha(
            source_acceptance_report,
            expected_config_sha=suite.config.sha256(),
            expected_source_commit=source_commit,
            expected_source_tree_sha=source_tree_sha,
        )
    classifier_sha = classifier_state_sha if frozen else ""
    feedback_sha = state_dict_sha(suite.feedback.state_dict()) if frozen else ""
    result = MainlineBundleV1(
        source_commit=source_commit,
        source_tree_sha=source_tree_sha,
        config_sha=suite.config.sha256(),
        classifier_profile=suite.config.model.classifier_profile,
        feedback_profile=(
            f"F-{suite.config.model.feedback_width}-"
            f"h{suite.config.model.feedback_heads}-b{suite.config.model.feedback_blocks}"
        ),
        classifier_checkpoint_sha=classifier_sha,
        feedback_checkpoint_sha=feedback_sha,
        frozen=frozen,
        source_bound=source_acceptance_report is not None,
        source_acceptance_sha=source_acceptance_sha,
        calibration_admission_sha=(
            str(calibration_admission["accepted_receipt_sha"])
            if calibration_admission is not None
            else ""
        ),
        authority=TrackAAuthority(),
    )
    result.validate()
    return result
