"""Research-only Track A classifier-to-feedback mainline adapter.

The adapter wires the existing V10 classifier and feedback student without
changing either architecture.  It keeps neural predictions separate from
deterministic certificates, requires bound CSM and residual-instance evidence,
uses the canonical V2.4 24-dimensional feedback gate, and projects the model
action through the recoverable logical support before registered bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import numpy as np
import torch

from ml_recon.data_v2_4.feedback_gate_contract import (
    FEEDBACK_GATE_BASE_DIM,
    FEEDBACK_GATE_DIM,
    compose_feedback_gate_features,
)
from ml_recon.physics_hybrid_v10.certificate_engine import CertificateResult
from ml_recon.physics_hybrid_v10.control_manifold import get_control_manifold
from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.physics_hybrid_v10.published_output_registry import (
    require_calibrated_feedback_outputs,
)
from ml_recon.physics_hybrid_v10.track_a_n8.feedback_n_student_csm import V10FeedbackNStudentCSM
from ml_recon.physics_hybrid_v10.track_a_n8.csm_context_builder import (
    build_csm_model_context,
    validate_csm_model_context,
)
from ml_recon.physics_hybrid_v10.csm_geometry import CSMGeometryResult
from ml_recon.physics_hybrid_v10.csm_recoverability import (
    geometry_recoverability_from_whitened_svd,
)
from ml_recon.physics_hybrid_v10.forward_instance import (
    CSMForwardInstanceContract,
    CSMResidualInstanceContract,
    object_sha,
)
from ml_recon.physics_hybrid_v10.partition_decoder import (
    PartitionDecoderConfig,
    decode_partition_candidates,
)
from ml_recon.physics_hybrid_v10.partition_structure import PartitionStructureV1

from ..contracts import MainlineBundleV1, TrackAAuthority
from ..contracts.feedback_envelope import (
    MEMORY_POLICY_NOT_AVAILABLE,
    V10FeedbackDecisionEnvelopeV1,
)
from .action_bounds import bound_method, bound_recoverable_action
from .bundles import TrackAModelSuite, state_dict_sha


MAINLINE_SCHEMA = "V10TrackAMainlineForwardV3"
MAINLINE_SUPPORTED_N = (3, 4)
FEEDBACK_TOKEN_DIM = 2
FEEDBACK_INPUT_DIM = 6
CONTROL_AMPLITUDE_BOUND = 0.35
CONTROL_PHASE_BOUND = 0.70
DEPLOYABLE_FEEDBACK_ARMS = (
    "A0_RAW",
    "A1_DENOISE",
    "A2_CLASS",
    "A3_DENOISE_CLASS",
)
CLASSIFIER_INPUT_FIELDS = frozenset(
    {
        "basis_descriptors",
        "certificate_features",
        "certificate_mask",
        "correlation_adjacency",
        "counts",
        "full_outcome_mask",
        "global_outcome_histogram",
        "incidence",
        "measurement_probability",
        "mode_features",
        "mode_mask",
        "pair_stat_axes",
        "pair_stat_coverage_mask",
        "pair_stat_expectation",
        "pair_stat_qubits",
        "pair_stat_shots",
        "pair_stat_variance",
        "physical_adjacency",
        "setting_mask",
        "setting_uncertainty",
        "shots",
    }
)


def _is_sha256(value: str) -> bool:
    return bool(
        isinstance(value, str)
        and value == value.strip().lower()
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_sha(value: torch.Tensor | np.ndarray) -> str:
    array = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    return sha256_of(array.tolist())


@dataclass(frozen=True)
class TrackAMainlineAdapterConfig:
    """Stable adapter settings; model capacity and MoE topology are unchanged."""

    feedback_arm: str = "A3_DENOISE_CLASS"
    partition_candidate_count: int = 8
    partition_temperature: float = 1.0
    amplitude_bound: float = CONTROL_AMPLITUDE_BOUND
    phase_bound: float = CONTROL_PHASE_BOUND

    def validate(self) -> None:
        if self.feedback_arm not in DEPLOYABLE_FEEDBACK_ARMS:
            raise ValueError(
                "Track A mainline accepts only deployable A0-A3 feedback arms"
            )
        if not 1 <= self.partition_candidate_count <= 1024:
            raise ValueError("partition candidate count must be in [1,1024]")
        if (
            not np.isfinite(self.partition_temperature)
            or self.partition_temperature <= 0.0
        ):
            raise ValueError("partition temperature must be finite and positive")
        if (
            not np.isfinite(self.amplitude_bound)
            or self.amplitude_bound <= 0.0
            or not np.isfinite(self.phase_bound)
            or self.phase_bound <= 0.0
        ):
            raise ValueError("registered action bounds must be finite and positive")


@dataclass(frozen=True)
class TrackAMainlineBatch:
    """One canonical deployable-observation batch and its bound geometry sidecars."""

    n: int
    sample_ids: tuple[str, ...]
    classifier_batch: Mapping[str, torch.Tensor]
    noisy_measurement_tokens: torch.Tensor
    measurement_covariance: torch.Tensor
    target_mode_tokens: torch.Tensor
    physical_mode_tokens: torch.Tensor
    base_gate_features: torch.Tensor
    analytic_action: torch.Tensor
    csm_geometries: tuple[CSMGeometryResult, ...]
    csm_forward_instances: tuple[CSMForwardInstanceContract, ...]
    rsm_residuals: tuple[np.ndarray, ...]
    rsm_residual_instances: tuple[CSMResidualInstanceContract, ...]
    rsm_metric_contracts: tuple[object, ...]
    deterministic_certificates: tuple[tuple[CertificateResult, ...], ...]
    measurement_contract_sha: str
    history_tokens: torch.Tensor | None = None
    history_mask: torch.Tensor | None = None
    physical_budget_summary: torch.Tensor | None = None


@dataclass(frozen=True)
class TrackARSMContext:
    """Target-conditioned RSM values computed on the bound CSM support."""

    sample_id: str
    forward_instance_sha: str
    residual_instance_sha: str
    metric_contract_sha: str
    csm_context_sha: str
    recoverable_basis: np.ndarray
    supported_singular_values: np.ndarray
    support_rank: int
    rank_fraction: float
    kappa_geometry: float
    epsilon_irrecoverable: float
    reason_code: str

    def validate(self, logical_dim: int) -> None:
        basis = np.asarray(self.recoverable_basis, dtype=np.float64)
        singular = np.asarray(self.supported_singular_values, dtype=np.float64).reshape(
            -1
        )
        if not self.sample_id or any(
            not _is_sha256(value)
            for value in (
                self.forward_instance_sha,
                self.residual_instance_sha,
                self.metric_contract_sha,
                self.csm_context_sha,
            )
        ):
            raise ValueError("RSM identity is incomplete")
        if (
            basis.shape != (logical_dim, self.support_rank)
            or singular.shape != (self.support_rank,)
            or not np.all(np.isfinite(basis))
            or not np.all(np.isfinite(singular))
        ):
            raise ValueError("RSM recoverable support has an invalid shape or value")
        if self.support_rank and not np.allclose(
            basis.T @ basis, np.eye(self.support_rank), atol=1e-8
        ):
            raise ValueError("RSM recoverable basis must be orthonormal")
        if not np.isclose(
            self.rank_fraction, self.support_rank / logical_dim, atol=1e-12
        ):
            raise ValueError("RSM rank fraction does not match its basis")
        if not 0.0 <= self.kappa_geometry <= 1.0:
            raise ValueError("RSM kappa must lie in [0,1]")
        if not 0.0 <= self.epsilon_irrecoverable <= 1.0:
            raise ValueError("RSM irrecoverable fraction must lie in [0,1]")
        if self.reason_code not in {
            "RECOVERABLE_SUPPORT_PRESENT",
            "NO_RECOVERABLE_SUPPORT",
        }:
            raise ValueError("unknown RSM reason code")


@dataclass(frozen=True)
class TrackAActionProjection:
    """Audit record for project-then-bound research action construction."""

    sample_id: str
    raw_logical_action: np.ndarray
    recoverable_logical_action: np.ndarray
    bounded_logical_action: np.ndarray
    bounded_mode_action: np.ndarray
    projection_order: tuple[str, str] = (
        "recoverable_basis",
        "registered_bounds",
    )
    analytic_action_application_count: int = 1
    projection_method: str = bound_method
    projection_sha: str = ""

    def __post_init__(self) -> None:
        if not self.projection_sha:
            object.__setattr__(self, "projection_sha", self.sha256())

    def sha256(self) -> str:
        return sha256_of(
            {
                "sample_id": self.sample_id,
                "raw_logical_action_sha": _array_sha(self.raw_logical_action),
                "recoverable_logical_action_sha": _array_sha(
                    self.recoverable_logical_action
                ),
                "bounded_logical_action_sha": _array_sha(self.bounded_logical_action),
                "bounded_mode_action_sha": _array_sha(self.bounded_mode_action),
                "projection_order": self.projection_order,
                "projection_method": self.projection_method,
                "analytic_action_application_count": self.analytic_action_application_count,
            }
        )

    def validate(self, n: int, *, amplitude_bound: float, phase_bound: float) -> None:
        logical_dim = 2 * n - 2
        raw = np.asarray(self.raw_logical_action, dtype=np.float64).reshape(-1)
        recoverable = np.asarray(
            self.recoverable_logical_action, dtype=np.float64
        ).reshape(-1)
        bounded = np.asarray(self.bounded_logical_action, dtype=np.float64).reshape(-1)
        mode = np.asarray(self.bounded_mode_action, dtype=np.float64)
        if (
            not self.sample_id
            or raw.shape != (logical_dim,)
            or recoverable.shape != (logical_dim,)
            or bounded.shape != (logical_dim,)
            or mode.shape != (n, 2)
        ):
            raise ValueError("action projection shape mismatch")
        if any(
            not np.all(np.isfinite(value))
            for value in (raw, recoverable, bounded, mode)
        ):
            raise ValueError("action projection contains non-finite values")
        if self.projection_sha != self.sha256():
            raise ValueError("action projection digest changed after construction")
        if np.any(np.abs(bounded[0::2]) > amplitude_bound + 1e-10) or np.any(
            np.abs(bounded[1::2]) > phase_bound + 1e-10
        ):
            raise ValueError("bounded logical action exceeds the registered limits")
        if self.projection_order != ("recoverable_basis", "registered_bounds"):
            raise ValueError("action must be projected before bounds")
        if self.analytic_action_application_count != 1:
            raise ValueError("analytic action must be applied exactly once")
        if self.projection_method != bound_method:
            raise ValueError("action requires the versioned support-preserving bound")
        expected_bounded = bound_recoverable_action(
            torch.from_numpy(recoverable),
            amplitude_bound=amplitude_bound,
            phase_bound=phase_bound,
        ).numpy()
        expected_mode = (
            get_control_manifold(N=n).logical_to_mode_lift() @ bounded
        ).reshape(n, 2)
        if not np.allclose(
            bounded, expected_bounded, atol=1e-7, rtol=1e-6
        ) or not np.allclose(mode, expected_mode, atol=1e-7, rtol=1e-6):
            raise ValueError(
                "action projection record does not follow the registered bounds/lift"
            )


@dataclass(frozen=True)
class TrackAMainlineRowProvenance:
    sample_id: str
    forward_instance_sha: str
    residual_instance_sha: str
    csm_context_sha: str
    partition_sha: str
    analytic_action_sha: str
    bounded_action_sha: str
    action_projection_sha: str

    def validate(self) -> None:
        if not self.sample_id or any(
            not _is_sha256(value)
            for value in (
                self.forward_instance_sha,
                self.residual_instance_sha,
                self.csm_context_sha,
                self.partition_sha,
                self.analytic_action_sha,
                self.bounded_action_sha,
                self.action_projection_sha,
            )
        ):
            raise ValueError("mainline row provenance is incomplete")


@dataclass(frozen=True)
class TrackAMainlineProvenance:
    source_commit: str
    source_tree_sha: str
    mainline_bundle_sha: str
    classifier_state_sha: str
    feedback_state_sha: str
    measurement_contract_sha: str
    row_bindings: tuple[TrackAMainlineRowProvenance, ...]
    source_acceptance_sha: str = ""
    schema: str = MAINLINE_SCHEMA
    source_bound: bool = False
    authority: TrackAAuthority = field(default_factory=TrackAAuthority)

    def validate(self) -> None:
        self.authority.validate()
        if self.schema != MAINLINE_SCHEMA or type(self.source_bound) is not bool:
            raise ValueError("mainline provenance schema/source-bound flag is invalid")
        for value in (self.source_commit, self.source_tree_sha):
            if (
                not isinstance(value, str)
                or value != value.strip().lower()
                or len(value) not in (40, 64)
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("source commit/tree identity is invalid")
        if any(
            not _is_sha256(value)
            for value in (
                self.mainline_bundle_sha,
                self.classifier_state_sha,
                self.feedback_state_sha,
                self.measurement_contract_sha,
            )
        ):
            raise ValueError("mainline provenance SHA-256 identity is invalid")
        if (self.source_bound and not _is_sha256(self.source_acceptance_sha)) or (
            not self.source_bound and self.source_acceptance_sha != ""
        ):
            raise ValueError(
                "source-bound mainline provenance requires exactly one acceptance SHA"
            )
        if not self.row_bindings:
            raise ValueError("mainline provenance requires row bindings")
        for row in self.row_bindings:
            row.validate()


@dataclass(frozen=True)
class TrackAMainlineOutput:
    classifier_prediction: Mapping[str, torch.Tensor | bool]
    deterministic_certificates: tuple[tuple[CertificateResult, ...], ...]
    legal_partitions: tuple[PartitionStructureV1, ...]
    csm_contexts: tuple[Mapping[str, Any], ...]
    rsm_contexts: tuple[TrackARSMContext, ...]
    feedback_inputs: Mapping[str, torch.Tensor]
    feedback_prediction: Mapping[str, torch.Tensor]
    bounded_research_action: torch.Tensor
    action_projections: tuple[TrackAActionProjection, ...]
    provenance: TrackAMainlineProvenance

    def validate(self, n: int, *, amplitude_bound: float, phase_bound: float) -> None:
        self.provenance.validate()
        count = int(self.bounded_research_action.shape[0])
        if tuple(self.bounded_research_action.shape) != (count, n, 2) or any(
            len(values) != count
            for values in (
                self.action_projections,
                self.provenance.row_bindings,
                self.csm_contexts,
                self.rsm_contexts,
            )
        ):
            raise ValueError("mainline output row coverage is inconsistent")
        for index, (projection, row) in enumerate(
            zip(self.action_projections, self.provenance.row_bindings)
        ):
            projection.validate(
                n, amplitude_bound=amplitude_bound, phase_bound=phase_bound
            )
            validate_csm_model_context(dict(self.csm_contexts[index]))
            if (
                projection.sample_id != row.sample_id
                or projection.projection_sha != row.action_projection_sha
                or _array_sha(projection.bounded_mode_action) != row.bounded_action_sha
                or _array_sha(self.bounded_research_action[index])
                != row.bounded_action_sha
            ):
                raise ValueError(
                    "mainline action projection is not bound to its actual output"
                )
            basis = np.asarray(self.rsm_contexts[index].recoverable_basis)
            raw = np.asarray(projection.raw_logical_action)
            expected = basis @ (basis.T @ raw) if basis.shape[1] else np.zeros_like(raw)
            if not np.allclose(
                projection.recoverable_logical_action, expected, atol=1e-7, rtol=1e-6
            ):
                raise ValueError(
                    "mainline action projection uses a different recoverable support"
                )


class TrackAMainlineForward:
    """Cohesive small-N Track A forward with fail-closed adapter boundaries."""

    def __init__(
        self,
        suite: TrackAModelSuite,
        bundle: MainlineBundleV1,
        *,
        adapter_config: TrackAMainlineAdapterConfig | None = None,
        calibration_admission_path: str | None = None,
        calibration_expected_bindings: Mapping[str, str] | None = None,
    ) -> None:
        bundle.validate()
        if bundle.frozen:
            if calibration_expected_bindings is None:
                require_calibrated_feedback_outputs()
        suite.config.validate()
        if suite.feedback_input_dim != FEEDBACK_INPUT_DIM:
            raise ValueError(
                "Track A mainline requires the canonical six-channel feedback input"
            )
        if suite.gate_input_dim != FEEDBACK_GATE_DIM:
            raise ValueError(
                "Track A mainline requires the formal 24-dimensional gate ABI"
            )
        if bundle.config_sha != suite.config.sha256():
            raise ValueError(
                "mainline bundle and model suite configuration identities disagree"
            )
        active_config = adapter_config or TrackAMainlineAdapterConfig(
            partition_candidate_count=suite.config.model.partition_candidate_count,
        )
        active_config.validate()
        self.suite = suite
        self.bundle = bundle
        self.adapter_config = active_config
        self.classifier_state_sha = state_dict_sha(suite.classifier.state_dict())
        self.feedback_state_sha = state_dict_sha(suite.feedback.state_dict())
        if bundle.frozen and (
            bundle.classifier_checkpoint_sha != self.classifier_state_sha
            or bundle.feedback_checkpoint_sha != self.feedback_state_sha
        ):
            raise ValueError(
                "frozen mainline bundle does not match the loaded model states"
            )
        self.calibration_admission_path = calibration_admission_path
        self.calibration_expected_bindings = (
            dict(calibration_expected_bindings)
            if calibration_expected_bindings is not None
            else None
        )
        if bundle.frozen:
            self._require_calibration_admission(self.classifier_state_sha)

    def _require_calibration_admission(self, classifier_state_sha: str) -> None:
        if self.calibration_expected_bindings is None:
            require_calibrated_feedback_outputs()
        bindings = dict(self.calibration_expected_bindings or {})
        actual_bindings = {
            "source_commit": self.bundle.source_commit,
            "source_tree_sha": self.bundle.source_tree_sha,
            "config_sha": self.bundle.config_sha,
            "model_state_sha": classifier_state_sha,
        }
        for name, value in actual_bindings.items():
            if name in bindings and bindings[name] != value:
                raise ValueError(
                    f"frozen calibration binding does not match loaded {name}"
                )
        bindings.update(actual_bindings)
        admission = require_calibrated_feedback_outputs(
            self.calibration_admission_path, bindings
        )
        if (
            not isinstance(admission, Mapping)
            or admission.get("accepted_receipt_sha")
            != self.bundle.calibration_admission_sha
        ):
            raise ValueError("frozen mainline calibration admission receipt mismatch")
        self.calibration_admission = dict(admission)

    @staticmethod
    def validate_deployable_policy_envelope(
        decision_envelope: V10FeedbackDecisionEnvelopeV1,
    ) -> None:
        """Validate the only deployable policy input boundary.

        Memory-aware policy routing has no registered model input or fusion map.
        A supplied extension therefore remains an explicit ``NOT_AVAILABLE``
        outcome; it cannot be ignored by the legacy feedback student.
        """
        if type(decision_envelope) is not V10FeedbackDecisionEnvelopeV1:
            raise ValueError("deployable policy requires V10FeedbackDecisionEnvelopeV1")
        decision_envelope.validate()
        if decision_envelope.memory_extension is not None:
            if decision_envelope.memory_policy_status != MEMORY_POLICY_NOT_AVAILABLE:
                raise ValueError("memory-aware policy status is not registered")
            raise ValueError("memory-aware deployable policy is NOT_AVAILABLE")

    def forward_deployable_policy(
        self,
        batch: TrackAMainlineBatch,
        decision_envelope: V10FeedbackDecisionEnvelopeV1,
    ) -> TrackAMainlineOutput:
        """Fail closed until a registered core-to-batch deployable adapter exists.

        ``forward`` is a research entry point over ``TrackAMainlineBatch``.  It
        does not consume a V2 decision core, so validating an unrelated envelope
        before forwarding that batch would not provide input protection.
        """
        self.validate_deployable_policy_envelope(decision_envelope)
        del batch
        raise ValueError(
            "deployable policy adapter is NOT_AVAILABLE; research forward cannot claim deployment"
        )

    @staticmethod
    def _require_tensor_shape(
        value: torch.Tensor,
        expected: tuple[int, ...],
        name: str,
    ) -> None:
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if value.is_floating_point() and not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{name} must be finite")

    def _validate_batch(self, batch: TrackAMainlineBatch) -> int:
        if batch.n not in MAINLINE_SUPPORTED_N:
            raise ValueError(
                "N>4 requires a registered large-N observation/semantic adapter; "
                "the small-N mainline will not fabricate one"
            )
        batch_size = len(batch.sample_ids)
        if batch_size <= 0 or len(set(batch.sample_ids)) != batch_size:
            raise ValueError("mainline sample_ids must be non-empty and unique")
        if not _is_sha256(batch.measurement_contract_sha):
            raise ValueError("measurement_contract_sha must be SHA-256")
        aligned = (
            batch.csm_geometries,
            batch.csm_forward_instances,
            batch.rsm_residuals,
            batch.rsm_residual_instances,
            batch.rsm_metric_contracts,
            batch.deterministic_certificates,
        )
        if any(len(value) != batch_size for value in aligned):
            raise ValueError(
                "every CSM/RSM/certificate sidecar must align with sample_ids"
            )

        unexpected = set(batch.classifier_batch) - CLASSIFIER_INPUT_FIELDS
        if unexpected:
            raise ValueError(
                f"classifier observation contains unregistered fields: {sorted(unexpected)}"
            )
        required_classifier = {
            "counts",
            "basis_descriptors",
            "shots",
            "setting_uncertainty",
            "setting_mask",
            "mode_features",
            "mode_mask",
            "physical_adjacency",
            "correlation_adjacency",
            "incidence",
            "certificate_features",
            "certificate_mask",
        }
        missing = required_classifier - set(batch.classifier_batch)
        if missing:
            raise ValueError(
                f"classifier observation is missing fields: {sorted(missing)}"
            )
        for name, value in batch.classifier_batch.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim < 1
                or value.shape[0] != batch_size
            ):
                raise ValueError(
                    f"classifier field {name} does not preserve batch rows"
                )
        mode_mask = batch.classifier_batch["mode_mask"]
        self._require_tensor_shape(mode_mask, (batch_size, batch.n), "mode_mask")
        if not bool(torch.all(mode_mask.bool())):
            raise ValueError("small-N mainline requires an unpadded exact-N mode panel")

        mode_shape = (batch_size, batch.n, FEEDBACK_TOKEN_DIM)
        self._require_tensor_shape(
            batch.noisy_measurement_tokens, mode_shape, "noisy_measurement_tokens"
        )
        self._require_tensor_shape(
            batch.measurement_covariance, mode_shape, "measurement_covariance"
        )
        if bool(torch.any(batch.measurement_covariance < 0.0)):
            raise ValueError("measurement_covariance cannot be negative")
        self._require_tensor_shape(
            batch.target_mode_tokens, mode_shape, "target_mode_tokens"
        )
        self._require_tensor_shape(
            batch.physical_mode_tokens, mode_shape, "physical_mode_tokens"
        )
        self._require_tensor_shape(batch.analytic_action, mode_shape, "analytic_action")
        self._require_tensor_shape(
            batch.base_gate_features,
            (batch_size, FEEDBACK_GATE_BASE_DIM),
            "base_gate_features",
        )
        if (batch.history_tokens is None) != (batch.history_mask is None):
            raise ValueError(
                "history_tokens and history_mask must be provided together"
            )
        if batch.history_tokens is not None and batch.history_mask is not None:
            if (
                batch.history_tokens.ndim != 4
                or batch.history_tokens.shape[0] != batch_size
                or batch.history_tokens.shape[2:] != (batch.n, 7)
                or batch.history_mask.shape != batch.history_tokens.shape[:2]
            ):
                raise ValueError("history must have shape [batch,rounds,N,7]")
        if batch.physical_budget_summary is not None:
            self._require_tensor_shape(
                batch.physical_budget_summary,
                (batch_size, 2),
                "physical_budget_summary",
            )
        return batch_size

    @staticmethod
    def _semantic_feedback_values(
        classifier_prediction: Mapping[str, torch.Tensor | bool],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = {
            "W_populations",
            "W_coherence_re",
            "W_coherence_im",
            "W_amplitude_variance",
            "W_phase_variance",
        }
        missing = required - set(classifier_prediction)
        if missing:
            raise ValueError(
                f"classifier semantic adapter is missing outputs: {sorted(missing)}"
            )
        population = classifier_prediction["W_populations"]
        coherence_re = classifier_prediction["W_coherence_re"][:, :, 0]
        coherence_im = classifier_prediction["W_coherence_im"][:, :, 0]
        relative_phase = torch.atan2(coherence_im, coherence_re) / torch.pi
        denoised = torch.stack((population, relative_phase), dim=-1)
        covariance = torch.stack(
            (
                classifier_prediction["W_amplitude_variance"],
                classifier_prediction["W_phase_variance"] / (torch.pi**2),
            ),
            dim=-1,
        )
        coherence = torch.sqrt(coherence_re.square() + coherence_im.square())
        confidence = torch.exp(-covariance.mean(-1)).clamp(0.0, 1.0)
        esm = torch.stack((coherence, confidence), dim=-1)
        return denoised, covariance, esm

    def _decode_partitions(
        self,
        batch: TrackAMainlineBatch,
        classifier_prediction: Mapping[str, torch.Tensor | bool],
    ) -> tuple[PartitionStructureV1, ...]:
        probabilities = classifier_prediction["partition_link_probs"]
        pair_evidence = classifier_prediction["pair_entanglement_probs"]
        hyperedge_evidence = classifier_prediction["hyperedge_entanglement_probs"]
        abstention = classifier_prediction["abstention_probability"]
        decoder_config = PartitionDecoderConfig(
            candidate_count=self.adapter_config.partition_candidate_count,
            posterior_temperature=self.adapter_config.partition_temperature,
        )
        source_checkpoint_sha = (
            self.bundle.classifier_checkpoint_sha if self.bundle.frozen else ""
        )
        structures = []
        for index in range(len(batch.sample_ids)):
            pair_probability = probabilities[index].detach().cpu().numpy()
            pair_sha = sha256_of(
                {
                    "partition_link_probability": pair_probability.tolist(),
                    "pair_entanglement_probability": (
                        pair_evidence[index].detach().cpu().numpy().tolist()
                    ),
                }
            )
            hyperedge_sha = _array_sha(hyperedge_evidence[index])
            structure = decode_partition_candidates(
                pair_probability,
                input_kind="probabilities",
                config=decoder_config,
                certificates=batch.deterministic_certificates[index],
                abstention_probability=float(abstention[index].detach().cpu()),
                pair_evidence_ref=pair_sha,
                hyperedge_evidence_ref=hyperedge_sha,
                source_checkpoint_sha=source_checkpoint_sha,
                measurement_contract_sha=batch.measurement_contract_sha,
                candidate_set_complete=False,
            )
            structures.append(structure)
        return tuple(structures)

    def _build_csm_rsm_contexts(
        self,
        batch: TrackAMainlineBatch,
    ) -> tuple[tuple[Mapping[str, Any], ...], tuple[TrackARSMContext, ...]]:
        csm_contexts = []
        rsm_contexts = []
        for index, sample_id in enumerate(batch.sample_ids):
            geometry = batch.csm_geometries[index]
            forward_instance = batch.csm_forward_instances[index]
            residual_instance = batch.rsm_residual_instances[index]
            residual = np.asarray(batch.rsm_residuals[index], dtype=np.float64).reshape(
                -1
            )
            target = batch.target_mode_tokens[index].detach().cpu().numpy()
            metric_contract = batch.rsm_metric_contracts[index]
            geometry.validate_bound()
            forward_instance.validate()
            residual_instance.validate()
            if geometry.N != batch.n or forward_instance.sample_id != sample_id:
                raise ValueError("CSM geometry/sample identity mismatch")
            if (
                geometry.forward_instance_sha != forward_instance.sha256()
                or geometry.contract.forward_instance_sha
                != geometry.forward_instance_sha
                or geometry.covariance_sha != forward_instance.covariance_sha
            ):
                raise ValueError(
                    "CSM geometry is not bound to the supplied forward instance"
                )
            if (
                residual_instance.geometry_forward_instance_sha
                != geometry.forward_instance_sha
                or residual_instance.target_sha != object_sha(target)
                or residual_instance.residual_sha != object_sha(residual)
                or residual_instance.metric_contract_sha != object_sha(metric_contract)
            ):
                raise ValueError(
                    "RSM residual instance is not bound to geometry/target/residual/metric"
                )

            whitened_logical = geometry.whiten_and_profile(geometry.logical_jacobian)
            whitened_residual = geometry.whiten_and_profile(residual)
            _, singular, vh = np.linalg.svd(whitened_logical, full_matrices=False)
            threshold = geometry.contract.singular_support.effective_threshold(
                float(singular.max(initial=0.0))
            )
            support_rank = int(np.sum(singular > threshold))
            logical_dim = geometry.contract.logical_dim
            basis = (
                vh[:support_rank].T
                if support_rank
                else np.zeros((logical_dim, 0), dtype=np.float64)
            )
            metrics = geometry_recoverability_from_whitened_svd(
                whitened_residual,
                whitened_logical,
                singular_support=geometry.contract.singular_support,
            )
            csm_context = build_csm_model_context(
                geometry,
                batch.n,
                require_bound=True,
                source_fields={
                    "sample_id": sample_id,
                    "forward_instance_sha": geometry.forward_instance_sha,
                    "covariance_sha": geometry.covariance_sha,
                    "csm_contract_sha": geometry.contract.sha256(require_bound=True),
                },
            )
            rsm_context = TrackARSMContext(
                sample_id=sample_id,
                forward_instance_sha=geometry.forward_instance_sha,
                residual_instance_sha=residual_instance.sha256(),
                metric_contract_sha=residual_instance.metric_contract_sha,
                csm_context_sha=str(csm_context["context_sha"]),
                recoverable_basis=basis,
                supported_singular_values=singular[:support_rank],
                support_rank=support_rank,
                rank_fraction=support_rank / logical_dim,
                kappa_geometry=float(metrics["kappa_tangent"]),
                epsilon_irrecoverable=float(metrics["epsilon_irr_geom"]),
                reason_code=(
                    "RECOVERABLE_SUPPORT_PRESENT"
                    if support_rank
                    else "NO_RECOVERABLE_SUPPORT"
                ),
            )
            rsm_context.validate(logical_dim)
            csm_contexts.append(csm_context)
            rsm_contexts.append(rsm_context)
        return tuple(csm_contexts), tuple(rsm_contexts)

    def _feedback_inputs(
        self,
        batch: TrackAMainlineBatch,
        classifier_prediction: Mapping[str, torch.Tensor | bool],
        csm_contexts: tuple[Mapping[str, Any], ...],
        rsm_contexts: tuple[TrackARSMContext, ...],
    ) -> Dict[str, torch.Tensor]:
        denoised, classifier_covariance, esm = self._semantic_feedback_values(
            classifier_prediction
        )
        arm = self.adapter_config.feedback_arm
        active_denoised = (
            denoised
            if arm in {"A1_DENOISE", "A3_DENOISE_CLASS"}
            else batch.noisy_measurement_tokens
        )
        active_covariance = (
            classifier_covariance
            if arm in {"A1_DENOISE", "A3_DENOISE_CLASS"}
            else batch.measurement_covariance
        )
        active_esm = (
            esm if arm in {"A2_CLASS", "A3_DENOISE_CLASS"} else torch.zeros_like(esm)
        )
        gate_values = np.stack(
            [
                compose_feedback_gate_features(
                    row.detach().cpu().numpy(),
                    arm=arm,
                )
                for row in batch.base_gate_features
            ]
        )
        gate_features = torch.as_tensor(
            gate_values,
            dtype=batch.base_gate_features.dtype,
            device=batch.base_gate_features.device,
        )
        result = {
            "noisy_measurement_tokens": batch.noisy_measurement_tokens,
            "denoised_measurement_tokens": active_denoised,
            "covariance": active_covariance,
            "target_mode_tokens": batch.target_mode_tokens,
            "esm_mode_tokens": active_esm,
            "physical_mode_tokens": batch.physical_mode_tokens,
            "gate_features": gate_features,
            "analytic_action": batch.analytic_action,
            "mode_mask": batch.classifier_batch["mode_mask"].bool(),
        }
        if batch.history_tokens is not None and batch.history_mask is not None:
            result["history_tokens"] = batch.history_tokens
            result["history_mask"] = batch.history_mask
        if isinstance(self.suite.feedback, V10FeedbackNStudentCSM):
            result["csm_node_context"] = torch.as_tensor(
                np.stack([context["csm_node_context"] for context in csm_contexts]),
                dtype=batch.noisy_measurement_tokens.dtype,
                device=batch.noisy_measurement_tokens.device,
            )
            result["csm_edge_context"] = torch.as_tensor(
                np.stack([context["csm_edge_context"] for context in csm_contexts]),
                dtype=batch.noisy_measurement_tokens.dtype,
                device=batch.noisy_measurement_tokens.device,
            )
            result["csm_global_context"] = torch.as_tensor(
                np.stack([context["csm_global_context"] for context in csm_contexts]),
                dtype=batch.noisy_measurement_tokens.dtype,
                device=batch.noisy_measurement_tokens.device,
            )
            result["structure_confidence"] = torch.as_tensor(
                [rsm.rank_fraction for rsm in rsm_contexts],
                dtype=batch.noisy_measurement_tokens.dtype,
                device=batch.noisy_measurement_tokens.device,
            )
            csm_global = result["csm_global_context"]
            esm_summary = active_esm.mean(1)
            rsm_summary = torch.as_tensor(
                [
                    [rsm.kappa_geometry, rsm.epsilon_irrecoverable]
                    for rsm in rsm_contexts
                ],
                dtype=batch.noisy_measurement_tokens.dtype,
                device=batch.noisy_measurement_tokens.device,
            )
            budget_summary = (
                batch.physical_budget_summary
                if batch.physical_budget_summary is not None
                else torch.zeros(
                    (len(rsm_contexts), 2),
                    dtype=esm_summary.dtype,
                    device=esm_summary.device,
                )
            )
            result["semantic_routing_context"] = torch.cat(
                (esm_summary, csm_global[:, :2], rsm_summary, budget_summary), dim=-1
            )
        return result

    def _project_and_bound_action(
        self,
        batch: TrackAMainlineBatch,
        feedback_prediction: Mapping[str, torch.Tensor],
        rsm_contexts: tuple[TrackARSMContext, ...],
    ) -> tuple[torch.Tensor, tuple[TrackAActionProjection, ...]]:
        raw_mode_action = torch.stack(
            (
                feedback_prediction["delta_log_amp"],
                feedback_prediction["delta_phase"],
            ),
            dim=-1,
        )
        bounded_actions = []
        records = []
        for index, rsm_context in enumerate(rsm_contexts):
            geometry = batch.csm_geometries[index]
            manifold = get_control_manifold(
                geometry.contract.control_manifold_id,
                batch.n,
            )
            dtype = (
                torch.float64
                if raw_mode_action.dtype == torch.float64
                else torch.float32
            )
            device = raw_mode_action.device
            mode_to_logical = torch.as_tensor(
                manifold.mode_to_logical_lift(), dtype=dtype, device=device
            )
            logical_to_mode = torch.as_tensor(
                manifold.logical_to_mode_lift(), dtype=dtype, device=device
            )
            raw_logical = mode_to_logical @ raw_mode_action[index].reshape(-1).to(
                dtype=dtype
            )
            basis = torch.as_tensor(
                rsm_context.recoverable_basis, dtype=dtype, device=device
            )
            recoverable = (
                basis @ (basis.transpose(0, 1) @ raw_logical)
                if basis.shape[1]
                else torch.zeros_like(raw_logical)
            )
            bounded = bound_recoverable_action(
                recoverable,
                amplitude_bound=self.adapter_config.amplitude_bound,
                phase_bound=self.adapter_config.phase_bound,
            )
            bounded_mode = (logical_to_mode @ bounded).reshape(batch.n, 2)
            record = TrackAActionProjection(
                sample_id=batch.sample_ids[index],
                raw_logical_action=raw_logical.detach().cpu().numpy(),
                recoverable_logical_action=recoverable.detach().cpu().numpy(),
                bounded_logical_action=bounded.detach().cpu().numpy(),
                bounded_mode_action=bounded_mode.detach().cpu().numpy(),
            )
            record.validate(
                batch.n,
                amplitude_bound=self.adapter_config.amplitude_bound,
                phase_bound=self.adapter_config.phase_bound,
            )
            bounded_actions.append(bounded_mode)
            records.append(record)
        return torch.stack(bounded_actions), tuple(records)

    def forward(self, batch: TrackAMainlineBatch) -> TrackAMainlineOutput:
        """Execute the research mainline with current-state provenance and no authority escalation."""

        self._validate_batch(batch)
        classifier_state_sha = state_dict_sha(self.suite.classifier.state_dict())
        feedback_state_sha = state_dict_sha(self.suite.feedback.state_dict())
        if self.bundle.frozen and (
            classifier_state_sha != self.bundle.classifier_checkpoint_sha
            or feedback_state_sha != self.bundle.feedback_checkpoint_sha
        ):
            raise ValueError(
                "frozen mainline model state changed after bundle validation"
            )
        if self.bundle.frozen:
            self._require_calibration_admission(classifier_state_sha)
        classifier_prediction = self.suite.classifier(
            dict(batch.classifier_batch),
            N=batch.n,
            fine_identifiable=False,
        )
        legal_partitions = self._decode_partitions(batch, classifier_prediction)
        csm_contexts, rsm_contexts = self._build_csm_rsm_contexts(batch)
        feedback_inputs = self._feedback_inputs(
            batch, classifier_prediction, csm_contexts, rsm_contexts
        )
        if tuple(feedback_inputs["gate_features"].shape) != (
            len(batch.sample_ids),
            FEEDBACK_GATE_DIM,
        ):
            raise ValueError(
                "canonical feedback gate composition did not produce 24 features"
            )
        feedback_prediction = self.suite.feedback(feedback_inputs)
        bounded_action, action_projections = self._project_and_bound_action(
            batch, feedback_prediction, rsm_contexts
        )
        row_bindings = tuple(
            TrackAMainlineRowProvenance(
                sample_id=sample_id,
                forward_instance_sha=rsm_contexts[index].forward_instance_sha,
                residual_instance_sha=rsm_contexts[index].residual_instance_sha,
                csm_context_sha=rsm_contexts[index].csm_context_sha,
                partition_sha=legal_partitions[index].sha256(),
                analytic_action_sha=_array_sha(batch.analytic_action[index]),
                bounded_action_sha=_array_sha(bounded_action[index]),
                action_projection_sha=action_projections[index].projection_sha,
            )
            for index, sample_id in enumerate(batch.sample_ids)
        )
        provenance = TrackAMainlineProvenance(
            source_commit=self.bundle.source_commit,
            source_tree_sha=self.bundle.source_tree_sha,
            mainline_bundle_sha=self.bundle.sha256(),
            classifier_state_sha=classifier_state_sha,
            feedback_state_sha=feedback_state_sha,
            measurement_contract_sha=batch.measurement_contract_sha,
            row_bindings=row_bindings,
            source_acceptance_sha=self.bundle.source_acceptance_sha,
            source_bound=self.bundle.source_bound,
        )
        provenance.validate()
        output = TrackAMainlineOutput(
            classifier_prediction=classifier_prediction,
            deterministic_certificates=batch.deterministic_certificates,
            legal_partitions=legal_partitions,
            csm_contexts=csm_contexts,
            rsm_contexts=rsm_contexts,
            feedback_inputs=feedback_inputs,
            feedback_prediction=feedback_prediction,
            bounded_research_action=bounded_action,
            action_projections=action_projections,
            provenance=provenance,
        )
        output.validate(
            batch.n,
            amplitude_bound=self.adapter_config.amplitude_bound,
            phase_bound=self.adapter_config.phase_bound,
        )
        return output


__all__ = [
    "CONTROL_AMPLITUDE_BOUND",
    "CONTROL_PHASE_BOUND",
    "DEPLOYABLE_FEEDBACK_ARMS",
    "MAINLINE_SCHEMA",
    "MAINLINE_SUPPORTED_N",
    "TrackAActionProjection",
    "TrackAMainlineAdapterConfig",
    "TrackAMainlineBatch",
    "TrackAMainlineForward",
    "TrackAMainlineOutput",
    "TrackAMainlineProvenance",
    "TrackAMainlineRowProvenance",
    "TrackARSMContext",
]
