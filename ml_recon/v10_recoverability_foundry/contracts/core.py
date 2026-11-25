"""Fail-closed data, model-bundle, and evidence contracts for Track A."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable

import numpy as np

from ml_recon.physics_hybrid_v10.contracts import sha256_of


FORBIDDEN_OBSERVATION_FIELDS = frozenset(
    {
        "clean_measurements",
        "clean_measurement_tokens",
        "clean_state",
        "clean_w_coefficients",
        "collision_arm_state",
        "current_coefficients_im",
        "current_coefficients_re",
        "epsilon_irr",
        "fixed_k_state",
        "mechanism_truth",
        "measurement_probabilities",
        "oracle_action",
        "oracle_action_mask",
        "partition_truth",
        "selected_functionals",
        "state_truth",
        "target_certificate_status",
        "target_gain",
        "target_harm",
        "target_ood",
        "target_recoverability",
        "target_recoverable_class",
        "target_stop",
        "true_esm_mode_tokens",
        "true_kappa",
        "true_mechanism",
        "true_noise_algorithm",
        "true_partition",
        "true_severity",
    }
)


def validate_hex_digest(
    value: str,
    *,
    field_name: str,
    allowed_lengths: Iterable[int] = (64,),
    allow_empty: bool = False,
) -> None:
    """Validate a project-controlled hexadecimal identity without accepting placeholders."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string identity")
    text = value
    lengths = tuple(int(length) for length in allowed_lengths)
    if allow_empty and not text:
        return
    if (
        text != text.strip().lower()
        or len(text) not in lengths
        or any(character not in "0123456789abcdef" for character in text)
    ):
        rendered_lengths = ",".join(str(length) for length in lengths)
        raise ValueError(
            f"{field_name} must be canonical lowercase hexadecimal with length in "
            f"{{{rendered_lengths}}}"
        )


def _forbidden_observation_paths(
    value: Any, path: str = "observation"
) -> tuple[str, ...]:
    """Return every recursively nested privileged field path in a deployable view."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            nested_path = f"{path}.{key}"
            if key.lower() in FORBIDDEN_OBSERVATION_FIELDS:
                found.append(nested_path)
            found.extend(_forbidden_observation_paths(nested, nested_path))
    elif isinstance(value, np.ndarray):
        if value.dtype == object:
            for index, nested in np.ndenumerate(value):
                found.extend(
                    _forbidden_observation_paths(
                        nested, f"{path}[{','.join(str(item) for item in index)}]"
                    )
                )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            found.extend(_forbidden_observation_paths(nested, f"{path}[{index}]"))
    return tuple(found)


@dataclass(frozen=True)
class TrackAAuthority:
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    ten_k_training_authorized: bool = False

    def validate(self) -> None:
        values = (
            self.research_only,
            self.selection_evidence,
            self.promotion_ready,
            self.hardware_action_authorized,
            self.ten_k_training_authorized,
        )
        if any(type(value) is not bool for value in values):
            raise ValueError("Track A authority fields must be strict booleans")
        if not self.research_only:
            raise ValueError("Track A is research-only")
        if any(
            (
                self.selection_evidence,
                self.promotion_ready,
                self.hardware_action_authorized,
                self.ten_k_training_authorized,
            )
        ):
            raise ValueError(
                "Track A packages cannot grant selection, promotion, hardware, or 10k authority"
            )


@dataclass(frozen=True)
class TrackASampleKey:
    split: str
    stage: str
    cell_id: str
    global_step: int
    rank: int
    microbatch: int
    slot: int
    global_seed: int

    @property
    def sample_id(self) -> str:
        return (
            f"track_a:{self.split}:{self.stage}:{self.cell_id}:step{self.global_step}:"
            f"rank{self.rank}:micro{self.microbatch}:slot{self.slot}:seed{self.global_seed}"
        )

    def validate(self) -> None:
        if not isinstance(self.split, str) or self.split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError("split must be train, validation, or test")
        if self.stage not in {f"D{index}" for index in range(8)}:
            raise ValueError("stage must be D0..D7")
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise ValueError("cell_id is required")
        coordinates = (
            self.global_step,
            self.rank,
            self.microbatch,
            self.slot,
            self.global_seed,
        )
        if any(type(value) is not int or value < 0 for value in coordinates):
            raise ValueError("counter-key coordinates must be non-negative")

    def sha256(self) -> str:
        self.validate()
        return sha256_of(asdict(self))


@dataclass(frozen=True)
class TrackASample:
    key: TrackASampleKey
    n: int
    k: int
    observation: Dict[str, Any]
    truth: Dict[str, Any]
    metadata: Dict[str, Any]

    def validate(self) -> None:
        self.key.validate()
        if (
            type(self.n) is not int
            or type(self.k) is not int
            or not 3 <= self.n <= 256
            or not 1 <= self.k <= min(5, self.n)
        ):
            raise ValueError("sample N/K outside the Track A contract")
        if (
            not isinstance(self.observation, dict)
            or not isinstance(self.truth, dict)
            or not isinstance(self.metadata, dict)
        ):
            raise ValueError(
                "sample observation, truth, and metadata must be dictionaries"
            )
        leaked = _forbidden_observation_paths(self.observation)
        if leaked:
            raise ValueError(
                f"deployable observation contains privileged fields: {sorted(leaked)}"
            )
        if self.metadata.get("sample_id") != self.key.sample_id:
            raise ValueError("sample metadata identity mismatch")
        if self.metadata.get("split") != self.key.split:
            raise ValueError("sample metadata split mismatch")
        validate_hex_digest(
            self.metadata.get("key_sha", ""), field_name="metadata.key_sha"
        )


@dataclass(frozen=True)
class MainlineBundleV1:
    source_commit: str
    source_tree_sha: str
    config_sha: str
    classifier_profile: str
    feedback_profile: str
    semantic_abi: str = "V10TrackAMainlineABI1"
    classifier_checkpoint_sha: str = ""
    feedback_checkpoint_sha: str = ""
    frozen: bool = False
    source_bound: bool = False
    source_acceptance_sha: str = ""
    calibration_admission_sha: str = ""
    authority: TrackAAuthority = field(default_factory=TrackAAuthority)

    def validate(self) -> None:
        self.authority.validate()
        source_identities = (self.source_commit, self.source_tree_sha)
        for field_name, value in zip(
            ("source_commit", "source_tree_sha"), source_identities
        ):
            validate_hex_digest(value, field_name=field_name, allowed_lengths=(40, 64))
        validate_hex_digest(self.config_sha, field_name="config_sha")
        if (
            not isinstance(self.classifier_profile, str)
            or not self.classifier_profile
            or not isinstance(self.feedback_profile, str)
            or not self.feedback_profile
            or self.semantic_abi != "V10TrackAMainlineABI1"
            or type(self.frozen) is not bool
            or type(self.source_bound) is not bool
        ):
            raise ValueError(
                "mainline profile, ABI, or frozen-state identity is invalid"
            )
        hashes = (
            self.config_sha,
            self.classifier_checkpoint_sha,
            self.feedback_checkpoint_sha,
        )
        for index, value in enumerate(hashes[1:], start=1):
            validate_hex_digest(
                value,
                field_name=("classifier_checkpoint_sha", "feedback_checkpoint_sha")[
                    index - 1
                ],
                allow_empty=True,
            )
        if self.frozen and not (
            self.classifier_checkpoint_sha and self.feedback_checkpoint_sha
        ):
            raise ValueError(
                "a frozen mainline bundle requires both checkpoint identities"
            )
        if not self.frozen and (
            self.classifier_checkpoint_sha or self.feedback_checkpoint_sha
        ):
            raise ValueError(
                "an unfrozen mainline bundle cannot claim checkpoint identities"
            )
        validate_hex_digest(
            self.source_acceptance_sha,
            field_name="source_acceptance_sha",
            allow_empty=not self.source_bound,
        )
        if self.source_bound != bool(self.source_acceptance_sha):
            raise ValueError(
                "source-bound mainline identity requires exactly one acceptance SHA"
            )
        validate_hex_digest(
            self.calibration_admission_sha,
            field_name="calibration_admission_sha",
            allow_empty=not self.frozen,
        )

    def sha256(self) -> str:
        self.validate()
        return sha256_of(asdict(self))


@dataclass(frozen=True)
class RecoverabilityBundleV1:
    sample_id: str
    csm_context_sha: str
    recoverable_basis: np.ndarray
    supported_singular_values: np.ndarray
    rank_fraction: float
    condition_number: float
    kappa_geometry: float
    epsilon_irrecoverable: float
    action_mean: np.ndarray
    action_covariance: np.ndarray
    action_snr: float
    direction_stability: float
    reason_code: str
    same_forward_bound: bool = False
    forward_instance_sha: str = ""
    logical_jacobian_sha: str = ""
    covariance_sha: str = ""
    authority: TrackAAuthority = field(default_factory=TrackAAuthority)

    def validate(self) -> None:
        self.authority.validate()
        if type(self.same_forward_bound) is not bool:
            raise ValueError("same_forward_bound must be a strict boolean")
        basis = np.asarray(self.recoverable_basis, dtype=np.float64)
        singular = np.asarray(self.supported_singular_values, dtype=np.float64).reshape(
            -1
        )
        action = np.asarray(self.action_mean, dtype=np.float64).reshape(-1)
        covariance = np.asarray(self.action_covariance, dtype=np.float64)
        if not self.sample_id:
            raise ValueError("recoverability bundle identity is incomplete")
        validate_hex_digest(self.csm_context_sha, field_name="csm_context_sha")
        for field_name, value in (
            ("forward_instance_sha", self.forward_instance_sha),
            ("logical_jacobian_sha", self.logical_jacobian_sha),
            ("covariance_sha", self.covariance_sha),
        ):
            validate_hex_digest(value, field_name=field_name, allow_empty=True)
        if self.same_forward_bound:
            if not (
                self.forward_instance_sha
                and self.logical_jacobian_sha
                and self.covariance_sha
            ):
                raise ValueError(
                    "same-forward bundles require complete bound provenance"
                )
        elif self.forward_instance_sha:
            raise ValueError("unbound recoverability cannot claim a forward instance")
        if (
            basis.ndim != 2
            or not action.size
            or basis.shape[0] != action.size
            or singular.size != basis.shape[1]
            or covariance.shape != (action.size, action.size)
        ):
            raise ValueError("recoverability basis/action covariance shape mismatch")
        arrays = (basis, singular, action, covariance)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("recoverability bundle contains non-finite arrays")
        if np.any(singular <= 0.0):
            raise ValueError("supported singular values must be strictly positive")
        if singular.size > 1 and np.any(np.diff(singular) > 1e-12):
            raise ValueError("supported singular values must be non-increasing")
        if basis.shape[1] and not np.allclose(
            basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8
        ):
            raise ValueError("recoverability basis must be orthonormal")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("action covariance must be symmetric")
        if float(np.linalg.eigvalsh(covariance).min(initial=0.0)) < -1e-10:
            raise ValueError("action covariance must be positive semidefinite")
        expected_rank_fraction = basis.shape[1] / action.size
        if not np.isclose(self.rank_fraction, expected_rank_fraction, atol=1e-12):
            raise ValueError("rank_fraction does not match the recoverability basis")
        if not 0.0 <= self.rank_fraction <= 1.0:
            raise ValueError("rank_fraction must lie in [0,1]")
        if not 0.0 <= self.kappa_geometry <= 1.0:
            raise ValueError("kappa_geometry must lie in [0,1]")
        if not 0.0 <= self.epsilon_irrecoverable <= 1.0:
            raise ValueError("epsilon_irrecoverable must lie in [0,1]")
        expected_condition = float(singular[0] / singular[-1]) if singular.size else 0.0
        if not np.isclose(self.condition_number, expected_condition, atol=1e-10):
            raise ValueError(
                "condition_number does not match supported singular values"
            )
        if (
            not np.isfinite(self.condition_number)
            or self.condition_number < 0.0
            or not np.isfinite(self.action_snr)
            or self.action_snr < 0.0
            or not np.isfinite(self.direction_stability)
            or not 0.0 <= self.direction_stability <= 1.0
        ):
            raise ValueError("invalid action SNR or direction stability")
        allowed_reason_codes = {
            "RECOVERABLE_SUPPORT_PRESENT",
            "RECOVERABILITY_ZERO_SUPPORT",
        }
        if self.reason_code not in allowed_reason_codes:
            raise ValueError(
                "recoverability reason_code is not a successful fail-closed state"
            )


class GateStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class GateRecord:
    gate_id: str
    status: GateStatus
    evidence_sha: str = ""
    notes: str = ""

    def validate(self) -> None:
        if not isinstance(self.gate_id, str) or not self.gate_id:
            raise ValueError("gate_id is required")
        if not isinstance(self.status, GateStatus):
            raise ValueError("gate status must use the GateStatus enum")
        if not isinstance(self.notes, str):
            raise ValueError("gate notes must be a string")
        terminal = self.status in {GateStatus.PASS, GateStatus.FAIL}
        if terminal and not self.evidence_sha:
            raise ValueError(
                "terminal gate decisions require a SHA-256 evidence identity"
            )
        validate_hex_digest(
            self.evidence_sha,
            field_name="evidence_sha",
            allow_empty=not terminal,
        )
