"""Canonical evidence identities for the synthetic Memory V4 contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from enum import Enum
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

NORMALIZATION_TOLERANCE = 2e-12
MODE_DESCRIPTOR_SCHEMA = "V10MemoryModeDescriptorV1"
PHASE_CONVENTION_SCHEMA = "V10MemoryRelativePhaseConventionV1"
COORDINATE_LAYOUT_SCHEMA = "V10MemoryCoordinateLayoutV1"
TARGET_STATE_SCHEMA = "V10MemoryTargetStateV1"
QUALIFICATION_EVIDENCE_SCHEMA = "V10MemoryQualificationEvidenceV1"
QUALIFICATION_RECEIPT_SCHEMA = "V10MemoryQualificationReceiptV1"
QUALIFICATION_EVALUATOR_RECIPE = {
    "schema": "V10SyntheticScenarioQualificationEvaluatorV1",
    "source": "past_observation_only",
}
QUALIFICATION_PREDICATE_RECIPE = {
    "schema": "V10SyntheticQualificationPredicateV1",
    "requires": ["certificate", "recoverability", "safety", "quality", "receipt_available_tick"],
}
QUALIFICATION_EVALUATOR_ID = "V10_SYNTHETIC_QUALIFICATION_EVALUATOR_V2"
QUALIFICATION_EVALUATOR_ALGORITHM = {
    "schema": "V10SyntheticQualificationEvaluatorAlgorithmV2",
    "certificate_rule": "certificate_score >= 0.5",
    "recoverability_rule": "recoverability_score >= 0.5",
    "safety_rule": "safety_score >= 0.5",
    "quality_rule": "quality_score",
    "apparatus_session_status": "NOT_APPLICABLE_SYNTHETIC",
}
QUALIFICATION_RECEIPT_V4_SCHEMA = "V10MemoryQualificationEvaluatorReceiptV4"
QUALIFICATION_ISSUER_ROLE = "SYNTHETIC_QUALIFICATION_ISSUER_V1"
QUALIFICATION_ALGORITHM_VERSION = "V4_SIGNED_EVALUATOR_BUNDLE"


def _canonical_json_value(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, Enum):
        return _canonical_json_value(value.value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical evidence does not allow non-finite floats")
        return value
    if type(value) is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("canonical evidence does not allow non-finite complex values")
        return {"complex": [value.real, value.imag]}
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("canonical evidence mappings require string keys")
        return {key: _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError(f"unsupported canonical evidence value: {type(value).__name__}")


def digest(value: object) -> str:
    encoded = json.dumps(
        _canonical_json_value(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        _canonical_json_value(payload), allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def evaluator_io_source_sha() -> str:
    """Hash the issuer-side evaluator I/O implementation only."""
    evaluator_path = Path(__file__).parents[1] / "physics" / "memory_qualification_evaluator.py"
    return hashlib.sha256(evaluator_path.read_bytes()).hexdigest()


def qualification_evaluator_source_sha() -> str:
    """Bind the complete executable parser/predicate/evaluator bundle."""
    return hashlib.sha256(evaluator_bundle_bytes()).hexdigest()


def _validated_scores(raw_input_json: str) -> dict[str, float]:
    try:
        raw = json.loads(raw_input_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("qualification raw input must be canonical JSON") from error
    if _canonical_json(raw) != raw_input_json or set(raw) != {
        "certificate_score", "quality_score", "recoverability_score", "safety_score"
    }:
        raise ValueError("qualification raw input must contain exactly the registered score payload")
    scores: dict[str, float] = {}
    for name in ("certificate_score", "recoverability_score", "safety_score", "quality_score"):
        value = raw[name]
        if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("qualification raw scores must be finite probabilities")
        scores[name] = float(value)
    return scores


def _qualification_outcome(raw_input_json: str) -> tuple[bool, bool, bool, float]:
    scores = _validated_scores(raw_input_json)
    return scores["certificate_score"] >= 0.5, scores["recoverability_score"] >= 0.5, scores["safety_score"] >= 0.5, scores["quality_score"]


class Raw_qualification_evidence_unavailable(ValueError):
    """No registered raw evidence supports formal resource qualification."""


def require_registered_raw_qualification_evidence(raw: Mapping[str, object]) -> None:
    """Reject score diagnostics until an independent raw-evidence law is registered.

    V1 contains caller-assigned scores, not trials/counts or simulator events.
    Its signature authenticates a statement but cannot establish the physical
    predicates. No raw producer/forward/uncertainty contract is implemented yet;
    unknown schemas must not become an alternative route to qualification.
    There is deliberately no runtime override or test-mode configuration.
    """
    if raw.get("schema") == "V10SyntheticQualificationObservationV1":
        raise Raw_qualification_evidence_unavailable(
            "RAW_QUALIFICATION_EVIDENCE_UNAVAILABLE: score-only V1 is a "
            "diagnostic, not independently recomputed qualification"
        )
    raise Raw_qualification_evidence_unavailable(
        "RAW_QUALIFICATION_EVIDENCE_UNAVAILABLE: raw qualification producer "
        "and evaluator contract are not registered"
    )


@dataclass(frozen=True)
class Synthetic_qualification_evaluator_receipt:
    """Legacy signed format, subject to independent raw-evidence admission.

    The private key belongs only to the independently invoked evaluator.  Queue,
    reducer and replay receive this immutable bundle and can only verify it.
    The score-only V1 format cannot currently pass construction or validation.
    """

    raw_observation_bytes: bytes
    issuer_manifest_bytes: bytes
    issuer_id: str
    issuer_role: str
    issuer_manifest_sha: str
    issuer_public_key: bytes
    signature: bytes
    evaluator_id: str
    evaluator_source_sha: str
    algorithm_version: str
    observation_tick: int
    receipt_available_tick: int
    evaluator_bundle_bytes: bytes
    derived_certificate_valid: bool
    derived_recoverability_valid: bool
    derived_safety_valid: bool
    derived_estimated_quality: float

    def __post_init__(self) -> None:
        if self.evaluator_id != QUALIFICATION_EVALUATOR_ID or self.algorithm_version != QUALIFICATION_ALGORITHM_VERSION:
            raise ValueError("qualification evaluator is not registered")
        if self.evaluator_source_sha != qualification_evaluator_source_sha():
            raise ValueError("qualification evaluator source identity is not registered")
        if self.issuer_role != QUALIFICATION_ISSUER_ROLE:
            raise ValueError("qualification issuer role is not registered")
        if type(self.issuer_id) is not str or not self.issuer_id:
            raise ValueError("qualification issuer identity is required")
        if type(self.issuer_manifest_sha) is not str or len(self.issuer_manifest_sha) != 64:
            raise ValueError("qualification issuer manifest SHA must be SHA-256")
        if type(self.raw_observation_bytes) is not bytes or type(self.issuer_manifest_bytes) is not bytes:
            raise ValueError("qualification receipt must embed immutable observation and issuer bytes")
        if type(self.issuer_public_key) is not bytes or len(self.issuer_public_key) != 32:
            raise ValueError("qualification issuer public key must be an Ed25519 key")
        if type(self.signature) is not bytes or len(self.signature) != 64:
            raise ValueError("qualification receipt signature must be Ed25519")
        if type(self.observation_tick) is not int or type(self.receipt_available_tick) is not int:
            raise ValueError("qualification observation and receipt ticks must be integers")
        if self.receipt_available_tick < self.observation_tick:
            raise ValueError("qualification receipt cannot predate its observation")
        raw = _parse_raw_observation(self.raw_observation_bytes)
        require_registered_raw_qualification_evidence(raw)
        bundle = _parse_evaluator_bundle(self.evaluator_bundle_bytes)
        issuer = _parse_issuer_manifest(self.issuer_manifest_bytes)
        if hashlib.sha256(self.issuer_manifest_bytes).hexdigest() != self.issuer_manifest_sha:
            raise ValueError("qualification issuer manifest bytes do not match their SHA")
        if (issuer["issuer_id"], issuer["issuer_role"], bytes.fromhex(issuer["public_key_hex"])) != (self.issuer_id, self.issuer_role, self.issuer_public_key):
            raise ValueError("qualification issuer manifest does not bind issuer identity, role and public key")
        if raw["observation_tick"] != self.observation_tick:
            raise ValueError("qualification receipt observation tick does not bind raw observation bytes")
        if issuer["evaluator_id"] != self.evaluator_id or issuer["evaluator_source_sha"] != self.evaluator_source_sha:
            raise ValueError("qualification issuer manifest does not bind evaluator identity")
        try:
            Ed25519PublicKey.from_public_bytes(self.issuer_public_key).verify(self.signature, self.canonical_payload_bytes)
        except Exception as error:
            raise ValueError("qualification receipt signature is invalid") from error
        derived = _bundle_qualification_outcome(raw["scores"], bundle)
        if (type(self.derived_certificate_valid) is not bool or type(self.derived_recoverability_valid) is not bool
                or type(self.derived_safety_valid) is not bool or type(self.derived_estimated_quality) is not float
                or not math.isfinite(self.derived_estimated_quality) or not 0 <= self.derived_estimated_quality <= 1):
            raise ValueError("qualification signed derived gates and quality have invalid types")
        if derived != (self.derived_certificate_valid, self.derived_recoverability_valid, self.derived_safety_valid, self.derived_estimated_quality):
            raise ValueError("qualification signed derived outcome does not match evaluator bundle")

    def validate(self) -> None:
        """Rehash embedded bytes and reverify the signature on every consumer boundary."""
        self.__post_init__()

    @property
    def raw_observation(self) -> dict:
        return _parse_raw_observation(self.raw_observation_bytes)

    @property
    def canonical_payload_bytes(self) -> bytes:
        return _canonical_json({
            "schema": QUALIFICATION_RECEIPT_V4_SCHEMA,
            "raw_observation_bytes_hex": self.raw_observation_bytes.hex(),
            "issuer_manifest_bytes_hex": self.issuer_manifest_bytes.hex(),
            "issuer_id": self.issuer_id,
            "issuer_role": self.issuer_role,
            "issuer_manifest_sha": self.issuer_manifest_sha,
            "issuer_public_key_hex": self.issuer_public_key.hex(),
            "evaluator_id": self.evaluator_id,
            "evaluator_source_sha": self.evaluator_source_sha,
            "algorithm_version": self.algorithm_version,
            "observation_tick": self.observation_tick,
            "receipt_available_tick": self.receipt_available_tick,
            "evaluator_bundle_bytes_hex": self.evaluator_bundle_bytes.hex(),
            "derived_certificate_valid": self.derived_certificate_valid,
            "derived_recoverability_valid": self.derived_recoverability_valid,
            "derived_safety_valid": self.derived_safety_valid,
            "derived_estimated_quality": self.derived_estimated_quality,
        }).encode()

    @property
    def certificate_valid(self) -> bool:
        return self.derived_certificate_valid

    @property
    def recoverability_valid(self) -> bool:
        return self.derived_recoverability_valid

    @property
    def safety_valid(self) -> bool:
        return self.derived_safety_valid

    @property
    def estimated_quality(self) -> float:
        return self.derived_estimated_quality

    @property
    def algorithm_sha(self) -> str:
        return hashlib.sha256(self.evaluator_bundle_bytes).hexdigest()

    @property
    def evidence_sha(self) -> str:
        return digest({
            "schema": QUALIFICATION_RECEIPT_V4_SCHEMA,
            "evaluator_id": self.evaluator_id,
            "evaluator_source_sha": self.evaluator_source_sha,
            "algorithm_sha": self.algorithm_sha,
            "raw_observation_sha": hashlib.sha256(self.raw_observation_bytes).hexdigest(),
            "issuer_manifest_sha": self.issuer_manifest_sha,
            "issuer_id": self.issuer_id,
            "issuer_role": self.issuer_role,
            "issuer_public_key_sha": hashlib.sha256(self.issuer_public_key).hexdigest(),
            "signature_sha": hashlib.sha256(self.signature).hexdigest(),
            "observation_tick": self.observation_tick,
            "receipt_available_tick": self.receipt_available_tick,
            "derived_outcome": {
                "certificate_valid": self.certificate_valid,
                "recoverability_valid": self.recoverability_valid,
                "safety_valid": self.safety_valid,
                "estimated_quality": self.estimated_quality,
            },
        })

    @property
    def receipt_sha(self) -> str:
        return digest({"schema": QUALIFICATION_RECEIPT_V4_SCHEMA, "evidence_sha": self.evidence_sha})


def evaluator_bundle_bytes() -> bytes:
    return _canonical_json({
        "schema": "V10SyntheticQualificationEvaluatorBundleV1",
        "evaluator_algorithm": QUALIFICATION_EVALUATOR_ALGORITHM,
        "memory_evidence_source_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluator_io_source_sha": evaluator_io_source_sha(),
    }).encode()


def _parse_evaluator_bundle(value: bytes) -> dict:
    try:
        bundle = json.loads(value)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("qualification evaluator bundle must be canonical JSON") from error
    if _canonical_json(bundle).encode() != value or bundle.get("schema") != "V10SyntheticQualificationEvaluatorBundleV1":
        raise ValueError("qualification evaluator bundle schema is invalid")
    if bundle.get("memory_evidence_source_sha") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or bundle.get("evaluator_io_source_sha") != evaluator_io_source_sha():
        raise ValueError("qualification evaluator bundle source identity is stale")
    if bundle.get("evaluator_algorithm") != QUALIFICATION_EVALUATOR_ALGORITHM:
        raise ValueError("qualification evaluator bundle algorithm is stale")
    return bundle


def _bundle_qualification_outcome(scores: Mapping[str, object], bundle: Mapping[str, object]) -> tuple[bool, bool, bool, float]:
    validated = _validated_scores(_canonical_json(scores))
    algorithm = bundle["evaluator_algorithm"]
    if algorithm != QUALIFICATION_EVALUATOR_ALGORITHM:
        raise ValueError("qualification evaluator algorithm is unsupported")
    return validated["certificate_score"] >= 0.5, validated["recoverability_score"] >= 0.5, validated["safety_score"] >= 0.5, validated["quality_score"]


def _parse_raw_observation(raw_bytes: bytes) -> dict:
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("qualification raw observation bytes must be canonical JSON") from error
    if _canonical_json(raw).encode() != raw_bytes or set(raw) != {"identity", "observation_tick", "scores", "schema"}:
        raise ValueError("qualification raw observation schema is invalid")
    if raw["schema"] != "V10SyntheticQualificationObservationV1" or type(raw["observation_tick"]) is not int:
        raise ValueError("qualification raw observation identity/tick is invalid")
    required_identity = {"subject_id", "resource_id", "source_id", "cycle_id", "trial_provenance_sha", "partition_sha", "mode_coefficients", "qualification_outcome"}
    if type(raw["identity"]) is not dict or set(raw["identity"]) != required_identity:
        raise ValueError("qualification raw observation must bind complete identity")
    _validated_scores(_canonical_json(raw["scores"]))
    return raw


def _parse_issuer_manifest(issuer_bytes: bytes) -> dict:
    try:
        issuer = json.loads(issuer_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("qualification issuer manifest bytes must be canonical JSON") from error
    if _canonical_json(issuer).encode() != issuer_bytes:
        raise ValueError("qualification issuer manifest must use canonical JSON")
    required = {"schema", "issuer_id", "issuer_role", "public_key_hex", "evaluator_id", "evaluator_source_sha"}
    if set(issuer) != required or issuer["schema"] != "V10SyntheticQualificationIssuerManifestV1":
        raise ValueError("qualification issuer manifest schema is invalid")
    if type(issuer["issuer_id"]) is not str or type(issuer["issuer_role"]) is not str:
        raise ValueError("qualification issuer manifest identity is invalid")
    try:
        key = bytes.fromhex(issuer["public_key_hex"])
    except (TypeError, ValueError) as error:
        raise ValueError("qualification issuer manifest public key is invalid") from error
    if len(key) != 32:
        raise ValueError("qualification issuer manifest public key is invalid")
    return issuer


def normalized_coefficients(value: object) -> tuple[complex, ...]:
    try:
        source = tuple(value)
        if any(isinstance(item, bool) for item in source):
            raise ValueError("boolean mode coefficient")
        coefficients = tuple(complex(item) for item in source)
    except (TypeError, ValueError) as error:
        raise ValueError("mode coefficients must be finite complex values") from error
    norm = sum(abs(item) ** 2 for item in coefficients)
    if len(coefficients) < 2 or any(not math.isfinite(item.real) or not math.isfinite(item.imag) for item in coefficients) or abs(norm - 1.0) > NORMALIZATION_TOLERANCE:
        raise ValueError("mode coefficients must be normalized")
    return coefficients


def coefficient_payload(coefficients: tuple[complex, ...]) -> list[tuple[float, float]]:
    return [(value.real, value.imag) for value in coefficients]


def mode_descriptor_sha(coefficients: object) -> str:
    return digest({"schema": MODE_DESCRIPTOR_SCHEMA, "coefficients": coefficient_payload(normalized_coefficients(coefficients))})


def phase_convention_sha(coefficients: object) -> str:
    values = normalized_coefficients(coefficients)
    reference = next(value for value in values if abs(value) > 0)
    relative = tuple(value * reference.conjugate() / abs(reference) for value in values)
    return digest({"schema": PHASE_CONVENTION_SCHEMA, "coefficients": coefficient_payload(relative)})


def coordinate_layout_sha(coefficients: object) -> str:
    values = normalized_coefficients(coefficients)
    return digest({"schema": COORDINATE_LAYOUT_SCHEMA, "mode_count": len(values), "coordinate_order": "RELATIVE_LOG_AMPLITUDE_THEN_RELATIVE_PHASE_HELMERT"})


def target_state_sha(coefficients: object) -> str:
    return digest({"schema": TARGET_STATE_SCHEMA, "coefficients": coefficient_payload(normalized_coefficients(coefficients))})


def calibration_domain_sha() -> str:
    return digest({"schema": "V10SyntheticMemoryCalibrationDomainV1", "source_tag": "SYNTHETIC"})


def qualification_evidence_sha(subject_id, resource_id, certificate_valid, recoverability_valid, safety_valid, receipt_available_tick, estimated_quality, source_provenance_sha, source_id, cycle_id, trial_provenance_sha, mode_coefficients, partition_sha, qualification_outcome):
    return digest({"schema": QUALIFICATION_EVIDENCE_SCHEMA, "evaluator_recipe_sha": digest(QUALIFICATION_EVALUATOR_RECIPE), "predicate_recipe_sha": digest(QUALIFICATION_PREDICATE_RECIPE), "trial_id": subject_id, "resource_id": resource_id, "source_id": source_id, "cycle_id": cycle_id, "trial_provenance_sha": trial_provenance_sha, "mode_coefficients": coefficient_payload(normalized_coefficients(mode_coefficients)), "partition_sha": partition_sha, "qualification_outcome": qualification_outcome, "certificate_valid": certificate_valid, "recoverability_valid": recoverability_valid, "safety_valid": safety_valid, "receipt_available_tick": receipt_available_tick, "estimated_quality": estimated_quality, "source_provenance_sha": source_provenance_sha})


def qualification_receipt_sha(evidence_sha: str) -> str:
    return digest({"schema": QUALIFICATION_RECEIPT_SCHEMA, "evidence_sha": evidence_sha})


def synthetic_source_session_apparatus_provenance_sha(source_id: str, cycle_id: str, resource_id: str) -> str:
    return digest({"schema": "V10MemorySourceSessionApparatusProvenanceV1", "status": "NOT_APPLICABLE_SYNTHETIC", "reason": "synthetic_queue_input_has_no_apparatus_session", "source_id": source_id, "cycle_id": cycle_id, "resource_id": resource_id})
