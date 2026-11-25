"""Machine-readable scientific and engineering predicates for Track A gates."""

from __future__ import annotations

import math
from dataclasses import asdict
from numbers import Real
from typing import Any, Dict

from ml_recon.physics_hybrid_v10.contracts import sha256_of

from ..config import GateConfig


TRACK_A_GATE_EVIDENCE_SCHEMA = "V10TrackAGateEvidenceV1"
TRACK_A_GATE_SPEC_SCHEMA = "V10TrackAGateSpecV1"

RESEARCH_AUTHORITY = {
    "research_only": True,
    "selection_evidence": False,
    "promotion_ready": False,
    "hardware_action_authorized": False,
    "ten_k_training_authorized": False,
}

GATE_SPEC_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "P0": {
        "evaluator_schema": "V10TrackAP0ContractEvidenceV1",
        "required_checks": (
            "subsystem_contract",
            "fixed_k_contract",
            "collision_contract",
            "partition_legality",
            "certificate_separation",
            "csm_rsm_same_forward",
            "observation_leakage_absent",
        ),
        "metric_rules": (),
        "minimum_seed_count": 1,
        "minimum_sample_rung": 0,
        "maximum_sample_rung": 0,
        "require_checkpoints": False,
        "require_dataset_ranges": False,
        "require_dataset_identity": False,
    },
    "E0": {
        "evaluator_schema": "V10TrackAE0ExecutionEvidenceV1",
        "required_checks": (
            "full_forward",
            "invariance",
            "conservation",
            "finite_gradients",
            "deterministic_resume",
            "checkpoint_reload_exact",
        ),
        "metric_rules": (
            ("gradient_nonfinite_count", "maximum", 0.0),
            ("resume_mismatch_count", "maximum", 0.0),
        ),
        "minimum_seed_count": 1,
        "minimum_sample_rung": 1,
        "maximum_sample_rung": 1_000,
        "require_checkpoints": True,
        "require_dataset_ranges": True,
        "require_dataset_identity": True,
    },
    "E1": {
        "evaluator_schema": "V10TrackAE1TeacherParityEvidenceV1",
        "required_checks": (
            "small_n_exact_teacher",
            "certificate_parity",
            "validation_test_disjoint",
            "worst_cell_complete",
        ),
        "metric_rules": (
            ("teacher_max_abs_error", "maximum", 1.0e-6),
            ("certificate_parity_rate", "minimum", 1.0),
        ),
        "minimum_seed_count": 1,
        "minimum_sample_rung": 1_000,
        "maximum_sample_rung": 1_000,
        "require_checkpoints": True,
        "require_dataset_ranges": True,
        "require_dataset_identity": True,
    },
    "E2": {
        "evaluator_schema": "V10TrackAE2OracleCeilingEvidenceV1",
        "required_checks": (
            "esm_oracle_ceiling",
            "csm_oracle_ceiling",
            "action_oracle_ceiling",
            "measurement_value_oracle_ceiling",
            "worst_cell_complete",
        ),
        "metric_rules": (
            ("fisher_rank_fraction_worst_cell", "minimum", "fisher_rank_fraction_minimum"),
            ("control_auroc_worst_cell", "minimum", "control_auroc_minimum"),
            ("action_cosine_worst_cell", "minimum", "action_cosine_minimum"),
            ("measurement_value_gain_lcb95", "strict_minimum", 0.0),
        ),
        "minimum_seed_count": 1,
        "minimum_sample_rung": 1_000,
        "maximum_sample_rung": 1_000,
        "require_checkpoints": True,
        "require_dataset_ranges": True,
        "require_dataset_identity": True,
    },
    "E3": {
        "evaluator_schema": "V10TrackAE3AntiCollapsePilotEvidenceV1",
        "required_checks": (
            "anti_collapse",
            "finite_gradients",
            "deterministic_resume",
            "worst_cell_complete",
            "no_gate_regression",
        ),
        "metric_rules": (
            ("collapse_fraction", "maximum", 0.0),
            (
                "posterior_coverage_error_worst_cell",
                "maximum",
                "posterior_coverage_error_maximum",
            ),
            ("serious_harm_u95", "maximum", "serious_harm_u95_maximum"),
        ),
        "minimum_seed_count": 3,
        "minimum_sample_rung": 2_000,
        "maximum_sample_rung": 10_000,
        "require_checkpoints": True,
        "require_dataset_ranges": True,
        "require_dataset_identity": True,
    },
    "S1": {
        "evaluator_schema": "V10TrackAS1GeneralizationEvidenceV1",
        "required_checks": (
            "held_out_n",
            "held_out_k",
            "held_out_mechanism",
            "held_out_backend",
            "held_out_schedule",
            "worst_cell_complete",
        ),
        "metric_rules": (
            ("control_auroc_worst_cell", "minimum", "control_auroc_minimum"),
            (
                "posterior_coverage_error_worst_cell",
                "maximum",
                "posterior_coverage_error_maximum",
            ),
        ),
        "minimum_seed_count": 3,
        "minimum_sample_rung": 10_000,
        "maximum_sample_rung": 2_000_000,
        "require_checkpoints": True,
        "require_dataset_ranges": True,
        "require_dataset_identity": True,
    },
    "S2": {
        "evaluator_schema": "V10TrackAS2StructureCertificateEvidenceV1",
        "required_checks": (
            "structure_evidence",
            "certificate_evidence",
            "multiplicity_control",
            "worst_cell_complete",
        ),
        "metric_rules": (
            ("structure_pr_auc_worst_cell", "minimum", "control_auroc_minimum"),
            ("certificate_precision_worst_cell", "minimum", 0.95),
        ),
        "minimum_seed_count": 3,
        "minimum_sample_rung": 10_000,
        "maximum_sample_rung": 2_000_000,
        "require_checkpoints": True,
        "require_dataset_ranges": True,
        "require_dataset_identity": True,
    },
    "S3": {
        "evaluator_schema": "V10TrackAS3RecoverabilityCalibrationEvidenceV1",
        "required_checks": (
            "same_forward_geometry",
            "rank_calibration",
            "posterior_calibration",
            "worst_cell_complete",
        ),
        "metric_rules": (
            ("fisher_rank_fraction_worst_cell", "minimum", "fisher_rank_fraction_minimum"),
            (
                "posterior_coverage_error_worst_cell",
                "maximum",
                "posterior_coverage_error_maximum",
            ),
        ),
        "minimum_seed_count": 3,
        "minimum_sample_rung": 10_000,
        "maximum_sample_rung": 2_000_000,
        "require_checkpoints": True,
        "require_dataset_ranges": True,
        "require_dataset_identity": True,
    },
    "S4": {
        "evaluator_schema": "V10TrackAS4DecisionSafetyEvidenceV1",
        "required_checks": (
            "action",
            "risk",
            "stop",
            "measurement_decision",
            "worst_cell_complete",
        ),
        "metric_rules": (
            ("action_cosine_worst_cell", "minimum", "action_cosine_minimum"),
            ("serious_harm_u95", "maximum", "serious_harm_u95_maximum"),
            ("normalized_decision_regret_u95", "maximum", 0.05),
        ),
        "minimum_seed_count": 3,
        "minimum_sample_rung": 10_000,
        "maximum_sample_rung": 2_000_000,
        "require_checkpoints": True,
        "require_dataset_ranges": True,
        "require_dataset_identity": True,
    },
    "S5": {
        "evaluator_schema": "V10TrackAS5ScalingConfirmationEvidenceV1",
        "required_checks": (
            "fixed_resource_comparison",
            "independent_confirmation",
            "capacity_ladder_complete",
            "worst_cell_complete",
        ),
        "metric_rules": (
            ("scaling_gain_lcb95", "strict_minimum", 0.0),
            ("confirmation_gain_lcb95", "strict_minimum", 0.0),
        ),
        "minimum_seed_count": 5,
        "minimum_sample_rung": 10_000,
        "maximum_sample_rung": 2_000_000,
        "require_checkpoints": True,
        "require_dataset_ranges": True,
        "require_dataset_identity": True,
    },
}


def _resolved_metric_rules(
    rules: tuple[tuple[str, str, str | float], ...],
    gate_config: GateConfig,
) -> tuple[Dict[str, Any], ...]:
    thresholds = asdict(gate_config)
    resolved = []
    for metric_name, comparator, threshold_reference in rules:
        threshold = (
            thresholds[threshold_reference]
            if isinstance(threshold_reference, str)
            else threshold_reference
        )
        resolved.append({
            "metric": metric_name,
            "comparator": comparator,
            "threshold": float(threshold),
        })
    return tuple(resolved)


def gate_spec(gate_id: str, gate_config: GateConfig) -> Dict[str, Any]:
    """Resolve one immutable Gate specification against checked-in thresholds."""

    gate_config.validate()
    if gate_id not in GATE_SPEC_TEMPLATES:
        raise ValueError(f"unknown Track A gate specification {gate_id}")
    template = GATE_SPEC_TEMPLATES[gate_id]
    payload = {
        "schema": TRACK_A_GATE_SPEC_SCHEMA,
        "gate_id": gate_id,
        "authorization_scope": f"track_a_{gate_id.lower()}_gate_closure_v1",
        "evaluator_schema": template["evaluator_schema"],
        "required_checks": template["required_checks"],
        "metric_rules": _resolved_metric_rules(
            template["metric_rules"], gate_config),
        "minimum_seed_count": template["minimum_seed_count"],
        "minimum_sample_rung": template["minimum_sample_rung"],
        "maximum_sample_rung": template["maximum_sample_rung"],
        "require_checkpoints": template["require_checkpoints"],
        "require_dataset_ranges": template["require_dataset_ranges"],
        "require_dataset_identity": template["require_dataset_identity"],
        "gate_thresholds": asdict(gate_config),
    }
    return {**payload, "gate_spec_sha": sha256_of(payload)}


def _require_digest(value: object, field_name: str, lengths: tuple[int, ...] = (64,)) -> str:
    if (not isinstance(value, str)
            or value != value.strip().lower()
            or len(value) not in lengths
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{field_name} must be canonical lowercase hexadecimal")
    return value


def _require_digest_map(
    value: object,
    field_name: str,
    *,
    allow_empty: bool,
) -> Dict[str, str]:
    if not isinstance(value, dict) or (not allow_empty and not value):
        raise ValueError(f"{field_name} must be a non-empty SHA-256 mapping")
    result: Dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        result[raw_name] = _require_digest(raw_digest, f"{field_name}.{raw_name}")
    return result


def _metric_passes(value: float, comparator: str, threshold: float) -> bool:
    if comparator == "minimum":
        return value >= threshold
    if comparator == "strict_minimum":
        return value > threshold
    if comparator == "maximum":
        return value <= threshold
    raise ValueError(f"unknown Gate metric comparator {comparator}")


def validate_gate_evidence(
    evidence: object,
    *,
    expected_gate_id: str,
    expected_config_sha: str,
    expected_source_tree_sha: str,
    gate_config: GateConfig,
) -> Dict[str, Any]:
    """Validate Gate-specific checks, metrics, data, model, and evaluator identity."""

    if not isinstance(evidence, dict):
        raise ValueError("gate evidence must be a JSON object")
    required_fields = {
        "schema",
        "gate_id",
        "gate_spec_sha",
        "evaluator_schema",
        "evaluator_sha",
        "source_commit",
        "source_tree_sha",
        "config_sha",
        "checkpoint_shas",
        "dataset_range_shas",
        "generator_sha",
        "contract_sha",
        "dataset_schema_sha",
        "seed_panel",
        "sample_rung",
        "checks",
        "metrics",
        "evidence_file_shas",
        "authority",
    }
    if set(evidence) != required_fields:
        missing = sorted(required_fields - set(evidence))
        extra = sorted(set(evidence) - required_fields)
        raise ValueError(f"gate evidence fields mismatch: missing={missing}, extra={extra}")

    spec = gate_spec(expected_gate_id, gate_config)
    if (evidence["schema"] != TRACK_A_GATE_EVIDENCE_SCHEMA
            or evidence["gate_id"] != expected_gate_id
            or evidence["gate_spec_sha"] != spec["gate_spec_sha"]
            or evidence["evaluator_schema"] != spec["evaluator_schema"]):
        raise ValueError("gate evidence schema, Gate, evaluator, or specification is stale")
    _require_digest(evidence["evaluator_sha"], "evaluator_sha")
    _require_digest(evidence["source_commit"], "source_commit", (40, 64))
    source_tree_sha = _require_digest(
        evidence["source_tree_sha"], "source_tree_sha", (40, 64))
    config_sha = _require_digest(evidence["config_sha"], "config_sha")
    if source_tree_sha != _require_digest(
            expected_source_tree_sha, "expected_source_tree_sha", (40, 64)):
        raise ValueError("gate evidence source tree identity mismatch")
    if config_sha != _require_digest(expected_config_sha, "expected_config_sha"):
        raise ValueError("gate evidence config identity mismatch")

    checkpoints = _require_digest_map(
        evidence["checkpoint_shas"],
        "checkpoint_shas",
        allow_empty=not spec["require_checkpoints"],
    )
    ranges = _require_digest_map(
        evidence["dataset_range_shas"],
        "dataset_range_shas",
        allow_empty=not spec["require_dataset_ranges"],
    )
    _require_digest_map(
        evidence["evidence_file_shas"],
        "evidence_file_shas",
        allow_empty=False,
    )
    identity_values = (
        evidence["generator_sha"],
        evidence["contract_sha"],
        evidence["dataset_schema_sha"],
    )
    if spec["require_dataset_identity"]:
        for field_name, value in zip(
                ("generator_sha", "contract_sha", "dataset_schema_sha"),
                identity_values):
            _require_digest(value, field_name)
    elif any(value != "" for value in identity_values):
        for field_name, value in zip(
                ("generator_sha", "contract_sha", "dataset_schema_sha"),
                identity_values):
            _require_digest(value, field_name)

    seed_panel = evidence["seed_panel"]
    if (not isinstance(seed_panel, list)
            or len(seed_panel) < spec["minimum_seed_count"]
            or any(type(seed) is not int or seed < 0 for seed in seed_panel)
            or seed_panel != sorted(set(seed_panel))):
        raise ValueError("gate evidence seed_panel is incomplete or non-canonical")
    sample_rung = evidence["sample_rung"]
    if (type(sample_rung) is not int
            or not spec["minimum_sample_rung"] <= sample_rung <= spec["maximum_sample_rung"]):
        raise ValueError("gate evidence sample_rung is outside the registered Gate interval")

    checks = evidence["checks"]
    required_checks = tuple(spec["required_checks"])
    if (not isinstance(checks, dict)
            or set(checks) != set(required_checks)
            or any(checks[name] is not True for name in required_checks)):
        raise ValueError("gate evidence required checks are missing, unexpected, or not all true")
    metrics = evidence["metrics"]
    metric_rules = tuple(spec["metric_rules"])
    required_metrics = {rule["metric"] for rule in metric_rules}
    if not isinstance(metrics, dict) or set(metrics) != required_metrics:
        raise ValueError("gate evidence metrics do not exactly match the Gate specification")
    for rule in metric_rules:
        metric_value = metrics[rule["metric"]]
        if (isinstance(metric_value, bool)
                or not isinstance(metric_value, Real)
                or not math.isfinite(float(metric_value))
                or not _metric_passes(
                    float(metric_value), rule["comparator"], rule["threshold"]
                )):
            raise ValueError(
                f"gate evidence metric {rule['metric']} failed "
                f"{rule['comparator']} threshold {rule['threshold']}"
            )
    if evidence["authority"] != RESEARCH_AUTHORITY:
        raise ValueError("gate evidence cannot grant selection, promotion, hardware, or 10k authority")
    if spec["require_checkpoints"] and not checkpoints:
        raise ValueError("gate evidence requires checkpoint identities")
    if spec["require_dataset_ranges"] and not ranges:
        raise ValueError("gate evidence requires dataset range identities")
    return dict(evidence)


__all__ = [
    "GATE_SPEC_TEMPLATES",
    "RESEARCH_AUTHORITY",
    "TRACK_A_GATE_EVIDENCE_SCHEMA",
    "TRACK_A_GATE_SPEC_SCHEMA",
    "gate_spec",
    "validate_gate_evidence",
]
