"""Consumer-shaped held-out calibration with source-bound protocol and update audit.

This evaluator measures errors; it does not issue a scientific Gate. Admission
additionally requires a frozen statistical protocol, the raw committed-update
audit, and an external review of the exact resulting report.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ml_recon.v10_recoverability_foundry.training.physical_supervision import (
    JOINT_COVARIANCE_NLL_CONVENTION,
    joint_covariance_gaussian_nll,
)

from .belief_geometry import (
    BELIEF_MEAN_SOURCE,
    Belief_validity_config,
    belief_residual_details,
    mode_marginal_variances,
)
from .physical_belief_contract import (
    derive_physical_k1_probability,
    validate_physical_belief_config,
)
from .contracts import sha256_of
from .control_coordinate_layout import (
    INTERLEAVED_CONTROL_LAYOUT_SHA,
    validate_interleaved_control_covariance,
)

LEGACY_CALIBRATION_ADMISSION_SCHEMA = "V10CalibrationAdmissionReceiptV3"
LEGACY_CALIBRATION_REPORT_SCHEMA = "V10HeldOutCalibrationReportV5"
LEGACY_CALIBRATION_EVIDENCE_SCHEMA = "V10HeldOutCalibrationEvidenceV5"
CALIBRATION_ADMISSION_SCHEMA = "V10CalibrationAdmissionReceiptV4"
CALIBRATION_REPORT_SCHEMA = "V10HeldOutCalibrationReportV6"
CALIBRATION_EVIDENCE_SCHEMA = "V10HeldOutCalibrationEvidenceV6"
REQUIRED_BINDING_KEYS = (
    "source_commit",
    "source_tree_sha",
    "classifier_checkpoint_sha",
    "model_state_sha",
    "config_sha",
    "target_contract_sha",
    "generator_sha",
    "data_fold_sha",
    "output_registry_sha",
    "coordinate_layout_sha",
)
SCOPE_KEYS = (
    "admitted_system",
    "admitted_n_mode",
    "source_float_dtype",
    "belief_validity_contract_sha",
)
PROFILE_PROVENANCE_KEYS = (
    "profile_contract_sha",
    "profile_calibration_source_sha",
)
PROFILE_FILE_KEYS = (
    "profile_contract_path",
    "profile_contract_file_sha",
)
SYSTEM_DIMENSIONS = {"3Q": (3, 4), "4Q": (4, 6)}
STATE_HEADS = ("R1_re", "R1_im", "W_populations", "W_coherence_re", "W_coherence_im")
VARIANCE_HEADS = ("W_amplitude_variance", "W_phase_variance")
ROW_KEYS = ("held_out_cell_ids", "seed_group_ids", "evaluation_group_ids", "system_ids")
MEMBERSHIP_SCHEMA = "V10CalibrationMembershipV2"
LEGACY_REQUIRED_PROTOCOL_METRICS = (
    "R1_frobenius_mse",
    "W_population_brier",
    "W_coherence_frobenius_mse",
    "W_amplitude_variance.mean_nll",
    "W_amplitude_variance.coverage_95",
    "W_phase_variance.mean_nll",
    "W_phase_variance.coverage_95",
    "joint_control_covariance.mean_nll",
    "joint_control_covariance.coverage_95",
)
REQUIRED_PROTOCOL_METRICS = (
    *LEGACY_REQUIRED_PROTOCOL_METRICS[:2],
    "derived_K1_brier",
    *LEGACY_REQUIRED_PROTOCOL_METRICS[2:],
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, name: str, length: int = 64) -> None:
    if (
        type(value) is not str
        or len(value) != length
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a canonical {length}-character digest")


def _profile_provenance(document, *, label):
    """Return optional profiled-estimand provenance; raw artifacts carry none."""
    values = tuple(document.get(name) for name in PROFILE_PROVENANCE_KEYS)
    if values == (None, None):
        return values
    if any(value is None for value in values):
        raise ValueError(f"{label} profile provenance must be complete or absent")
    for name, value in zip(PROFILE_PROVENANCE_KEYS, values):
        _sha(value, name)
    return values


def _validate_profile_provenance(expected_bindings, document, *, label):
    """Bind profile identity and its calibration source through every artifact."""
    expected_contract = expected_bindings.get("profile_contract_sha")
    expected_source = expected_bindings.get("profile_calibration_source_sha")
    actual = _profile_provenance(document, label=label)
    if expected_contract is None and expected_source is None:
        if actual != (None, None):
            raise ValueError(f"RAW_CSM {label} must not carry profile provenance")
        return actual
    if expected_contract is None or expected_source is None:
        raise ValueError("PROFILED_CSM bindings require contract and calibration-source SHA")
    _sha(expected_contract, "profile_contract_sha")
    _sha(expected_source, "profile_calibration_source_sha")
    if actual != (expected_contract, expected_source):
        raise ValueError(f"{label} profile provenance mismatch")
    return actual


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate calibration JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite calibration JSON constant: {value}")


def _read_json(path, expected_sha, label):
    _sha(expected_sha, label)
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"{label} bytes do not match declared SHA-256")
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} bytes cannot be read") from error
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError(f"{label} bytes do not match declared SHA-256")
    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if type(result) is not dict:
        raise ValueError(f"{label} must be an object")
    return result


def _contained_relative_file(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} requires a contained relative path")
    relative_path = Path(relative)
    root_path = root.resolve(strict=True)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label} path escapes the calibration root")
    candidate = (root_path / relative_path).resolve(strict=True)
    if not candidate.is_file() or not candidate.is_relative_to(root_path):
        raise ValueError(f"{label} path escapes the calibration root")
    return candidate


def _load_profile_contract(
    root: Path,
    document: Mapping[str, object],
    expected_bindings: Mapping[str, object],
    *,
    label: str,
):
    """Rebuild and byte-verify the exact V2 profile contract and its source."""
    from .csm_profile_contract import Profiled_csm_contract

    profile = _validate_profile_provenance(
        expected_bindings, document, label=label
    )
    file_values = tuple(document.get(name) for name in PROFILE_FILE_KEYS)
    if profile == (None, None):
        if file_values != (None, None):
            raise ValueError(f"RAW_CSM {label} must not carry a profile contract file")
        return None
    expected_path = expected_bindings.get("profile_contract_path")
    expected_file_sha = expected_bindings.get("profile_contract_file_sha")
    if not isinstance(expected_path, str) or not expected_path:
        raise ValueError("PROFILED_CSM bindings require a profile contract path")
    _sha(expected_file_sha, "profile_contract_file_sha")
    if file_values != (expected_path, expected_file_sha):
        raise ValueError(f"{label} profile contract file binding mismatch")
    contract_path = _contained_relative_file(root, expected_path, label + " profile contract")
    contract_document = _read_json(
        contract_path, expected_file_sha, label + " profile contract"
    )
    contract = Profiled_csm_contract.from_mapping(contract_document)
    if (
        contract.sha256() != profile[0]
        or contract.calibration_source_sha != profile[1]
    ):
        raise ValueError(f"{label} profile contract content identity mismatch")
    calibration_source_path = contract.verify_calibration_source(root)
    if calibration_source_path == contract_path:
        raise ValueError("profile contract and calibration source must be distinct files")
    return contract


def _array(value, name, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain numeric values, not booleans or strings")
    result = raw.astype(np.float64)
    if (
        not result.size
        or not np.isfinite(result).all()
        or shape is not None
        and result.shape != shape
    ):
        raise ValueError(f"{name} must be finite with exact shape {shape}")
    return result


def _mask(value, shape):
    raw = np.asarray(value)
    if raw.dtype != np.bool_ or raw.shape != shape or not raw.all():
        raise ValueError(
            "required calibration masks must be strict bool and fully available"
        )


def _strings(values, label, *, nonempty=True):
    if (
        type(values) is not list
        or nonempty
        and not values
        or any(
            type(item) is not str or not item or item.strip() != item for item in values
        )
    ):
        raise ValueError(f"{label} must be a list of nonempty identities")
    return tuple(values)


def _read_membership(root, relative, expected_sha, label):
    candidate = _contained_relative_file(Path(root), relative, label)
    document = _read_json(candidate, expected_sha, label)
    if set(document) != {"schema", "row_ids", "acquisition_group_ids"} or document["schema"] != MEMBERSHIP_SCHEMA:
        raise ValueError(f"{label} has an invalid membership schema")
    rows = _strings(document["row_ids"], label + ".row_ids")
    groups = _strings(document["acquisition_group_ids"], label + ".acquisition_group_ids")
    if (
        len(rows) != len(groups)
        or len(set(rows)) != len(rows)
        or rows != tuple(sorted(rows))
    ):
        raise ValueError(
            f"{label} requires sorted unique row IDs paired one-to-one with groups"
        )
    return rows, groups


def _head(entry, role, rows, system, signature):
    if not isinstance(entry, Mapping):
        raise ValueError("every required output needs raw held-out evidence")
    data_keys = (
        {"prediction", "target"}
        if role == "state"
        else {"variance", "mean_source"}
        if role == "variance"
        else {"covariance", "mean_source"}
    )
    if set(entry) != set(ROW_KEYS) | {"target_mask"} | data_keys:
        raise ValueError(
            "head fields must match the actual consumer role; independent surrogate means are forbidden"
        )
    current = tuple(_strings(entry[name], name) for name in ROW_KEYS)
    if (
        any(len(values) != rows for values in current)
        or set(current[-1]) != {system}
        or signature is not None
        and current != signature
    ):
        raise ValueError(
            "all outputs must bind identical rows, groups, seeds and system"
        )
    if len(set(current[0])) != rows:
        raise ValueError("held-out row identities must be unique")
    if role != "state" and entry["mean_source"] != BELIEF_MEAN_SOURCE:
        raise ValueError(
            "uncertainty is centred on the actual registered R1 point belief"
        )
    return current


def _aggregate_sector_head(entry, rows, system, signature):
    """Read physical [rest,K1] evidence with the common held-out identity."""
    current = _head(entry, "state", rows, system, signature)
    prediction = _array(
        entry["prediction"], "sector_aggregate_posterior.prediction", (rows, 2)
    )
    target = _array(
        entry["target"], "sector_aggregate_posterior.target", (rows, 2)
    )
    _mask(entry["target_mask"], (rows, 2))
    return current, prediction, target


def _config_and_scope(evidence, *, physical_route):
    system, n_mode = evidence.get("admitted_system"), evidence.get("admitted_n_mode")
    if (
        system not in SYSTEM_DIMENSIONS
        or type(n_mode) is not int
        or n_mode != SYSTEM_DIMENSIONS[system][0]
    ):
        raise ValueError("exact calibration system and N scope are required")
    config_values = evidence.get("belief_validity_config")
    if type(config_values) is not dict or set(config_values) != set(
        asdict(Belief_validity_config())
    ):
        raise ValueError(
            "calibration must include the full registered belief validity configuration"
        )
    config = Belief_validity_config(**config_values)
    if physical_route:
        validate_physical_belief_config(config)
    else:
        config.validate()
    if evidence.get("belief_validity_contract_sha") != config.sha256():
        raise ValueError("belief validity SHA does not bind its actual configuration")
    dtype_name = evidence.get("source_float_dtype")
    if dtype_name not in {"float32", "float64"}:
        raise ValueError("calibration must declare actual prediction precision")
    dtype = torch.float32 if dtype_name == "float32" else torch.float64
    return config, system, n_mode, dtype


def _normal_metrics(residual, variance, groups):
    if residual.shape != variance.shape or np.any(variance <= 0):
        raise ValueError("positive marginal variance and residual shapes must align")
    scaled_square = residual**2 / variance
    per_row_nll = 0.5 * np.mean(np.log(2 * math.pi * variance) + scaled_square, axis=1)
    per_row_coverage = np.mean(scaled_square <= 1.959963984540054**2, axis=1)
    if not np.isfinite(per_row_nll).all():
        raise ValueError("nonfinite marginal calibration score")
    return {
        "count": int(residual.size),
        "mean_nll": float(per_row_nll.mean()),
        "coverage_95": float(per_row_coverage.mean()),
        "coverage_95_by_group": {
            group: float(per_row_coverage[np.asarray(groups) == group].mean())
            for group in sorted(set(groups))
        },
    }


def evaluate_held_out_calibration(evidence, required_outputs=None):
    from .published_output_registry import (
        legacy_feedback_required_outputs,
        physical_feedback_required_outputs,
    )

    required = (
        tuple(required_outputs)
        if required_outputs is not None
        else legacy_feedback_required_outputs
    )
    physical_route = required == physical_feedback_required_outputs
    if not physical_route and required != legacy_feedback_required_outputs:
        raise ValueError("calibration cannot omit a consumer-required output")
    expected_evidence_schema = (
        CALIBRATION_EVIDENCE_SCHEMA
        if physical_route
        else LEGACY_CALIBRATION_EVIDENCE_SCHEMA
    )
    if (
        evidence.get("schema") != expected_evidence_schema
        or evidence.get("coordinate_layout_sha") != INTERLEAVED_CONTROL_LAYOUT_SHA
    ):
        raise ValueError("unsupported calibration evidence or coordinate layout")
    for name in REQUIRED_BINDING_KEYS:
        _sha(
            evidence.get(name),
            name,
            40 if name in {"source_commit", "source_tree_sha"} else 64,
        )
    profile_contract_sha, profile_calibration_source_sha = _profile_provenance(
        evidence, label="raw calibration evidence"
    )
    config, system, n_mode, dtype = _config_and_scope(
        evidence, physical_route=physical_route
    )
    if physical_route:
        if "prediction_confidence" in evidence:
            raise ValueError("physical calibration rejects unregistered prediction_confidence")
        if "prediction_sector_posterior" in evidence:
            raise ValueError("physical calibration rejects legacy prediction_sector_posterior")
    heads = evidence.get("heads")
    if not isinstance(heads, Mapping) or set(heads) != set(required):
        raise ValueError("raw evidence must contain exactly every required output")
    predicted_real = _array(heads["R1_re"].get("prediction"), "R1_re")
    if predicted_real.ndim != 3 or predicted_real.shape[1:] != (n_mode, n_mode):
        raise ValueError("R1 evidence must have actual [rows,N,N] consumer shape")
    rows = len(predicted_real)
    signature = None
    matrices = {}
    for name in STATE_HEADS:
        entry = heads[name]
        signature = _head(entry, "state", rows, system, signature)
        shape = (rows, n_mode) if name == "W_populations" else (rows, n_mode, n_mode)
        prediction = _array(entry["prediction"], name + ".prediction", shape)
        target = _array(entry["target"], name + ".target", shape)
        _mask(entry["target_mask"], shape)
        matrices[name] = (prediction, target)
    training_rows = set(
        _strings(evidence.get("training_row_ids"), "training_row_ids")
    )
    training_groups = set(
        _strings(evidence.get("training_group_ids"), "training_group_ids")
    )
    if training_rows.intersection(signature[0]) or training_groups.intersection(
        signature[2]
    ):
        raise ValueError(
            "calibration rows and acquisition groups must be disjoint from training"
        )
    tolerance = float(torch.finfo(dtype).eps) * n_mode * 16
    for side in (0, 1):
        real, imag = matrices["R1_re"][side], matrices["R1_im"][side]
        diagonal = np.diagonal(real, axis1=1, axis2=2)
        trace = diagonal.sum(-1)
        if np.any(trace <= 0):
            raise ValueError("R1 trace must be positive")
        population = diagonal / trace[:, None]
        scale = np.sqrt(
            np.clip(diagonal[:, :, None] * diagonal[:, None, :], 0, None)
        ).clip(1e-12)
        coherence = (real + 1j * imag) / scale
        if not np.allclose(
            matrices["W_populations"][side], population, atol=tolerance, rtol=0
        ) or not np.allclose(
            matrices["W_coherence_re"][side] + 1j * matrices["W_coherence_im"][side],
            coherence,
            atol=tolerance,
            rtol=0,
        ):
            raise ValueError(
                "population and coherence heads must be derived from the same subnormalized R1"
            )

    def as_tensor(value):
        return torch.as_tensor(value, dtype=dtype)

    if physical_route:
        aggregate_entry = heads.get("sector_aggregate_posterior")
        if not isinstance(aggregate_entry, Mapping):
            raise ValueError("physical calibration requires sector_aggregate_posterior evidence")
        _, aggregate_prediction, aggregate_target = _aggregate_sector_head(
            aggregate_entry, rows, system, signature
        )
        predicted_k1 = derive_physical_k1_probability(
            as_tensor(matrices["R1_re"][0]), as_tensor(aggregate_prediction)
        )
        target_k1 = derive_physical_k1_probability(
            as_tensor(matrices["R1_re"][1]), as_tensor(aggregate_target)
        )
        details = belief_residual_details(
            as_tensor(matrices["R1_re"][0]),
            as_tensor(matrices["R1_im"][0]),
            as_tensor(matrices["R1_re"][1]),
            as_tensor(matrices["R1_im"][1]),
            torch.ones((rows, n_mode), dtype=torch.bool),
            config=config,
            sector_k1_probability=predicted_k1,
        )
    else:
        confidence = evidence.get("prediction_confidence")
        sector = evidence.get("prediction_sector_posterior")
        details = belief_residual_details(
            as_tensor(matrices["R1_re"][0]),
            as_tensor(matrices["R1_im"][0]),
            as_tensor(matrices["R1_re"][1]),
            as_tensor(matrices["R1_im"][1]),
            torch.ones((rows, n_mode), dtype=torch.bool),
            config=config,
            confidence=None
            if confidence is None
            else as_tensor(_array(confidence, "prediction_confidence")),
            sector_posterior=None
            if sector is None
            else as_tensor(_array(sector, "prediction_sector_posterior")),
        )
    if not bool(details.valid_mask.all()):
        raise ValueError(
            "held-out point beliefs do not satisfy their registered full-D validity scope"
        )
    dimension = 2 * (n_mode - 1)
    joint = heads["joint_control_covariance"]
    _head(joint, "joint", rows, system, signature)
    covariance = _array(
        joint["covariance"], "joint.covariance", (rows, dimension, dimension)
    )
    _mask(joint["target_mask"], (rows, dimension))
    if not np.allclose(
        covariance, covariance.transpose(0, 2, 1), atol=tolerance, rtol=0
    ):
        raise ValueError("full-D covariance must be symmetric")
    covariance = (covariance + covariance.transpose(0, 2, 1)) / 2
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("full-D covariance must be positive definite") from error
    mode_variances = mode_marginal_variances(
        torch.as_tensor(covariance, dtype=dtype), n_mode
    )
    mode_residual = details.mode_residual.detach().cpu().double().numpy()
    metrics = {
        "R1_frobenius_mse": float(
            np.mean(
                np.sum(
                    (matrices["R1_re"][0] - matrices["R1_re"][1]) ** 2
                    + (matrices["R1_im"][0] - matrices["R1_im"][1]) ** 2,
                    axis=(1, 2),
                )
            )
        ),
        "W_population_brier": float(
            np.mean(
                np.sum(
                    (matrices["W_populations"][0] - matrices["W_populations"][1]) ** 2,
                    axis=1,
                )
            )
        ),
        "W_coherence_frobenius_mse": float(
            np.mean(
                np.sum(
                    (matrices["W_coherence_re"][0] - matrices["W_coherence_re"][1]) ** 2
                    + (matrices["W_coherence_im"][0] - matrices["W_coherence_im"][1])
                    ** 2,
                    axis=(1, 2),
                )
            )
        ),
    }
    if physical_route:
        metrics["derived_K1_brier"] = float(
            np.mean(
                (
                    predicted_k1.detach().cpu().double().numpy()
                    - target_k1.detach().cpu().double().numpy()
                )
                ** 2
            )
        )
    for index, name in enumerate(VARIANCE_HEADS):
        entry = heads[name]
        _head(entry, "variance", rows, system, signature)
        variance = _array(entry["variance"], name + ".variance", (rows, n_mode))
        _mask(entry["target_mask"], (rows, n_mode))
        expected = mode_variances[index].detach().cpu().double().numpy()
        if not np.allclose(variance, expected, atol=tolerance, rtol=0):
            raise ValueError(
                "published mode variances must be marginals of the same lifted full-D covariance"
            )
        metrics[name] = _normal_metrics(
            mode_residual[..., index], variance, signature[2]
        )
    residual = details.control_residual.detach().cpu().double().numpy()
    validate_interleaved_control_covariance(
        details.control_residual,
        torch.as_tensor(covariance, dtype=dtype),
        layout_sha=evidence["coordinate_layout_sha"],
    )
    residual_tensor = torch.as_tensor(residual, dtype=dtype)
    covariance_tensor = torch.as_tensor(covariance, dtype=dtype)
    nll = joint_covariance_gaussian_nll(
        residual_tensor, covariance_tensor, torch.ones(rows, dtype=torch.bool)
    )
    cholesky = np.linalg.cholesky(covariance)
    standardized = np.linalg.solve(cholesky, residual[..., None])[..., 0]
    mahalanobis = np.sum(standardized**2, axis=-1)
    covered = mahalanobis <= {4: 9.487729036781154, 6: 12.591587243743977}[dimension]
    if not bool(torch.isfinite(nll)):
        raise ValueError("nonfinite full-D calibration score")
    metrics["joint_control_covariance"] = {
        "count": rows,
        "dimension": dimension,
        "nll_convention": JOINT_COVARIANCE_NLL_CONVENTION,
        "mean_nll": float(nll.detach().cpu()),
        "coverage_95": float(covered.mean()),
        "coverage_95_by_group": {
            group: float(covered[np.asarray(signature[2]) == group].mean())
            for group in sorted(set(signature[2]))
        },
        "min_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
    }
    return {
        "schema": CALIBRATION_REPORT_SCHEMA
        if physical_route
        else LEGACY_CALIBRATION_REPORT_SCHEMA,
        **{name: evidence[name] for name in REQUIRED_BINDING_KEYS + SCOPE_KEYS},
        "heads": metrics,
        "row_count": rows,
        "independent_group_count": len(set(signature[2])),
        "held_out_rows_sha": sha256_of(signature[0]),
        "held_out_groups_sha": sha256_of(signature[2]),
        "status": "MEASURED_NOT_ADMITTED",
        "profile_contract_sha": profile_contract_sha,
        "profile_calibration_source_sha": profile_calibration_source_sha,
    }


def _timestamp(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("calibration timestamps must be ISO values") from error
    if result.tzinfo is None:
        raise ValueError("calibration timestamps need explicit timezones")
    return result


def _validate_protocol(protocol, evaluated, report):
    if protocol.get("schema") != "V10HeldOutCalibrationProtocolV1":
        raise ValueError("a preregistered calibration protocol is required")
    if _timestamp(protocol.get("frozen_at")) > _timestamp(
        report.get("evaluation_started_at")
    ):
        raise ValueError("calibration thresholds were not frozen before evaluation")
    for name in SCOPE_KEYS:
        if protocol.get(name) != evaluated[name]:
            raise ValueError("statistical protocol and calibrated scope disagree")
    for name, observed in (
        ("minimum_rows", evaluated["row_count"]),
        ("minimum_independent_groups", evaluated["independent_group_count"]),
    ):
        threshold = protocol.get(name)
        if type(threshold) is not int or threshold < 2 or observed < threshold:
            raise ValueError(
                "calibration evidence does not meet registered sample/group requirements"
            )
    limits = protocol.get("metric_limits")
    required_metrics = (
        REQUIRED_PROTOCOL_METRICS
        if evaluated.get("schema") == CALIBRATION_REPORT_SCHEMA
        else LEGACY_REQUIRED_PROTOCOL_METRICS
        if evaluated.get("schema") == LEGACY_CALIBRATION_REPORT_SCHEMA
        else ()
    )
    if type(limits) is not dict or set(limits) != set(required_metrics):
        raise ValueError(
            "calibration protocol must bound state error, NLL, and coverage"
        )
    for name, bound in limits.items():
        if type(bound) is not dict or not bound or set(bound) - {"minimum", "maximum"}:
            raise ValueError("invalid registered calibration metric bound")
        observed = evaluated["heads"]
        for key in name.split("."):
            observed = observed[key]
        for comparator, limit in bound.items():
            if type(limit) not in (int, float) or not math.isfinite(limit):
                raise ValueError("calibration metric bounds must be finite numbers")
            if (
                comparator == "minimum"
                and observed < limit
                or comparator == "maximum"
                and observed > limit
            ):
                raise ValueError(
                    "held-out calibration fails its registered statistical limits"
                )


def validate_calibration_admission(admission_path, expected_bindings, required_outputs):
    from .calibration_update_evidence import validate_output_update_evidence
    from .published_output_registry import (
        legacy_feedback_required_outputs,
        physical_feedback_required_outputs,
        physical_output_training_manifest_sha,
    )

    required_outputs = tuple(required_outputs)
    physical_route = required_outputs == physical_feedback_required_outputs
    if not physical_route and required_outputs != legacy_feedback_required_outputs:
        raise ValueError("every actual feedback output is required")
    expected_receipt_schema = (
        CALIBRATION_ADMISSION_SCHEMA
        if physical_route
        else LEGACY_CALIBRATION_ADMISSION_SCHEMA
    )
    if (
        physical_route
        and expected_bindings.get("output_registry_sha")
        != physical_output_training_manifest_sha()
    ):
        raise ValueError(
            "physical calibration output_registry_sha does not bind the current manifest"
        )
    receipt_sha = sha256_file(admission_path)
    receipt = _read_json(admission_path, receipt_sha, "calibration admission receipt")
    if (
        receipt.get("schema") != expected_receipt_schema
        or tuple(receipt.get("required_outputs", ())) != required_outputs
    ):
        raise ValueError("unsupported calibration receipt or consumer set")
    for name in REQUIRED_BINDING_KEYS:
        expected = expected_bindings.get(name)
        _sha(expected, name, 40 if name in {"source_commit", "source_tree_sha"} else 64)
        if receipt.get(name) != expected:
            raise ValueError(f"calibration source binding mismatch for {name}")
    root = Path(admission_path).resolve().parent
    receipt_contract = _load_profile_contract(
        root,
        receipt,
        expected_bindings,
        label="calibration admission receipt",
    )
    receipt_profile = _profile_provenance(
        receipt, label="calibration admission receipt"
    )
    report = _read_json(
        _contained_relative_file(
            root,
            receipt.get("evaluation_report_path", ""),
            "calibration report",
        ),
        receipt.get("evaluation_report_sha"),
        "calibration report",
    )
    evidence = _read_json(
        _contained_relative_file(
            root,
            report.get("raw_evidence_path", ""),
            "raw calibration evidence",
        ),
        report.get("raw_evidence_sha"),
        "raw calibration evidence",
    )
    membership_keys = (
        "fit_membership_path", "fit_membership_sha",
        "held_out_membership_path", "held_out_membership_sha",
    )
    if any(key not in evidence for key in membership_keys):
        raise ValueError("calibration evidence requires file-backed fit and held-out memberships")
    fit_rows, fit_groups = _read_membership(
        root, evidence["fit_membership_path"], evidence["fit_membership_sha"], "fit membership"
    )
    held_rows, held_groups = _read_membership(
        root, evidence["held_out_membership_path"], evidence["held_out_membership_sha"], "held-out membership"
    )
    if set(fit_rows) & set(held_rows) or set(fit_groups) & set(held_groups):
        raise ValueError("fit and held-out calibration memberships overlap")
    heads = evidence.get("heads")
    reference_head = (
        heads.get(required_outputs[0]) if isinstance(heads, Mapping) else None
    )
    if not isinstance(reference_head, Mapping):
        raise ValueError("calibration evidence lacks the held-out membership signature")
    if (
        fit_rows
        != _strings(evidence.get("training_row_ids"), "training_row_ids")
        or fit_groups
        != _strings(evidence.get("training_group_ids"), "training_group_ids")
        or held_rows
        != _strings(reference_head.get("held_out_cell_ids"), "held_out_cell_ids")
        or held_groups
        != _strings(
            reference_head.get("evaluation_group_ids"), "evaluation_group_ids"
        )
    ):
        raise ValueError(
            "file-backed memberships differ from the raw fit/held-out row signature"
        )
    if any(receipt.get(key) != evidence[key] for key in membership_keys) or any(
        report.get(key) != evidence[key] for key in membership_keys
    ):
        raise ValueError("receipt and report must bind the exact calibration memberships")
    evaluated = evaluate_held_out_calibration(evidence, required_outputs)
    if any(report.get(name) != value for name, value in evaluated.items()):
        raise ValueError(
            "calibration report is not an exact raw-evidence recomputation"
        )
    for name in REQUIRED_BINDING_KEYS:
        if evaluated[name] != expected_bindings[name]:
            raise ValueError("raw calibration source binding mismatch")
    evidence_contract = _load_profile_contract(
        root, evidence, expected_bindings, label="raw calibration evidence"
    )
    report_contract = _load_profile_contract(
        root, report, expected_bindings, label="calibration report"
    )
    evidence_profile = _profile_provenance(
        evidence, label="raw calibration evidence"
    )
    report_profile = _profile_provenance(report, label="calibration report")
    if receipt_profile != evidence_profile or report_profile != evidence_profile:
        raise ValueError("profile provenance must agree across receipt/report/evidence")
    if any(
        contract != receipt_contract
        for contract in (evidence_contract, report_contract)
    ):
        raise ValueError("profile contract content differs across admission artifacts")
    if any(receipt.get(name) != evaluated[name] for name in SCOPE_KEYS):
        raise ValueError(
            "receipt cannot widen raw calibration system, precision, or validity scope"
        )
    protocol = _read_json(
        _contained_relative_file(
            root,
            report.get("calibration_protocol_path", ""),
            "calibration protocol",
        ),
        report.get("calibration_protocol_sha"),
        "calibration protocol",
    )
    _validate_protocol(protocol, evaluated, report)
    update_summary = validate_output_update_evidence(
        _contained_relative_file(
            root,
            report.get("optimizer_audit_path", ""),
            "optimizer audit",
        ),
        report.get("optimizer_audit_sha"),
        expected_bindings,
        required_outputs,
        expected_fit_row_ids=fit_rows,
        expected_fit_group_ids=fit_groups,
        require_physical_objective_coverage=physical_route,
    )
    if report.get("optimizer_update_summary") != update_summary:
        raise ValueError(
            "optimizer summary must be recomputed from the raw bound update audit"
        )
    review = _read_json(
        _contained_relative_file(
            root,
            receipt.get("external_review_path", ""),
            "external calibration review",
        ),
        receipt.get("external_review_sha"),
        "external calibration review",
    )
    if (
        review.get("schema") != "V10CalibrationExternalReviewV3"
        or review.get("decision") != "ACCEPTED"
        or review.get("reviewed_report_sha") != receipt.get("evaluation_report_sha")
        or type(review.get("reviewer_id")) is not str
        or not review["reviewer_id"]
    ):
        raise ValueError("exact calibration report lacks external acceptance")
    if any(
        review.get(name) != receipt[name] for name in REQUIRED_BINDING_KEYS + SCOPE_KEYS
    ):
        raise ValueError("external calibration review source/scope mismatch")
    if any(review.get(key) != evidence[key] for key in membership_keys):
        raise ValueError("external calibration review membership binding mismatch")
    review_contract = _load_profile_contract(
        root, review, expected_bindings, label="external calibration review"
    )
    review_profile = _profile_provenance(
        review, label="external calibration review"
    )
    if review_profile != receipt_profile:
        raise ValueError("external calibration review profile provenance mismatch")
    if review_contract != receipt_contract:
        raise ValueError("external calibration review profile contract differs")
    return {
        "accepted_receipt_sha": receipt_sha,
        "required_outputs": required_outputs,
        **{name: evaluated[name] for name in SCOPE_KEYS},
        "calibration_protocol_sha": report["calibration_protocol_sha"],
        "optimizer_audit_sha": report["optimizer_audit_sha"],
        "fit_membership_sha": evidence["fit_membership_sha"],
        "held_out_membership_sha": evidence["held_out_membership_sha"],
        "profile_contract_sha": receipt_profile[0],
        "profile_calibration_source_sha": receipt_profile[1],
        "profile_contract_path": receipt.get("profile_contract_path"),
        "profile_contract_file_sha": receipt.get("profile_contract_file_sha"),
    }
