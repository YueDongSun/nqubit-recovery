"""Fidelity and rho-estimator helpers for Data UI density-matrix variants."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


VARIANT_LABELS = (
    ("raw", "raw"),
    ("idler", "idler"),
    ("retrieval", "retr"),
    ("coin_signal", "coin/signal"),
    ("retrieval_sqrt", "retr_sqrt"),
    ("shifted_idler", "shift_idler"),
)

ESTIMATOR_METRICS = (
    ("F_target_amp", "F amp"),
    ("ood_target_amp", "OOD amp"),
    ("F_target_sqrt_amp", "F sqrt"),
    ("ood_target_sqrt_amp", "OOD sqrt"),
    ("F_target_amp_squared", "F amp^2"),
    ("ood_target_amp_squared", "OOD amp^2"),
    ("F_target_sinc_amp", "F sin/amp"),
    ("ood_target_sinc_amp", "OOD sin/amp"),
    ("F_depolarized_corrected", "F depol"),
    ("depolarized_p", "p signal"),
    ("F_phase_opt_target_amp", "F phase target"),
    ("F_phase_opt_balanced", "F phase balanced"),
    ("F_best_W_subspace", "F best W"),
    ("purity", "Purity"),
    ("core_purity", "Core purity"),
    ("ood", "OOD"),
    ("multi_click_leakage", "Multi leak"),
    ("noise_floor", "Noise floor"),
)

TARGET_AMP_VARIANTS = (
    ("amp", "amp"),
    ("sqrt_amp", "sqrt_amp"),
    ("amp_squared", "amp_squared"),
    ("sinc_amp", "sinc_amp"),
)

_EPS = 1e-12
_DRIFT_ERROR_LABELS = {
    "phase_drift",
    "amp_imbalance",
    "amplitude_drift",
    "retrieval_imbalance",
    "retrieval_drift",
}


def w_state_indices(num_qubits: int) -> list[int]:
    num_qubits = int(num_qubits)
    return [2 ** (num_qubits - 1 - qubit) for qubit in range(num_qubits)]


def ideal_w_state_vector(num_qubits: int) -> np.ndarray:
    dim = 2 ** int(num_qubits)
    psi = np.zeros(dim, dtype=complex)
    amp = 1.0 / np.sqrt(float(num_qubits))
    for index in w_state_indices(num_qubits):
        psi[index] = amp
    return psi


def target_w_state_vector(state_params: Mapping[str, Any], num_qubits: int) -> np.ndarray:
    """Build the target W-state vector from UI/main automation state params."""

    num_qubits = int(num_qubits)
    amps, phases = _state_params_amps_phases(state_params, num_qubits)
    return _w_state_vector_from_coefficients(amps, phases, num_qubits)


def target_w_state_vector_for_amp_variant(
    state_params: Mapping[str, Any],
    num_qubits: int,
    variant_key: str,
) -> np.ndarray:
    amps, phases = _state_params_amps_phases(state_params, num_qubits)
    coeffs = _transform_target_amplitudes(amps, variant_key)
    return _w_state_vector_from_coefficients(coeffs, phases, num_qubits)


def _w_state_vector_from_coefficients(coefficients: Any, phases: Any, num_qubits: int) -> np.ndarray:
    num_qubits = int(num_qubits)
    dim = 2 ** num_qubits
    psi = np.zeros(dim, dtype=complex)
    amps = _normalized_coefficients(coefficients, num_qubits)
    phases = _coerce_phase_array(phases, num_qubits)
    for qubit, index in enumerate(w_state_indices(num_qubits)):
        psi[index] = amps[qubit] * np.exp(1j * phases[qubit])
    return psi


def target_density_matrix(state_params: Mapping[str, Any], num_qubits: int) -> np.ndarray:
    psi = target_w_state_vector(state_params, num_qubits)
    return np.outer(psi, psi.conj())


def density_fidelity_to_ideal_w(rho: np.ndarray, num_qubits: int) -> float:
    rho = np.asarray(rho, dtype=complex)
    psi = ideal_w_state_vector(num_qubits)
    fidelity = np.vdot(psi, rho @ psi)
    return float(np.clip(np.real(fidelity), 0.0, 1.0))


def density_fidelity_to_target(rho: np.ndarray, rho_target: np.ndarray) -> float:
    rho = np.asarray(rho, dtype=complex)
    rho_target = np.asarray(rho_target, dtype=complex)
    fidelity = np.trace(rho_target @ rho)
    return float(np.clip(np.real(fidelity), 0.0, 1.0))


def density_purity(rho: np.ndarray) -> float:
    rho = np.asarray(rho, dtype=complex)
    return float(np.clip(np.real(np.trace(rho @ rho)), 0.0, 1.0))


def core_density_matrix(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    indices = w_state_indices(num_qubits)
    return rho[np.ix_(indices, indices)]


def core_trace(rho: np.ndarray, num_qubits: int) -> float:
    return float(np.clip(np.real(np.trace(core_density_matrix(rho, num_qubits))), 0.0, 1.0))


def normalized_core_density(rho: np.ndarray, num_qubits: int) -> tuple[np.ndarray, float]:
    core = core_density_matrix(rho, num_qubits)
    trace = float(np.real(np.trace(core)))
    if trace <= _EPS:
        return core, 0.0
    return core / trace, float(np.clip(trace, 0.0, 1.0))


def purity_limited_fidelity_ceiling(purity: float, dim: int) -> float:
    """Largest possible pure-state overlap for a density matrix with purity P."""

    dim = int(dim)
    if dim <= 1:
        return 1.0
    lower = 1.0 / float(dim)
    purity = float(np.clip(purity, lower, 1.0))
    radicand = max(0.0, (dim - 1.0) * (dim * purity - 1.0))
    return float(np.clip((1.0 + np.sqrt(radicand)) / float(dim), lower, 1.0))


def depolarized_corrected_fidelity(
    *,
    fidelity: float,
    core_purity: float,
    core_dim: int,
    core_population: float = 1.0,
) -> dict[str, Any]:
    """Estimate target fidelity after removing a uniform W-core noise floor.

    This is a diagnostic estimator for the model
    rho_core = p * rho_signal + (1 - p) * I / d_core.  It is not a replacement
    for the raw target overlap.
    """

    core_dim = max(1, int(core_dim))
    noise_floor = 1.0 / float(core_dim)
    core_purity = float(core_purity)
    core_population = float(np.clip(core_population, 0.0, 1.0))
    if core_purity <= noise_floor + _EPS:
        return {
            "fidelity": 0.0,
            "p": 0.0,
            "noise_floor": noise_floor,
            "low_confidence": True,
        }
    p = np.sqrt((core_purity - noise_floor) / max(_EPS, 1.0 - noise_floor))
    p = float(np.clip(p, 0.0, 1.0))
    if p <= _EPS:
        corrected = 0.0
        low_confidence = True
    else:
        noise_contribution = core_population * (1.0 - p) * noise_floor
        corrected = (float(fidelity) - noise_contribution) / p
        corrected = float(np.clip(corrected, 0.0, 1.0))
        low_confidence = p < 0.05
    return {
        "fidelity": corrected,
        "p": p,
        "noise_floor": noise_floor,
        "low_confidence": low_confidence,
    }


def phase_optimized_w_overlap(
    rho: np.ndarray,
    amplitudes: Any,
    num_qubits: int,
    *,
    phase_hint: Optional[Any] = None,
) -> float:
    """Optimize only W-core phases for fixed amplitudes and return overlap."""

    num_qubits = int(num_qubits)
    core = core_density_matrix(rho, num_qubits)
    core = 0.5 * (core + core.conj().T)
    amps = _normalized_amplitudes(amplitudes, num_qubits)
    starts = [np.zeros(num_qubits, dtype=float)]
    if phase_hint is not None:
        starts.append(_coerce_phase_array(phase_hint, num_qubits))
    try:
        eigvals, eigvecs = np.linalg.eigh(core)
        vec = eigvecs[:, int(np.argmax(np.real(eigvals)))]
        starts.append(np.angle(vec))
    except np.linalg.LinAlgError:
        pass

    best = 0.0
    for start in starts:
        phases = np.asarray(start, dtype=float).copy()
        for _ in range(80):
            changed = False
            vector = amps * np.exp(1j * phases)
            for index in range(num_qubits):
                coupling = np.dot(core[index, :], vector) - core[index, index] * vector[index]
                if abs(coupling) > _EPS:
                    new_phase = float(np.angle(coupling))
                    changed = changed or abs(np.angle(np.exp(1j * (new_phase - phases[index])))) > 1e-10
                    phases[index] = new_phase
                    vector[index] = amps[index] * np.exp(1j * new_phase)
            if not changed:
                break
        vector = amps * np.exp(1j * phases)
        overlap = float(np.real(np.vdot(vector, core @ vector)))
        best = max(best, overlap)
    return float(np.clip(best, 0.0, 1.0))


def best_w_subspace_overlap(rho: np.ndarray, num_qubits: int) -> float:
    core = core_density_matrix(rho, num_qubits)
    core = 0.5 * (core + core.conj().T)
    try:
        values = np.linalg.eigvalsh(core)
    except np.linalg.LinAlgError:
        return 0.0
    return float(np.clip(np.max(np.real(values)), 0.0, 1.0))


def estimate_multi_click_leakage(data: Optional[Mapping[str, Any]]) -> float:
    """Estimate leakage out of the single-excitation manifold from multi-clicks."""

    if not data:
        return 0.0

    multi2 = _float_value(data.get("multi2_count"))
    multi3 = _float_value(data.get("multi3_count"))
    multi4 = _float_value(data.get("multi4_count"))
    if any(value > 0.0 for value in (multi2, multi3, multi4)):
        weighted_multi = multi2 + 2.0 * multi3 + 3.0 * multi4
    else:
        weighted_multi = _float_value(data.get("multi_click"))

    denominator = (
        _float_value(data.get("read_count"))
        or _float_value(data.get("write_count"))
        or _float_value(data.get("signal_interval"))
        or _safe_sum(data.get("signal_bins"))
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(weighted_multi / denominator, 0.0, 1.0))


def adjusted_ideal_fidelity_ceiling(
    *,
    purity: float,
    dim: int,
    ood_probability: Optional[float],
    multi_click_leakage: float,
) -> float:
    """Estimate an experiment-realistic ideal-F ceiling."""

    ceiling = purity_limited_fidelity_ceiling(purity, dim)
    if ood_probability is not None:
        ceiling *= 1.0 - float(np.clip(ood_probability, 0.0, 1.0))
    ceiling *= 1.0 - float(np.clip(multi_click_leakage, 0.0, 1.0))
    return float(np.clip(ceiling, 0.0, 1.0))


def summarize_rho_variants(
    rho_variants: Mapping[str, Any],
    *,
    num_qubits: int,
    data: Optional[Mapping[str, Any]] = None,
    ml_interface: Any = None,
    target_state_params: Optional[Mapping[str, Any]] = None,
    rho_target: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    num_qubits = int(num_qubits)
    multi_click_leakage = estimate_multi_click_leakage(data)
    dim = 2 ** num_qubits
    core_dim = num_qubits
    variants: dict[str, dict[str, Any]] = {}
    target_info = _resolve_target_info(
        num_qubits=num_qubits,
        target_state_params=target_state_params,
        rho_target=rho_target,
    )
    target_rho = target_info["rho_target"]
    target_amps = target_info["target_amplitudes"]
    target_input_amps = target_info["target_input_amplitudes"]
    target_phases = target_info["target_phases_rad"]
    target_variants = _target_amp_variant_infos(target_input_amps, target_phases, num_qubits)

    for key, _label in VARIANT_LABELS:
        rho = rho_variants.get(key)
        if rho is None:
            continue
        rho = np.asarray(rho, dtype=complex)
        analysis = _analyze_with_fallback(rho, target_rho, num_qubits)
        target_results: dict[str, dict[str, Any]] = {}
        for target_key, info in target_variants.items():
            variant_rho_target = info["rho_target"]
            variant_diagnosis = _diagnose(ml_interface, rho, variant_rho_target)
            target_results[target_key] = {
                "fidelity": density_fidelity_to_target(rho, variant_rho_target),
                "diagnosis": variant_diagnosis,
                "ood": (
                    _optional_float(variant_diagnosis.get("unknown_probability"))
                    if variant_diagnosis.get("available")
                    else None
                ),
                "model_fidelity": _optional_float(variant_diagnosis.get("fidelity_estimate")),
                "model_purity": _optional_float(variant_diagnosis.get("purity_estimate")),
            }
        primary_target = target_results["amp"]
        diagnosis = primary_target["diagnosis"]
        ood = primary_target["ood"]
        fidelity = float(analysis["fidelity"])
        purity = float(analysis["purity"])
        core_purity = float(analysis["core_purity"])
        core_pop = float(analysis["core_population"])
        depol = depolarized_corrected_fidelity(
            fidelity=fidelity,
            core_purity=core_purity,
            core_dim=core_dim,
            core_population=core_pop,
        )
        values = {
            "fidelity": fidelity,
            "F_target_phase": fidelity,
            "F_target_amp": float(primary_target["fidelity"]),
            "F_depolarized_corrected": float(depol["fidelity"]),
            "depolarized_p": float(depol["p"]),
            "depolarized_low_confidence": bool(depol["low_confidence"]),
            "F_phase_opt_target_amp": phase_optimized_w_overlap(
                rho,
                target_amps,
                num_qubits,
                phase_hint=target_phases,
            ),
            "F_phase_opt_balanced": phase_optimized_w_overlap(
                rho,
                np.ones(num_qubits, dtype=float),
                num_qubits,
            ),
            "F_best_W_subspace": best_w_subspace_overlap(rho, num_qubits),
            "noise_floor": float(depol["noise_floor"]),
            "multi_click_leakage": multi_click_leakage,
            "purity": purity,
            "core_purity": core_purity,
            "core_population": core_pop,
            "trace": float(analysis.get("trace", np.real(np.trace(rho)))),
            "ood": ood,
            "OOD": ood,
            "diagnosis": diagnosis,
            "target_amp_variant_results": target_results,
            "model_fidelity": primary_target["model_fidelity"],
            "model_purity": primary_target["model_purity"],
            "ideal_fidelity_ceiling": adjusted_ideal_fidelity_ceiling(
                purity=purity,
                dim=dim,
                ood_probability=ood,
                multi_click_leakage=multi_click_leakage,
            ),
        }
        for target_key, result in target_results.items():
            values[f"F_target_{target_key}"] = float(result["fidelity"])
            values[f"ood_target_{target_key}"] = result["ood"]
            values[f"OOD_target_{target_key}"] = result["ood"]
            values[f"model_fidelity_target_{target_key}"] = result["model_fidelity"]
            values[f"model_purity_target_{target_key}"] = result["model_purity"]
        variants[key] = values

    summary: dict[str, Any] = {
        "variants": variants,
        "multi_click_leakage": multi_click_leakage,
        "ideal_w_fidelity": 1.0,
        "target_source": target_info["target_source"],
        "target_state_params": target_info["target_state_params"],
        "target_description": target_info["target_description"],
        "target_amplitudes": target_amps.tolist(),
        "target_input_amplitudes": target_input_amps.tolist(),
        "target_amp_variants": {
            target_key: {
                "label": info["label"],
                "amplitudes": info["amplitudes"].tolist(),
                "description": info["description"],
            }
            for target_key, info in target_variants.items()
        },
        "target_phases_rad": target_phases.tolist(),
        "target_phases_deg": np.rad2deg(target_phases).tolist(),
        "rho_target": target_rho,
    }
    summary["model_audit"] = audit_model_consistency(summary)
    return summary


def audit_model_consistency(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Separate mode/error diagnosis from state-family classification."""

    _variant_key, metrics = primary_variant(summary)
    diagnosis = (metrics or {}).get("diagnosis", {}) or {}
    if not diagnosis.get("available"):
        return {
            "status": "unavailable",
            "message": diagnosis.get("message", "diagnosis unavailable"),
            "diagnosis_backend": diagnosis.get("backend"),
            "diagnosis_model_path": diagnosis.get("model_path"),
            "state_family_model_path": diagnosis.get("state_family_model_path"),
        }

    top_mode = _top_probability(diagnosis.get("mode_probabilities", {}))
    top_error = _top_probability(diagnosis.get("error_source_probabilities", {}))
    top_family = _top_probability(diagnosis.get("state_family_probabilities", {}))
    top_group = _top_probability(diagnosis.get("state_family_group_probabilities", {}))
    ood = _optional_float(diagnosis.get("unknown_probability")) or 0.0

    status = "ok"
    message = "mode/error and state-family are separate diagnostic heads"
    group_label = str((top_group or {}).get("label") or "").lower()
    family_label = str((top_family or {}).get("label") or "").lower()
    error_label = str((top_error or {}).get("label") or "").lower()
    error_prob = float((top_error or {}).get("probability") or 0.0)

    if ood > 0.35:
        status = "low_confidence"
        message = "mode/error low-confidence because OOD > 0.35"
    elif (group_label == "w" or family_label.startswith("w")) and error_label in _DRIFT_ERROR_LABELS:
        status = "not_contradiction"
        message = "not contradiction: W-like state with calibration drift"
    elif group_label and group_label != "w" and error_prob >= 0.7:
        status = "true_conflict"
        message = "true conflict: inspect model input/training domain"

    return {
        "status": status,
        "message": message,
        "ood": ood,
        "top_mode": top_mode,
        "top_error": top_error,
        "top_family": top_family,
        "top_group": top_group,
        "diagnosis_backend": diagnosis.get("backend"),
        "diagnosis_model_path": diagnosis.get("model_path"),
        "state_family_backend": diagnosis.get("state_family_backend"),
        "state_family_model_path": diagnosis.get("state_family_model_path"),
    }


def format_model_audit(summary: Mapping[str, Any]) -> str:
    audit = summary.get("model_audit", {}) or {}
    message = audit.get("message") or audit.get("status") or "--"
    top_error = audit.get("top_error") or {}
    top_group = audit.get("top_group") or {}
    pieces = [str(message)]
    if top_error:
        pieces.append(f"error {top_error.get('label')}: {float(top_error.get('probability') or 0.0) * 100.0:.1f}%")
    if top_group:
        pieces.append(f"group {top_group.get('label')}: {float(top_group.get('probability') or 0.0) * 100.0:.1f}%")
    return "; ".join(pieces)


def format_variant_metric(summary: Mapping[str, Any], metric: str, *, percent: bool = False) -> str:
    variants = summary.get("variants", {}) or {}
    pieces = []
    for key, label in VARIANT_LABELS:
        values = variants.get(key)
        if not values or values.get(metric) is None:
            continue
        value = float(values[metric])
        if percent:
            pieces.append(f"{label} {value * 100.0:.1f}%")
        else:
            pieces.append(f"{label} {value:.3f}")
    return " | ".join(pieces) if pieces else "--"


def top_fidelity_fits(summary: Mapping[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    """Return the highest explicitly labelled F diagnostics across rho variants."""

    limit = max(0, int(limit))
    variants = summary.get("variants", {}) or {}
    variant_labels = dict(VARIANT_LABELS)
    metric_labels = dict(ESTIMATOR_METRICS)
    candidates: list[dict[str, Any]] = []
    for variant_key, values in variants.items():
        if not isinstance(values, Mapping):
            continue
        for metric_key, value in values.items():
            if not str(metric_key).startswith("F_") or value is None:
                continue
            try:
                fidelity = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fidelity):
                continue
            candidates.append({
                "variant": str(variant_key),
                "variant_label": variant_labels.get(str(variant_key), str(variant_key)),
                "metric": str(metric_key),
                "metric_label": metric_labels.get(str(metric_key), str(metric_key)),
                "fidelity": fidelity,
            })
    candidates.sort(key=lambda item: item["fidelity"], reverse=True)
    return candidates[:limit]


def primary_variant(summary: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    variants = summary.get("variants", {}) or {}
    for key in ("idler", "retrieval", "raw", "retrieval_sqrt", "shifted_idler"):
        if key in variants:
            return key, variants[key]
    return None, None


def _resolve_target_info(
    *,
    num_qubits: int,
    target_state_params: Optional[Mapping[str, Any]],
    rho_target: Optional[np.ndarray],
) -> dict[str, Any]:
    if rho_target is not None:
        target = np.asarray(rho_target, dtype=complex)
        params = dict(target_state_params or {})
        amps, phases = _target_amps_phases_from_rho_or_params(target, params, num_qubits)
        input_amps = _target_input_amps_from_params_or_normalized(params, amps, num_qubits)
        return {
            "rho_target": target,
            "target_source": "provided_rho_target",
            "target_state_params": params,
            "target_description": _target_description(params, "provided_rho_target", amps, phases),
            "target_amplitudes": amps,
            "target_input_amplitudes": input_amps,
            "target_phases_rad": phases,
        }
    if target_state_params:
        params = dict(target_state_params)
        amps, phases = _state_params_amps_phases(params, num_qubits)
        return {
            "rho_target": target_density_matrix(params, num_qubits),
            "target_source": "feedback_context",
            "target_state_params": params,
            "target_description": _target_description(params, "feedback_context", amps, phases),
            "target_amplitudes": _normalized_amplitudes(amps, num_qubits),
            "target_input_amplitudes": amps,
            "target_phases_rad": phases,
        }
    params = {
        "name": "fallback_zero_phase_w",
        "amplitudes": [1.0] * int(num_qubits),
        "phases_deg": [0.0] * int(num_qubits),
    }
    amps, phases = _state_params_amps_phases(params, num_qubits)
    return {
        "rho_target": target_density_matrix(params, num_qubits),
        "target_source": "fallback_zero_phase_w",
        "target_state_params": params,
        "target_description": _target_description(params, "fallback_zero_phase_w", amps, phases),
        "target_amplitudes": _normalized_amplitudes(amps, num_qubits),
        "target_input_amplitudes": amps,
        "target_phases_rad": phases,
    }


def _target_description(
    state_params: Mapping[str, Any],
    source: str,
    amps: Optional[np.ndarray] = None,
    phases_rad: Optional[np.ndarray] = None,
) -> str:
    name = state_params.get("name") or source
    if amps is None or phases_rad is None:
        amps, phases_rad = _state_params_amps_phases(state_params, len(state_params.get("amplitudes", [])) or 3)
    phases_deg = np.rad2deg(phases_rad)
    amp_text = ",".join(f"{value:.3g}" for value in np.asarray(amps, dtype=float)) if amps.size else "--"
    phase_text = ",".join(f"{value:.3g}" for value in np.asarray(phases_deg, dtype=float)) if phases_deg.size else "--"
    return f"{source}: {name}; amps {amp_text}; phases_deg {phase_text}"


def _analyze_with_fallback(rho: np.ndarray, rho_target: np.ndarray, num_qubits: int) -> dict[str, float]:
    normalized_core, core_pop = normalized_core_density(rho, num_qubits)
    core_purity_value = density_purity(normalized_core) if core_pop > _EPS else 0.0
    return {
        "fidelity": density_fidelity_to_target(rho, rho_target),
        "purity": density_purity(rho),
        "core_purity": core_purity_value,
        "core_population": core_pop,
        "trace": float(np.real(np.trace(rho))),
    }


def _diagnose(ml_interface: Any, rho: np.ndarray, rho_target: np.ndarray) -> dict[str, Any]:
    if ml_interface is None:
        return {"available": False, "message": "ML interface unavailable"}
    try:
        return ml_interface.diagnose_density_matrix(rho, rho_target)
    except Exception as exc:
        return {"available": False, "message": f"Quantum diagnosis failed: {exc}"}


def _state_params_amps_phases(state_params: Mapping[str, Any], num_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    num_qubits = int(num_qubits)
    amps = np.asarray(state_params.get("amplitudes", [1.0] * num_qubits), dtype=float)
    if "phases_rad" in state_params:
        phases = np.asarray(state_params.get("phases_rad", [0.0] * num_qubits), dtype=float)
    else:
        phases = np.deg2rad(np.asarray(state_params.get("phases_deg", [0.0] * num_qubits), dtype=float))
    amps = np.pad(amps, (0, max(0, num_qubits - amps.size)), constant_values=1.0)[:num_qubits]
    phases = np.pad(phases, (0, max(0, num_qubits - phases.size)), constant_values=0.0)[:num_qubits]
    return amps, phases


def _target_amps_phases_from_rho_or_params(
    rho_target: np.ndarray,
    params: Mapping[str, Any],
    num_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    if params:
        amps, phases = _state_params_amps_phases(params, num_qubits)
        return _normalized_amplitudes(amps, num_qubits), phases
    core = core_density_matrix(rho_target, num_qubits)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (core + core.conj().T))
    vector = eigvecs[:, int(np.argmax(np.real(eigvals)))]
    return _normalized_amplitudes(np.abs(vector), num_qubits), np.angle(vector)


def _target_input_amps_from_params_or_normalized(
    params: Mapping[str, Any],
    normalized_amps: Any,
    num_qubits: int,
) -> np.ndarray:
    if params:
        amps, _phases = _state_params_amps_phases(params, num_qubits)
        return amps
    return _normalized_amplitudes(normalized_amps, num_qubits)


def _target_amp_variant_infos(
    input_amplitudes: Any,
    phases: Any,
    num_qubits: int,
) -> dict[str, dict[str, Any]]:
    phases = _coerce_phase_array(phases, num_qubits)
    infos: dict[str, dict[str, Any]] = {}
    for key, label in TARGET_AMP_VARIANTS:
        coeffs = _transform_target_amplitudes(input_amplitudes, key)
        normalized = _normalized_coefficients(coeffs, num_qubits)
        psi = _w_state_vector_from_coefficients(normalized, phases, num_qubits)
        infos[key] = {
            "label": label,
            "amplitudes": normalized,
            "rho_target": np.outer(psi, psi.conj()),
            "description": f"{label}; normalized coefficients",
        }
    return infos


def _transform_target_amplitudes(amplitudes: Any, variant_key: str) -> np.ndarray:
    raw = np.asarray(amplitudes, dtype=float)
    amps = np.abs(raw)
    if variant_key == "amp":
        return raw
    if variant_key == "sqrt_amp":
        return np.sqrt(np.clip(amps, 0.0, None))
    if variant_key == "amp_squared":
        return raw * raw
    if variant_key == "sinc_amp":
        return np.where(np.abs(raw) <= _EPS, 1.0, np.sin(raw) / raw)
    raise ValueError(f"unknown target amplitude variant: {variant_key}")


def _normalized_coefficients(coefficients: Any, num_qubits: int) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=float)
    coeffs = np.pad(coeffs, (0, max(0, int(num_qubits) - coeffs.size)), constant_values=1.0)[: int(num_qubits)]
    norm = float(np.linalg.norm(coeffs))
    if norm <= _EPS:
        coeffs = np.ones(int(num_qubits), dtype=float)
        norm = float(np.linalg.norm(coeffs))
    return coeffs / norm


def _normalized_amplitudes(amplitudes: Any, num_qubits: int) -> np.ndarray:
    amps = np.asarray(amplitudes, dtype=float)
    amps = np.pad(amps, (0, max(0, int(num_qubits) - amps.size)), constant_values=1.0)[: int(num_qubits)]
    amps = np.abs(amps)
    norm = float(np.linalg.norm(amps))
    if norm <= _EPS:
        amps = np.ones(int(num_qubits), dtype=float)
        norm = float(np.linalg.norm(amps))
    return amps / norm


def _coerce_phase_array(phases: Any, num_qubits: int) -> np.ndarray:
    values = np.asarray(phases, dtype=float)
    return np.pad(values, (0, max(0, int(num_qubits) - values.size)), constant_values=0.0)[: int(num_qubits)]


def _top_probability(values: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    if not values:
        return None
    label, probability = max(values.items(), key=lambda item: float(item[1] or 0.0))
    return {"label": label, "probability": float(probability or 0.0)}


def _float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_sum(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(np.sum(np.asarray(value, dtype=float)))
    except Exception:
        return 0.0
