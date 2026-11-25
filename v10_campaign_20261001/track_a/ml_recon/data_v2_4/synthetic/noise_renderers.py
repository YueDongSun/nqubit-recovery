"""Executable mechanism selection and actual D3 matching for V2.4 samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ml_recon.physics_hybrid_v10.dense_pauli_panel import (
    Dense_Pauli_Panel,
    compile_dense_pauli_panel,
    max_dense_pauli_panel_qubits,
)
from ml_recon.physics_hybrid_v10.noise_interventions import (
    CORRECTABLE_INTERVENTIONS,
    IRRECOVERABLE_INTERVENTIONS,
    CompositeNoiseIntervention,
    LockedOODIntervention,
    V24NoiseIntervention,
    intervention_registry,
    registered_feedback_settings,
    w_density_matrix,
)
from ml_recon.physics_hybrid_v10.noise_matching import hellinger_squared, state_fidelity
from ml_recon.data_v2_4.hashing import sha256_json
from ml_recon.data_v2_4.version import NOISE_INTERVENTION_SHA


CORRECTABLE_MECHANISMS = CORRECTABLE_INTERVENTIONS
IRRECOVERABLE_MECHANISMS = IRRECOVERABLE_INTERVENTIONS


@dataclass(frozen=True)
class NoiseTruth:
    mechanism_name: str
    mechanism_category: str
    intervention: V24NoiseIntervention
    correctable_multihot: np.ndarray
    irrecoverable_multihot: np.ndarray
    severity: np.ndarray
    epsilon_irr: float
    eta_target: float
    eta_achieved: float
    matching_residual: float
    matching_converged: bool
    physical_parameter: float
    state_infidelity: float
    train_fit_sha: str
    heldout_error: float
    intervention_forward_sha: str
    rho_noisy: np.ndarray
    measurement_clean: np.ndarray
    measurement_noisy: np.ndarray
    eta_ctrl_target: float = 0.0
    eta_irr_target: float = 0.0
    eta_ctrl_achieved: float = 0.0
    eta_irr_achieved: float = 0.0
    kappa_inj: float = 1.0
    kappa_achieved: float = 1.0
    kappa_forward_achieved: float = 1.0
    kappa_target: float = 1.0
    kappa_component_achieved: float = 1.0
    kappa_component_mask: bool = False
    kappa_operational: float = 0.0
    kappa_operational_mask: bool = False
    control_parameter: float = 0.0
    irrecoverable_parameter: float = 0.0
    control_mechanism_name: str = ""
    irrecoverable_mechanism_name: str = ""


def _select_mechanism(
    rng: np.random.Generator, names: Tuple[str, ...], rho: np.ndarray,
    settings: Tuple[Tuple[str, ...], ...], clean_law: np.ndarray,
    registry: Dict[str, V24NoiseIntervention],
    minimum_eta: float = 0.0,
    *,
    panel: Dense_Pauli_Panel | None = None,
) -> V24NoiseIntervention:
    """Choose a deterministic-RNG rotated observable registered mechanism."""
    start_index = int(rng.integers(len(names)))
    observable = []
    for offset in range(len(names)):
        name = names[(start_index + offset) % len(names)]
        intervention = registry[name]
        maximum = hellinger_squared(
            clean_law,
            intervention.measurement_law(
                rho,
                settings,
                intervention.maximum_parameter,
                panel=panel,
            ),
        )
        if maximum > 1e-10:
            observable.append((maximum, intervention))
        if maximum + 1e-10 >= minimum_eta and maximum > 1e-10:
            return intervention
    if observable:
        return max(observable, key=lambda item: item[0])[1]
    raise RuntimeError("no physical mechanism affects the registered panel")

def _render_composite_noise_truth(
    rng: np.random.Generator, rho: np.ndarray,
    settings: Tuple[Tuple[str, ...], ...], eta_target: float,
    kappa_inj: float, noise_family: str, clean_law: np.ndarray,
    registry: Dict[str, V24NoiseIntervention],
    *,
    panel: Dense_Pauli_Panel | None = None,
) -> NoiseTruth:
    """Match a composite channel with a declared kappa injection budget."""
    correctable = _select_mechanism(
        rng, CORRECTABLE_MECHANISMS, rho, settings, clean_law, registry,
        eta_target * kappa_inj, panel=panel)
    irrecoverable = _select_mechanism(
        rng, IRRECOVERABLE_MECHANISMS, rho, settings, clean_law, registry,
        eta_target * (1.0 - kappa_inj), panel=panel)
    maximum_pair_eta = hellinger_squared(
        clean_law, CompositeNoiseIntervention(
            correctable, irrecoverable,
            correctable.maximum_parameter,
            irrecoverable.maximum_parameter).measurement_law(
                rho, settings, panel=panel))
    if maximum_pair_eta + 1e-7 < eta_target:
        control_order = rng.permutation(len(CORRECTABLE_MECHANISMS))
        irrecoverable_order = rng.permutation(len(IRRECOVERABLE_MECHANISMS))
        best = (maximum_pair_eta, correctable, irrecoverable)
        selected = None
        for control_index in control_order:
            candidate_control = registry[
                CORRECTABLE_MECHANISMS[int(control_index)]]
            for irrecoverable_index in irrecoverable_order:
                candidate_irrecoverable = registry[
                    IRRECOVERABLE_MECHANISMS[int(irrecoverable_index)]]
                candidate_eta = hellinger_squared(
                    clean_law, CompositeNoiseIntervention(
                        candidate_control, candidate_irrecoverable,
                        candidate_control.maximum_parameter,
                        candidate_irrecoverable.maximum_parameter,
                    ).measurement_law(rho, settings, panel=panel))
                if candidate_eta > best[0]:
                    best = (candidate_eta, candidate_control,
                            candidate_irrecoverable)
                if candidate_eta + 1e-7 >= eta_target:
                    selected = (candidate_control, candidate_irrecoverable)
                    break
            if selected is not None:
                break
        correctable, irrecoverable = selected or (best[1], best[2])
    max_ctrl = hellinger_squared(
        clean_law, correctable.measurement_law(
            rho, settings, correctable.maximum_parameter, panel=panel))
    max_irr = hellinger_squared(
        clean_law, irrecoverable.measurement_law(
            rho, settings, irrecoverable.maximum_parameter, panel=panel))
    requested = float(eta_target)
    target_ctrl = min(max_ctrl, requested * kappa_inj)
    target_irr = min(max_irr, requested * (1.0 - kappa_inj))
    remaining = max(0.0, requested - target_ctrl - target_irr)
    ctrl_extra = min(max_ctrl - target_ctrl, remaining)
    target_ctrl += ctrl_extra
    target_irr += min(max_irr - target_irr, remaining - ctrl_extra)
    ctrl = correctable.severity_matching(
        rho, settings, target_ctrl, panel=panel)
    irr = irrecoverable.severity_matching(
        rho, settings, target_irr, panel=panel)
    base_ctrl_parameter = float(ctrl["physical_parameter"])
    base_irr_parameter = float(irr["physical_parameter"])

    def at_factor(factor: float):
        candidate = CompositeNoiseIntervention(
            correctable, irrecoverable,
            min(correctable.maximum_parameter, base_ctrl_parameter * factor),
            min(irrecoverable.maximum_parameter, base_irr_parameter * factor))
        candidate_eta = hellinger_squared(
            clean_law, candidate.measurement_law(
                rho, settings, panel=panel))
        return candidate, candidate_eta

    factor_limits = [1.0]
    if base_ctrl_parameter > 1e-12:
        factor_limits.append(
            correctable.maximum_parameter / base_ctrl_parameter)
    if base_irr_parameter > 1e-12:
        factor_limits.append(
            irrecoverable.maximum_parameter / base_irr_parameter)
    maximum_factor = max(factor_limits)
    maximum_channel, maximum_achieved = at_factor(maximum_factor)
    if maximum_achieved + 1e-7 < requested:
        channel, achieved = maximum_channel, maximum_achieved
    else:
        lower_factor, upper_factor = 0.0, maximum_factor
        channel, achieved = at_factor(1.0)
        for _ in range(32):
            middle = 0.5 * (lower_factor + upper_factor)
            channel, achieved = at_factor(middle)
            if achieved < requested:
                lower_factor = middle
            else:
                upper_factor = middle
    eta_ctrl = hellinger_squared(
        clean_law, correctable.measurement_law(
            rho, settings, channel.correctable_parameter, panel=panel))
    eta_irr = hellinger_squared(
        clean_law, irrecoverable.measurement_law(
            rho, settings, channel.irrecoverable_parameter, panel=panel))
    rho_noisy = channel.apply_state_channel(rho)
    irrecoverable_only = irrecoverable.apply_state_channel(
        rho, channel.irrecoverable_parameter)
    severity = np.zeros(
        len(CORRECTABLE_MECHANISMS) + len(IRRECOVERABLE_MECHANISMS),
        dtype=np.float32)
    severity[CORRECTABLE_MECHANISMS.index(correctable.mechanism_name)] = channel.correctable_parameter
    severity[len(CORRECTABLE_MECHANISMS) + IRRECOVERABLE_MECHANISMS.index(
        irrecoverable.mechanism_name)] = channel.irrecoverable_parameter
    total = max(eta_ctrl + eta_irr, 1e-15)
    calibration_sha = sha256_json({
        "kind": "composite_noise_v1", "noise_family": noise_family,
        "correctable": correctable.mechanism_name,
        "irrecoverable": irrecoverable.mechanism_name,
        "kappa_inj": kappa_inj, "eta_target": eta_target,
    })
    correctable_mask = np.zeros(len(CORRECTABLE_MECHANISMS), dtype=np.float32)
    irrecoverable_mask = np.zeros(len(IRRECOVERABLE_MECHANISMS), dtype=np.float32)
    correctable_mask[CORRECTABLE_MECHANISMS.index(correctable.mechanism_name)] = 1.0
    irrecoverable_mask[IRRECOVERABLE_MECHANISMS.index(irrecoverable.mechanism_name)] = 1.0
    return NoiseTruth(
        mechanism_name=channel.mechanism_name, mechanism_category="composite",
        intervention=channel, correctable_multihot=correctable_mask,
        irrecoverable_multihot=irrecoverable_mask, severity=severity,
        epsilon_irr=1.0 - state_fidelity(rho, irrecoverable_only),
        eta_target=requested, eta_achieved=float(achieved),
        matching_residual=abs(float(achieved) - requested),
        matching_converged=abs(float(achieved) - requested) <= 1e-6,
        physical_parameter=channel.physical_parameter,
        state_infidelity=1.0 - state_fidelity(rho, rho_noisy),
        train_fit_sha=calibration_sha, heldout_error=0.0,
        intervention_forward_sha=NOISE_INTERVENTION_SHA,
        rho_noisy=rho_noisy, measurement_clean=clean_law,
        measurement_noisy=channel.measurement_law(
            rho, settings, panel=panel),
        eta_ctrl_target=requested * kappa_inj,
        eta_irr_target=requested * (1.0 - kappa_inj),
        eta_ctrl_achieved=eta_ctrl, eta_irr_achieved=eta_irr,
        kappa_inj=kappa_inj, kappa_achieved=eta_ctrl / total,
        kappa_forward_achieved=eta_ctrl / total,
        kappa_target=kappa_inj,
        kappa_component_achieved=eta_ctrl / total,
        kappa_component_mask=True,
        kappa_operational=0.0, kappa_operational_mask=False,
        control_parameter=channel.correctable_parameter,
        irrecoverable_parameter=channel.irrecoverable_parameter,
        control_mechanism_name=correctable.mechanism_name,
        irrecoverable_mechanism_name=irrecoverable.mechanism_name,
    )

def render_noise_truth(
    rng: np.random.Generator,
    *,
    rho_clean: np.ndarray | None = None,
    settings: Tuple[Tuple[str, ...], ...] | None = None,
    force_irrecoverable: bool = False,
    noise_family: str = "smoke_mixed",
    eta_target: float | None = None,
    correctable_fraction: float = 0.5,
    composite: bool = False,
    locked_ood: bool = False,
) -> NoiseTruth:
    if rho_clean is None:
        coefficients = np.ones(3, dtype=np.complex128) / np.sqrt(3.0)
        rho_clean = w_density_matrix(coefficients)
    rho = np.asarray(rho_clean, dtype=np.complex128)
    N = int(round(np.log2(rho.shape[0])))
    if settings is None:
        settings = registered_feedback_settings(N)
    if not 0.0 <= correctable_fraction <= 1.0:
        raise ValueError("correctable_fraction must be in [0, 1]")
    if eta_target is not None and not 0.0 <= eta_target <= 1.0:
        raise ValueError("eta_target must be in [0, 1]")
    panel = (
        compile_dense_pauli_panel(N, settings)
        if 1 <= N <= max_dense_pauli_panel_qubits
        else None
    )
    registry = intervention_registry()
    identity = V24NoiseIntervention("identity", "none", "none", 0.0, "none")
    clean_law = identity.measurement_law(rho, settings, 0.0, panel=panel)
    if locked_ood:
        intervention = LockedOODIntervention()
        noisy_law = intervention.measurement_law(rho, settings, panel=panel)
        rho_noisy = intervention.apply_state_channel(rho)
        achieved = hellinger_squared(clean_law, noisy_law)
        zeros_correctable = np.zeros(len(CORRECTABLE_MECHANISMS), dtype=np.float32)
        zeros_irrecoverable = np.zeros(len(IRRECOVERABLE_MECHANISMS), dtype=np.float32)
        return NoiseTruth(
            mechanism_name=intervention.mechanism_name,
            mechanism_category=intervention.category,
            intervention=intervention,
            correctable_multihot=zeros_correctable,
            irrecoverable_multihot=zeros_irrecoverable,
            severity=np.zeros(
                len(CORRECTABLE_MECHANISMS) + len(IRRECOVERABLE_MECHANISMS),
                dtype=np.float32),
            epsilon_irr=1.0 - state_fidelity(rho, rho_noisy),
            eta_target=float(achieved), eta_achieved=float(achieved),
            matching_residual=0.0, matching_converged=True,
            physical_parameter=intervention.physical_parameter,
            state_infidelity=1.0 - state_fidelity(rho, rho_noisy),
            train_fit_sha=sha256_json({
                "kind": "locked_ood_v1",
                "intervention": intervention.mechanism_name,
            }),
            heldout_error=0.0,
            intervention_forward_sha=NOISE_INTERVENTION_SHA,
            rho_noisy=rho_noisy, measurement_clean=clean_law,
            measurement_noisy=noisy_law,
            kappa_inj=0.0, kappa_achieved=0.0, kappa_forward_achieved=0.0,
            kappa_target=0.0, kappa_component_achieved=0.0,
            kappa_component_mask=False,
        )
    if composite and noise_family != "standard" and (eta_target or 0.0) > 0.0:
        return _render_composite_noise_truth(
            rng, rho, settings, float(eta_target), correctable_fraction,
            noise_family, clean_law, registry, panel=panel)
    if noise_family == "standard" or eta_target == 0.0:
        zeros_correctable = np.zeros(len(CORRECTABLE_MECHANISMS), dtype=np.float32)
        zeros_irrecoverable = np.zeros(len(IRRECOVERABLE_MECHANISMS), dtype=np.float32)
        calibration_sha = sha256_json({
            "mechanism": "identity", "noise_family": noise_family, "N": N})
        return NoiseTruth(
            mechanism_name="identity", mechanism_category="none",
            intervention=identity, correctable_multihot=zeros_correctable,
            irrecoverable_multihot=zeros_irrecoverable,
            severity=np.zeros(len(CORRECTABLE_MECHANISMS)
                              + len(IRRECOVERABLE_MECHANISMS), dtype=np.float32),
            epsilon_irr=0.0, eta_target=0.0, eta_achieved=0.0,
            matching_residual=0.0, matching_converged=True,
            physical_parameter=0.0, state_infidelity=0.0,
            train_fit_sha=calibration_sha, heldout_error=0.0,
            intervention_forward_sha=NOISE_INTERVENTION_SHA,
            rho_noisy=rho.copy(), measurement_clean=clean_law,
            measurement_noisy=clean_law.copy(),
            kappa_inj=correctable_fraction,
            kappa_achieved=0.0,
            kappa_forward_achieved=0.0,
            kappa_target=correctable_fraction,
            kappa_component_achieved=0.0,
            kappa_component_mask=False)

    choose_irrecoverable = force_irrecoverable \
        or rng.random() >= correctable_fraction
    first_names = (IRRECOVERABLE_MECHANISMS if choose_irrecoverable
                   else CORRECTABLE_MECHANISMS)
    category_panels = [first_names]
    if not force_irrecoverable:
        category_panels.append(
            CORRECTABLE_MECHANISMS if choose_irrecoverable
            else IRRECOVERABLE_MECHANISMS)
    mechanism_name = ""
    intervention = None
    maximum_eta = 0.0
    for names in category_panels:
        start_index = int(rng.integers(len(names)))
        for offset in range(len(names)):
            candidate_name = names[(start_index + offset) % len(names)]
            candidate = registry[candidate_name]
            candidate_law = candidate.measurement_law(
                rho, settings, candidate.maximum_parameter, panel=panel)
            candidate_eta = hellinger_squared(clean_law, candidate_law)
            requested_eta = float(eta_target) if eta_target is not None else None
            if candidate_eta > 1e-10 and (
                    requested_eta is None or candidate_eta + 1e-10 >= requested_eta):
                mechanism_name = candidate_name
                intervention = candidate
                maximum_eta = candidate_eta
                choose_irrecoverable = names is IRRECOVERABLE_MECHANISMS
                break
        if intervention is not None:
            break
    if intervention is None:
        raise RuntimeError(
            "no registered physical mechanism has an observable effect for this "
            "state and measurement panel")
    correctable = np.zeros(len(CORRECTABLE_MECHANISMS), dtype=np.float32)
    irrecoverable = np.zeros(len(IRRECOVERABLE_MECHANISMS), dtype=np.float32)
    if choose_irrecoverable:
        irrecoverable[IRRECOVERABLE_MECHANISMS.index(mechanism_name)] = 1.0
    else:
        correctable[CORRECTABLE_MECHANISMS.index(mechanism_name)] = 1.0
    matched_eta_target = (float(rng.uniform(0.15, 0.70) * maximum_eta)
                          if eta_target is None else float(eta_target))
    matched = intervention.severity_matching(
        rho, settings, matched_eta_target, panel=panel)
    severity = np.zeros(
        len(CORRECTABLE_MECHANISMS) + len(IRRECOVERABLE_MECHANISMS),
        dtype=np.float32)
    offset = (len(CORRECTABLE_MECHANISMS) if choose_irrecoverable else 0)
    mechanism_index = ((IRRECOVERABLE_MECHANISMS if choose_irrecoverable
                        else CORRECTABLE_MECHANISMS).index(mechanism_name))
    severity[offset + mechanism_index] = float(matched["physical_parameter"])
    epsilon_irr = (max(float(matched["state_infidelity"]),
                       float(matched["eta_achieved"]))
                   if choose_irrecoverable else 0.0)
    return NoiseTruth(
        mechanism_name=mechanism_name,
        mechanism_category=intervention.category,
        intervention=intervention,
        correctable_multihot=correctable,
        irrecoverable_multihot=irrecoverable,
        severity=severity,
        epsilon_irr=epsilon_irr,
        eta_target=matched_eta_target,
        eta_achieved=float(matched["eta_achieved"]),
        matching_residual=float(matched["matching_residual"]),
        matching_converged=bool(matched["converged"]),
        physical_parameter=float(matched["physical_parameter"]),
        state_infidelity=float(matched["state_infidelity"]),
        train_fit_sha=str(matched["train_fit_sha"]),
        heldout_error=float(matched["heldout_error"]),
        intervention_forward_sha=str(matched["intervention_forward_sha"]),
        rho_noisy=np.asarray(matched["rho_noisy"], dtype=np.complex128),
        measurement_clean=np.asarray(matched["measurement_clean"], dtype=np.float64),
        measurement_noisy=np.asarray(matched["measurement_noisy"], dtype=np.float64),
        kappa_inj=correctable_fraction,
        kappa_achieved=float(not choose_irrecoverable),
        kappa_forward_achieved=float(not choose_irrecoverable),
        kappa_target=correctable_fraction,
        kappa_component_achieved=float(not choose_irrecoverable),
        kappa_component_mask=True,
    )
