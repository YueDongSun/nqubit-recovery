"""Independent synthetic math audit for the unequal-concentration ABI.

The audit is deliberately bounded and research-only. It cross-checks the
native beta-binomial terms against an independent scalar log-PMF, and checks
the nested four-cell moments against a small exact enumeration. It consumes
an explicit native READ tensor; it never infers READ from observed cells,
constructs an optimizer, fits parameters, writes a checkpoint, or grants
posterior or hardware authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from ml_recon.physics_hybrid_v8.fpga_joint_observation import _beta_binomial_log_probability
from .losses import _query_mask
from .unequal_concentration_main_effects import (
    branch_names,
    concentration_by_branch,
    family_names,
    unequal_concentration_four_cell_nll,
    unequal_nested_four_cell_moments,
)


audit_schema = "TrackBUnequalConcentrationMathAuditV1"
maximum_audit_queries = 32
maximum_reference_read_trials = 64
default_absolute_tolerance = 1.0e-10


@dataclass(frozen=True)
class Unequal_concentration_math_audit_config:
    maximum_queries: int = maximum_audit_queries
    maximum_read_trials: int = maximum_reference_read_trials
    absolute_tolerance: float = default_absolute_tolerance
    schema: str = audit_schema

    def validate(self) -> None:
        if self.schema != audit_schema or type(self.maximum_queries) is not int \
                or not 1 <= self.maximum_queries <= maximum_audit_queries \
                or type(self.maximum_read_trials) is not int \
                or not 1 <= self.maximum_read_trials <= maximum_reference_read_trials \
                or type(self.absolute_tolerance) is not float \
                or not math.isfinite(self.absolute_tolerance) or self.absolute_tolerance <= 0:
            raise ValueError("unequal concentration audit configuration is invalid")


def independent_beta_binomial_log_pmf(
    success: int, trials: int, probability: float, concentration: float,
) -> float:
    """Evaluate one beta-binomial log-PMF without the native torch kernel."""
    if type(success) is not int or type(trials) is not int or not 0 <= success <= trials:
        raise ValueError("independent beta-binomial counts are invalid")
    if not all(type(value) is float and math.isfinite(value) for value in (probability, concentration)) \
            or not 0 < probability < 1 or concentration <= 0:
        raise ValueError("independent beta-binomial parameters are invalid")
    alpha = probability * concentration
    beta = (1.0 - probability) * concentration
    return math.lgamma(trials + 1.0) - math.lgamma(success + 1.0) \
        - math.lgamma(trials - success + 1.0) \
        + math.lgamma(success + alpha) + math.lgamma(trials - success + beta) \
        - math.lgamma(trials + alpha + beta) + math.lgamma(alpha + beta) \
        - math.lgamma(alpha) - math.lgamma(beta)


def _enumerated_nested_moments(
    read_trials: int, signal_probability: float, coin_probability: float,
    idler_probability: float, concentrations: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate a small nested tree as an independent moment reference."""
    if type(read_trials) is not int or not 0 <= read_trials <= maximum_reference_read_trials:
        raise ValueError("enumerated READ exceeds the bounded audit domain")
    mean = [0.0] * 4
    second = [[0.0] * 4 for _ in range(4)]
    total_probability = 0.0
    for signal in range(read_trials + 1):
        signal_log_pmf = independent_beta_binomial_log_pmf(
            signal, read_trials, signal_probability, concentrations[0],
        )
        for coin in range(signal + 1):
            coin_log_pmf = independent_beta_binomial_log_pmf(
                coin, signal, coin_probability, concentrations[1],
            )
            for idler in range(read_trials - signal + 1):
                idler_log_pmf = independent_beta_binomial_log_pmf(
                    idler, read_trials - signal, idler_probability, concentrations[2],
                )
                probability = math.exp(signal_log_pmf + coin_log_pmf + idler_log_pmf)
                cells = (read_trials - signal - idler, signal - coin, idler, coin)
                total_probability += probability
                for index, value in enumerate(cells):
                    mean[index] += probability * value
                    for other, other_value in enumerate(cells):
                        second[index][other] += probability * value * other_value
    if abs(total_probability - 1.0) > 100 * default_absolute_tolerance:
        raise ArithmeticError("independent nested reference does not normalize")
    mean_tensor = torch.tensor(mean, dtype=torch.float64)
    second_tensor = torch.tensor(second, dtype=torch.float64)
    return mean_tensor, second_tensor - torch.outer(mean_tensor, mean_tensor)


def audit_unequal_concentration_math(
    prediction: Mapping[str, torch.Tensor], batch: Mapping[str, Any],
    theta: torch.Tensor, read_trials: torch.Tensor, *,
    config: Unequal_concentration_math_audit_config = Unequal_concentration_math_audit_config(),
) -> dict[str, Any]:
    """Run a bounded parity audit for one unequal-concentration input.

    The returned Hessian is a local derivative diagnostic and is never a
    posterior covariance. No training or calibration conclusion is emitted.
    """
    config.validate()
    selected = _query_mask(batch)
    if selected.device.type != "cpu" or selected.ndim != 2 or not bool(selected.any()):
        raise ValueError("math audit requires a nonempty CPU query mask")
    if not isinstance(read_trials, torch.Tensor) or read_trials.device.type != "cpu" \
            or read_trials.shape != selected.shape or read_trials.dtype == torch.bool \
            or read_trials.is_complex() or read_trials.dtype.is_floating_point:
        raise ValueError("math audit requires an explicit aligned integer native READ tensor")
    if not bool(torch.isfinite(read_trials.to(torch.float64)).all()) \
            or bool((read_trials < 0).any()) \
            or bool((read_trials != read_trials.round()).any()):
        raise ValueError("math audit READ values must be finite nonnegative integers")
    selected_reads = read_trials[selected].to(torch.int64)
    if len(selected_reads) > config.maximum_queries or bool((selected_reads > config.maximum_read_trials).any()):
        raise ValueError("math audit query or READ bound exceeded")

    counts = batch.get("four_cell_counts")
    if not isinstance(counts, torch.Tensor) or counts.device.type != "cpu" \
            or counts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64) \
            or counts.shape != (*selected.shape, 4):
        raise ValueError("math audit counts must be native integer [batch,operation,4]")
    raw_selected_counts = counts[selected]
    if bool((raw_selected_counts < 0).any()) or bool(
        (raw_selected_counts > config.maximum_read_trials).any()
    ):
        raise ValueError("math audit four-cell counts exceed the bounded native READ domain")
    selected_counts = raw_selected_counts.to(torch.int64)
    selected_count_totals = selected_counts.sum(dim=-1, dtype=torch.int64)
    if bool((selected_count_totals != selected_reads).any()):
        raise ValueError("math audit four-cell totals must equal the explicit native READ tensor")

    native = unequal_concentration_four_cell_nll(prediction, batch, theta)
    expansion = concentration_by_branch(prediction, batch, theta)
    selected_concentration = expansion["concentration_by_branch"][selected]
    selected_probabilities = tuple(
        prediction[name][selected].to(torch.float64)
        for name in (
            "signal_probability", "coin_given_signal_probability", "idler_given_no_signal_probability",
        )
    )
    independent_nll = 0.0
    maximum_pmf_error = 0.0
    informative_branch_count = 0
    for row, cells in enumerate(selected_counts.tolist()):
        n00, n10, n01, n11 = cells
        signal = n10 + n11
        no_signal = n00 + n01
        for branch, (success, trials, probability) in enumerate((
            (signal, signal + no_signal, selected_probabilities[0][row]),
            (n11, signal, selected_probabilities[1][row]),
            (n01, no_signal, selected_probabilities[2][row]),
        )):
            if trials == 0:
                continue
            informative_branch_count += 1
            probability_value = float(probability)
            concentration_value = float(selected_concentration[row, branch])
            expected = independent_beta_binomial_log_pmf(
                success, trials, probability_value, concentration_value,
            )
            native_value = float(_beta_binomial_log_probability(
                torch.tensor([float(success)], dtype=torch.float64),
                torch.tensor([float(trials)], dtype=torch.float64),
                torch.tensor([probability_value], dtype=torch.float64),
                torch.tensor([concentration_value], dtype=torch.float64),
            )[0])
            maximum_pmf_error = max(maximum_pmf_error, abs(native_value - expected))
            independent_nll -= expected
    independent_nll /= len(selected_counts)

    def objective(value: torch.Tensor) -> torch.Tensor:
        return unequal_concentration_four_cell_nll(prediction, batch, value)["loss"]

    differentiable_theta = theta.detach().to(dtype=torch.float64, device="cpu").requires_grad_(True)
    gradient = torch.autograd.grad(objective(differentiable_theta), differentiable_theta, create_graph=False)[0]
    hessian = torch.autograd.functional.hessian(objective, differentiable_theta)

    moment = unequal_nested_four_cell_moments(
        read_trials=selected_reads,
        signal_probability=selected_probabilities[0],
        coin_given_signal_probability=selected_probabilities[1],
        idler_given_no_signal_probability=selected_probabilities[2],
        concentration_by_branch=selected_concentration,
    )
    operation_families = batch.get("operation_families")
    if type(operation_families) is not tuple or len(operation_families) != selected.shape[1] \
            or set(operation_families) != set(family_names):
        raise ValueError("math audit requires the complete registered operation-family tuple")
    selected_operations = torch.nonzero(selected, as_tuple=False).tolist()
    reference_means = []
    reference_covariances = []
    for index, (_row, _operation) in enumerate(selected_operations):
        concentrations = tuple(float(value) for value in selected_concentration[index])
        means, covariance = _enumerated_nested_moments(
            int(selected_reads[index]),
            float(selected_probabilities[0][index]),
            float(selected_probabilities[1][index]),
            float(selected_probabilities[2][index]),
            concentrations,
        )
        reference_means.append(means)
        reference_covariances.append(covariance)
    reference_mean_tensor = torch.stack(reference_means)
    reference_covariance_tensor = torch.stack(reference_covariances)
    mean_error = float((moment["four_cell_mean"] - reference_mean_tensor).abs().max())
    covariance_error = float((moment["four_cell_covariance"] - reference_covariance_tensor).abs().max())
    fixed_total_error = float(moment["four_cell_covariance"].sum(-1).abs().max())
    native_loss_error = abs(float(native["loss"]) - independent_nll)
    tolerance = config.absolute_tolerance
    errors = {
        "native_nll_vs_independent": native_loss_error,
        "pmf_log_probability_max_absolute": maximum_pmf_error,
        "mean_max_absolute": mean_error,
        "covariance_max_absolute": covariance_error,
        "fixed_total_covariance_max_absolute": fixed_total_error,
    }
    if any(value > tolerance for value in errors.values()) or not bool(torch.isfinite(gradient).all()) \
            or not bool(torch.isfinite(hessian).all()):
        raise ArithmeticError(f"unequal concentration math audit failed: {errors}")
    return {
        "schema": audit_schema,
        "status": "COMPUTED_REQUIRES_NUMERICAL_REVIEW_NOT_FIT_OR_CALIBRATION",
        "selected_query_count": len(selected_counts),
        "informative_branch_count": informative_branch_count,
        "errors": errors,
        "gradient": gradient.detach().tolist(),
        "hessian": hessian.detach().tolist(),
        "theta": differentiable_theta.detach().tolist(),
        "branch_names": list(branch_names),
        "operation_family_vocabulary": list(family_names),
        "optimizer_constructed": False,
        "parameters_updated": False,
        "means_frozen": True,
        "curvature_is_posterior_covariance": False,
        "scientific_gate_issued": False,
        "hardware_action_authorized": False,
    }
