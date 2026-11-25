"""Research-only unequal-concentration main effects for frozen Track B means.

This module is a separate ABI. It never mutates the V2/V3 prediction mapping,
constructs an optimizer, writes a checkpoint, or grants posterior/hardware
authority. The six coordinates are one shared effect, three sum-zero family
contrasts and two sum-zero branch contrasts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from ml_recon.physics_hybrid_v8.fpga_joint_observation import _beta_binomial_log_probability
from .detected_count_forward import require_probability_tensor
from .family_loss_decomposition import branch_names, probability_names
from .losses import _query_mask


unequal_concentration_schema = "TrackBUnequalConcentrationMainEffectsV1"
family_names = ("info", "pair", "single", "triple")
branch_count = len(branch_names)
parameter_count = 6


@dataclass(frozen=True)
class Unequal_concentration_config:
    maximum_log_delta: float = math.log(2.0)
    schema: str = unequal_concentration_schema

    def validate(self) -> None:
        if self.schema != unequal_concentration_schema:
            raise ValueError("unequal concentration schema is invalid")
        if not math.isfinite(self.maximum_log_delta) or self.maximum_log_delta <= 0:
            raise ValueError("maximum_log_delta must be finite positive")

    @property
    def coefficient_bound(self) -> float:
        self.validate()
        return self.maximum_log_delta / 6.0


def _checked_theta(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor) or theta.shape != (parameter_count,) \
            or theta.device.type != "cpu" or theta.dtype not in (torch.float32, torch.float64) \
            or not bool(torch.isfinite(theta).all()):
        raise ValueError("main effects require one finite CPU six-vector")
    return theta.to(torch.float64)


def expand_main_effects(
    theta: torch.Tensor, *, config: Unequal_concentration_config = Unequal_concentration_config(),
) -> dict[str, torch.Tensor]:
    """Expand bounded six-vector coordinates with explicit gauge constraints."""
    config.validate()
    checked = _checked_theta(theta)
    bounded = config.coefficient_bound * torch.tanh(checked)
    family = torch.stack((bounded[1], bounded[2], bounded[3], -(bounded[1] + bounded[2] + bounded[3])))
    branch = torch.stack((bounded[4], bounded[5], -(bounded[4] + bounded[5])))
    delta = bounded[0] + family[:, None] + branch[None, :]
    if bool((delta.abs() > config.maximum_log_delta + 1e-12).any()):
        raise RuntimeError("bounded main effects exceeded the registered log-delta domain")
    return {
        "theta": checked,
        "bounded_theta": bounded,
        "family_effect": family,
        "branch_effect": branch,
        "log_delta": delta,
    }


def _family_indices(batch: Mapping[str, Any], operation_count: int) -> torch.Tensor:
    values = batch.get("operation_families")
    if type(values) is not tuple or len(values) != operation_count or set(values) != set(family_names):
        raise ValueError("unequal concentration requires exactly four registered operation families")
    return torch.tensor([family_names.index(value) for value in values], dtype=torch.long)


def concentration_by_branch(
    prediction: Mapping[str, Any], batch: Mapping[str, Any], theta: torch.Tensor,
    *, config: Unequal_concentration_config = Unequal_concentration_config(),
) -> dict[str, torch.Tensor]:
    """Return [batch, operation, branch] concentrations without changing means."""
    config.validate()
    base = prediction.get("four_cell_concentration")
    operation_mask = batch.get("operation_mask")
    if not isinstance(base, torch.Tensor) or base.ndim != 1 or base.device.type != "cpu" \
            or not bool(torch.isfinite(base).all()) or bool((base <= 0).any()):
        raise ValueError("prediction must provide finite positive CPU row concentration")
    if not isinstance(operation_mask, torch.Tensor) or operation_mask.dtype != torch.bool \
            or operation_mask.ndim != 2 or operation_mask.shape[0] != base.shape[0]:
        raise ValueError("operation mask must be CPU bool [batch,operation]")
    expansion = expand_main_effects(theta, config=config)
    indices = _family_indices(batch, operation_mask.shape[1])
    row_delta = expansion["log_delta"].index_select(0, indices)
    concentrations = base.to(torch.float64)[:, None, None] * row_delta[None].exp()
    concentrations = torch.where(operation_mask[..., None], concentrations, torch.zeros_like(concentrations))
    return {**expansion, "concentration_by_branch": concentrations}


def unequal_concentration_four_cell_nll(
    prediction: Mapping[str, Any], batch: Mapping[str, Any], theta: torch.Tensor,
    *, config: Unequal_concentration_config = Unequal_concentration_config(),
) -> dict[str, Any]:
    """Score frozen probabilities with the unequal-kappa native conditional tree."""
    selected = _query_mask(batch)
    if selected.device.type != "cpu" or not bool(torch.any(selected)):
        raise ValueError("unequal concentration requires nonempty CPU query selection")
    counts = batch.get("four_cell_counts")
    if not isinstance(counts, torch.Tensor) or counts.device.type != "cpu" or counts.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64,
    ) or counts.ndim != 3 or counts.shape[-1] != 4 or counts.shape[:2] != selected.shape:
        raise ValueError("unequal concentration counts must be native integer [batch,operation,4]")
    if bool((counts[selected] < 0).any()):
        raise ValueError("unequal concentration counts cannot be negative")
    for name in probability_names:
        value = prediction.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != selected.shape \
                or value.device.type != "cpu" or not value.dtype.is_floating_point:
            raise ValueError("unequal concentration probabilities must be CPU floating [batch,operation]")
        if not bool(torch.isfinite(value).all()) or bool(((value <= 0) | (value >= 1)).any()):
            raise ValueError(f"{name} must be strictly between zero and one for finite NLL")
    cells = batch["four_cell_counts"][selected].double()
    n00, n10, n01, n11 = cells.unbind(-1)
    signal, no_signal = n10 + n11, n00 + n01
    parents = (signal + no_signal, signal, no_signal)
    successes = (signal, n11, n01)
    values = prediction
    for name in probability_names:
        if not isinstance(values.get(name), torch.Tensor) or values[name].shape != selected.shape:
            raise ValueError("prediction branch probabilities do not match the query shape")
    expansion = concentration_by_branch(prediction, batch, theta, config=config)
    concentrations = expansion["concentration_by_branch"][selected]
    probability_values = [values[name][selected].double() for name in probability_names]
    terms = []
    for index, (success, parent, probability) in enumerate(zip(successes, parents, probability_values, strict=True)):
        informative = parent > 0
        branch_terms = torch.zeros_like(parent, dtype=torch.float64)
        if bool(informative.any()):
            branch_terms[informative] = -_beta_binomial_log_probability(
                success[informative], parent[informative], probability[informative],
                concentrations[informative, index],
            )
        terms.append(branch_terms)
    nll = torch.stack(terms, -1).sum() / int(selected.sum())
    # Keep a legal no-information loss connected to theta so its gradient and
    # Hessian are semantic zeros instead of an autograd failure.
    nll = nll + (expansion["theta"] * 0.0).sum()
    if not bool(torch.isfinite(nll)):
        raise FloatingPointError("unequal concentration native NLL became nonfinite")
    return {
        "schema": unequal_concentration_schema,
        "four_cell_nll": nll,
        "loss": nll,
        "concentration_by_branch": concentrations,
        "log_delta": expansion["log_delta"],
        "theta": expansion["theta"],
        "optimizer_constructed": False,
        "parameters_updated": False,
        "means_frozen": True,
        "scientific_gate_issued": False,
        "hardware_action_authorized": False,
    }


def unequal_nested_four_cell_moments(
    *, read_trials: torch.Tensor, signal_probability: torch.Tensor,
    coin_given_signal_probability: torch.Tensor,
    idler_given_no_signal_probability: torch.Tensor,
    concentration_by_branch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Exact unequal-kappa nested-tree moments with fixed-total covariance."""
    signal = require_probability_tensor("signal_probability", signal_probability).to(torch.float64)
    coin = require_probability_tensor("coin_given_signal_probability", coin_given_signal_probability).to(torch.float64)
    idler = require_probability_tensor("idler_given_no_signal_probability", idler_given_no_signal_probability).to(torch.float64)
    if read_trials.dtype == torch.bool or read_trials.is_complex():
        raise TypeError("read trials must be nonnegative integers")
    trials = read_trials.to(torch.float64)
    concentration = concentration_by_branch.to(torch.float64)
    if concentration.ndim < 1 or concentration.shape[-1] != branch_count:
        raise ValueError("concentration_by_branch requires final branch dimension three")
    if not bool(torch.isfinite(trials).all()) or bool(((trials < 0) | (trials != trials.round())).any()):
        raise ValueError("read trials must be finite nonnegative integers")
    if not bool(torch.isfinite(concentration).all()) or bool((concentration <= 0).any()):
        raise ValueError("concentration_by_branch must be finite positive")
    try:
        base_shape = torch.broadcast_shapes(
            trials.shape, signal.shape, coin.shape, idler.shape, concentration.shape[:-1],
        )
    except RuntimeError as error:
        raise ValueError("unequal concentration moment inputs are not broadcastable") from error
    trials = torch.broadcast_to(trials, base_shape)
    signal = torch.broadcast_to(signal, base_shape)
    coin = torch.broadcast_to(coin, base_shape)
    idler = torch.broadcast_to(idler, base_shape)
    concentration = torch.broadcast_to(concentration, (*base_shape, branch_count))
    k_s, k_c, k_t = concentration.unbind(-1)
    signal_mean = trials * signal
    no_signal_mean = trials * (1 - signal)
    signal_variance = trials * signal * (1 - signal) * (trials + k_s) / (1 + k_s)
    coin_variance = coin.square() * signal_variance + coin * (1 - coin) * (
        signal_variance + signal_mean.square() + k_c * signal_mean
    ) / (1 + k_c)
    idler_variance = idler.square() * signal_variance + idler * (1 - idler) * (
        signal_variance + no_signal_mean.square() + k_t * no_signal_mean
    ) / (1 + k_t)
    signal_idler = -idler * signal_variance
    signal_coin = coin * signal_variance
    idler_coin = -idler * coin * signal_variance
    branch_covariance = torch.stack((
        torch.stack((signal_variance, signal_idler, signal_coin), -1),
        torch.stack((signal_idler, idler_variance, idler_coin), -1),
        torch.stack((signal_coin, idler_coin, coin_variance), -1),
    ), -2)
    transform = trials.new_tensor(((-1, -1, 0), (1, 0, -1), (0, 1, 0), (0, 0, 1)))
    covariance = transform @ branch_covariance @ transform.T
    mean = torch.stack((
        no_signal_mean * (1 - idler), signal_mean * (1 - coin),
        no_signal_mean * idler, signal_mean * coin,
    ), -1)
    return {"four_cell_mean": mean, "four_cell_covariance": covariance}


def unequal_concentration_zero_fallback(
    prediction: Mapping[str, Any], batch: Mapping[str, Any],
    *, config: Unequal_concentration_config = Unequal_concentration_config(),
) -> dict[str, Any]:
    """Validate that six-vector zero exactly recovers the parent shared-kappa ABI."""
    zero = torch.zeros(parameter_count, dtype=torch.float64)
    result = unequal_concentration_four_cell_nll(prediction, batch, zero, config=config)
    if not bool(torch.equal(result["log_delta"], torch.zeros((len(family_names), branch_count), dtype=torch.float64))):
        raise AssertionError("zero main effects did not produce exact zero log deltas")
    return result
