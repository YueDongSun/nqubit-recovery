"""FPGA-native joint observation contract for the V8 DLCZ sidecar.

The measured operation statistics are not three unrelated count vectors.
For every operation and authoritative read trial count ``R`` the FPGA records
herald presence ``S``, singleton-idler presence ``I``, and their coincidence
``C``.  They therefore define the four mutually exclusive cells

``(no herald, no singleton), (herald only), (singleton only), (coin)``.

The read multi-click registers are deliberately absent from this module: they
are global 10 ns detector-clock observations and are not operation-local
labels.  In particular, ``multi2`` is a mixture of source n>=2,
source--background and background--background events, not a true-double label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - NumPy-only repository installs
    torch = None


FPGA_JOINT_OBSERVATION_SCHEMA = "v8_fpga_joint_four_cell_observation_v1"
FPGA_JOINT_CELL_NAMES: Tuple[str, str, str, str] = (
    "no_herald_no_singleton_idler",
    "herald_no_singleton_idler",
    "no_herald_singleton_idler",
    "herald_singleton_idler_coin",
)
EXTERNAL_HERALD_SINGLETON_GATE_S = 200.0e-9
READ_MULTI_SIMULTANEITY_CLOCK_S = 10.0e-9
READ_WINDOW_ACCEPTED_CLOCK_COUNT = 19
READ_WINDOW_ACCEPTED_SUPPORT_S = (
    READ_WINDOW_ACCEPTED_CLOCK_COUNT * READ_MULTI_SIMULTANEITY_CLOCK_S
)
FPGA_MULTI_ORIGIN_NAMES: Tuple[str, str, str, str] = (
    "no_herald_background_window",
    "herald_n1_window",
    "herald_n2_window",
    "herald_n_ge_3_window",
)


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the FPGA joint likelihood")


@dataclass(frozen=True)
class FPGAJointFourCellCounts:
    """Validated four-cell sufficient statistics in canonical cell order."""

    counts: "torch.Tensor"
    valid_mask: "torch.Tensor"
    schema: str = FPGA_JOINT_OBSERVATION_SCHEMA
    cell_names: Tuple[str, str, str, str] = FPGA_JOINT_CELL_NAMES


@dataclass(frozen=True)
class FPGAReadWindowStatistics:
    """Exact window-OR and per-clock expectations for the read FPGA.

    ``getidler`` is sticky over the 19 accepted 100 MHz clocks and is set when
    any accepted clock has exactly one of the four read SPADs high.  In
    contrast, the six pair counters and multi2/3/4 are incremented separately
    on every accepted clock.  The returned clock-pattern expectation therefore
    sums to 19, not one.
    """

    singleton_given_signal_probability: "torch.Tensor"
    singleton_given_no_signal_probability: "torch.Tensor"
    singleton_probability_by_signal_sector: "torch.Tensor"
    expected_pattern_clock_counts_per_trial: "torch.Tensor"
    expected_exact_pair_clock_counts_per_trial: "torch.Tensor"
    expected_multi_clock_counts_per_trial: "torch.Tensor"
    expected_pattern_clock_counts_per_trial_by_origin: "torch.Tensor"
    expected_exact_pair_clock_counts_per_trial_by_origin: "torch.Tensor"
    expected_multi_clock_counts_per_trial_by_origin: "torch.Tensor"
    multi_origin_names: Tuple[str, str, str, str] = FPGA_MULTI_ORIGIN_NAMES
    accepted_clock_count: int = READ_WINDOW_ACCEPTED_CLOCK_COUNT


def combine_independent_on_off_patterns(
    first_probability: "torch.Tensor",
    second_probability: "torch.Tensor",
) -> "torch.Tensor":
    """Combine two independent four-SPAD patterns by detector-wise OR."""

    _require_torch()
    if first_probability.shape != second_probability.shape:
        raise ValueError("independent pattern tensors must have equal shape")
    if first_probability.ndim < 1 or first_probability.shape[-1] != 16:
        raise ValueError("four-SPAD pattern tensors must end in 16 patterns")
    completed = []
    for name, value in (("first", first_probability), ("second", second_probability)):
        if torch.any(~torch.isfinite(value)) or torch.any(value < 0.0):
            raise ValueError(f"{name} pattern probability must be finite and non-negative")
        mass = value.sum(dim=-1)
        if torch.any(mass > 1.0 + 1.0e-5) or torch.any(mass <= 0.0):
            raise ValueError(f"{name} pattern probability must have mass in (0,1]")
        no_click = torch.zeros_like(value)
        no_click[..., 0] = (1.0 - mass).clamp_min(0.0)
        completed.append(value + no_click)
    first, second = completed
    output = torch.zeros_like(first)
    for first_index in range(16):
        for second_index in range(16):
            output[..., first_index | second_index] = (
                output[..., first_index | second_index]
                + first[..., first_index]
                * second[..., second_index]
            )
    return output / output.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def fpga_read_window_statistics(
    signal_probability: "torch.Tensor",
    signal_sector_probability: "torch.Tensor",
    background_clock_pattern_probability: "torch.Tensor",
    single_source_clock_pattern_probability: "torch.Tensor",
    double_source_same_clock_pattern_probability: "torch.Tensor",
    tail_source_clock_pattern_probability: "torch.Tensor",
    *,
    double_source_same_clock_probability: float | "torch.Tensor" = 1.0,
    accepted_clock_count: int = READ_WINDOW_ACCEPTED_CLOCK_COUNT,
) -> FPGAReadWindowStatistics:
    """Map source sectors to the FPGA's asymmetric read-window statistics.

    ``signal_sector_probability`` is conditional on an external herald and is
    ordered ``(n=1, n=2, n>=3)``.  Source-clock pattern inputs already include
    same-clock background/dark OR.  For ``n=2`` the photons may share one
    10 ns clock or occupy two clocks; this temporal overlap is explicit rather
    than being confused with the 200 ns external-herald association window.
    """

    _require_torch()
    if int(accepted_clock_count) < 2:
        raise ValueError("accepted_clock_count must be at least two")
    prefix = signal_probability.shape
    if signal_sector_probability.shape != (*prefix, 3):
        raise ValueError("signal_sector_probability must end in (n1,n2,tail)")
    pattern_shape = (*prefix, 16)
    pattern_values = (
        background_clock_pattern_probability,
        single_source_clock_pattern_probability,
        double_source_same_clock_pattern_probability,
        tail_source_clock_pattern_probability,
    )
    if any(value.shape != pattern_shape for value in pattern_values):
        raise ValueError("all read-clock pattern tensors must match signal shape plus 16")
    if torch.any(~torch.isfinite(signal_probability)) or torch.any(
        (signal_probability < 0.0) | (signal_probability > 1.0)
    ):
        raise ValueError("signal_probability must lie in [0,1]")
    if torch.any(~torch.isfinite(signal_sector_probability)) or torch.any(
        signal_sector_probability < 0.0
    ):
        raise ValueError("signal-sector probabilities must be finite and non-negative")
    if torch.any(torch.abs(signal_sector_probability.sum(dim=-1) - 1.0) > 1.0e-5):
        raise ValueError("signal-sector probabilities must sum to one")
    for value in pattern_values:
        if torch.any(~torch.isfinite(value)) or torch.any(value < 0.0):
            raise ValueError("read-clock patterns must be finite and non-negative")
        if torch.any(torch.abs(value.sum(dim=-1) - 1.0) > 1.0e-5):
            raise ValueError("read-clock patterns must sum to one")

    overlap = torch.as_tensor(
        double_source_same_clock_probability,
        dtype=signal_probability.dtype,
        device=signal_probability.device,
    )
    try:
        overlap = torch.broadcast_to(overlap, prefix)
    except RuntimeError as exc:
        raise ValueError("double-source clock overlap cannot broadcast to signal shape") from exc
    if torch.any(~torch.isfinite(overlap)) or torch.any(
        (overlap < 0.0) | (overlap > 1.0)
    ):
        raise ValueError("double-source same-clock probability must lie in [0,1]")

    clock_count = int(accepted_clock_count)
    background = background_clock_pattern_probability
    single = single_source_clock_pattern_probability
    double_same = double_source_same_clock_pattern_probability
    tail = tail_source_clock_pattern_probability
    double_clock_expectation = (
        overlap[..., None] * (double_same + (clock_count - 1) * background)
        + (1.0 - overlap[..., None])
        * (2.0 * single + (clock_count - 2) * background)
    )
    sector_clock_expectation = torch.stack(
        (
            single + (clock_count - 1) * background,
            double_clock_expectation,
            tail + (clock_count - 1) * background,
        ),
        dim=-2,
    )
    signal_origin_pattern = (
        signal_probability[..., None, None]
        * signal_sector_probability[..., :, None]
        * sector_clock_expectation
    )
    background_origin_pattern = (
        (1.0 - signal_probability)[..., None]
        * clock_count
        * background
    )
    expected_pattern_by_origin = torch.cat(
        (background_origin_pattern[..., None, :], signal_origin_pattern),
        dim=-2,
    )
    expected_pattern = expected_pattern_by_origin.sum(dim=-2)
    if torch.any(torch.abs(expected_pattern.sum(dim=-1) - clock_count) > 1.0e-4):
        raise RuntimeError("FPGA clock-pattern expectation does not conserve clocks")

    one_click_indices = torch.as_tensor((1, 2, 4, 8), device=signal_probability.device)
    q_background = background.index_select(-1, one_click_indices).sum(dim=-1)
    q_single = single.index_select(-1, one_click_indices).sum(dim=-1)
    q_double = double_same.index_select(-1, one_click_indices).sum(dim=-1)
    q_tail = tail.index_select(-1, one_click_indices).sum(dim=-1)
    no_singleton_background = torch.pow(1.0 - q_background, clock_count)
    singleton_no_signal = 1.0 - no_singleton_background
    singleton_n1 = 1.0 - (1.0 - q_single) * torch.pow(
        1.0 - q_background,
        clock_count - 1,
    )
    singleton_n2_same = 1.0 - (1.0 - q_double) * torch.pow(
        1.0 - q_background,
        clock_count - 1,
    )
    singleton_n2_separate = 1.0 - torch.square(1.0 - q_single) * torch.pow(
        1.0 - q_background,
        clock_count - 2,
    )
    singleton_n2 = overlap * singleton_n2_same + (1.0 - overlap) * singleton_n2_separate
    singleton_tail = 1.0 - (1.0 - q_tail) * torch.pow(
        1.0 - q_background,
        clock_count - 1,
    )
    singleton_signal = (
        signal_sector_probability
        * torch.stack((singleton_n1, singleton_n2, singleton_tail), dim=-1)
    ).sum(dim=-1)

    pair_indices = torch.as_tensor((3, 5, 9, 6, 10, 12), device=signal_probability.device)
    pattern_indices = torch.arange(16, device=signal_probability.device)
    click_count = torch.stack(
        tuple((pattern_indices >> bit) & 1 for bit in range(4)),
        dim=-1,
    ).sum(dim=-1)
    expected_pairs = expected_pattern.index_select(-1, pair_indices)
    expected_pairs_by_origin = expected_pattern_by_origin.index_select(
        -1,
        pair_indices,
    )
    expected_multi = torch.stack(
        (
            expected_pattern[..., click_count == 2].sum(dim=-1),
            expected_pattern[..., click_count == 3].sum(dim=-1),
            expected_pattern[..., click_count == 4].sum(dim=-1),
        ),
        dim=-1,
    )
    expected_multi_by_origin = torch.stack(
        (
            expected_pattern_by_origin[..., click_count == 2].sum(dim=-1),
            expected_pattern_by_origin[..., click_count == 3].sum(dim=-1),
            expected_pattern_by_origin[..., click_count == 4].sum(dim=-1),
        ),
        dim=-1,
    )
    if not torch.allclose(expected_pairs.sum(dim=-1), expected_multi[..., 0], atol=1e-6):
        raise RuntimeError("six exact-pair clocks do not sum to multi2")
    if not torch.allclose(
        expected_pattern_by_origin.sum(dim=-2),
        expected_pattern,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError("FPGA pattern-origin decomposition lost clock mass")
    if not torch.allclose(
        expected_pairs_by_origin.sum(dim=-2),
        expected_pairs,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError("FPGA pair-origin decomposition lost exact-two mass")
    if not torch.allclose(
        expected_multi_by_origin.sum(dim=-2),
        expected_multi,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError("FPGA multi-origin decomposition lost multi-click mass")
    return FPGAReadWindowStatistics(
        singleton_given_signal_probability=singleton_signal.clamp(0.0, 1.0),
        singleton_given_no_signal_probability=singleton_no_signal.clamp(0.0, 1.0),
        singleton_probability_by_signal_sector=torch.stack(
            (singleton_n1, singleton_n2, singleton_tail),
            dim=-1,
        ).clamp(0.0, 1.0),
        expected_pattern_clock_counts_per_trial=expected_pattern,
        expected_exact_pair_clock_counts_per_trial=expected_pairs,
        expected_multi_clock_counts_per_trial=expected_multi,
        expected_pattern_clock_counts_per_trial_by_origin=(
            expected_pattern_by_origin
        ),
        expected_exact_pair_clock_counts_per_trial_by_origin=(
            expected_pairs_by_origin
        ),
        expected_multi_clock_counts_per_trial_by_origin=expected_multi_by_origin,
    )


def build_fpga_joint_four_cell_counts(
    read_count: "torch.Tensor",
    signal_count: "torch.Tensor",
    singleton_idler_count: "torch.Tensor",
    coin_count: "torch.Tensor",
    *,
    operation_mask: Optional["torch.Tensor"] = None,
) -> FPGAJointFourCellCounts:
    """Build and fail-closed validate the FPGA joint four-cell table.

    ``read_count`` may be ``[B]``, ``[B, 1]`` or ``[B, O]``.  The remaining
    tensors must be ``[B, O]`` and contain non-negative integer-valued counts.
    """

    _require_torch()
    if signal_count.ndim != 2:
        raise ValueError("signal_count must have shape [batch, operation]")
    expected_shape = signal_count.shape
    for name, value in (
        ("singleton_idler_count", singleton_idler_count),
        ("coin_count", coin_count),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if read_count.shape == (expected_shape[0],):
        read = read_count[:, None].expand(expected_shape)
    elif read_count.shape == (expected_shape[0], 1):
        read = read_count.expand(expected_shape)
    elif read_count.shape == expected_shape:
        read = read_count
    else:
        raise ValueError(
            "read_count must have shape [batch], [batch, 1], or [batch, operation]"
        )
    dtype = coin_count.dtype
    device = coin_count.device
    read = read.to(device=device, dtype=dtype)
    signal = signal_count.to(device=device, dtype=dtype)
    singleton = singleton_idler_count.to(device=device, dtype=dtype)
    coin = coin_count.to(device=device, dtype=dtype)
    mask = (
        torch.ones(expected_shape, dtype=torch.bool, device=device)
        if operation_mask is None
        else operation_mask.to(device=device, dtype=torch.bool)
    )
    if mask.shape != expected_shape:
        raise ValueError(f"operation_mask must have shape {expected_shape}")
    for name, value in (
        ("read_count", read),
        ("signal_count", signal),
        ("singleton_idler_count", singleton),
        ("coin_count", coin),
    ):
        if torch.any(mask & (~torch.isfinite(value) | (value < 0.0))):
            raise ValueError(f"valid {name} must be finite and non-negative")
        if torch.any(mask & (torch.abs(value - torch.round(value)) > 1.0e-6)):
            raise ValueError(f"valid {name} must be integer-valued")
    cells = torch.stack(
        (
            read - signal - singleton + coin,
            signal - coin,
            singleton - coin,
            coin,
        ),
        dim=-1,
    )
    if torch.any(mask[..., None] & (cells < 0.0)):
        raise ValueError(
            "FPGA joint counts violate C<=S,I or S+I-C<=R support"
        )
    if not torch.equal(
        torch.round(cells.sum(dim=-1)[mask]),
        torch.round(read[mask]),
    ):
        raise RuntimeError("FPGA four-cell counts do not conserve read trials")
    return FPGAJointFourCellCounts(counts=cells, valid_mask=mask)


def fpga_joint_probabilities(
    signal_probability: "torch.Tensor",
    singleton_given_signal_probability: "torch.Tensor",
    singleton_given_no_signal_probability: "torch.Tensor",
    *,
    probability_epsilon: float = 1.0e-6,
) -> "torch.Tensor":
    """Return the four exclusive FPGA event probabilities.

    The parameterization follows the physical conditional tree
    ``P(S) P(I_singleton | S)`` and ``P(I_singleton | not S)``.  It avoids
    pretending that signal, idler and coin are independent likelihood terms.
    """

    _require_torch()
    if not (
        signal_probability.shape
        == singleton_given_signal_probability.shape
        == singleton_given_no_signal_probability.shape
    ):
        raise ValueError("all FPGA joint probabilities must have equal shape")
    eps = float(probability_epsilon)
    if not 0.0 < eps < 0.01:
        raise ValueError("probability_epsilon must lie in (0, 0.01)")
    signal = signal_probability.clamp(eps, 1.0 - eps)
    singleton_signal = singleton_given_signal_probability.clamp(eps, 1.0 - eps)
    singleton_no_signal = singleton_given_no_signal_probability.clamp(
        eps,
        1.0 - eps,
    )
    probability = torch.stack(
        (
            (1.0 - signal) * (1.0 - singleton_no_signal),
            signal * (1.0 - singleton_signal),
            (1.0 - signal) * singleton_no_signal,
            signal * singleton_signal,
        ),
        dim=-1,
    )
    return probability / probability.sum(dim=-1, keepdim=True)


def _beta_binomial_log_probability(
    success: "torch.Tensor",
    trials: "torch.Tensor",
    probability: "torch.Tensor",
    concentration: "torch.Tensor",
) -> "torch.Tensor":
    if not bool(torch.all(torch.isfinite(probability)).detach()) or bool(
        torch.any((probability <= 0.0) | (probability >= 1.0)).detach()
    ):
        raise ValueError("beta-binomial probability must be finite and strictly interior")
    if not bool(torch.all(torch.isfinite(concentration)).detach()) or bool(
        torch.any(concentration <= 0.0).detach()
    ):
        raise ValueError("beta-binomial concentration must be finite and positive")
    alpha = probability * concentration
    beta = (1.0 - probability) * concentration
    return (
        torch.lgamma(trials + 1.0)
        - torch.lgamma(success + 1.0)
        - torch.lgamma(trials - success + 1.0)
        + torch.lgamma(success + alpha)
        + torch.lgamma(trials - success + beta)
        - torch.lgamma(trials + alpha + beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )


def nested_beta_binomial_log_likelihood(
    four_cell_counts: FPGAJointFourCellCounts,
    signal_probability: "torch.Tensor",
    singleton_given_signal_probability: "torch.Tensor",
    singleton_given_no_signal_probability: "torch.Tensor",
    concentration: "torch.Tensor",
) -> "torch.Tensor":
    """Overdispersed FPGA joint likelihood, summed over operations per row.

    It factorizes the exact four-cell table into ``S|R``, ``C|S`` and
    ``(I-C)|(R-S)``.  This is a conditional composite likelihood with one
    shared concentration, not three independent count objectives.
    """

    _require_torch()
    counts = four_cell_counts.counts
    expected_shape = counts.shape[:-1]
    for name, value in (
        ("signal_probability", signal_probability),
        ("singleton_given_signal_probability", singleton_given_signal_probability),
        ("singleton_given_no_signal_probability", singleton_given_no_signal_probability),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if concentration.ndim == 0:
        concentration_value = concentration.expand(expected_shape)
    elif concentration.shape == (expected_shape[0],):
        concentration_value = concentration[:, None].expand(expected_shape)
    elif concentration.shape == (expected_shape[0], 1):
        concentration_value = concentration.expand(expected_shape)
    elif concentration.shape == expected_shape:
        concentration_value = concentration
    else:
        raise ValueError("concentration has an incompatible shape")
    n00, n10, n01, n11 = counts.unbind(dim=-1)
    signal_trials = n10 + n11
    no_signal_trials = n00 + n01
    read_trials = signal_trials + no_signal_trials
    log_probability = (
        _beta_binomial_log_probability(
            signal_trials,
            read_trials,
            signal_probability,
            concentration_value,
        )
        + _beta_binomial_log_probability(
            n11,
            signal_trials,
            singleton_given_signal_probability,
            concentration_value,
        )
        + _beta_binomial_log_probability(
            n01,
            no_signal_trials,
            singleton_given_no_signal_probability,
            concentration_value,
        )
    )
    log_probability = torch.where(
        four_cell_counts.valid_mask,
        log_probability,
        torch.zeros_like(log_probability),
    )
    return log_probability.sum(dim=-1)


def multinomial_four_cell_log_likelihood(
    four_cell_counts: FPGAJointFourCellCounts,
    four_cell_probability: "torch.Tensor",
) -> "torch.Tensor":
    """Strict multinomial comparator for the same FPGA four-cell table."""

    _require_torch()
    if four_cell_probability.shape != four_cell_counts.counts.shape:
        raise ValueError("four_cell_probability must match four-cell counts")
    probability = four_cell_probability.clamp_min(1.0e-12)
    probability = probability / probability.sum(dim=-1, keepdim=True)
    counts = four_cell_counts.counts
    trials = counts.sum(dim=-1)
    log_probability = (
        torch.lgamma(trials + 1.0)
        - torch.lgamma(counts + 1.0).sum(dim=-1)
        + (counts * torch.log(probability)).sum(dim=-1)
    )
    log_probability = torch.where(
        four_cell_counts.valid_mask,
        log_probability,
        torch.zeros_like(log_probability),
    )
    return log_probability.sum(dim=-1)


__all__ = [
    "EXTERNAL_HERALD_SINGLETON_GATE_S",
    "FPGA_JOINT_CELL_NAMES",
    "FPGA_JOINT_OBSERVATION_SCHEMA",
    "FPGAJointFourCellCounts",
    "FPGAReadWindowStatistics",
    "READ_MULTI_SIMULTANEITY_CLOCK_S",
    "READ_WINDOW_ACCEPTED_CLOCK_COUNT",
    "READ_WINDOW_ACCEPTED_SUPPORT_S",
    "build_fpga_joint_four_cell_counts",
    "combine_independent_on_off_patterns",
    "fpga_joint_probabilities",
    "fpga_read_window_statistics",
    "multinomial_four_cell_log_likelihood",
    "nested_beta_binomial_log_likelihood",
]
