"""TRAIN-only structure comparators and separately calibrated single-action FD diagnostics."""

import math
from collections import Counter
from functools import lru_cache

import numpy as np
import torch

from .fd_policy_evaluation import (
    fd_reliability_rows,
    select_fd_actions,
    summarize_fd_reliability,
)
from .long_projector_data import CHUNK, bank_seed, build_bank
from .long_projector_prior import training_seen_mask
from .training_ideas_data import _states, public_projector_inputs
from .training_ideas_geometry import public_control_candidates

STRONG_CALIBRATION = (0.025, 0.65, 0.002, 0.35, 0.002)
WEAK_CALIBRATION = (0.025, 0.65, 0.002, 0.04, 0.002)
TARGET_COVERAGES = (0.10, 0.25, 0.50)
CAL_SCHEMA = "projector_fd_strong_cal_v1"


@lru_cache(maxsize=3)
def _legal_pairs(n):
    """Lexicographically ordered restricted-growth partitions on the unordered pair axis."""
    if n not in (3, 4, 8):
        raise ValueError("the matrix structure study supports N=3,4,8")
    upper = np.triu_indices(n, 1)
    patterns = []

    def visit(labels):
        if len(labels) == n:
            group = np.asarray(labels)
            patterns.append(group[upper[0]] == group[upper[1]])
            return
        for block in range(max(labels) + 2):
            visit((*labels, block))

    visit((0,))
    return np.asarray(patterns, dtype=np.bool_)


def _decode(probabilities, legal):
    # The prediction's squared norm is constant across candidate partitions.
    candidates = legal.astype(np.float64)
    distances = candidates.sum(1)[None, :] - 2 * probabilities @ candidates.T
    return legal[np.argmin(distances, axis=1)]


@lru_cache(maxsize=24)
def structure_priors(n, rows, step, train_namespace):
    """Use only distinct visited TRAIN states; statistics never read evaluation labels."""
    seen = training_seen_mask(n, rows, step).numpy()
    upper = np.triu_indices(n, 1)
    groups = {"all_train": Counter(), "hidden_train": Counter()}
    for chunk, active in enumerate(seen.reshape(-1, CHUNK)):
        if active.any():
            rng = np.random.Generator(
                np.random.Philox(bank_seed(n, "train", chunk, train_namespace))
            )
            _, _, classes, labels, _, _ = _states(n, CHUNK, rng)
            for group, select in (
                ("all_train", active),
                ("hidden_train", active & (classes == 5)),
            ):
                for bits in labels[select][:, upper[0], upper[1]]:
                    groups[group][bits.tobytes()] += 1
    priors = {}
    legal = _legal_pairs(n)
    for name, counts in groups.items():
        total = sum(counts.values())
        if total:
            frequency = (
                sum(
                    np.frombuffer(k, dtype=np.bool_) * amount
                    for k, amount in counts.items()
                )
                / total
            )
            # Ties use the same legal partition ordering as the network's decoder.
            modal = max(legal, key=lambda pattern: counts.get(pattern.tobytes(), 0))
            priors[name] = (total, frequency, modal)
        else:
            priors[name] = (0, None, None)
    return priors


def _structure_metrics(predicted, binary, truth, mask, include):
    active_pairs = mask & include[:, None]
    complete = active_pairs.all(1)
    pair_counts = active_pairs.sum(1)
    sse = np.where(
        active_pairs, (np.asarray(predicted, dtype=np.float64) - truth) ** 2, 0.0
    ).sum(1)
    exact = (binary == truth).all(1) & complete
    one_block = complete & truth.all(1)
    false_split = one_block & ~binary.all(1)
    positive = np.asarray(predicted)[truth & active_pairs]
    negative = np.sort(np.asarray(predicted)[~truth & active_pairs])
    auc = None
    if len(positive) and len(negative):
        low = np.searchsorted(negative, positive)
        high = np.searchsorted(negative, positive, side="right")
        auc = float(((low + high) / (2 * len(negative))).mean())
    return {
        "n_states": int(include.sum()),
        "n_labeled_pairs": int(pair_counts.sum()),
        "n_positive_pairs": len(positive),
        "n_negative_pairs": len(negative),
        "pair_auc_with_half_credit_for_ties": auc,
        "pair_brier": float(sse.sum() / pair_counts.sum())
        if pair_counts.any()
        else None,
        "n_complete_partitions": int(complete.sum()),
        "n_exact": int(exact.sum()),
        "exact_fraction": float(exact.sum() / complete.sum())
        if complete.any()
        else None,
        "n_true_one_block": int(one_block.sum()),
        "n_false_splits": int(false_split.sum()),
        "false_split_fraction": float(false_split.sum() / one_block.sum())
        if one_block.any()
        else None,
    }, {
        "pair_squared_error_sum": sse.tolist(),
        "pair_count": pair_counts.tolist(),
        "exact": exact.tolist(),
        "complete_partition": complete.tolist(),
        "true_one_block": one_block.tolist(),
        "false_split": false_split.tolist(),
    }


def _structure(output, bank, n, rows, step, namespace):
    upper = np.triu_indices(n, 1)
    probabilities = output["pair_entanglement_probs"].detach().cpu().double().numpy()
    probs = ((probabilities + probabilities.transpose(0, 2, 1)) / 2)[
        :, upper[0], upper[1]
    ]
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("invalid pair probabilities")
    truth = bank["target_partition"].cpu().numpy()[:, upper[0], upper[1]].astype(bool)
    mask = (
        bank["target_partition_mask"].cpu().numpy()[:, upper[0], upper[1]].astype(bool)
    )
    hidden = bank["target_operational_noise_mask"].cpu().numpy().astype(bool) & (
        bank["target_operational_noise"].cpu().numpy() == 5
    )
    legal = _legal_pairs(n)
    ones = np.ones_like(truth)
    comparators = {
        "network": (probs, _decode(probs, legal)),
        "all_in_one": (ones, ones),
    }
    priors = structure_priors(n, rows, step, namespace)
    for name, (_, frequency, modal) in priors.items():
        if frequency is not None:
            comparators[name + "_pair_frequency"] = (
                np.broadcast_to(frequency, truth.shape),
                np.broadcast_to(_decode(frequency[None], legal), truth.shape),
            )
            comparators[name + "_modal_partition"] = (
                np.broadcast_to(modal, truth.shape),
            ) * 2
    table, state_rows = {}, {}
    for group, include in (
        ("all_families", np.ones(len(probs), bool)),
        ("hidden_bipartition", hidden),
    ):
        table[group], state_rows[group] = {}, {}
        for name, (prediction, binary) in comparators.items():
            if group == "all_families" and name.startswith("hidden_train"):
                continue
            table[group][name], state_rows[group][name] = _structure_metrics(
                prediction, binary, truth, mask, include
            )
    targets = {
        k: {
            "unique_train_states": v[0],
            "pair_frequency": None if v[1] is None else v[1].tolist(),
            "modal_partition_pairs": None if v[2] is None else v[2].tolist(),
        }
        for k, v in priors.items()
    }
    report = {
        "train_states": {k: v[0] for k, v in priors.items()},
        "train_priors": targets,
        "pair_order": np.column_stack(upper).tolist(),
        "legal_partitions": len(legal),
        "comparisons": table,
        "conditional_prior_role": "diagnostic restricted to the hidden family",
    }
    return report, state_rows


def _fd_panel(model, n, states, role, namespace, device, calibration):
    bank = build_bank(
        n,
        states,
        role,
        0,
        False,
        split_namespace=namespace,
        fixed_calibration=calibration,
    )
    public = {k: v.to(device) for k, v in public_projector_inputs(bank).items()}
    candidates = public_control_candidates(
        states, n, device=torch.device("cpu"), dtype=torch.float64
    ).float()
    bank["public_candidate_actions"] = candidates
    public.update(
        public_candidate_actions=candidates.to(device),
        public_candidate_valid_mask=torch.ones(
            states, 7, dtype=torch.bool, device=device
        ),
        public_noop_candidate_index=torch.zeros(
            states, dtype=torch.long, device=device
        ),
    )
    output = model(public, N=n)
    decisions = select_fd_actions(output, public)
    return bank, decisions


def _coverage_thresholds(decisions):
    """At most floor(c × all CAL states); accept tied scores together or not at all."""
    ranked = decisions["selected"]["learned"]
    ids = torch.arange(len(ranked), device=ranked.device)
    proposed = ranked != decisions["noop"]
    probabilities = decisions["predicted_harm"][ids, ranked][proposed].double().cpu()
    unique, counts = torch.unique(probabilities, sorted=True, return_counts=True)
    cumulative = counts.cumsum(0)
    thresholds = []
    for fraction in TARGET_COVERAGES:
        goal = math.floor(fraction * len(ranked))
        possible = (cumulative <= goal).nonzero().flatten()
        accepted = int(cumulative[possible[-1]]) if possible.numel() else 0
        threshold = float(unique[possible[-1]]) if possible.numel() else None
        thresholds.append(
            {
                "target": fraction,
                "target_count_floor": goal,
                "threshold": threshold,
                "states": len(ranked),
                "proposed_nonzero": int(proposed.sum()),
                "cal_executed": accepted,
                "cal_coverage": accepted / len(ranked),
                "target_count_attained": accepted == goal and goal > 0,
                "zero_coverage": accepted == 0,
            }
        )
    return thresholds


@torch.inference_mode()
def register_matrix_calibration(model, n, split_namespace, device, *, cal_states=768):
    """Bind this result to the candidate before generating CONFIRMATION rows for the roster."""
    if model.training:
        raise ValueError("CAL requires a frozen eval-mode model")
    receipt = {
        "schema": CAL_SCHEMA,
        "n": n,
        "split_namespace": split_namespace,
        "role": "calibration",
        "cal_states": cal_states,
        "strong_calibration": list(STRONG_CALIBRATION),
        "weak_calibration": list(WEAK_CALIBRATION),
        "available": model.training_ideas_config.fd,
    }
    if receipt["available"]:
        _, decisions = _fd_panel(
            model,
            n,
            cal_states,
            "calibration",
            split_namespace,
            device,
            STRONG_CALIBRATION,
        )
        receipt["thresholds"] = _coverage_thresholds(decisions)
    return receipt


def _score_frozen_thresholds(decisions, gain, receipt):
    rows = fd_reliability_rows(decisions, gain)
    supplied = [
        point["threshold"]
        for point in receipt["thresholds"]
        if point["threshold"] is not None
    ]
    reliability = summarize_fd_reliability(rows, harm_thresholds=supplied)
    ranked = decisions["selected"]["learned"]
    ids = torch.arange(len(ranked), device=ranked.device)
    noop = decisions["noop"]
    probability = decisions["predicted_harm"][ids, ranked]
    per_state = {
        "candidate_rows": {name: value.tolist() for name, value in rows.items()}
    }
    results = []
    for point in receipt["thresholds"]:
        cutoff = point["threshold"]
        execute = (
            (ranked != noop) & (probability <= cutoff)
            if cutoff is not None
            else torch.zeros_like(noop, dtype=torch.bool)
        )
        chosen = torch.where(execute, ranked, noop)
        actual = gain[ids, chosen]
        harmful = actual < -1e-12
        count = int(execute.sum())
        results.append(
            {
                "target": point["target"],
                "threshold": cutoff,
                "states": len(ranked),
                "n_executed": count,
                "n_harmed": int(harmful.sum()),
                "coverage": count / len(ranked),
                "zero_coverage": count == 0,
                "gain_all_states": float(actual.mean()),
                "harm_all_states": float(harmful.double().mean()),
                "gain_given_execution": float(actual[execute].mean())
                if count
                else None,
                "harm_given_execution": float(harmful[execute].double().mean())
                if count
                else None,
            }
        )
        per_state[str(point["target"])] = {
            "candidate": chosen.tolist(),
            "executed": execute.tolist(),
            "gain": actual.tolist(),
            "harm": harmful.tolist(),
        }
    valid = decisions["valid"]
    nonzero = valid & (
        torch.arange(valid.shape[1], device=valid.device)[None] != noop[:, None]
    )

    def brier(mask):
        return (
            float(
                (
                    (decisions["predicted_harm"][mask] - (gain[mask] < -1e-12).double())
                    ** 2
                ).mean()
            )
            if mask.any()
            else None
        )

    return (
        {
            "by_target_coverage": results,
            "reliability_bins": reliability["reliability_bins"],
            "eligible_candidate_harm_brier": brier(valid),
            "nonzero_candidate_harm_brier": brier(nonzero),
        },
        per_state,
    )


@torch.inference_mode()
def score_matrix_diagnostics(
    model,
    confirmation_bank,
    n,
    rows,
    step,
    train_namespace,
    split_namespace,
    device,
    *,
    calibration_receipt,
):
    """Read frozen thresholds, score nominal ESM and paired strong/weak one-step decisions."""
    if model.training:
        raise ValueError("confirmation requires a frozen eval-mode model")
    receipt = calibration_receipt
    if (
        receipt["schema"] != CAL_SCHEMA
        or receipt["n"] != n
        or receipt["split_namespace"] != split_namespace
        or receipt["role"] != "calibration"
        or receipt["available"] != model.training_ideas_config.fd
        or receipt["strong_calibration"] != list(STRONG_CALIBRATION)
        or receipt["weak_calibration"] != list(WEAK_CALIBRATION)
    ):
        raise ValueError("incompatible preregistered calibration receipt")
    public = {
        k: v.to(device) for k, v in public_projector_inputs(confirmation_bank).items()
    }
    structure, structure_rows = _structure(
        model(public, N=n), confirmation_bank, n, rows, step, train_namespace
    )
    metrics, per_state = (
        {"esm_train_baselines": structure},
        {"esm_train_baselines": structure_rows},
    )
    if not receipt["available"]:
        metrics["fd_strong_weak"] = {"available": False}
        return metrics, per_state
    if [p["target"] for p in receipt["thresholds"]] != list(TARGET_COVERAGES):
        raise ValueError("incomplete CAL target registration")
    reference_mask = None
    shifted, shifted_rows = {}, {}
    for name, calibration in (
        ("strong", STRONG_CALIBRATION),
        ("weak", WEAK_CALIBRATION),
    ):
        bank, decisions = _fd_panel(
            model,
            n,
            len(public["counts"]),
            "confirmation",
            split_namespace,
            device,
            calibration,
        )
        for key in (
            "prepared_re",
            "prepared_im",
            "target_R1_re",
            "target_R1_im",
            "shots",
        ):
            if not torch.equal(bank[key], confirmation_bank[key].cpu()):
                raise ValueError(
                    f"shifted panel does not match confirmation states/exposures: {key}"
                )
        if reference_mask is not None and not torch.equal(
            reference_mask, bank["setting_mask"]
        ):
            raise ValueError("strong and weak panels must use identical query masks")
        reference_mask = bank["setting_mask"]
        if not torch.equal(
            confirmation_bank["public_candidate_actions"].cpu(),
            bank["public_candidate_actions"],
        ):
            raise ValueError(
                "counterfactual gain indices do not match the registered candidates"
            )
        shifted[name], shifted_rows[name] = _score_frozen_thresholds(
            decisions, confirmation_bank["target_fd_utility"].to(device), receipt
        )
    metrics["fd_strong_weak"] = {
        "available": True,
        "endpoint": "one_step_conditional_gain",
        "conditions": shifted,
    }
    per_state["fd_strong_weak"] = shifted_rows
    return metrics, per_state
