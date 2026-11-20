"""Select seven-candidate FD interventions from frozen public inputs and score them offline.

The legacy learned rule selects the highest-utility valid candidate (including
NOOP), then replaces that choice by NOOP when predicted harm exceeds 0.5.
A different threshold must be fixed on separate development data. Candidate
gains describe conditional single-excitation fidelity under the declared action model.
"""

from __future__ import annotations

import torch

from .training_ideas_experiment import _candidate_plugin_scores


@torch.no_grad()
def select_fd_actions(
    output: dict,
    public: dict,
    *,
    harm_threshold: float = 0.5,
    supported_candidates: torch.Tensor | None = None,
) -> dict:
    """Select four policies before consulting any counterfactual outcomes.

    An optional public support mask excludes unexecutable candidates identically
    from every policy and the later oracle. Candidate coordinates are never
    projected or modified after having been scored.
    """
    if not 0 <= harm_threshold <= 1:
        raise ValueError("harm threshold must lie in [0,1]")
    actions = public["public_candidate_actions"]
    valid = public["public_candidate_valid_mask"]
    if (
        actions.ndim != 4
        or actions.shape[1] != 7
        or actions.shape[3] != 2
        or valid.shape != actions.shape[:2]
        or valid.dtype != torch.bool
        or not len(actions)
    ):
        raise ValueError("expected seven public candidates and one Boolean valid mask per state")
    if supported_candidates is not None:
        if supported_candidates.shape != valid.shape or supported_candidates.dtype != torch.bool:
            raise ValueError("supported-candidate mask must match the Boolean validity mask")
        valid = valid & supported_candidates
    noop = torch.as_tensor(public["public_noop_candidate_index"], device=actions.device)
    if noop.dtype not in (torch.int32, torch.int64) or noop.shape not in ((), (len(actions),)):
        raise ValueError("NOOP index must be integer, scalar or one per state")
    noop = noop.long().expand(len(actions))
    if ((noop < 0) | (noop >= 7)).any():
        raise ValueError("NOOP outside candidate library")
    if not torch.isfinite(actions[valid]).all():
        raise ValueError("valid actions must be finite")
    zeros = (actions == 0).all(dim=(-1, -2))
    ids = torch.arange(len(actions), device=actions.device)
    if not (valid[ids, noop] & zeros[ids, noop]).all() or ((valid & zeros).sum(-1) != 1).any():
        raise ValueError("each row must contain exactly one supported exact NOOP")
    utility = output["idea_fd_logits"]
    harm_logits = output["idea_fd_harm_logits"]
    if (
        utility.shape != valid.shape
        or harm_logits.shape != valid.shape
        or not torch.isfinite(utility[valid]).all()
        or not torch.isfinite(harm_logits[valid]).all()
    ):
        raise ValueError(
            "finite utility and harm predictions are required for every valid candidate"
        )
    # Exclude any private fields carried alongside public inputs in historical batches.
    fields = {
        "public_control_target": public["public_control_target"],
        "public_candidate_actions": actions,
        "public_candidate_valid_mask": valid,
    }
    plugin = _candidate_plugin_scores(output, fields)
    if not torch.isfinite(plugin[valid]).all():
        raise ValueError("nonfinite physical plug-in score on valid candidate")
    ranked = utility.masked_fill(~valid, -torch.inf).argmax(-1)
    probability = harm_logits.sigmoid()
    gated = torch.where(probability[ids, ranked] > harm_threshold, noop, ranked)
    return {
        "selected": {
            "noop": noop,
            "physical_plugin": plugin.argmax(-1),
            "learned": ranked,
            "learned_gated": gated,
        },
        "valid": valid,
        "noop": noop,
        "predicted_gain": utility,
        "predicted_harm": probability,
        "harm_threshold": harm_threshold,
    }


@torch.no_grad()
def score_fd_actions(decisions: dict, true_gain: torch.Tensor) -> dict:
    """Score selected actions after their IDs are frozen, preserving paired per-state values."""
    valid, noop = decisions["valid"], decisions["noop"]
    if true_gain.shape != valid.shape or not torch.isfinite(true_gain[valid]).all():
        raise ValueError("counterfactual gain must be finite for every eligible candidate")
    ids = torch.arange(len(noop), device=noop.device)
    if not (true_gain[ids, noop].abs() <= 1e-12).all():
        raise ValueError("NOOP must have zero counterfactual gain")
    oracle = true_gain.masked_fill(~valid, -torch.inf).amax(-1)
    harm = true_gain < -1e-12  # Same harmful-action label as the projector FD teacher.
    policies = {}
    for name, choice in decisions["selected"].items():
        if choice.shape != noop.shape or ((choice < 0) | (choice >= valid.shape[1])).any():
            raise ValueError("invalid selected candidate index")
        if not valid[ids, choice].all():
            raise ValueError("selected candidate outside the public executable set")
        executed = choice != noop
        gain = true_gain[ids, choice]
        bad = harm[ids, choice]
        regret = oracle - gain
        n = int(executed.sum())
        policies[name] = {
            "rows": {
                "candidate": choice,
                "executed": executed,
                "gain": gain,
                "harm": bad,
                "regret": regret,
            },
            "summary": {
                "n_states": len(ids),
                "n_executed": n,
                "n_harmed": int(bad.sum()),
                "execution_coverage": n / len(ids),
                "gain_all_states": float(gain.mean()),
                "harm_all_states": float(bad.double().mean()),
                "regret_all_states": float(regret.mean()),
                "gain_given_execution": float(gain[executed].mean()) if n else None,
                "harm_given_execution": float(bad[executed].double().mean()) if n else None,
                "regret_given_execution": float(regret[executed].mean()) if n else None,
            },
        }
    nonzero = valid.clone()
    nonzero[ids, noop] = False
    calibration = {}
    for label, mask in (("eligible_candidates", valid), ("eligible_nonzero_candidates", nonzero)):
        count = int(mask.sum())
        predicted = decisions["predicted_harm"][mask]
        actual = harm[mask].to(predicted.dtype)
        residual = decisions["predicted_gain"][mask] - true_gain[mask]
        calibration[label] = {
            "count": count,
            "utility_rmse": float(residual.square().mean().sqrt()) if count else None,
            "harm_brier": float((predicted - actual).square().mean()) if count else None,
            "mean_predicted_harm": float(predicted.mean()) if count else None,
            "observed_harm": float(actual.mean()) if count else None,
        }
    return {"policies": policies, "candidate_calibration": calibration, "oracle_gain": oracle}


@torch.no_grad()
def fd_reliability_rows(decisions: dict, true_gain: torch.Tensor) -> dict[str, torch.Tensor]:
    """One record per eligible (state index, candidate); retain NOOP and pre-gate rank.

    State indices refer to the paired order in the same frozen evaluation file.
    Private gains and harm labels are attached only after selection has finished.
    """
    valid, noop = decisions["valid"], decisions["noop"]
    if true_gain.shape != valid.shape or not torch.isfinite(true_gain[valid]).all():
        raise ValueError("each eligible candidate needs a finite counterfactual gain")
    state, candidate = valid.nonzero(as_tuple=True)
    ranked = decisions["selected"]["learned"]
    is_noop = candidate == noop[state]
    gain = true_gain[state, candidate]
    if (gain[is_noop].abs() > 1e-12).any():
        raise ValueError("NOOP must have zero counterfactual gain")
    return {
        "state_index": state,
        "candidate_index": candidate,
        "is_noop": is_noop,
        "ranked_before_gate": candidate == ranked[state],
        "predicted_harm": decisions["predicted_harm"][state, candidate],
        "actual_harm": gain < -1e-12,
        "predicted_delta_f": decisions["predicted_gain"][state, candidate],
        "actual_delta_f": gain,
    }


@torch.no_grad()
def summarize_fd_reliability(rows: dict, *, harm_thresholds: list[float]) -> dict:
    """Fixed deciles for candidates and thresholded risks per independent state.

    The fit's pre-gate first choice is retained at equality with each supplied
    ex ante risk threshold. Rejecting a proposal executes NOOP. Bins use [lo,hi)
    except for the closed right endpoint in the final bin.
    """
    p = rows["predicted_harm"].double()
    harm = rows["actual_harm"]
    ranked = rows["ranked_before_gate"]
    active = ~rows["is_noop"]
    if not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("candidate harm probabilities must lie in [0,1]")
    states = rows["state_index"].unique()
    if not len(states) or not torch.equal(
        rows["state_index"][ranked].sort().values, states.sort().values
    ):
        raise ValueError("expected exactly one ranked candidate per independent state")
    # p=1 belongs in the last interval, while p=0.5 begins the sixth interval.
    bin_index = (10 * p).floor().long().clamp_max(9)
    bins = {}
    for name, group in (
        ("eligible_nonzero_candidates", active),
        ("ranked_candidates_including_noop", ranked),
        ("ranked_nonzero_candidates", ranked & active),
    ):
        records = []
        for i in range(10):
            keep = group & (bin_index == i)
            n = int(keep.sum())
            predicted, observed = p[keep], harm[keep].double()
            records.append(
                {
                    "lo": i / 10,
                    "hi": (i + 1) / 10,
                    "n": n,
                    "predicted_rate": float(predicted.mean()) if n else None,
                    "observed_rate": float(observed.mean()) if n else None,
                    "brier": float((predicted - observed).square().mean()) if n else None,
                }
            )
        bins[name] = records
    gates = []
    for threshold in harm_thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("predeclared harm threshold must lie in [0,1]")
        acted = ranked & active & (p <= threshold)
        n_executed, n_harmed = int(acted.sum()), int((acted & harm).sum())
        gates.append(
            {
                "harm_threshold": float(threshold),
                "n_states": len(states),
                "n_executed": n_executed,
                "n_harmed": n_harmed,
                "execution_coverage": n_executed / len(states),
                "harm_all_states": n_harmed / len(states),
                "harm_given_execution": n_harmed / n_executed if n_executed else None,
            }
        )
    return {"reliability_bins": bins, "thresholds": gates}
