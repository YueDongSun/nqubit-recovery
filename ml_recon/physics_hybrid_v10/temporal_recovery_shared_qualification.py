"""Qualify the supervised pair head on public native observations before routing its predicted partitions."""

from __future__ import annotations
import argparse
import hashlib
import itertools
import numpy as np
from pathlib import Path
import torch
from scipy.stats import beta

from nqrec.physics.single_event_mc import sample_counts
from .effective_projector_law import effective_probabilities
from .effective_projector_observation import make_public_observation
from .temporal_recovery_environment import (
    CALIBRATION,
    PRIOR_SD,
    PreparationEnvironment,
    allocation,
    centered,
    state_vector,
)
from .temporal_recovery_evaluation import atomic_json, paired_interval
from .temporal_recovery_policy import legal_map
from .temporal_recovery_training import load_classifier
from .temporal_recovery_long_training import checkpoint_digest

QUAL_VERSION = "shared_native_pair_head_qualification_v1"


def campaign_seed_base(namespace):
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("a new nonempty campaign namespace is required")
    token = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:4], "big")
    return 30_000_000_000_000 + token * 100_000_000


def check_source(export, lane, campaign_namespace=None):
    identity = export.get("identity", {})
    arm = {"b0": "state", "b2": "joint"}[lane]
    expected_lane = lane
    expected_seed = 73
    if campaign_namespace is not None:
        campaign_seed_base(campaign_namespace)
        expected_lane = f"{arm}_cs_32768_s7301"
        expected_seed = 7301
        if (identity.get("split_namespace") != campaign_namespace
                or identity.get("train_states_per_n") != 32768
                or identity.get("train_namespace", campaign_namespace) != campaign_namespace):
            raise ValueError("wrong matrix cohort or TRAIN bank for the new shared anchor")
    if (
        identity.get("lane") != expected_lane
        or export.get("updates") != 32768
        or identity.get("seed") != expected_seed
        or identity.get("sizes") != [3, 4, 8]
        or export["classifier_config"]["profile"] != "C-S"
        or export.get("law") != "effective_single_event_v1"
        or identity.get("arm") != {"b0": "state", "b2": "joint"}[lane]
    ):
        raise ValueError("expected the registered 32768-update shared C-S B0/B2 core")


def qualified_route(receipt, lane, digest):
    """Recompute the registered decision, including independent units and interval endpoints."""
    if (
        receipt.get("schema") != QUAL_VERSION
        or receipt.get("lane") != lane
        or receipt.get("core_digest") != digest
        or receipt.get("upstream_updates") != 32768
        or receipt.get("pair_output") != "pair_entanglement_logits"
        or type(receipt.get("allow_partition")) != bool
    ):
        raise ValueError("unmatched supervised pair qualification")
    if lane == "b0":
        if receipt["allow_partition"] or receipt["systems"]:
            raise ValueError(
                "state-primary upstream cannot supply qualified partition scores"
            )
        return False
    if set(receipt["systems"]) != {"3", "4"}:
        raise ValueError("both systems require a structural qualification result")
    decisions = []
    for row in receipt["systems"].values():
        if row["dev_hidden_bipartitions"] != 256 or row["dev_pure_controls"] != 256:
            raise ValueError("incorrect ESM DEV independent state count")
        brier = row["within_bipartition_pair_brier_minus_train_edge_prior"]
        legal = row["legal_map_exact_minus_train_modal"]
        if brier["family_size"] != 6 or legal["family_size"] != 6:
            raise ValueError("incorrect structure qualification family")
        passed = (
            brier["family_ci95"][1] < 0
            and legal["family_ci95"][0] > 0
            and row["false_split_upper"] <= 0.10
        )
        if passed != row["qualified"]:
            raise ValueError("stored B2 decision disagrees with its DEV metrics")
        decisions.append(passed)
    if all(decisions) != receipt["allow_partition"]:
        raise ValueError(
            "shared partition flag disagrees with both-system qualification"
        )
    return all(decisions)


def panel(n, seed):
    env = PreparationEnvironment(n)
    rng = np.random.default_rng(seed)
    x = centered(torch.tensor(rng.normal(size=(512, n, 2)) * np.asarray(PRIOR_SD)))
    c = state_vector(x)
    rho = c[:, :, None] * c[:, None, :].conj()
    candidates = [
        (0, *suffix)
        for suffix in itertools.product([0, 1], repeat=n - 1)
        if any(suffix)
    ]
    chosen = rng.integers(len(candidates), size=256)
    labels = torch.tensor([(0,) * n] * 256 + [candidates[j] for j in chosen])
    truth = labels[:, :, None] == labels[:, None, :]
    rho = rho * truth
    settings = env.analyzers.expand(512, -1, -1)
    calib = torch.tensor(CALIBRATION, dtype=torch.float64).expand(512, -1)
    cells = effective_probabilities(rho, settings, calib)
    shots = int(allocation(n)[0])
    counts = torch.from_numpy(sample_counts(cells, shots, rng))
    public = make_public_observation(
        counts=counts,
        shots=counts.sum(-1),
        analyzers=env.analyzers,
        prepared=torch.ones(512, n, dtype=torch.float64) / n**0.5,
        calibration=calib,
        visible_mask=torch.ones(512, len(env.analyzers), dtype=torch.bool),
    )
    return public, truth


def qualify(core_path, lane, output, campaign_namespace=None):
    torch.set_num_threads(1)
    model, export = load_classifier(core_path, "cpu")
    check_source(export, lane, campaign_namespace)
    rec = {
        "schema": QUAL_VERSION,
        "lane": lane,
        "source_identity": export["identity"],
        "upstream_updates": export["updates"],
        "core_digest": checkpoint_digest(core_path),
        "pair_output": "pair_entanglement_logits",
        "decision_class": "synthetic DEV qualification; no control confirmation",
        "allow_partition": False,
        "systems": {},
    }
    if lane == "b0":
        rec["reason"] = (
            "Upstream state-primary objective has no labelled pair supervision."
        )
    else:
        for n in (3, 4):
            offset = campaign_seed_base(campaign_namespace) if campaign_namespace else 0
            _, tr = panel(n, offset + 941308 + n)
            public, truth = panel(n, offset + 941407 + n)
            mask = torch.ones(n, n, dtype=torch.bool).triu(1)
            train_pairs = tr[256:, mask]
            prior = train_pairs.float().mean(0)
            unique, counts = torch.unique(train_pairs, dim=0, return_counts=True)
            modal = unique[counts.argmax()]
            positives = []
            maps = []
            for start in range(0, len(truth), 32):
                with torch.inference_mode():
                    batch = {k: v[start : start + 32] for k, v in public.items()}
                    out = model(batch, N=n, fine_identifiable=False)
                    logits = out["pair_entanglement_logits"]
                    probs = out["pair_entanglement_probs"]
                    positives.append(probs[:, mask])
                    maps.append(legal_map(logits)[:, mask].bool())
            prob = torch.cat(positives)
            pred = torch.cat(maps)
            actual = truth[256:, mask]
            brier = (
                (
                    (prob[256:] - actual.float()) ** 2 - (prior - actual.float()) ** 2
                ).mean(1)
            ).numpy()
            accuracy = (
                pred[256:].eq(actual).all(1).float() - modal.eq(actual).all(1).float()
            ).numpy()
            diff1 = paired_interval(brier, 6)
            diff2 = paired_interval(accuracy, 6)
            false_count = int((~pred[:256].all(1)).sum())
            ub = (
                1.0
                if false_count == 256
                else float(beta.ppf(1 - 0.025 / 6, false_count + 1, 256 - false_count))
            )
            passed = (
                diff1["family_ci95"][1] < 0
                and diff2["family_ci95"][0] > 0
                and ub <= 0.10
            )
            rec["systems"][str(n)] = {
                "dev_hidden_bipartitions": 256,
                "dev_pure_controls": 256,
                "within_bipartition_pair_brier_minus_train_edge_prior": diff1,
                "legal_map_exact_minus_train_modal": diff2,
                "false_split_fraction_on_pure": false_count / 256,
                "false_split_upper": ub,
                "brier_model": float(((prob[256:] - actual.float()) ** 2).mean()),
                "legal_map_accuracy": float(
                    pred[256:].eq(actual).all(1).float().mean()
                ),
                "qualified": passed,
            }
        rec["allow_partition"] = all(x["qualified"] for x in rec["systems"].values())
    output.parent.mkdir(exist_ok=True, parents=True)
    atomic_json(output, rec)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core", type=Path, required=True)
    p.add_argument("--lane", choices=("b0", "b2"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--campaign-namespace", help="qualify the new seed-7301 matrix anchors")
    a = p.parse_args()
    qualify(a.core, a.lane, a.output, a.campaign_namespace)
