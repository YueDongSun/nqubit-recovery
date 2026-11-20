"""Measure a frozen classifier's response to nested corrected-law observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from nqrec.physics.single_event_mc import sample_counts

from .classifier_n_full import V10ClassifierNFull
from .effective_inference_metrics import (
    classical_estimate,
    command_and_truth,
    fit_prior,
    paired_estimate,
    prior_density,
    state_risk,
)
from .effective_projector_law import effective_probabilities
from .effective_projector_observation import EFFECTIVE_LAW, PUBLIC_SCHEMA, make_public_observation
from .projector_core_export import CLASSIFIER_SCHEMA, classifier_configuration
from .training_ideas_data import allocate_integer_budget, make_projector_training_batch
from .training_ideas_objective import normalized_state


BUDGETS = (1024, 4096, 16384)
BUDGET_PAIRS = ((1024, 4096), (4096, 16384), (1024, 16384))
NAMESPACES = {"short": 202611207110, "long": 202611207208}
ROWS = 192
BOOTSTRAP_DRAWS = 4000
FORWARD_BATCH = 32
THREADS = 2


def measurement_panels(n, rows, namespace):
    """Keep the same states, apparatus and information boundary at each budget."""
    base = make_projector_training_batch(
        n, rows, BUDGETS[0], namespace + n, include_teachers=False, observation_law=EFFECTIVE_LAW
    )
    _, truth = command_and_truth(base)
    analyzer = torch.complex(base["analyzer_re"], base["analyzer_im"])
    prepared = torch.complex(base["prepared_re"], base["prepared_im"])
    calibration = base["effective_calibration"]
    family = torch.where(base["target_is_nominal"], -1, base["target_operational_noise"]).numpy()
    probability = effective_probabilities(truth, analyzer, calibration)
    settings = analyzer.shape[1]
    query_size = min(settings - 1, max(1, round(settings * 0.25)))
    mask_rng = np.random.default_rng(np.random.SeedSequence([namespace, n, 17]))
    visible = torch.ones((rows, settings), dtype=torch.bool)
    for row in range(rows):
        visible[row, mask_rng.choice(settings, size=query_size, replace=False)] = False

    count_rng = np.random.default_rng(np.random.SeedSequence([namespace, n, 29]))
    cumulative = np.zeros(probability.shape, dtype=np.int64)
    previous = np.zeros(settings, dtype=np.int64)
    panels = {}
    for budget in BUDGETS:
        allocation = allocate_integer_budget(budget, settings)
        increment = allocation - previous
        if (increment <= 0).any():
            raise ArithmeticError("every setting must receive new observations at the next budget")
        for trials in np.unique(increment):
            selected = increment == trials
            cumulative[:, selected, :] += sample_counts(
                probability[:, selected, :], int(trials), count_rng
            )
        counts = torch.from_numpy(cumulative.copy())
        shots = torch.from_numpy(allocation.copy()).expand(rows, -1).contiguous()
        observation = make_public_observation(
            counts=counts,
            shots=shots,
            analyzers=analyzer,
            prepared=prepared,
            calibration=calibration,
            visible_mask=visible,
        )
        panels[budget] = (observation, torch.where(visible[..., None], 0, counts))
        previous = allocation
    return truth, family, panels


def read_core(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != CLASSIFIER_SCHEMA
        or payload.get("law") != EFFECTIVE_LAW
        or payload.get("observation_schema") != PUBLIC_SCHEMA
        or payload.get("sector_family") != "K0_K1_reference"
        or type(payload.get("updates")) is not int
        or payload["updates"] <= 0
    ):
        raise ValueError("expected a complete corrected-law conditional K1 core")
    return payload


def build_model(payload, profile):
    config, weights = classifier_configuration(payload)
    if config.get("profile") != profile or config.get("observation_law") != EFFECTIVE_LAW:
        raise ValueError("wrong architecture or readout law for this study")
    model = V10ClassifierNFull(**config)
    model.load_state_dict(weights, strict=True)
    return model.eval().requires_grad_(False)


def validate_long_binding(payload, selection, lane, arm):
    """Bind this experiment to the already selected matched training rung."""
    from .long_projector_data import LARGE, SIZES

    identity = payload["identity"]
    step = payload["updates"]
    if (
        identity.get("lane") != lane
        or identity.get("arm") != arm
        or identity.get("train_states_per_n") != LARGE
        or identity.get("sizes") != list(SIZES)
        or step < 256
        or step & (step - 1)
        or selection.get("lane") != lane
        or selection.get("source") != identity.get("source")
        or selection.get("selected", {}).get("matched") != f"snapshots/step_{step:09d}/core.pt"
    ):
        raise ValueError(f"{lane} core and locked matched registration disagree")


def matrix_budget_namespace(campaign):
    """Domain-separated nested WRITE/READ increments and query masks for this fresh campaign."""
    if not campaign:
        raise ValueError("a nonempty frozen campaign namespace is required")
    token = hashlib.sha256(("matrix-nested-reads-v1:" + campaign).encode()).digest()
    return int.from_bytes(token[:7], "big")


def matrix_budget_candidates(root, frozen):
    """Use only the predeclared shared-N seed-7301 large-bank anchors at update 32768."""
    from .long_projector_prior import actual_training_prior
    from .long_projector_data import train_bank_namespace
    from .temporal_recovery_shared_qualification import check_source

    root = Path(root)
    namespace = frozen["roster"]["namespace"]
    step = 32768
    models = {}
    evidence = {}
    for lane, arm in (("b0", "state"), ("b2", "joint")):
        run = f"{arm}_cs_32768_s7301"
        entry = frozen["candidates"][run]
        path = root / run / f"snapshots/step_{step:09d}/core.pt"
        if entry["choices"].get("step_32768") != step:
            raise ValueError(f"{run}: required 32768-update snapshot is not in the frozen roster")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["weights"][str(step)]:
            raise ValueError(f"{run}: frozen weight identity mismatch")
        payload = read_core(path)
        if (
            payload["identity"] != entry["identity"]
            or payload["identity"]["source"] != frozen["roster"]["source"]
        ):
            raise ValueError(f"{run}: checkpoint identity differs from frozen source")
        check_source(payload, lane, namespace)
        name = arm + "_network"
        models[name] = build_model(payload, "C-S")
        evidence[name] = {
            "run_id": run,
            "upstream_lane": lane,
            "seed": 7301,
            "profile": "C-S",
            "updates": step,
            "sha256": digest,
            "train_rows_per_n": 32768,
            "train_namespace": train_bank_namespace(entry["identity"]),
        }
    if len({e["train_namespace"] for e in evidence.values()}) != 1:
        raise ValueError("matrix budget anchors use different TRAIN banks")
    per_size = {}
    for n in (3, 4, 8):
        retention, _, seen = actual_training_prior(n, 32768, step, namespace)
        per_size[n] = (models, retention, seen)
    return per_size, evidence


def load_candidates(args):
    per_size = {}
    evidence = {}
    if args.stage == "short":
        if args.short_root is None:
            raise ValueError("the short study needs --short-root")
        for n in (3, 4):
            folder = args.short_root / f"n{n}_baseline_s61_r4"
            payload = read_core(folder / "core.pt")
            identity = payload["identity"]
            terminal = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            if (
                tuple(identity[k] for k in ("n", "arm", "seed", "train", "budget", "rank"))
                != (n, "baseline", 61, 256, 4096, 4)
                or payload["updates"] != identity["steps"]
                or payload["updates"] != 96
                or terminal["identity"] != identity
                or terminal["actual_updates"] != payload["updates"]
            ):
                raise ValueError(f"invalid terminal seed-61 C-TEST core for N={n}")
            model = build_model(payload, "C-TEST")
            train = make_projector_training_batch(
                n, identity["train"], identity["budget"], identity["split_seeds"]["train"],
                include_teachers=False, observation_law=EFFECTIVE_LAW,
            )
            retention, _ = fit_prior(train)
            per_size[n] = ({"network": model}, retention, identity["train"])
            evidence[str(n)] = {
                "source": identity["source"], "seed": identity["seed"], "profile": "C-TEST",
                "updates": payload["updates"], "checkpoint_folder": folder.name,
                "train_namespace": identity["split_seeds"]["train"],
            }
        return per_size, evidence

    from .long_projector_confirmation import actual_training_prior

    bindings = (
        ("state_network", args.state_core, args.state_selection, "b0", "state"),
        ("joint_network", args.joint_core, args.joint_selection, "b2", "joint"),
    )
    models, matches = {}, []
    for name, checkpoint, registration, lane, arm in bindings:
        if checkpoint is None or registration is None:
            raise ValueError("the long study needs both cores and matched selection registrations")
        payload = read_core(checkpoint)
        selection = json.loads(registration.read_text(encoding="utf-8"))
        validate_long_binding(payload, selection, lane, arm)
        models[name] = build_model(payload, "C-S")
        identity = payload["identity"]
        matches.append(
            (payload["updates"], identity["source"], identity["seed"], identity["train_states_per_n"],
             identity["batch"], identity["learning_rate"])
        )
        evidence[name] = {
            "lane": lane, "source": identity["source"], "seed": identity["seed"], "profile": "C-S",
            "updates": payload["updates"], "checkpoint": selection["selected"]["matched"],
        }
    if matches[0] != matches[1]:
        raise ValueError("long models differ in their matched run and sampler identity")
    step, _, _, train_rows, _, _ = matches[0]
    for n in (3, 4, 8):
        retention, _, seen = actual_training_prior(n, train_rows, step)
        per_size[n] = (models, retention, seen)
    return per_size, evidence


@torch.inference_mode()
def predict(model, observation, n):
    return torch.cat(
        [
            normalized_state(model({k: v[start:start + FORWARD_BATCH] for k, v in observation.items()}, N=n))
            for start in range(0, len(observation["counts"]), FORWARD_BATCH)
        ]
    )


@torch.no_grad()
def query_score(rho, query, observation):
    analyzer = torch.complex(observation["analyzer_re"], observation["analyzer_im"])
    probability = effective_probabilities(rho, analyzer, observation["effective_calibration"])
    return (-(query * probability.log()).sum((1, 2)) / query.sum((1, 2))).numpy()


def evaluate_size(n, models, retention, prior_rows, namespace):
    truth, family, panels = measurement_panels(n, ROWS, namespace)
    scores, data = {}, {}

    def interval(values):
        return paired_estimate(values, family, draws=BOOTSTRAP_DRAWS)

    by_budget = {}
    for budget, (observation, query) in panels.items():
        prepare = torch.complex(observation["prepared_re"], observation["prepared_im"])
        command = prepare.unsqueeze(-1) * prepare.conj().unsqueeze(-2)
        prior = prior_density(command, retention)
        predictions = {
            "command": command,
            "train_shrinkage": prior,
            "classical": classical_estimate(observation, prior),
            **{name: predict(model, observation, n) for name, model in models.items()},
        }
        by_budget[budget] = {
            "state_risk": {k: state_risk(v, truth).numpy() for k, v in predictions.items()},
            "query_cross_entropy_per_read": {k: query_score(v, query, observation) for k, v in predictions.items()},
        }
        query_reads = query.sum((1, 2)).numpy()
        result = {
            "settings": observation["counts"].shape[1],
            "query_settings_per_state": int((~observation["setting_mask"]).sum(-1)[0]),
            "visible_reads_per_state_range": [int(budget - query_reads.max()), int(budget - query_reads.min())],
        }
        raw = {}
        for metric, values in by_budget[budget].items():
            contrasts = {
                f"{name}_minus_{control}": interval(values[name] - values[control])
                for name in models
                for control in ("command", "train_shrinkage", "classical")
            }
            if "joint_network" in values:
                contrasts["joint_minus_state"] = interval(values["joint_network"] - values["state_network"])
            result[metric] = {"estimate": {k: interval(v) for k, v in values.items()}, "paired": contrasts}
            raw[metric] = {k: v.tolist() for k, v in values.items()}
        scores[str(budget)] = result
        data[str(budget)] = raw

    changes = {}
    for low, high in BUDGET_PAIRS:
        row = {}
        for metric, high_values in by_budget[high].items():
            delta = {k: v - by_budget[low][metric][k] for k, v in high_values.items()}
            row[metric] = {k: interval(v) for k, v in delta.items()}
            if "joint_network" in delta:
                row[metric]["joint_response_minus_state_response"] = interval(
                    delta["joint_network"] - delta["state_network"]
                )
        changes[f"{high}_minus_{low}"] = row
    return (
        {"prior_retention": retention, "train_states_used_for_prior": prior_rows,
         "budgets": scores, "changes": changes},
        {"family": family.tolist(), "budgets": data},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("short", "long", "matrix"), required=True)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--frozen-registry", type=Path)
    parser.add_argument("--short-root", type=Path)
    parser.add_argument("--state-core", type=Path)
    parser.add_argument("--state-selection", type=Path)
    parser.add_argument("--joint-core", type=Path)
    parser.add_argument("--joint-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True, help="worktree source revision and relevant dirty state")
    args = parser.parse_args()
    result_path = args.output / f"{args.stage}_result.json"
    state_path = args.output / f"{args.stage}_states.json"
    if result_path.exists() or state_path.exists():
        raise FileExistsError("this output already contains a scored panel")
    torch.set_num_threads(THREADS)
    started = time.perf_counter()
    if args.stage == "matrix":
        if not args.campaign_root or not args.frozen_registry:
            parser.error("--stage matrix requires --campaign-root and --frozen-registry")
        frozen = json.loads(args.frozen_registry.read_text())
        if args.source != frozen["roster"]["source"]:
            raise ValueError("requested source differs from frozen matrix source")
        per_size, candidates = matrix_budget_candidates(args.campaign_root, frozen)
        namespace = matrix_budget_namespace(frozen["roster"]["namespace"])
    else:
        per_size, candidates = load_candidates(args)
        namespace = NAMESPACES[args.stage]
    systems, state_records = {}, {}
    for n, (models, retention, prior_rows) in per_size.items():
        systems[str(n)], state_records[str(n)] = evaluate_size(n, models, retention, prior_rows, namespace)
    record = {
        "schema": "frozen_projector_budget_response_v1",
        "scope": args.stage,
        "source": args.source,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cpu_threads": THREADS,
        "law": EFFECTIVE_LAW,
        "total_paid_reads": BUDGETS,
        "fresh_namespace": namespace,
        "states_per_n": ROWS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "uncertainty_unit": "state; paired family-stratified pointwise intervals",
        "actual_weight_updates": 0,
        "candidate_provenance": candidates,
        "systems": systems,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with state_path.open("x", encoding="utf-8") as stream:
        json.dump(state_records, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{args.stage} scored on CPU in {record['wall_seconds']:.2f} s")


if __name__ == "__main__":
    main()
