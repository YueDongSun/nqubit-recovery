"""Evaluate the six frozen high-N amplitude fits on a prospective synthetic split."""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

if __package__:
    from . import amp_data, amp_fit, long_campaign, long_fit, long_model
else:
    import amp_data
    import amp_fit
    import long_campaign
    import long_fit
    import long_model


ARMS = ("first_order", "second_order")
SEEDS = (61, 62, 63)
ENDPOINTS = ("selected", "terminal")
SIZES = (32, 64, 128, 256)
BUDGETS = ((30, 1), (300, 16))
BASELINES = ("no_action", "calibration_only", "observed_likelihood")
TEST_ROLE = "long_test"
TEST_ROLE_ID = 25
TEST_START = 2_000_000
TEST_ROWS = 512
TEST_POINT = (256, 30, 1)
BOUND = 0.25
HARM_TOLERANCE = 1e-12
PRACTICAL_GAIN = 0.001
EXCESS_HARM_MARGIN = 0.02
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20261118
CONFIDENCE = 0.95
COMPAT_LONG_FIT = "947c7257dd390a4eb805f79b1442528da1cfd7cde0805bb53ba555b135268411"
TRAINING_BANK_ID = "ed5cc3f82cb0fd82df5f99170e7ec07ac499e4fd9059541eb9472a685bc089f7"
SOURCE_FILES = ("amp_data.py", "amp_fit.py", "long_campaign.py", "long_fit.py", "long_model.py")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def matches(left: float, right: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and math.isclose(
        left, right, rel_tol=1e-12, abs_tol=1e-12
    )


def verify_protocol(bundle: Path) -> dict:
    protocol = read_json(bundle / "evaluation" / "protocol.json")
    require(protocol["schema"] == "high_n_amplitude_long_test_protocol_v1", "Wrong protocol schema")
    require(protocol["training_bank_id"] == TRAINING_BANK_ID, "Wrong frozen bank identity")
    data = protocol["test_data"]
    require(data["role"] == TEST_ROLE and data["role_id"] == TEST_ROLE_ID and
            data["start_index"] == TEST_START and data["latents_per_n"] == TEST_ROWS and
            tuple(data["sizes"]) == SIZES and tuple(map(tuple, data["budget_shots_rounds"])) == BUDGETS,
            "TEST split differs from the frozen protocol")
    require(tuple(protocol["arms"]) == ARMS and tuple(protocol["fixed_training_seeds"]) == SEEDS and
            tuple(protocol["endpoints"]) == ENDPOINTS and
            tuple(protocol["action_and_outcomes"]["baselines"]) == BASELINES and
            tuple(protocol["action_and_outcomes"]["n256_low_focus"]) == TEST_POINT,
            "Campaign axes differ from the frozen protocol")
    bootstrap = protocol["bootstrap"]
    decision = protocol["operational_decision"]
    require(bootstrap["replicates"] == BOOTSTRAP_REPLICATES and bootstrap["seed"] == BOOTSTRAP_SEED and
            bootstrap["confidence"] == CONFIDENCE and
            decision["minimum_practical_selected_gain"] == PRACTICAL_GAIN and
            decision["maximum_selected_excess_harm"] == EXCESS_HARM_MARGIN,
            "Statistical criteria differ from the frozen protocol")
    return protocol


def verify_sources(bundle: Path, expected: dict, local_source: Path) -> dict:
    require(set(SOURCE_FILES).issubset(expected), "Frozen source map is incomplete")
    effective = {}
    with ZipFile(bundle / "amplitude_long.zip") as archive:
        for name, expected_hash in expected.items():
            members = [item for item in archive.namelist() if Path(item).name == name]
            require(len(members) == 1, f"Missing or ambiguous archive member {name}")
            original = archive.read(members[0])
            require(digest(original) == expected_hash, f"Archive digest differs for {name}")
            local = (local_source / name).read_bytes()
            local_hash = digest(local)
            if name == "long_fit.py" and local_hash == COMPAT_LONG_FIT:
                require(ast.dump(ast.parse(local)) == ast.dump(ast.parse(original)),
                        "Python 3.10 compatibility edit changes the training AST")
            else:
                require(local_hash == expected_hash, f"Local source digest differs for {name}")
            effective[name] = local_hash
    return effective


def verify_bank(bundle: Path, nqrec_root: Path) -> None:
    bank_dir = bundle / "banks"
    bank_id, banks = long_campaign.load_banks(bank_dir, qualification=False)
    require(bank_id == TRAINING_BANK_ID, "Training bank identity mismatch")
    del banks
    manifest = read_json(bank_dir / "bank_manifest.json")
    excluded = {"numpy", "python"}
    actual = long_campaign.source_record(nqrec_root)
    require({key: value for key, value in manifest["source"].items() if key not in excluded} ==
            {key: value for key, value in actual.items() if key not in excluded},
            "Local generator source differs from the generator that produced the training bank")
    require(TEST_ROLE not in amp_data.ROLE_IDS and TEST_ROLE_ID not in amp_data.ROLE_IDS.values() and
            TEST_ROLE_ID not in long_campaign.ROLE_IDS.values(), "TEST role overlaps training roles")


def scheduled_jobs(bundle: Path) -> dict:
    queue = read_json(bundle / "jobs.json")
    require(queue["schema"] == "high_n_amplitude_long_queue_v1" and
            queue["bank_id"] == TRAINING_BANK_ID, "Queue identity mismatch")
    jobs = {(job["arm"], job["seed"]): job for job in queue["jobs"]}
    require(set(jobs) == {(arm, seed) for arm in ARMS for seed in SEEDS} and
            len(jobs) == len(queue["jobs"]), "Queue must contain exactly the six frozen jobs")
    for (arm, seed), job in jobs.items():
        require(job["job_id"] == f"{arm}_s{seed}", "Queue job ID mismatch")
    return jobs


def expected_recipe(job: dict, source_sha: dict) -> dict:
    command = job["commands"]["train"]
    microbatch = int(command[command.index("--microbatch") + 1])
    recipe = long_fit.make_recipe(
        argparse.Namespace(mode="train", arm=job["arm"], seed=job["seed"],
                           microbatch=microbatch, device="cpu"), TRAINING_BANK_ID
    )
    recipe["device"] = "cuda"
    if job["seed"] == 61:
        source_sha = {**source_sha, "long_fit.py": COMPAT_LONG_FIT}
    recipe["source_sha256"] = source_sha
    return recipe


def verify_recipe(recipe: dict, job: dict, source_sha: dict) -> None:
    expected = expected_recipe(job, source_sha)
    runtime_fields = {"device_name", "torch", "cuda", "numpy", "python"}
    require(all(isinstance(recipe.get(key), str) and recipe[key] for key in runtime_fields),
            f"Incomplete runtime provenance for {job['job_id']}")
    require({key: value for key, value in recipe.items() if key not in runtime_fields} ==
            {key: value for key, value in expected.items() if key not in runtime_fields},
            f"Frozen recipe mismatch for {job['job_id']}")
    if job["seed"] == 61:
        require(recipe["python"].startswith("3.10."), "Unexpected Python on the compatibility host")


def objective(record: dict) -> float:
    return float(record["objective"])


def verify_completion(result: dict, status: dict, job_id: str) -> dict:
    steps = long_fit.FORMAL_STEPS
    require(result["state"] == "completed" and result["step"] == steps and
            status == {"state": "completed", "step": steps,
                       "deadline_unix": result["origin"]["deadline_unix"]},
            f"Missing matching terminal completion for {job_id}")
    require(result["updates_per_cell"] == steps // (len(SIZES) * len(BUDGETS)) and
            result["rows_per_update"] == long_fit.BATCH, f"Unbalanced updates for {job_id}")
    history = result["validation_history"]
    require([row["step"] for row in history] == list(range(long_fit.VALID_INTERVAL, steps + 1, long_fit.VALID_INTERVAL)) and
            all(math.isfinite(objective(row)) for row in history), f"Incomplete validation history for {job_id}")
    selected = min(history, key=objective)
    require(result["selected_step"] == selected["step"] and objective(result["selected_validation"]) == objective(selected) and
            objective(result["terminal_validation"]) == objective(history[-1]), f"Selection or terminal history mismatch for {job_id}")
    return selected


def create_model(arm: str, weights: dict) -> torch.nn.Module:
    model = long_model.AmplitudeSetModel(arm, width=long_fit.WIDTH).double().eval()
    for name, tensor in weights.items():
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float64 and
                bool(tensor.isfinite().all()), f"Invalid checkpoint tensor {name}")
    model.load_state_dict(weights, strict=True)
    return model


def verify_checkpoint(bundle: Path, job: dict, source_sha: dict) -> tuple[dict, dict]:
    folder = bundle / "recovered" / job["job_id"]
    result = read_json(folder / "result.json")
    status = read_json(folder / "status.json")
    recipe = result["recipe"]
    verify_recipe(recipe, job, source_sha)
    best = verify_completion(result, status, job["job_id"])
    latest = torch.load(folder / "latest.pt", map_location="cpu", weights_only=True)
    selected = torch.load(folder / "selected.pt", map_location="cpu", weights_only=True)
    require(latest["recipe"] == selected["recipe"] == recipe and
            latest["step"] == result["step"] and selected["step"] == best["step"] and
            latest["validation_history"] == result["validation_history"] and
            latest["best"]["step"] == best["step"] and matches(latest["best"]["objective"], objective(best)),
            f"Checkpoint identity or selection mismatch for {job['job_id']}")
    picked, latest_best = selected["weights"], latest["best"]["weights"]
    require(picked.keys() == latest_best.keys() and
            all(torch.equal(picked[key], latest_best[key]) for key in picked),
            f"Selected checkpoint differs from saved best weights for {job['job_id']}")
    weights = {"selected": picked, "terminal": latest["weights"]}
    for state in weights.values():
        model = create_model(job["arm"], state)
        require(sum(tensor.numel() for tensor in model.parameters()) == result["parameter_count"],
                f"Parameter count mismatch for {job['job_id']}")
    return result, weights


def test_data(n: int, shots: int, rounds: int, nqrec_root: Path) -> dict:
    with patch.dict(amp_data.ROLE_IDS, {TEST_ROLE: TEST_ROLE_ID}):
        return amp_data.generate(n, TEST_ROWS, TEST_ROLE, nqrec_root=nqrec_root,
                                 start=TEST_START, shots=shots, rounds=rounds, family="structured")


def baseline_scores(data: dict, truth: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    estimates = amp_fit.likelihood_estimates(data)
    safe_p = torch.as_tensor(estimates["calibration_prior_no_counts"], dtype=torch.float64)
    count_p = torch.as_tensor(estimates["template_grid_posterior_mean"], dtype=torch.float64)
    base_f = long_model.torch_conditional_fidelity(truth, torch.zeros_like(truth))
    scores = [[np.zeros(len(truth)), np.full(len(truth), np.nan)]]
    for prob in (safe_p, count_p):
        require(bool((prob >= 0).all() and torch.isfinite(prob).all() and
                     (prob.sum(dim=-1) > 0).all()), "Invalid baseline probability")
        prob = prob.clamp_min(torch.finfo(prob.dtype).tiny)
        logp = (prob / prob.sum(dim=-1, keepdim=True)).log()
        action = long_model.torch_centered_action(logp, BOUND)
        gain = long_model.torch_conditional_fidelity(truth, action) - base_f
        kl = torch.nn.functional.kl_div(logp, truth, reduction="none").sum(dim=-1)
        scores.append([gain.numpy(), kl.numpy()])
    scores = np.asarray(scores)
    return scores[:, 0], scores[:, 1]


@torch.inference_mode()
def score_all(nqrec_root: Path, models: dict) -> dict:
    model_shape = (len(ENDPOINTS), len(ARMS), len(SEEDS), len(SIZES), len(BUDGETS), TEST_ROWS)
    baseline_shape = (len(BASELINES), len(SIZES), len(BUDGETS), TEST_ROWS)
    gains = np.empty(model_shape)
    kl = np.empty(model_shape)
    baseline_gains = np.empty(baseline_shape)
    baseline_kl = np.empty(baseline_shape)
    for size_index, n in enumerate(SIZES):
        first_budget = None
        for budget_index, (shots, rounds) in enumerate(BUDGETS):
            data = test_data(n, shots, rounds, nqrec_root)
            latents = (data["population"], data["log_weight"])
            if first_budget is None:
                first_budget = latents
            else:
                require(all(np.array_equal(x, y) for x, y in zip(first_budget, latents)),
                        f"Budget pairing changed the latents at N={n}")
            features = torch.as_tensor(amp_fit.encode_observations(data), dtype=torch.float64)
            truth = torch.as_tensor(data["population"], dtype=torch.float64)
            baseline_gains[:, size_index, budget_index], baseline_kl[:, size_index, budget_index] = baseline_scores(data, truth)
            for indices, model in models.items():
                location = (*indices, size_index, budget_index)
                for start in range(0, TEST_ROWS, 64):
                    part = slice(start, start + 64)
                    batch_kl, batch_gain, _ = long_fit.batch_scores(model, features[part], truth[part])
                    gains[(*location, part)] = batch_gain.numpy()
                    kl[(*location, part)] = batch_kl.numpy()
            print(f"Scored N={n} shots={shots} rounds={rounds}", flush=True)
    for values in (gains, kl, baseline_gains, baseline_kl[1:]):
        require(np.isfinite(values).all(), "Nonfinite TEST metric")
    return {"gain": gains, "kl": kl, "baseline_gain": baseline_gains, "baseline_kl": baseline_kl}


def bootstrap_weights(n_sizes: int, n_rows: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_sizes, replicates, n_rows), dtype=np.float64)
    for block in weights:
        draws = rng.integers(n_rows, size=(replicates, n_rows))
        np.add.at(block, (np.arange(replicates)[:, None], draws), 1)
    return weights / n_rows


def interval(values: np.ndarray, weights: np.ndarray) -> dict:
    per_n_draws = np.einsum("nij,nj->ni", weights, values)
    tails = ((1 - CONFIDENCE) / 2, (1 + CONFIDENCE) / 2)
    lower, upper = np.quantile(per_n_draws.mean(axis=0), tails)
    return {"mean": float(values.mean()), "ci95": [float(lower), float(upper)]}


def paired_summary(gain: np.ndarray, other: np.ndarray, weights: np.ndarray) -> dict:
    diff = gain - other
    harm = (gain < -HARM_TOLERANCE).astype(float) - (other < -HARM_TOLERANCE).astype(float)
    by_cell = []
    for size_index, n in enumerate(SIZES):
        for budget_index, (shots, rounds) in enumerate(BUDGETS):
            rows = (size_index, budget_index)
            by_cell.append({"n": n, "shots": shots, "rounds": rounds,
                            "gain_difference": interval(diff[..., *rows, :].reshape(-1, TEST_ROWS).mean(axis=0)[None], weights[size_index:size_index + 1]),
                            "excess_harm": interval(harm[..., *rows, :].reshape(-1, TEST_ROWS).mean(axis=0)[None], weights[size_index:size_index + 1])})
    return {"gain_difference": interval(diff.reshape(-1, len(SIZES), len(BUDGETS), TEST_ROWS).mean(axis=(0, 2)), weights),
            "excess_harm": interval(harm.reshape(-1, len(SIZES), len(BUDGETS), TEST_ROWS).mean(axis=(0, 2)), weights),
            "by_cell": by_cell}


def observed_summary(gain: np.ndarray, kl: np.ndarray) -> dict:
    return {"mean_gain": float(gain.mean()), "harm_rate": float((gain < -HARM_TOLERANCE).mean()),
            "mean_kl": None if np.isnan(kl).all() else float(kl.mean())}


def summarize(metrics: dict) -> dict:
    gains, kl = metrics["gain"], metrics["kl"]
    baseline_gains, baseline_kl = metrics["baseline_gain"], metrics["baseline_kl"]
    weights = bootstrap_weights(len(SIZES), TEST_ROWS, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    endpoints = {}
    for end_index, endpoint in enumerate(ENDPOINTS):
        end_gains, end_kl = gains[end_index], kl[end_index]
        comparison = paired_summary(end_gains[1], end_gains[0], weights)
        arms = {}
        for arm_index, arm in enumerate(ARMS):
            arms[arm] = {
                "observed": observed_summary(end_gains[arm_index], end_kl[arm_index]),
                "versus_baselines": {baseline: paired_summary(end_gains[arm_index], baseline_gains[index], weights)
                                     for index, baseline in enumerate(BASELINES)},
            }
        endpoints[endpoint] = {
            "second_minus_first": comparison,
            "by_seed": {str(seed): paired_summary(end_gains[1, index], end_gains[0, index], weights)
                        for index, seed in enumerate(SEEDS)},
            "arms": arms,
        }
    primary = endpoints["selected"]["second_minus_first"]
    low_focus = next(row for row in primary["by_cell"] if (row["n"], row["shots"], row["rounds"]) == TEST_POINT)
    effect = primary["gain_difference"]["ci95"][0] > PRACTICAL_GAIN
    harm = primary["excess_harm"]["ci95"][1] < EXCESS_HARM_MARGIN
    low_harm = low_focus["excess_harm"]["ci95"][1] < EXCESS_HARM_MARGIN
    return {"endpoints": endpoints,
            "baselines": {baseline: observed_summary(baseline_gains[index], baseline_kl[index])
                          for index, baseline in enumerate(BASELINES)},
            "selected_n256_low": low_focus,
            "operational_decision": {"passes_practical_gain": effect, "passes_aggregate_harm": harm,
                                     "passes_n256_low_harm": low_harm,
                                     "disposition": "support_second_order_for_further_synthetic_evaluation"
                                     if effect and harm and low_harm else "retain_first_order"}}


def write_json(path: Path, result: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--nqrec-root", type=Path, required=True)
    parser.add_argument("--original-source", type=Path, required=True, help="Unmodified training scripts from the historical six-fit archive")
    parser.add_argument("--check-only", action="store_true", help="Verify provenance and completion before generating any TEST rows")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    require(args.threads > 0, "Use a positive CPU thread count")
    torch.set_num_threads(args.threads)
    bundle = args.bundle_dir.resolve()
    protocol = verify_protocol(bundle)
    local_source = verify_sources(bundle, protocol["training_source_sha256"], args.original_source)
    verify_bank(bundle, args.nqrec_root.resolve())
    jobs = scheduled_jobs(bundle)
    for job in jobs.values():
        folder = bundle / "recovered" / job["job_id"]
        for name in ("status.json", "result.json", "latest.pt", "selected.pt"):
            require((folder / name).is_file(), f"TEST release requires {folder / name}")
    models, completed = {}, {}
    for arm_index, arm in enumerate(ARMS):
        for seed_index, seed in enumerate(SEEDS):
            job = jobs[arm, seed]
            result, weights = verify_checkpoint(bundle, job, protocol["training_source_sha256"])
            completed[job["job_id"]] = {key: result[key] for key in ("recipe", "origin", "step", "selected_step", "selected_validation", "terminal_validation", "cumulative_wall_s", "parameter_count")}
            if not args.check_only:
                for end_index, endpoint in enumerate(ENDPOINTS):
                    models[end_index, arm_index, seed_index] = create_model(arm, weights[endpoint])
            print(f"Verified {job['job_id']} at step {result['step']}", flush=True)
    del weights
    gc.collect()
    if args.check_only:
        return 0
    output = bundle / "evaluation" / "test_result"
    output.mkdir(exist_ok=False)
    try:
        metrics = score_all(args.nqrec_root.resolve(), models)
        summary = summarize(metrics)
        np.savez_compressed(output / "row_metrics.npz", **metrics)
        write_json(output / "summary.json", {"state": "completed", "protocol": protocol,
                   "local_source_sha256": local_source, "completed_jobs": completed,
                   "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "threads": args.threads},
                   **summary})
    except Exception as error:
        write_json(output / "failed.json", {"state": "failed", "error": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
