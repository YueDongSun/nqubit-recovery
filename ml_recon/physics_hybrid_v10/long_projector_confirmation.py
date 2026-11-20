"""Evaluate locked C-S choices at one common update and their validation-selected steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .classifier_n_full import V10ClassifierNFull
from .effective_inference_experiment import write_json
from .effective_inference_metrics import (
    classical_estimate,
    command_and_truth,
    paired_estimate,
    prior_density,
    score_model,
)
from .long_projector_data import (
    SIZES,
    SPLIT,
    build_bank,
    train_bank_namespace,
)
from .long_projector_prior import actual_training_prior
from .long_projector_training import to_device
from .matrix_campaign import manifest_digest as _manifest_digest
from .matrix_campaign import register_once


def run(root, lane, common, device, *, use_matrix=False, frozen=None, prepare_calibration=False,
        inference_device=None, forward_batch=32, cache_root=None):
    if (inference_device is not None or cache_root is not None) and (not use_matrix or device.type != "cpu"):
        raise ValueError("accelerated matrix scoring keeps all fits and scientific metrics on CPU")
    if use_matrix and frozen is None:
        raise ValueError("matrix confirmation requires the globally frozen candidate roster and CAL seal")
    if use_matrix:
        official = json.loads((root / "frozen_candidates.json").read_text())
        if _manifest_digest(official) != _manifest_digest(frozen):
            raise ValueError("candidate list differs from the global frozen roster")
        if not prepare_calibration:
            seal = json.loads((root / "matrix_calibration_complete.json").read_text())
            if seal["candidate_manifest_sha256"] != _manifest_digest(frozen):
                raise ValueError("CAL roster is not bound to the frozen candidates")
    home = root / lane
    done = (frozen["candidates"][lane]["training"] if frozen is not None else
            json.loads((home / "training_done.json").read_text()))
    records = json.loads((home / "registration.json").read_text())
    split_namespace = records["identity"].get("split_namespace")
    if use_matrix != bool(split_namespace):
        raise ValueError("matrix confirmation requires the training run's own split namespace")
    if records["identity"]["lane"] != lane:
        raise ValueError("confirmation directory does not match the registered run")
    if frozen is None:
        if common is None or common < 256 or common & (common - 1) or common > done["step"]:
            raise ValueError("common update must be a completed prespecified rung")
        choices = {"matched": common, "validation_selected": int(done["best_validation_step"])}
        selected = {"matched": f"snapshots/step_{common:09d}/core.pt",
                    "validation_selected": done["best_core"]}
        aliases = {}
    else:
        if not use_matrix or frozen["candidates"][lane]["identity"] != records["identity"]:
            raise ValueError("frozen candidate and fitted run differ")
        entry = frozen["candidates"][lane]
        choices = entry["choices"]
        selected, aliases, canonical = {}, {}, {}
        for key, step in choices.items():
            path = f"snapshots/step_{step:09d}/core.pt"
            if str(step) not in entry["weights"]:
                raise ValueError("candidate file was not registered")
            if step in canonical:
                aliases[key] = canonical[step]
            else:
                canonical[step] = key
                selected[key] = path
    dest = home / "confirmation"
    dest.mkdir(exist_ok=True)
    identity = dict(
        lane=lane,
        source=records["identity"]["source"],
        selected=selected,
        states_per_n=768,
        namespace=SPLIT["confirmation"],
        partition_score_version="labeled_unordered_pairs_v2",
    )
    if use_matrix:
        identity["split_namespace"] = split_namespace
        identity["module_report_version"] = 1
        if "train_namespace" in records["identity"]:
            identity["train_namespace"] = records["identity"]["train_namespace"]
    if frozen is not None:
        identity.update(weights=entry["weights"], candidate_steps=choices, aliases=aliases)
    registration = dest / "registration.json"
    register_once(registration, identity)
    if (dest / "result.json").exists() and not prepare_calibration:
        return
    calibration = {}
    if frozen is not None:
        from .matrix_precal import prepare_core
        for key in selected:
            step = choices[key]
            early = prepare_core(home, records["identity"], step, entry["weights"][str(step)],
                                 allow_calculation=prepare_calibration)
            for n, receipt in early.items():
                cal_path = dest / f"n{n}_{key}_cal.json"
                if prepare_calibration:
                    register_once(cal_path, receipt)
                if json.loads(cal_path.read_text()) != receipt:
                    raise ValueError("official CAL receipt differs from its immutable precomputed scores")
                calibration[n, key] = receipt
        if prepare_calibration:
            register_once(dest / "calibration_complete.json", {"candidate_manifest_sha256": _manifest_digest(frozen),
                          "weights": entry["weights"], "sizes": list(SIZES)})
            return
    scores = {}
    rows = {}
    for n in SIZES:
        pending = [key for key in selected if not (dest / f"n{n}_{key}.json").is_file()]
        if pending:
            from .matrix_score_acceleration import cached_tensor_tree, prior_digest, FrozenPublicForward
            panel_key = f"{_manifest_digest(frozen)}_n{n}" if frozen is not None else ""
            cpu_bank = cached_tensor_tree(
                cache_root, panel_key + "_bank",
                lambda: build_bank(n, 768, "confirmation", 0, True, split_namespace=split_namespace),
            )
            # Cross-device paired intervals require the identical observations
            # and query split. Compare the integer bytes before GPU transfer.
            count_digest = hashlib.sha256()
            for field in ("counts", "query_counts", "shots", "setting_mask", "measurement_query_mask"):
                count_digest.update(cpu_bank[field].contiguous().numpy().tobytes())
            observed_count_identity = count_digest.hexdigest()
            bank = to_device(cpu_bank, device)
            command, _ = command_and_truth(bank)
        for key in pending:
            path = home / selected[key]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["identity"] != records["identity"]:
                raise ValueError("wrong complete core")
            step = int(payload["updates"])
            if step != choices[key]:
                raise ValueError("core does not match its declared selected update")
            model = V10ClassifierNFull(**payload["classifier_config"]).to(device)
            model.load_state_dict(payload["classifier_state_dict"], strict=True)
            model.eval()
            retention, pair_prior, seen = actual_training_prior(
                n, records["identity"]["train_states_per_n"], step,
                split_namespace=train_bank_namespace(records["identity"])
            )
            prior = prior_density(command, retention)
            estimate = cached_tensor_tree(
                cache_root, panel_key + "_classical_" + prior_digest(prior),
                lambda: classical_estimate(bank, prior),
            ) if cache_root is not None else classical_estimate(bank, prior)
            if inference_device is not None:
                model = FrozenPublicForward(model, bank, n, inference_device, forward_batch)
            metrics, states = score_model(
                model, bank, n=n, retention=retention, pair_prior=pair_prior, classical=estimate
            )
            if use_matrix:
                from .module_representation_report import representation_report

                with torch.inference_mode():
                    frozen_output = model(bank, N=n)
                    from .long_projector_prior import visited_noise_class_counts
                    train_classes = visited_noise_class_counts(
                        n, records["identity"]["train_states_per_n"], step,
                        train_bank_namespace(records["identity"]),
                    )
                    metrics["module_representations"] = representation_report(
                        frozen_output, bank, train_class_counts=train_classes
                    )
                    if "idea_fd_logits" in frozen_output:
                        from .fd_policy_evaluation import (
                            fd_reliability_rows,
                            score_fd_actions,
                            select_fd_actions,
                            summarize_fd_reliability,
                        )

                        decisions = select_fd_actions(frozen_output, bank)
                        scored = score_fd_actions(decisions, bank["target_fd_utility"])
                        metrics["fd_candidate_calibration"] = scored["candidate_calibration"]
                        metrics["fd_harm_gate"] = decisions["harm_threshold"]
                        metrics["fd_policies"] = {
                            name: values["summary"] for name, values in scored["policies"].items()
                        }
                        states["fd_policies"] = {
                            name: {key: tensor.tolist() for key, tensor in values["rows"].items()}
                            for name, values in scored["policies"].items()
                        }
                        candidate_rows = fd_reliability_rows(decisions, bank["target_fd_utility"])
                        metrics["fd_reliability"] = summarize_fd_reliability(
                            candidate_rows, harm_thresholds=[decisions["harm_threshold"]]
                        )
                        states["fd_candidate_rows"] = {
                            key: tensor.tolist() for key, tensor in candidate_rows.items()
                        }
            if frozen is not None:
                from .matrix_diagnostics import score_matrix_diagnostics
                diagnostics, per_state = score_matrix_diagnostics(
                    model, bank, n, records["identity"]["train_states_per_n"], step,
                    train_bank_namespace(records["identity"]), split_namespace, device,
                    calibration_receipt=calibration[n, key]["cal"],
                )
                metrics["dev_selected_baseline"] = calibration[n, key]["dev_baseline"]
                metrics["matrix_diagnostics"] = diagnostics
                states["matrix_diagnostics"] = per_state
                from .fd_drift_evaluation import score_drift_endpoint
                drift, drift_rows = score_drift_endpoint(model, n, len(bank["counts"]), split_namespace, device,
                                                        calibration[n, key]["cal"])
                metrics["fd_strong_drift_snapshot_t10"] = drift
                states["fd_strong_drift_snapshot_t10"] = drift_rows
            from .training_ideas_experiment import _rho8_diagnostics

            metrics["rho8"] = _rho8_diagnostics(bank)
            state_path = dest / f"n{n}_{key}_states.json"
            write_json(state_path, states)
            write_json(
                dest / f"n{n}_{key}.json",
                dict(
                    updates=step,
                    actual_seen_train=seen,
                    train_prior_retention=retention,
                    train_pair_prior=pair_prior,
                    state_file=state_path.name,
                    count_panel_identity=observed_count_identity,
                    scoring_device=str(inference_device or device),
                    scoring_gpu_uuid=(os.environ.get("CUDA_VISIBLE_DEVICES")
                                      if inference_device is not None and inference_device.type == "cuda" else None),
                    classical_fit_device=str(device),
                    forward_microbatch=forward_batch if inference_device is not None else None,
                    metrics=metrics,
                ),
            )
        for key in selected:
            record = json.loads((dest / f"n{n}_{key}.json").read_text())
            scores.setdefault(key, {})[str(n)] = record
            rows.setdefault(key, {})[str(n)] = record["state_file"]
    for alias, key in aliases.items():
        scores[alias], rows[alias] = scores[key], rows[key]
    write_json(
        dest / "result.json", dict(identity=identity, training=done, candidates=scores, states=rows)
    )


def aggregate(root, output):
    arms = {}
    for lane in ("b0", "b1", "b2", "b3"):
        d = root / lane / "confirmation"
        result = json.loads((d / "result.json").read_text())
        arms[lane] = result
    sources = {r["identity"]["source"] for r in arms.values()}
    common = {r["candidates"]["matched"]["3"]["updates"] for r in arms.values()}
    if len(sources) != 1 or len(common) != 1:
        raise ValueError("different registered candidate source or common-step selection")
    contrasts = {}
    for first, reference in (("b1", "b0"), ("b2", "b0"), ("b2", "b1"), ("b3", "b2")):
        key = f"{first}_minus_{reference}"
        contrasts[key] = {}
        for select in ("matched", "validation_selected"):
            contrasts[key][select] = {}
            for n in SIZES:
                a, b = [
                    json.loads(
                        (root / l / "confirmation" / arms[l]["states"][select][str(n)]).read_text()
                    )
                    for l in (first, reference)
                ]
                count_ids = {arms[l]["candidates"][select][str(n)]["count_panel_identity"]
                             for l in (first, reference)}
                if len(count_ids) != 1:
                    raise ValueError("confirmation count tables/exposure/query masks differ")
                if not np.allclose(a["estimators"]["command"], b["estimators"]["command"], atol=2e-6, rtol=1e-6):
                    raise ValueError("public command and latent state pairing disagree")
                if a["family"] != b["family"]:
                    raise ValueError("confirmation state identities disagree")
                delta = np.asarray(a["estimators"]["network"]) - b["estimators"]["network"]
                contrasts[key][select][str(n)] = paired_estimate(delta, a["family"])
    staged = arms["b1"]
    boundary = staged["training"]["stage_boundary"]
    staged_work = {}
    for key, step in {
        "terminal": staged["training"]["step"],
        "matched": staged["candidates"]["matched"]["3"]["updates"],
        "validation_selected": staged["candidates"]["validation_selected"]["3"]["updates"],
    }.items():
        staged_work[key] = dict(updates=step, state_primary_updates=min(step, boundary),
                                joint_updates=max(0, step-boundary), reached_joint=step>boundary)
    write_json(
        output,
        dict(
            source=sources.pop(),
            common_update=common.pop(),
            contrasts=contrasts,
            realized_staged_optimizer_work=staged_work,
            actual_total_updates=sum(v["training"]["step"] for v in arms.values()),
            arms=arms,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--lane", choices=("b0", "b1", "b2", "b3"))
    route.add_argument("--run-id", help="Score a new matrix run using its own registered data namespace.")
    parser.add_argument("--common-step", type=int)
    parser.add_argument("--frozen-roster", type=Path)
    parser.add_argument("--prepare-calibration", action="store_true")
    parser.add_argument("--precal-available", action="store_true",
                        help="Only independent CAL/DEV on locally verified immutable cores; never opens confirmation")
    parser.add_argument("--precal-until-utc", help="Check for additional returned cores every minute until this UTC deadline or the global roster freeze")
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-device", help="matrix only: neural forward device; scores and classical fits use CPU")
    parser.add_argument("--forward-batch", type=int, default=32)
    parser.add_argument("--shared-cache", type=Path, help="same-host cache for globally sealed matrix panels and priors")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if (args.inference_device or args.shared_cache) and not args.frozen_roster:
        parser.error("shared scoring requires a frozen matrix roster")
    if args.precal_available:
        if args.frozen_roster or args.prepare_calibration or args.aggregate or args.common_step or args.lane:
            parser.error("--precal-available is independent of confirmation and its frozen-manifest modes")
        from .matrix_precal import available
        until = None
        if args.precal_until_utc:
            stop = datetime.fromisoformat(args.precal_until_utc.replace("Z","+00:00"))
            if stop.utcoffset() is None or stop.utcoffset().total_seconds()!=0:
                parser.error("--precal-until-utc requires an explicit UTC timestamp")
            until = stop.timestamp()
        while True:
            available(args.output_root, args.run_id)
            if until is None or time.time() >= until or (args.output_root/"frozen_candidates.json").exists():
                break
            time.sleep(min(60, max(0,until-time.time())))
        return
    if args.precal_until_utc:
        parser.error("--precal-until-utc requires --precal-available")
    if args.prepare_calibration and not args.frozen_roster:
        parser.error("--prepare-calibration requires a frozen matrix roster")
    if args.frozen_roster:
        if not args.run_id or args.aggregate or args.common_step is not None:
            parser.error("--frozen-roster requires --run-id without --aggregate or --common-step")
        run(args.output_root, args.run_id, None, torch.device(args.device), use_matrix=True,
            frozen=json.loads(args.frozen_roster.read_text()), prepare_calibration=args.prepare_calibration,
            inference_device=torch.device(args.inference_device) if args.inference_device else None,
            forward_batch=args.forward_batch, cache_root=args.shared_cache)
        return
    if args.aggregate and args.run_id:
        parser.error("legacy four-lane aggregation does not accept matrix run IDs")
    if args.aggregate:
        aggregate(args.output_root, args.aggregate)
    elif args.lane and args.common_step:
        run(args.output_root, args.lane, args.common_step, torch.device(args.device))
    elif args.run_id and args.common_step:
        if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
            parser.error("invalid matrix run ID")
        run(args.output_root, args.run_id, args.common_step, torch.device(args.device), use_matrix=True)
    else:
        parser.error("supply --lane and registered --common-step or --aggregate")


if __name__ == "__main__":
    main()
