"""Prepare only independent CAL predictions and DEV comparators for returned immutable cores."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import torch

from .classifier_n_full import V10ClassifierNFull
from .effective_inference_metrics import (
    classical_estimate,
    command_and_truth,
    prior_density,
)
from .long_projector_data import SIZES, bank_seed, build_bank, train_bank_namespace
from .long_projector_prior import actual_training_prior
from .long_projector_training import constructor
from .matrix_campaign import manifest_digest, register_once
from .matrix_diagnostics import (
    CAL_SCHEMA,
    STRONG_CALIBRATION,
    TARGET_COVERAGES,
    register_matrix_calibration,
)

SCHEMA = "returned_core_cal_dev_v1"
CAL_ROWS = 768
DEV_ROWS = 192


@lru_cache(maxsize=1024)
def _digest_file(path, inode, size, modified_ns, changed_ns):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_file(path):
    stat = path.stat()
    return _digest_file(
        path, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    )


def binding_for(identity, step, weights):
    if step < 1:
        raise ValueError("an untrained model is not a calibration candidate")
    return {
        "schema": SCHEMA,
        "source": identity["source"],
        "run": identity["lane"],
        "step": step,
        "weights": weights,
        "training_identity_sha256": manifest_digest(identity),
        "train_namespace": train_bank_namespace(identity),
        "split_namespace": identity["split_namespace"],
        "cal_schema": CAL_SCHEMA,
        "cal_target_coverage": list(TARGET_COVERAGES),
        "cal_strong_parameters": list(STRONG_CALIBRATION),
        "cal_states": CAL_ROWS,
        "dev_states": DEV_ROWS,
        "dev_comparators": ["command", "train_prior", "classical"],
        "baseline_selection": "smallest mean DEV conditional squared Frobenius risk, listed order breaks ties",
        "calculation_device": "cpu",
    }


@lru_cache(maxsize=256)
def dev_comparators(source, train_namespace, split_namespace, n, rows, step):
    """Shared only if fitted TRAIN rows, DEV split, source and estimator recipe all coincide."""
    retention, _, seen = actual_training_prior(n, rows, step, train_namespace)
    dev = build_bank(
        n, DEV_ROWS, "validation", 0, False, split_namespace=split_namespace
    )
    command, truth = command_and_truth(dev)
    prior = prior_density(command, retention)
    fitted = classical_estimate(dev, prior)
    risks = {
        name: float((value - truth).abs().square().sum((-1, -2)).mean())
        for name, value in (
            ("command", command),
            ("train_prior", prior),
            ("classical", fitted),
        )
    }
    return retention, seen, risks


def read_receipts(folder, binding):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["binding"] != binding or set(manifest["sizes"]) != {
        str(n) for n in SIZES
    }:
        raise ValueError(
            "early CAL manifest identity differs from the model, split or registered policy"
        )
    receipts = {}
    for n in SIZES:
        digest = manifest["sizes"][str(n)]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid CAL content digest")
        rec = json.loads((folder / f"n{n}_{digest}.json").read_text())
        if (
            manifest_digest(rec) != digest
            or rec["binding"] != binding
            or rec["n"] != n
            or rec["weights"] != binding["weights"]
            or rec["cal"]["n"] != n
            or rec["cal"]["split_namespace"] != binding["split_namespace"]
        ):
            raise ValueError(
                "CAL/DEV scores or threshold identity changed after registration"
            )
        receipts[n] = rec
    return receipts


def prepare_core(home, identity, step, expected_sha, *, allow_calculation):
    """The caller supplies either the independent transfer digest or the frozen roster digest."""
    core = home / f"snapshots/step_{step:09d}/core.pt"
    if digest_file(core) != expected_sha:
        raise ValueError(f"published core differs from its registered digest: {core}")
    binding = binding_for(identity, step, expected_sha)
    folder = home / "precal" / f"step_{step:09d}"
    if (folder / "manifest.json").exists():
        return read_receipts(folder, binding)
    if not allow_calculation:
        raise FileNotFoundError(f"required sealed CAL/DEV receipt is missing: {folder}")
    payload = torch.load(core, map_location="cpu", weights_only=True)
    if (
        payload["identity"] != identity
        or int(payload["updates"]) != step
        or payload["classifier_config"] != identity["constructor"]
    ):
        raise ValueError(
            "candidate weights and complete registered training identity disagree"
        )
    model = V10ClassifierNFull(**payload["classifier_config"])
    model.load_state_dict(payload["classifier_state_dict"], strict=True)
    del payload
    model.eval()
    digests = {}
    for n in SIZES:
        cal = register_matrix_calibration(
            model,
            n,
            identity["split_namespace"],
            torch.device("cpu"),
            cal_states=CAL_ROWS,
        )
        retention, seen, risks = dev_comparators(
            identity["source"],
            train_bank_namespace(identity),
            identity["split_namespace"],
            n,
            identity["train_states_per_n"],
            step,
        )
        rec = {
            "binding": binding,
            "weights": expected_sha,
            "n": n,
            "cal": cal,
            "train_prior_retention": retention,
            "actual_seen_train": seen,
            "cal_seed": bank_seed(n, "calibration", 0, identity["split_namespace"]),
            "dev_seed": bank_seed(n, "validation", 0, identity["split_namespace"]),
            "dev_baseline_risks": risks,
            "dev_baseline": min(risks, key=risks.get),
        }
        digest = manifest_digest(rec)
        register_once(folder / f"n{n}_{digest}.json", rec)
        digests[str(n)] = digest
    register_once(folder / "manifest.json", {"binding": binding, "sizes": digests})
    return read_receipts(folder, binding)


def available(root, run_id=None):
    """Compute locally verified, fixed rungs and completed selected cores using CAL/DEV alone."""
    roster = json.loads((root / "matrix_roster.json").read_text())
    if run_id is not None and run_id not in roster["runs"]:
        raise ValueError(f"run not in the preregistered roster: {run_id}")
    returned = {}
    for path in sorted((root.parent / "hosts").rglob("retrieved_files.json")):
        for rel, record in json.loads(path.read_text()).items():
            if (
                not rel.endswith("/core.pt")
                or record["source_version"] != roster["source"]
            ):
                continue
            if rel in returned and returned[rel] != record["sha256"]:
                raise ValueError(
                    f"receiving sources disagree on returned weights: {rel}"
                )
            returned[rel] = record["sha256"]
    plans = []
    for name, entry in roster["runs"].items():
        if run_id is not None and name != run_id:
            continue
        home = root / name
        path = home / "registration.json"
        if not path.exists():
            continue
        identity = json.loads(path.read_text())["identity"]
        spec = entry["fit"]
        if (
            identity["lane"] != name
            or identity["source"] != roster["source"]
            or identity["split_namespace"] != roster["namespace"]
            or identity.get("train_namespace") != spec.get("train_namespace")
            or identity["constructor"]
            != constructor(
                spec["arm"], "factor_gram_v2", spec["capacity"], spec["idea_profile"]
            )
            or (
                identity["arm"],
                identity["seed"],
                identity["train_states_per_n"],
                identity["fixed_updates"],
                identity["learning_rate"],
                identity["staged_primary_updates"],
            )
            != (
                spec["arm"],
                spec["seed"],
                spec["rows"],
                spec["steps"],
                spec["lr"],
                spec["stage"],
            )
        ):
            raise ValueError(
                f"returned registration differs from the experiment roster: {name}"
            )
        terminal = home / "training_done.json"
        steps = set(entry["fixed_steps"])
        if terminal.exists():
            done = json.loads(terminal.read_text())
            if (
                done["status"] != "training_complete"
                or done["lane"] != name
                or done["step"] != spec["steps"]
                or not 0 < done["best_validation_step"] <= spec["steps"]
            ):
                raise ValueError(f"invalid completed training record: {name}")
            best = int(done["best_validation_step"])
            if done["best_core"] != f"snapshots/step_{best:09d}/core.pt":
                raise ValueError(
                    "DEV selection does not name the corresponding immutable core"
                )
            steps.add(best)
        for step in steps:
            rel = f"{name}/snapshots/step_{step:09d}/core.pt"
            if rel in returned and (root / rel).is_file():
                plans.append(
                    (
                        not (terminal.exists() and name.startswith("module_")),
                        step != 8192,
                        step != 4096,
                        name,
                        step,
                        identity,
                        returned[rel],
                    )
                )
    print(
        json.dumps(
            {
                "stage": "precal_only",
                "ready_runs": len({p[3] for p in plans}),
                "available_returned_cores": len(plans),
                "confirmation_opened": False,
            }
        ),
        flush=True,
    )
    for _, _, _, name, step, identity, sha in sorted(plans, key=lambda p: p[:5]):
        existing = (
            root / name / "precal" / f"step_{step:09d}" / "manifest.json"
        ).is_file()
        prepare_core(root / name, identity, step, sha, allow_calculation=True)
        if not existing:
            print(
                json.dumps(
                    {
                        "run": name,
                        "step": step,
                        "sizes": list(SIZES),
                        "weights": sha,
                        "stage": "precal_registered",
                    }
                ),
                flush=True,
            )
    return len(plans)
