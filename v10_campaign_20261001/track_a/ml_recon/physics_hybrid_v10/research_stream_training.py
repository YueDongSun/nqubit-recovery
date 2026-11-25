"""User-authorized V10 Track A research stream training route.

Campaign: v10_campaign_20261001 (explicit user task instruction, 2026-10-01).

This module is a narrowly scoped, versioned research execution route. It
reuses the registered V2.4 stream physics core unchanged:

* ``plan_registered_rung`` (validated rung plan, deterministic cell layout)
* ``classifier_stream_datasets`` (stream-only deterministic sample rendering)
* ``build_seeded_classifier`` (seeded C-S / registered profile init)
* ``classifier_registered_multi_cell_step`` (registered mask-aware objective)
* ``V24CellHomogeneousBatchSampler`` (homogeneous-cell batching)

The default task pool is ``small_n_classifier`` (the registered 3Q/4Q
classifier rungs; plan IDs are unchanged for historical launches).
``--task-pool large_n_structure_pilot`` selects the bounded N=5/N=6
``general_n_structure`` pilot rung — same research-only authority, same
route, no new physics or model code. ``--task-pool large_n_structure_n{N}``
selects the single-N structure rungs (one cell, shared row identity with the
pilot pool's cell for that N).

and mirrors the V10Trainer optimizer semantics exactly (AdamW lr/weight
decay, cosine schedule per epoch, AMP GradScaler with the registered initial
scale, gradient clipping, hard failure on non-finite loss/gradients, committed
optimizer-step accounting, mean-of-batches aggregate validation loss, best
metric checkpointing, early-stopping patience).

It deliberately does NOT modify or replace the gated formal route
(``train_streaming_classifier`` + ``V10StreamPilotAdmissionV2``); that route
and its requirements remain intact. This research route substitutes the
historical admission/branch ceremony with direct readiness verification
(source identity, validated plan, deterministic recipe identity, runtime and
device check, finite forward/backward probe, fresh owned output directory,
bounded budget) because the current user task authorization explicitly
directs execution without the historical receipt chain.

All outputs are engineering evidence only:
``research_only=true``, ``scientific_gate_evaluated=false``,
``promotion_ready=false``, ``selection_evidence=false``,
``hardware_action_authorized=false``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_recon.data_v2_4.multi_cell import V24CellHomogeneousBatchSampler
from ml_recon.data_v2_4.rung_plan import (
    SINGLE_N_STRUCTURE_POOLS, plan_registered_rung,
)
from ml_recon.data_v2_4.streaming import classifier_stream_datasets

from .classifier_n_full import CLASSIFIER_PROFILES, V10ClassifierNFull
from .loss_profiles import LOSS_WEIGHT_PROFILE_NAMES
from .research_run_ownership import (
    ARCHIVE_NAME, LOCK_NAME, ExclusiveRunOwnership,
    claim_exclusive_run_ownership,
)
from .research_resource_budget import Resource_budget, Resource_limit_exceeded
from .train_classifier import (
    CLASSIFIER_AMP_INITIAL_SCALE, build_seeded_classifier,
    classifier_registered_multi_cell_step,
)

ROUTE_ID = "user_authorized_research_route_v1"
AUTHORIZATION = "user task instruction 2026-10-03 (Track A MC training continuation)"
AUTHORITY_FLAGS = {
    "research_only": True,
    "scientific_gate_evaluated": False,
    "promotion_ready": False,
    "selection_evidence": False,
    "hardware_action_authorized": False,
}
MAX_EPOCHS = 96
DEFAULT_TRAIN_ROWS = 1000
DEFAULT_VALIDATION_ROWS = 200
RESEARCH_TASK_POOLS = ("small_n_classifier", "large_n_structure_pilot",
                       *SINGLE_N_STRUCTURE_POOLS)
PER_CELL_REPORT_KEYS = (
    "loss", "small_n_family", "measurement", "pair_map", "hyperedge",
    "partition", "depth", "intactness", "stretchability", "coarse",
)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_provenance() -> Dict[str, str]:
    root = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> str:
        try:
            return subprocess.check_output(("git", *arguments), cwd=root,
                                           text=True).strip()
        except (subprocess.CalledProcessError, OSError):
            return ""

    return {
        "repository_root": str(root),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree_sha": git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
    }


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(value, device) for value in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def _make_loader(dataset, *, shuffle: bool, epoch: int, batch_size: int,
                 seed: int, num_workers: int,
                 allowed_cells=None) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + (17 if shuffle else 29) + 1_000_003 * int(epoch))
    batch_sampler = V24CellHomogeneousBatchSampler(
        dataset, batch_size=batch_size, shuffle=shuffle,
        seed=seed + 1_000_003 * int(epoch), allowed_cells=allowed_cells)
    worker_arguments: Dict[str, Any] = {"num_workers": num_workers,
                                        "generator": generator}
    if num_workers:
        worker_arguments["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, batch_sampler=batch_sampler, **worker_arguments)


def _run_epoch(model, step_function, loader, *, training: bool, device,
               optimizer, scaler, amp_active: bool, clip_norm: float,
               state: Dict[str, Any], epoch: int,
               resource_budget: Resource_budget) -> Dict[str, float]:
    model.train(training)
    totals: Dict[str, float] = {}
    per_cell_totals: Dict[str, Dict[str, float]] = {}
    count = 0
    for raw_batch in loader:
        resource_budget.check()
        batch = _move_to_device(raw_batch, device)
        if training:
            state["attempts"] += 1
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_active):
            losses = step_function(model, batch)
            loss = losses["loss"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite {'train' if training else 'validation'} loss")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients:
                raise RuntimeError("training loss produced no gradients")
            if not all(bool(torch.all(torch.isfinite(g))) for g in gradients):
                invalid = [n for n, p in model.named_parameters()
                           if p.grad is not None and not bool(torch.all(torch.isfinite(p.grad)))]
                raise FloatingPointError(f"non-finite gradient detected; epoch={epoch}; params={invalid[:16]}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            state["overflow_count"] += int(scaler.get_scale() < previous_scale)
            state["last_gradient_norm"] = float(gradient_norm)
        cell_ids = batch.get("cell_id")
        cell_key = str(cell_ids[0]) if isinstance(cell_ids, (list, tuple)) and cell_ids else ""
        cell_bucket = per_cell_totals.setdefault(cell_key, {})
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                scalar = float(value.detach().cpu())
                totals[key] = totals.get(key, 0.0) + scalar
                cell_bucket[key] = cell_bucket.get(key, 0.0) + scalar
        per_cell_totals.setdefault(cell_key, {})["__count__"] = \
            cell_bucket.get("__count__", 0.0) + 1.0
        count += 1
    if count == 0:
        raise ValueError("loader produced zero batches")
    result = {key: value / count for key, value in totals.items()}
    result["batches"] = float(count)
    per_cell = {}
    for cell_key, bucket in per_cell_totals.items():
        cell_count = max(bucket.pop("__count__", 0.0), 1.0)
        per_cell[cell_key] = {key: value / cell_count for key, value in bucket.items()}
    result["per_cell"] = per_cell
    return result


def _run_epoch_with_resource_boundary(model, step_function, loader, *,
                                      training: bool, device, optimizer,
                                      scaler, amp_active: bool, clip_norm: float,
                                      state: Dict[str, Any], epoch: int,
                                      resource_budget: Resource_budget):
    """Return no metrics for an epoch interrupted before full completion."""
    initial_step = state["global_step"]
    initial_attempts = state["attempts"]
    try:
        metrics = _run_epoch(
            model, step_function, loader, training=training, device=device,
            optimizer=optimizer, scaler=scaler, amp_active=amp_active,
            clip_norm=clip_norm, state=state, epoch=epoch,
            resource_budget=resource_budget)
    except Resource_limit_exceeded as exc:
        return None, {
            "reason": str(exc), "interrupted_epoch": epoch,
            "partial_optimizer_steps": state["global_step"] - initial_step,
            "partial_optimizer_attempts": state["attempts"] - initial_attempts,
        }
    return metrics, None


def _stream_model_build_spec(model, model_profile: str) -> Dict[str, Any]:
    """Checkpoint build spec for the stream route.

    Deliberately NOT ``model.checkpoint_build_spec()`` wholesale: that dict
    carries ``architecture_class`` and exact constructor dimensions, which
    would route loaders through ``classifier_from_build_spec`` and its exact
    field-set contract.  The profile string rebuilds the identical
    architecture; the hyperedge fusion wiring version is carried alongside
    so a ``-HE`` checkpoint records which forward it trained under.
    """
    spec = {"model_id": "C_N_FULL_V10", "profile": model_profile}
    wiring_version = model.checkpoint_build_spec().get(
        "hyperedge_fusion_wiring_version")
    if wiring_version is not None:
        spec["hyperedge_fusion_wiring_version"] = wiring_version
    return spec


def _validate_resume_identity(checkpoint: Dict[str, Any], *,
                              provenance: Dict[str, Any],
                              expected_model_build_spec: Dict[str, Any]) -> None:
    """Reject a resume whose data, source or model identity changed."""
    metadata = checkpoint.get("metadata") or {}
    saved_provenance = metadata.get("provenance") or {}
    required_pairs = (
        ("seed", saved_provenance.get("seed"), provenance.get("seed")),
        ("init_seed", saved_provenance.get(
            "init_seed", saved_provenance.get("seed")), provenance.get("init_seed")),
        ("rung_plan_sha256", saved_provenance.get("rung_plan_sha256"),
         provenance.get("rung_plan_sha256")),
        ("stream_recipe_sha256", saved_provenance.get("stream_recipe_sha256"),
         provenance.get("stream_recipe_sha256")),
    )
    for name, saved_value, current_value in required_pairs:
        if saved_value is None or current_value is None or saved_value != current_value:
            raise ValueError(
                f"resume identity mismatch on {name}: checkpoint has "
                f"{saved_value!r}, current run has {current_value!r}")
    saved_git = saved_provenance.get("git") or {}
    current_git = provenance.get("git") or {}
    for name in ("source_commit", "source_tree_sha"):
        saved_value = saved_git.get(name)
        current_value = current_git.get(name)
        if not saved_value or not current_value or saved_value != current_value:
            raise ValueError(
                f"resume identity mismatch on git.{name}: checkpoint has "
                f"{saved_value!r}, current run has {current_value!r}")
    saved_spec = checkpoint.get("model_build_spec")
    if saved_spec is None or saved_spec != expected_model_build_spec:
        raise ValueError(
            "resume identity mismatch on model_build_spec: checkpoint has "
            f"{saved_spec!r}, current run expects {expected_model_build_spec!r}")


def _checkpoint_payload(model, optimizer, scheduler, scaler, *, epoch: int,
                        state: Dict[str, Any], provenance: Dict[str, Any],
                        config: Dict[str, Any], best_metric: float,
                        model_profile: str) -> Dict[str, Any]:
    return {
        "metadata": {
            "schema": "V10ResearchStreamCheckpointV1",
            "route_id": ROUTE_ID,
            "authorization": AUTHORIZATION,
            **AUTHORITY_FLAGS,
            "epoch": epoch,
            "global_step": state["global_step"],
            "elapsed_seconds_total": state.get("elapsed_seconds_total", 0.0),
            "best_metric": best_metric,
            "metric_name": "validation_loss",
            "metric_direction": "min",
            "provenance": provenance,
            "config": config,
        },
        "model_build_spec": _stream_model_build_spec(model, model_profile),
        "model_state_dict": {key: value.detach().cpu()
                             for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
        },
    }


def _reload_and_verify(path: Path, model, probe_batch,
                       cell_n: Dict[str, int], device) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    model.eval()
    probe_batch = _move_to_device(probe_batch, device)
    cell_id = str(probe_batch["cell_id"][0])
    if cell_id not in cell_n:
        raise RuntimeError(f"verification batch references unknown cell {cell_id!r}")
    n_qubits = cell_n[cell_id]
    with torch.no_grad():
        expected_output = model(probe_batch, N=n_qubits)
        expected = {key: value.detach().float().cpu()
                    for key, value in expected_output.items()
                    if isinstance(value, torch.Tensor)}
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        gc.collect()
        actual_output = model(probe_batch, N=n_qubits)
    actual = {key: value.detach().float().cpu()
              for key, value in actual_output.items()
              if isinstance(value, torch.Tensor)}
    if set(expected) != set(actual):
        raise RuntimeError(
            "checkpoint inference tensor keys differ: "
            f"trained={sorted(expected)}, checkpoint={sorted(actual)}")
    differences = []
    for key in sorted(expected):
        if expected[key].shape != actual[key].shape:
            raise RuntimeError(
                f"checkpoint inference shape mismatch for {key}: "
                f"trained={tuple(expected[key].shape)}, "
                f"checkpoint={tuple(actual[key].shape)}")
        if not bool(torch.isfinite(expected[key]).all()) or not bool(torch.isfinite(actual[key]).all()):
            raise RuntimeError(f"checkpoint inference output is non-finite: {key}")
        differences.append(float((expected[key] - actual[key]).abs().max()))
        # FP32 CPU and accelerator inference can differ slightly by kernel;
        # require close parity with an explicit absolute and relative bound.
        if not torch.allclose(expected[key], actual[key], rtol=1e-6, atol=1e-6):
            raise RuntimeError(
                f"checkpoint inference mismatch for {key}: "
                f"max_abs_diff={differences[-1]:.8g}, tolerance=1e-6")
    if not differences:
        raise RuntimeError("checkpoint inference verification produced no tensor outputs")
    max_abs_diff = max(differences)
    return {
        "reloaded": False,
        "checkpoint_state_loaded": True,
        "load_mode": "state_dict_into_existing_trained_model",
        "comparison": "checkpoint_latest_vs_preload_trained_model_inference",
        "probe_batch": "deterministic_train_batch_epoch_0",
        "outputs_compared": sorted(expected),
        "all_outputs_finite": True,
        "equality_tolerance": {"rtol": 1e-6, "atol": 1e-6},
        "schema": metadata.get("schema"),
        "epoch": metadata.get("epoch"),
        "global_step": metadata.get("global_step"),
        "best_metric": metadata.get("best_metric"),
        "reload_max_abs_diff": max_abs_diff,
    }


def _atomic_torch_save(payload, path: Path) -> None:
    """Write a checkpoint via temp file + atomic replace so a mid-save kill
    (host memory-pressure reap) can never leave a truncated checkpoint; a
    stale ``.tmp`` residue from such a kill is simply overwritten by the
    next save."""
    temp_path = path.parent / (path.name + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def _rebind_cosine_horizon(scheduler, optimizer, epochs: int) -> bool:
    """Re-bind a restored CosineAnnealingLR to a longer schedule horizon.

    A schedule-extension resume (``--resume-from`` with larger ``--epochs``)
    builds the scheduler for the new horizon, but ``load_state_dict`` restores
    the saved ``T_max`` together with the annealed-to-end learning rate; the
    added epochs would then run on the wrong, out-of-domain cosine curve
    (pinned at ``eta_min`` or re-rising past the old horizon, depending on
    scheduler internals). This keeps the restored progress (``last_epoch``),
    applies the new ``T_max``, and recomputes the learning rate at the resume
    point from the closed form. The jump onto the new curve is the accepted
    warm-restart discontinuity of an extension resume. Returns True when a
    rebind happened.
    """
    if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        return False
    horizon = max(epochs, 1)
    if int(scheduler.T_max) == horizon:
        return False
    scheduler.T_max = horizon
    progress = min(int(scheduler.last_epoch), horizon)
    for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
        group["lr"] = (
            scheduler.eta_min
            + (base_lr - scheduler.eta_min)
            * (1.0 + math.cos(math.pi * progress / horizon)) / 2.0
        )
    return True


def run_research_stream_training(
    *, output_dir: str, run_id: str, seed: int = 20261001, device: str = "cuda",
    epochs: int = 12, batch_size: int = 16, learning_rate: float = 3e-4,
    weight_decay: float = 1e-5, scheduler_kind: str = "cosine",
    model_profile: str = "C-S", loss_weight_profile: str = "default",
    train_rows: int = DEFAULT_TRAIN_ROWS,
    validation_rows: int = DEFAULT_VALIDATION_ROWS,
    amp_enabled: bool = True, patience: int = 10, num_workers: int = 0,
    gradient_clip_norm: float = 1.0, max_minutes: float = 0.0,
    init_seed: Optional[int] = None,
    curriculum_standard_epochs: int = 0,
    resume_from: Optional[str] = None,
    task_pool: str = "small_n_classifier",
    min_available_memory_mb: float = 0.0,
    max_process_memory_mb: float = 0.0,
) -> Dict[str, Any]:
    # ``seed`` owns the deterministic data stream (common random numbers across
    # a seed panel); ``init_seed`` optionally varies ONLY model initialization
    # and optimizer seeding so multi-seed panels keep identical data membership.
    init_seed = seed if init_seed is None else init_seed
    # --- bounded research budget -------------------------------------------
    if not (1 <= epochs <= MAX_EPOCHS):
        raise ValueError(f"research route epochs must be within 1..{MAX_EPOCHS}")
    if not (0 <= curriculum_standard_epochs < epochs):
        raise ValueError(
            "curriculum_standard_epochs must satisfy 0 <= value < epochs "
            "(a curriculum covering every epoch would never train noisy cells)")
    if batch_size <= 0 or learning_rate <= 0.0 or patience < 1:
        raise ValueError("batch size, learning rate, and patience must be positive")
    if scheduler_kind not in ("cosine", "constant"):
        raise ValueError("scheduler_kind must be cosine or constant")
    if model_profile not in CLASSIFIER_PROFILES:
        raise ValueError("model_profile must be a registered classifier profile")
    if loss_weight_profile not in LOSS_WEIGHT_PROFILE_NAMES:
        raise ValueError("loss_weight_profile must be registered")
    if task_pool not in RESEARCH_TASK_POOLS:
        raise ValueError(
            "task_pool must be one of " + ", ".join(RESEARCH_TASK_POOLS))
    if curriculum_standard_epochs and task_pool != "small_n_classifier":
        raise ValueError(
            "curriculum_standard_epochs is defined for the small_n_classifier "
            "pool only (its standard/noisy cell split does not exist here)")

    output = Path(output_dir)
    if resume_from is None and output.exists() and any(output.iterdir()):
        contents = {path.name for path in output.iterdir()}
        if contents <= {LOCK_NAME, ARCHIVE_NAME}:
            # A run claimed the directory but never wrote an artifact
            # (hard-killed between claim and the first write); point the
            # operator at the stale record instead of the generic gate.
            raise ValueError(
                f"output directory contains only a stale ownership record "
                f"({sorted(contents)}); a prior run claimed it but never "
                "wrote artifacts. Verify the recorded process is stopped, "
                "then delete the record — or relaunch with --resume-from "
                "if the run has a checkpoint to continue from")
        raise ValueError("research route requires an empty new output directory")
    resume_path = Path(resume_from) if resume_from is not None else None
    if resume_path is not None:
        if resume_path.parent.resolve() != output.resolve():
            raise ValueError(
                "resume checkpoint must live inside the resumed run's "
                "output directory")
        if not (output / "metrics.jsonl").is_file():
            raise ValueError("resume requires the run's existing metrics.jsonl")
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive ownership fence (external review 2026-10-02, round 2): the
    # empty-directory check above is not atomic and resume had no fencing at
    # all, so a resume launched while the original trainer still runs would
    # interleave metrics.jsonl / checkpoint writes.  The claim fails closed
    # on a live owner; a hard kill leaves a stale record that the next
    # resume launch archives after verifying the recorded PID is dead.
    # Release runs on every exit path below, including failures.
    ownership = claim_exclusive_run_ownership(output, run_id)
    try:
        return _run_claimed_research_stream_training(
            output=output, ownership=ownership, resume_path=resume_path,
            run_id=run_id, seed=seed, epochs=epochs, device=device,
            batch_size=batch_size, learning_rate=learning_rate,
            weight_decay=weight_decay, scheduler_kind=scheduler_kind,
            model_profile=model_profile,
            loss_weight_profile=loss_weight_profile,
            train_rows=train_rows, validation_rows=validation_rows,
            amp_enabled=amp_enabled, patience=patience,
            num_workers=num_workers,
            gradient_clip_norm=gradient_clip_norm,
            max_minutes=max_minutes, init_seed=init_seed,
            curriculum_standard_epochs=curriculum_standard_epochs,
            resume_from=resume_from, task_pool=task_pool,
            min_available_memory_mb=min_available_memory_mb,
            max_process_memory_mb=max_process_memory_mb)
    finally:
        ownership.release()


def _run_claimed_research_stream_training(
        *, output: Path, ownership: ExclusiveRunOwnership,
        resume_path: Optional[Path], run_id: str, seed: int, epochs: int,
        device: str, batch_size: int, learning_rate: float,
        weight_decay: float, scheduler_kind: str, model_profile: str,
        loss_weight_profile: str, train_rows: int, validation_rows: int,
        amp_enabled: bool, patience: int, num_workers: int,
        gradient_clip_norm: float, max_minutes: float,
        init_seed: Optional[int], curriculum_standard_epochs: int,
        resume_from: Optional[str], task_pool: str,
        min_available_memory_mb: float, max_process_memory_mb: float
        ) -> Dict[str, Any]:
    """Run the training body under an already-claimed ownership fence."""
    metrics_path = output / "metrics.jsonl"

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    resource_budget = Resource_budget(
        min_available_memory_mb=min_available_memory_mb,
        max_process_memory_mb=max_process_memory_mb,
        max_seconds=max_minutes * 60.0)

    # --- validated inputs: registered plan + deterministic stream recipe ----
    plan = plan_registered_rung(task_pool, train_rows, validation_rows)
    train, validation, recipe = classifier_stream_datasets(plan, global_seed=seed)
    cell_n = {cell.cell_id: cell.N for cell in plan.cell_allocations}
    standard_cells = {
        cell_id for cell_id in cell_n if "noisy" not in cell_id}
    if curriculum_standard_epochs and not standard_cells:
        raise ValueError("curriculum requested but no standard cells found")
    recipe_sha = str(recipe["recipe_sha256"])

    provenance = {
        "route_id": ROUTE_ID,
        "authorization": AUTHORIZATION,
        **AUTHORITY_FLAGS,
        "run_id": run_id,
        "git": _git_provenance(),
        "torch_version": torch.__version__,
        "device": str(torch_device),
        "device_name": torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda" else "cpu",
        "seed": seed,
        "init_seed": init_seed,
        "run_ownership": ownership.describe(),
        "rung_plan_sha256": plan.sha256(),
        "stream_recipe_sha256": recipe_sha,
        "stream_version": recipe.get("stream_version"),
        "input_delivery": recipe.get("input_delivery"),
        "cells": cell_n,
        "train_rows": len(train),
        "validation_rows": len(validation),
    }
    config = {
        "model_id": "C_N_FULL_V10", "model_profile": model_profile,
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "scheduler_kind": scheduler_kind, "amp_enabled": amp_enabled,
        "amp_initial_scale": CLASSIFIER_AMP_INITIAL_SCALE,
        "gradient_clip_norm": gradient_clip_norm, "patience": patience,
        "num_workers": num_workers, "loss_weight_profile": loss_weight_profile,
        "train_rows": train_rows, "validation_rows": validation_rows,
        "max_minutes": max_minutes,
        "min_available_memory_mb": min_available_memory_mb,
        "max_process_memory_mb": max_process_memory_mb,
        "curriculum_standard_epochs": curriculum_standard_epochs,
        "resume_from": resume_from,
        "task_pool": task_pool,
    }
    resume_checkpoint = None
    if resume_from is not None:
        resume_checkpoint = torch.load(resume_path, map_location="cpu",
                                       weights_only=False)
        saved_config = (resume_checkpoint.get("metadata") or {}).get("config") or {}
        for field in ("model_profile", "batch_size", "learning_rate",
                      "weight_decay", "scheduler_kind", "loss_weight_profile",
                      "train_rows", "validation_rows",
                      "curriculum_standard_epochs"):
            if field in saved_config:
                if saved_config[field] != config[field]:
                    raise ValueError(
                        f"resume config mismatch on {field}: checkpoint has "
                        f"{saved_config[field]!r}, this launch requests "
                        f"{config[field]!r}")
            elif config[field] != 0:
                # Checkpoints saved before the option existed ran with its
                # neutral default; resuming may not silently introduce it.
                raise ValueError(
                    f"resume cannot introduce {field}={config[field]!r}: "
                    "the checkpoint predates this option (absent == 0)")
        saved_pool = saved_config.get("task_pool", "small_n_classifier")
        if saved_pool != task_pool:
            # String-valued counterpart of the numeric loop above: checkpoints
            # saved before task_pool existed ran the small_n_classifier pool.
            raise ValueError(
                f"resume config mismatch on task_pool: checkpoint has "
                f"{saved_pool!r}, this launch requests {task_pool!r}")
        resume_after_epoch = int(resume_checkpoint["metadata"]["epoch"])
        if resume_after_epoch + 1 >= epochs:
            raise ValueError(
                "nothing to resume: checkpoint epoch "
                f"{resume_after_epoch} already covers the requested "
                f"{epochs} epochs")
    (output / "research_provenance.json").write_text(
        json.dumps({"provenance": provenance, "config": config}, indent=2,
                   default=str), encoding="utf-8")

    resource_budget.check()
    # --- model / objective / optimizer (mirror V10Trainer semantics) --------
    model = build_seeded_classifier(model_profile, init_seed)
    if resume_checkpoint is not None:
        _validate_resume_identity(
            resume_checkpoint, provenance=provenance,
            expected_model_build_spec=_stream_model_build_spec(
                model, model_profile))
    _seed_all(seed)  # restore common runtime RNG (CRN); only init used init_seed
    model.to(torch_device)
    step_function = classifier_registered_multi_cell_step(cell_n, loss_weight_profile)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)) if scheduler_kind == "cosine"
        else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0))
    amp_active = bool(amp_enabled and torch_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_active,
                                  init_scale=CLASSIFIER_AMP_INITIAL_SCALE)
    state: Dict[str, Any] = {"global_step": 0, "attempts": 0, "overflow_count": 0,
                             "last_gradient_norm": float("nan"),
                             "elapsed_seconds_total": 0.0}
    optimizer.register_step_post_hook(
        lambda opt, args, kwargs: state.update(global_step=state["global_step"] + 1))

    # --- readiness probe: finite forward/backward before committing budget --
    resource_budget.check()
    _seed_all(seed)
    probe_loader = _make_loader(train, shuffle=True, epoch=0, batch_size=batch_size,
                                seed=seed, num_workers=0)
    probe_batch = _move_to_device(next(iter(probe_loader)), torch_device)
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=torch_device.type, enabled=amp_active):
        probe_losses = step_function(model, probe_batch)
    probe_losses["loss"].backward()
    probe_gradients = [p.grad for p in model.parameters() if p.grad is not None]
    readiness = {
        "probe_loss": float(probe_losses["loss"].detach().cpu()),
        "probe_loss_finite": bool(torch.isfinite(probe_losses["loss"])),
        "probe_gradients_reached": len(probe_gradients) > 0,
        "probe_gradients_finite": all(
            bool(torch.all(torch.isfinite(g))) for g in probe_gradients),
        "amp_active": amp_active,
        "amp_initial_scale": scaler.get_scale(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    if not (readiness["probe_loss_finite"] and readiness["probe_gradients_reached"]
            and readiness["probe_gradients_finite"]):
        raise FloatingPointError(f"readiness probe failed: {readiness}")
    (output / "research_readiness.json").write_text(
        json.dumps(readiness, indent=2), encoding="utf-8")

    # Re-seed and rebuild optimizer state cleanly after the probe backward.
    del probe_losses, probe_gradients, probe_batch, probe_loader
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    del optimizer, model
    gc.collect()
    resource_budget.check()
    _seed_all(seed)
    model = build_seeded_classifier(model_profile, init_seed)
    _seed_all(seed)  # restore common runtime RNG (CRN); only init used init_seed
    model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)) if scheduler_kind == "cosine"
        else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_active,
                                  init_scale=CLASSIFIER_AMP_INITIAL_SCALE)
    state = {"global_step": 0, "attempts": 0, "overflow_count": 0,
             "last_gradient_norm": float("nan"), "elapsed_seconds_total": 0.0}
    optimizer.register_step_post_hook(
        lambda opt, args, kwargs: state.update(global_step=state["global_step"] + 1))

    # --- training loop -------------------------------------------------------
    started = time.perf_counter()
    best_metric = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    start_epoch = 0
    status = "COMPLETED"
    resource_stop = None
    resume_info = None
    schedule_extended = False
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        model.to(torch_device)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        schedule_extended = _rebind_cosine_horizon(scheduler, optimizer, epochs)
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        metadata = resume_checkpoint["metadata"]
        state["global_step"] = int(metadata["global_step"])
        rng_state = resume_checkpoint.get("rng_state") or {}
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
        if rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])
        if rng_state.get("torch_cpu") is not None:
            torch.set_rng_state(rng_state["torch_cpu"])
        if (torch_device.type == "cuda" and torch.cuda.is_available()
                and rng_state.get("torch_cuda")):
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        # Rebuild loop bookkeeping from the persisted epoch entries and drop
        # any stale tail written after the resume checkpoint (kill window
        # between the metrics append and the checkpoint save).
        entries = [json.loads(line) for line in
                   metrics_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        start_epoch = int(metadata["epoch"]) + 1
        kept = [entry for entry in entries
                if int(entry["epoch"]) < start_epoch]
        history = kept
        best_metric = (float(kept[-1]["best_metric"])
                       if kept else float(metadata["best_metric"]))
        improved_flags = [bool(entry.get("improved")) for entry in kept]
        best_epoch = max((int(entry["epoch"]) for entry, flag
                          in zip(kept, improved_flags) if flag), default=-1)
        bad_epochs = 0
        for flag in reversed(improved_flags):
            if flag:
                break
            bad_epochs += 1
        if kept:
            state["overflow_count"] = int(kept[-1].get("amp_overflow_count", 0))
            # Cumulative compute time is rebuilt from the persisted epoch
            # entries (checkpoints predating this field carried no clock).
            state["elapsed_seconds_total"] = float(sum(
                float(entry.get("epoch_seconds", 0.0)) for entry in kept))
        if len(kept) != len(entries):
            metrics_path.write_text(
                "".join(json.dumps(entry, default=str) + "\n"
                        for entry in kept), encoding="utf-8")
        resume_info = {
            "source": str(resume_path),
            "resumed_after_epoch": int(metadata["epoch"]),
            "resumed_global_step": int(metadata["global_step"]),
            "epochs_carried_over": len(kept),
            "stale_entries_dropped": len(entries) - len(kept),
            "best_metric_carried": best_metric,
            "schedule_extended": schedule_extended,
        }
        print(json.dumps({"resuming": resume_info}), flush=True)
    metrics_path = output / "metrics.jsonl"
    segment_start_step = state["global_step"]
    for epoch in range(start_epoch, epochs):
        # Epoch-boundary ownership fence: if another launcher legitimately
        # took the directory over (this owner was superseded mid-run), stop
        # writing metrics/checkpoints immediately instead of interleaving.
        ownership.assert_owned()
        epoch_started = time.perf_counter()
        # Noise-severity curriculum (research lever, default off): restrict
        # TRAINING to the standard cells for the first K epochs so the model
        # learns the clean signal before the noisy cells join. Validation
        # always covers every cell so best-epoch selection stays comparable.
        curriculum_cells = (
            standard_cells if epoch < curriculum_standard_epochs else None)
        train_loader = _make_loader(train, shuffle=True, epoch=epoch,
                                    batch_size=batch_size, seed=seed,
                                    num_workers=num_workers,
                                    allowed_cells=curriculum_cells)
        train_metrics, resource_stop = _run_epoch_with_resource_boundary(
            model, step_function, train_loader, training=True,
            device=torch_device, optimizer=optimizer, scaler=scaler,
            amp_active=amp_active, clip_norm=gradient_clip_norm,
            state=state, epoch=epoch, resource_budget=resource_budget)
        if train_metrics is not None:
            with torch.no_grad():
                validation_loader = _make_loader(
                    validation, shuffle=False, epoch=epoch,
                    batch_size=batch_size, seed=seed, num_workers=num_workers)
                validation_metrics, resource_stop = _run_epoch_with_resource_boundary(
                    model, step_function, validation_loader, training=False,
                    device=torch_device, optimizer=optimizer, scaler=scaler,
                    amp_active=amp_active, clip_norm=gradient_clip_norm,
                    state=state, epoch=epoch, resource_budget=resource_budget)
        if resource_stop is not None:
            status = "STOPPED_RESOURCE_LIMIT"
            resource_stop["last_completed_checkpoint"] = (
                "checkpoint_latest.pt"
                if (output / "checkpoint_latest.pt").is_file() else None)
            break
        scheduler.step()
        metric = float(validation_metrics["loss"])
        improved = metric < best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        epoch_seconds = time.perf_counter() - epoch_started
        entry = {
            "epoch": epoch, "global_step": state["global_step"],
            "train": {k: v for k, v in train_metrics.items() if k != "per_cell"},
            "validation": {k: v for k, v in validation_metrics.items() if k != "per_cell"},
            "validation_per_cell": {
                cell: {k: v for k, v in values.items() if k in PER_CELL_REPORT_KEYS}
                for cell, values in validation_metrics.get("per_cell", {}).items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "improved": improved, "best_metric": best_metric,
            "amp_scale": float(scaler.get_scale()),
            "amp_overflow_count": state["overflow_count"],
            "gradient_norm": state["last_gradient_norm"],
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "elapsed_seconds_cumulative": state["elapsed_seconds_total"] + epoch_seconds,
        }
        if torch_device.type == "cuda":
            entry["gpu_memory"] = {
                "allocated_bytes": torch.cuda.memory_allocated(torch_device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(torch_device)}
        history.append(entry)
        state["elapsed_seconds_total"] += epoch_seconds
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
        _atomic_torch_save(_checkpoint_payload(
            model, optimizer, scheduler, scaler, epoch=epoch, state=state,
            provenance=provenance, config=config, best_metric=best_metric,
            model_profile=model_profile), output / "checkpoint_latest.pt")
        if improved:
            _atomic_torch_save(_checkpoint_payload(
                model, optimizer, scheduler, scaler, epoch=epoch, state=state,
                provenance=provenance, config=config, best_metric=best_metric,
                model_profile=model_profile), output / "checkpoint_best.pt")
        print(json.dumps({"epoch": epoch, "global_step": state["global_step"],
                          "validation_loss": metric, "best_metric": best_metric,
                          "epoch_seconds": epoch_seconds}, default=str), flush=True)
        if bad_epochs >= patience:
            status = "EARLY_STOP_PATIENCE"
            break
        if max_minutes and (time.perf_counter() - started) / 60.0 > max_minutes:
            status = "EARLY_STOP_WALL_CLOCK"
            break

    elapsed = time.perf_counter() - started
    completed_epochs = len(history)
    segment_optimizer_steps = state["global_step"] - segment_start_step
    durable_checkpoint_optimizer_steps = (
        int(history[-1]["global_step"]) if history else
        int((resume_checkpoint or {}).get("metadata", {}).get("global_step", 0)))
    if durable_checkpoint_optimizer_steps > state["global_step"]:
        raise RuntimeError(
            "durable checkpoint optimizer step count exceeds current committed updates")
    summary = {
        "schema": "V10ResearchStreamTrainingSummaryV1",
        "route_id": ROUTE_ID,
        "authorization": AUTHORIZATION,
        **AUTHORITY_FLAGS,
        "run_id": run_id,
        "status": status,
        "resource_stop": resource_stop,
        "resource_budget": resource_budget.summary(),
        "seed": seed,
        "epochs_completed": completed_epochs,
        "epochs_requested": epochs,
        "committed_optimizer_steps": state["global_step"],
        "optimizer_step_count_definition": (
            "successful optimizer updates counted by the optimizer post-step hook; "
            "AMP-overflow-skipped attempts are excluded"),
        "durable_checkpoint_optimizer_steps": durable_checkpoint_optimizer_steps,
        "uncheckpointed_optimizer_steps": (
            state["global_step"] - durable_checkpoint_optimizer_steps),
        "segment_optimizer_steps": segment_optimizer_steps,
        "optimizer_attempts": state["attempts"],
        "amp_overflow_count": state["overflow_count"],
        "sample_visitation": {
            "train_samples_per_epoch": len(train),
            "train_sample_visits": len(train) * completed_epochs,
            "validation_sample_visits": len(validation) * completed_epochs,
        },
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "resume": resume_info,
        "final_validation_loss": history[-1]["validation"]["loss"] if history else None,
        "final_validation_per_cell": history[-1].get("validation_per_cell", {}) if history else {},
        "metric_name": "validation_loss",
        "metric_direction": "min",
        "wall_clock_seconds": elapsed,
        "wall_clock_seconds_cumulative": state["elapsed_seconds_total"],
        # Segment throughput only: dividing cumulative steps by this
        # process's clock made resumed runs appear faster than
        # uninterrupted ones (external review 2026-10-02).
        "throughput_steps_per_second": (
            segment_optimizer_steps / elapsed if elapsed else 0.0),
        "throughput_steps_per_second_cumulative": (
            state["global_step"] / state["elapsed_seconds_total"]
            if state["elapsed_seconds_total"] else 0.0),
        "provenance": provenance,
        "config": config,
        "readiness": readiness,
        "history": history,
    }
    latest_path = output / "checkpoint_latest.pt"
    if latest_path.is_file() and resource_stop is None:
        verify_loader = _make_loader(train, shuffle=True, epoch=0,
                                     batch_size=batch_size, seed=seed,
                                     num_workers=0)
        verify_batch = next(iter(verify_loader))
        summary["reload_verification"] = _reload_and_verify(
            latest_path, model, verify_batch, cell_n, torch_device)
    elif latest_path.is_file() and resource_stop is not None:
        summary["reload_verification"] = {
            "reloaded": False,
            "skipped_reason": (
                "resource stop interrupted an epoch; the trained in-memory "
                "model contains partial-epoch updates and cannot be compared "
                "with the last completed checkpoint"),
            "last_completed_checkpoint": "checkpoint_latest.pt",
        }
    # Terminal-state fence: the last epoch-boundary assert ran at the start
    # of the final epoch, so assert once more before writing the summary —
    # a superseded final-epoch owner must not emit the completion marker.
    ownership.assert_owned()
    (output / "research_training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--init-seed", type=int, default=None,
                        help="vary ONLY model init (default: same as --seed); "
                             "data stream stays bound to --seed for CRN panels")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", dest="scheduler_kind",
                        choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--model-profile", choices=tuple(CLASSIFIER_PROFILES),
                        default="C-S")
    parser.add_argument("--loss-weight-profile", choices=tuple(LOSS_WEIGHT_PROFILE_NAMES),
                        default="default")
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--validation-rows", type=int, default=DEFAULT_VALIDATION_ROWS)
    parser.add_argument("--no-amp", dest="amp_enabled", action="store_false")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--min-available-memory-mb", type=float, default=0.0)
    parser.add_argument("--max-process-memory-mb", type=float, default=0.0)
    parser.add_argument("--curriculum-standard-epochs", type=int, default=0,
                        help="train only standard (non-noisy) cells for the "
                             "first K epochs; 0 disables (default)")
    parser.add_argument("--resume-from", default=None,
                        help="checkpoint_latest.pt of an interrupted run in "
                             "the same output directory; restores model/"
                             "optimizer/scheduler/scaler state and continues "
                             "from the next epoch")
    parser.add_argument("--task-pool", choices=RESEARCH_TASK_POOLS,
                        default="small_n_classifier",
                        help="registered rung task pool to train "
                             "(large_n_structure_pilot = the bounded N=5/N=6 "
                             "general_n_structure pilot rung; "
                             "large_n_structure_n{N} = single-N structure rung)")
    arguments = parser.parse_args(argv)
    summary = run_research_stream_training(**vars(arguments))
    print(json.dumps({key: summary[key] for key in (
        "status", "epochs_completed", "committed_optimizer_steps", "best_metric",
        "best_epoch", "final_validation_loss", "wall_clock_seconds",
        "throughput_steps_per_second", "reload_verification",
        "resource_stop", "resource_budget")
        if key in summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
