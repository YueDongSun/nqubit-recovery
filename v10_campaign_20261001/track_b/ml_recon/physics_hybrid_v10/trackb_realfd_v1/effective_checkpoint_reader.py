"""CPU weights-only V2 checkpoint inspection, with no optimizer construction.

This component binds file bytes to caller-supplied expectations and verifies
the stored bounded runtime contract. It is not a producer-process terminal
receipt, a replay of dataset admission, or an attestation of reader-code parity.
Those remain responsibilities of the source-bound evaluation entry point.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import io
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .contracts import canonical_json_bytes, canonical_sha256, require_git_oid, require_sha256
from .checkpoint_rng_validation import validate_checkpoint_rng_structure
from .effective_observation_loss import Effective_observation_loss_config
from .effective_observation_v2 import Effective_observation_config, Effective_singleton_observation, validate_global_initialization_buffers
from .pilot_diagnostics import Pilot_diagnostic_config
from .read_only_pilot_audit import (
    _CHECKPOINT_FIELDS, _CHECKPOINT_METADATA_FIELDS, _require_exact_mapping,
    _runtime_environment_from_dict, _validate_model_state,
)
from .training import (
    TrackBRealRuntimeConfigV1, TrackBRealRuntimeIdentityV1,
    _assert_finite_runtime_state, trackb_model_state_sha,
)


V2_METADATA_FIELDS = _CHECKPOINT_METADATA_FIELDS | {
    "runtime_config", "loss_config", "diagnostic_config", "response_scope", "idler_event",
}
TERMINAL_LOSS_FIELDS = {
    "loss", "four_cell_nll", "global_multi_nll", "global_multi_available_rows",
    "effective_response_prior", "prepared_state_prior",
}


def _typed_config(config_type: type, payload: object):
    if not isinstance(payload, dict) or set(payload) != {field.name for field in fields(config_type)}:
        raise ValueError("checkpoint configuration preimage fields are invalid")
    result = config_type(**payload)
    result.validate()
    if canonical_json_bytes(asdict(result)) != canonical_json_bytes(payload):
        raise ValueError("checkpoint configuration changes under typed parsing")
    return result


def _expected_visits(identity: TrackBRealRuntimeIdentityV1, steps: int) -> dict[str, int]:
    samples = identity.batch_plan_sample_ids
    batch_size = identity.batch_plan_batch_size
    batches_per_pass = (len(samples) + batch_size - 1) // batch_size
    passes, remainder = divmod(steps, batches_per_pass)
    return {sample: visits for index, sample in enumerate(samples)
            if (visits := passes + int(index < remainder * batch_size)) > 0}


def _validate_terminal_state(payload: Mapping[str, Any], identity: TrackBRealRuntimeIdentityV1,
                             config: TrackBRealRuntimeConfigV1, model: torch.nn.Module) -> None:
    for name in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_state"):
        if not isinstance(payload[name], Mapping):
            raise ValueError(f"{name} must be a mapping")
        _assert_finite_runtime_state(payload[name], name=name)
    optimizer = payload["optimizer_state_dict"]
    groups, slots = optimizer.get("param_groups"), optimizer.get("state")
    parameters = list(model.parameters())
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict) \
            or groups[0].get("params") != list(range(len(parameters))) \
            or not isinstance(slots, dict) or not slots:
        raise ValueError("bounded AdamW parameter topology is incomplete")
    for index, slot in slots.items():
        if type(index) is not int or not 0 <= index < len(parameters) or not isinstance(slot, dict) \
                or set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("bounded AdamW slot identity is invalid")
        step = slot["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 \
                or not 1 <= float(step) <= config.optimizer_steps or float(step) != int(float(step)):
            raise ValueError("bounded AdamW slot step is invalid")
        for name in ("exp_avg", "exp_avg_sq"):
            if not isinstance(slot[name], torch.Tensor) or slot[name].shape != parameters[index].shape \
                    or slot[name].dtype != parameters[index].dtype:
                raise ValueError("bounded AdamW slot tensor does not match the model")
        if bool((slot["exp_avg_sq"] < 0).any()):
            raise ValueError("bounded AdamW second moment cannot be negative")
    scheduler = payload["scheduler_state_dict"]
    current_lr = groups[0].get("lr")
    last_lrs = scheduler.get("_last_lr")
    if type(current_lr) not in (int, float) or not math.isfinite(current_lr) or current_lr < 0 \
            or type(last_lrs) is not list or len(last_lrs) != 1 \
            or type(last_lrs[0]) not in (int, float) or not math.isfinite(last_lrs[0]) or last_lrs[0] < 0:
        raise ValueError("terminal learning rates must be finite nonnegative numbers")
    eta_min = scheduler.get("eta_min")
    if type(eta_min) not in (int, float) or eta_min != 0 \
            or not math.isclose(current_lr, eta_min, rel_tol=0, abs_tol=1e-15):
        raise ValueError("terminal cosine learning rate must match registered eta_min=0")
    if scheduler.get("T_max") != config.optimizer_steps or scheduler.get("last_epoch") != config.optimizer_steps \
            or scheduler.get("_step_count") != config.optimizer_steps + 1 \
            or scheduler.get("base_lrs") != [config.learning_rate] \
            or scheduler.get("_last_lr") != [groups[0].get("lr")]:
        raise ValueError("bounded scheduler terminal cursor or horizon is invalid")
    if groups[0].get("weight_decay") != config.weight_decay or groups[0].get("initial_lr") != config.learning_rate:
        raise ValueError("bounded optimizer configuration differs from its preimage")
    if payload["scaler_state_dict"] != {}:
        raise ValueError("CPU FP32 checkpoint must have an empty disabled scaler")
    validate_checkpoint_rng_structure(payload["rng_state"])
    visits = payload["row_visits"]
    if not isinstance(visits, dict) or any(type(value) is not int or value <= 0 for value in visits.values()) \
            or visits != _expected_visits(identity, config.optimizer_steps):
        raise ValueError("terminal row visits do not match the exact committed batch schedule")
    losses = payload["last_loss_components"]
    if not isinstance(losses, dict) or set(losses) != TERMINAL_LOSS_FIELDS \
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in losses.values()):
        raise ValueError("terminal effective loss components are invalid")
    gradient = payload["last_gradient_norm"]
    if type(gradient) not in (int, float) or not math.isfinite(gradient) or gradient < 0:
        raise ValueError("terminal gradient norm is invalid")


@dataclass(frozen=True)
class Effective_checkpoint_replay:
    identity: TrackBRealRuntimeIdentityV1
    runtime_config: TrackBRealRuntimeConfigV1
    loss_config: Effective_observation_loss_config
    diagnostic_config: Pilot_diagnostic_config
    initial_model: Effective_singleton_observation
    final_model: Effective_singleton_observation
    final_model_state_sha: str
    file_sha: str
    row_visits: tuple[tuple[str, int], ...]

    def summary(self) -> dict[str, Any]:
        for model in (self.initial_model, self.final_model):
            if model.config.sha256() != self.identity.model_config_sha \
                    or any(module.training for module in model.modules()) \
                    or any(parameter.requires_grad or parameter.device.type != "cpu" for parameter in model.parameters()) \
                    or any(buffer.device.type != "cpu" for buffer in model.buffers()):
                raise ValueError("reloaded model configuration or CPU eval/frozen state changed")
            validate_global_initialization_buffers(model)
        if trackb_model_state_sha(self.initial_model) != self.identity.initial_model_state_sha \
                or trackb_model_state_sha(self.final_model) != self.final_model_state_sha:
            raise ValueError("reloaded model changed after checkpoint inspection")
        return {
            "schema": "TrackBEffectiveCheckpointInspectionV1",
            "scope": "CHECKPOINT_STRUCTURE_AND_CPU_MODEL_RELOAD_ONLY",
            "checkpoint_file_sha": self.file_sha, "identity_sha": self.identity.sha256(),
            "source_commit": self.identity.source_commit, "source_tree": self.identity.source_tree,
            "system_type": self.identity.system_type,
            "model_config": self.final_model.config.to_dict(),
            "global_initialization_kind": "SEED_ONLY_HEAD" if self.final_model.config.global_initialization is None else "TRAIN_FOLD_COUNT_BASELINE",
            "runtime_config": asdict(self.runtime_config), "loss_config": asdict(self.loss_config),
            "diagnostic_config": asdict(self.diagnostic_config),
            "initial_model_state_sha": trackb_model_state_sha(self.initial_model),
            "final_model_state_sha": self.final_model_state_sha,
            "global_step": self.runtime_config.optimizer_steps,
            "checkpoint_completed": True,
            "row_visit_total": sum(value for _, value in self.row_visits),
            "unique_rows_visited": len(self.row_visits),
            "optimizer_constructed": False, "runtime_resumed": False, "scientific_gate_issued": False,
            "rng_check_scope": "PYTHON_NUMPY_CPU_TORCH_FORMAT_AND_CUDA_BYTE_VECTOR_STRUCTURE_ONLY",
            "unverified": ["producer_process_terminal_receipt", "dataset_admission_replay", "reader_source_code_parity", "diagnostic_history_files"],
        }


def load_effective_checkpoint_bytes(
    checkpoint_bytes: bytes, *, expected_file_sha: str, expected_identity_sha: str,
    expected_source_commit: str, expected_source_tree: str,
) -> Effective_checkpoint_replay:
    """Parse exactly one externally identified byte snapshot, without resume."""
    require_sha256(expected_file_sha, "expected_file_sha")
    require_sha256(expected_identity_sha, "expected_identity_sha")
    require_git_oid(expected_source_commit, "expected_source_commit")
    require_git_oid(expected_source_tree, "expected_source_tree")
    if type(checkpoint_bytes) is not bytes or hashlib.sha256(checkpoint_bytes).hexdigest() != expected_file_sha:
        raise ValueError("effective checkpoint byte identity mismatch")
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    _require_exact_mapping(payload, fields=_CHECKPOINT_FIELDS, name="effective checkpoint")
    metadata = _require_exact_mapping(payload["metadata"], fields=frozenset(V2_METADATA_FIELDS), name="effective checkpoint metadata")
    from .effective_family_context_v3 import CHECKPOINT_SCHEMA as FAMILY_CHECKPOINT_SCHEMA
    if metadata["schema"] not in {"TrackBEffectiveObservationCheckpointV2", FAMILY_CHECKPOINT_SCHEMA}:
        raise ValueError("not a registered effective observation checkpoint")
    identity = TrackBRealRuntimeIdentityV1.from_dict(metadata["identity"])
    if identity.sha256() != expected_identity_sha or metadata["identity_sha"] != expected_identity_sha \
            or (identity.source_commit, identity.source_tree) != (expected_source_commit, expected_source_tree):
        raise ValueError("effective checkpoint producer identity mismatch")
    config = Effective_observation_config.from_dict(metadata["model_config"])
    if (metadata["schema"] == FAMILY_CHECKPOINT_SCHEMA) != (config.measurement_context is not None):
        raise ValueError("effective checkpoint schema and measurement-context configuration disagree")
    runtime = _typed_config(TrackBRealRuntimeConfigV1, metadata["runtime_config"])
    loss = _typed_config(Effective_observation_loss_config, metadata["loss_config"])
    diagnostic = _typed_config(Pilot_diagnostic_config, metadata["diagnostic_config"])
    if config.global_initialization is not None:
        from .global_count_initialization import validate_global_initialization_binding

        validate_global_initialization_binding(config.global_initialization, identity)
    environment = _runtime_environment_from_dict(metadata["runtime_environment"])
    if runtime.schema != "TrackBRealRuntimeConfigV1" or runtime.execution_mode != "pilot" or runtime.precision != "fp32" \
            or environment.device_type != "cpu" or environment.precision != "fp32" or environment.amp_enabled:
        raise ValueError("effective reader admits bounded CPU FP32 pilots only")
    if config.sha256() != metadata["model_config_sha"] or config.sha256() != identity.model_config_sha \
            or runtime.sha256() != metadata["runtime_config_sha"] \
            or canonical_sha256(asdict(loss)) != metadata["loss_weights_sha"] \
            or environment.sha256() != metadata["runtime_environment_sha"] \
            or config.system_type != identity.system_type or runtime.batch_size != identity.batch_plan_batch_size:
        raise ValueError("effective checkpoint configuration binding mismatch")
    if type(metadata["global_step"]) is not int or metadata["global_step"] != runtime.optimizer_steps:
        raise ValueError("effective checkpoint has not exhausted its registered budget")
    for name, expected in {"completed": True, "research_only": True, "selection_evidence": False,
                           "hardware_action_authorized": False, "long_training_started": False,
                           "long_authorization_sha": None}.items():
        if metadata[name] is not expected:
            raise ValueError("effective checkpoint completion or authority flags are invalid")
    if metadata["response_scope"] != "CONDITIONAL_ON_NATIVE_EFFECTIVE_READ" \
            or metadata["idler_event"] != "STICKY_OR_OF_EXACT_SINGLETON_OVER_19_ACCEPTED_CLOCKS":
        raise ValueError("effective checkpoint event contract differs")
    previous_dtype = torch.get_default_dtype()
    try:
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.set_default_dtype(torch.float32)
            torch.random.default_generator.manual_seed(identity.seed)
            initial = Effective_singleton_observation(config)
            if trackb_model_state_sha(initial) != identity.initial_model_state_sha:
                raise ValueError("effective checkpoint M0 initialization cannot be reproduced")
            final = copy.deepcopy(initial)
    finally:
        torch.set_default_dtype(previous_dtype)
    _validate_model_state(final, payload["model_state_dict"], expected_sha=metadata["model_state_sha"])
    validate_global_initialization_buffers(final)
    _validate_terminal_state(payload, identity, runtime, final)
    initial.eval().requires_grad_(False)
    final.eval().requires_grad_(False)
    return Effective_checkpoint_replay(identity, runtime, loss, diagnostic, initial, final,
                                       metadata["model_state_sha"], expected_file_sha,
                                       tuple(sorted(payload["row_visits"].items())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--expected-file-sha", required=True)
    parser.add_argument("--expected-identity-sha", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    args = parser.parse_args(argv)
    replay = load_effective_checkpoint_bytes(
        Path(args.checkpoint).read_bytes(), expected_file_sha=args.expected_file_sha,
        expected_identity_sha=args.expected_identity_sha, expected_source_commit=args.expected_source_commit,
        expected_source_tree=args.expected_source_tree,
    )
    print(canonical_json_bytes(replay.summary()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
