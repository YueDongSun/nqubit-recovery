"""End-to-end engineering smoke for the isolated Track A package."""

from __future__ import annotations

import io
from typing import Any, Dict

import numpy as np
import torch

from ml_recon.data_v2_4.feedback_gate_contract import (
    FEEDBACK_GATE_BASE_DIM,
    FEEDBACK_GATE_DIM,
    compose_feedback_gate_features,
)
from ml_recon.physics_hybrid_v10.contracts import sha256_of

from ..config import TrackAConfig
from ..contracts import TrackAAuthority
from ..data import TrackAStream
from ..models import TrackAModelSuite, state_dict_sha
from ..physics import CollisionArm, build_collision_arm, build_recoverability_bundle
from ..physics.fixed_k import random_fixed_k_state
from .gates import TrackAGateLedger, evidence_sha
from .runtime import TrackARuntimeAdapter


def _classifier_batch(
    batch_size: int = 2,
    n: int = 4,
    settings: int = 3,
) -> Dict[str, torch.Tensor]:
    mode_mask = torch.ones(batch_size, n, dtype=torch.bool)
    adjacency = torch.ones(batch_size, n, n) - torch.eye(n).unsqueeze(0)
    incidence = torch.zeros(batch_size, n, 2)
    incidence[:, :2, 0] = 1.0
    incidence[:, 2:, 1] = 1.0
    return {
        "counts": torch.ones(batch_size, settings, 4),
        "basis_descriptors": torch.zeros(batch_size, settings, 15),
        "shots": torch.full((batch_size, settings), 64.0),
        "setting_uncertainty": torch.full((batch_size, settings), 0.1),
        "setting_mask": torch.ones(batch_size, settings, dtype=torch.bool),
        "mode_mask": mode_mask,
        "mode_features": torch.zeros(batch_size, n, 12),
        "physical_adjacency": adjacency,
        "correlation_adjacency": adjacency.clone(),
        "incidence": incidence,
        "certificate_features": torch.zeros(batch_size, 1, 10),
        "certificate_mask": torch.ones(batch_size, 1, dtype=torch.bool),
    }


def _feedback_inputs(batch_size: int = 2, n: int = 4, settings: int = 3) -> Dict[str, torch.Tensor]:
    base_gate = np.zeros(FEEDBACK_GATE_BASE_DIM, dtype=np.float32)
    gate_features = np.stack([
        compose_feedback_gate_features(base_gate, arm="A3_DENOISE_CLASS")
        for _ in range(batch_size)
    ])
    return {
        "noisy_measurement_tokens": torch.zeros(batch_size, settings, 2),
        "denoised_measurement_tokens": torch.zeros(batch_size, settings, 2),
        "covariance": torch.ones(batch_size, settings, 2) * 0.1,
        "target_mode_tokens": torch.zeros(batch_size, n, 2),
        "esm_mode_tokens": torch.zeros(batch_size, n, 2),
        "physical_mode_tokens": torch.zeros(batch_size, n, 2),
        "gate_features": torch.from_numpy(gate_features),
        "analytic_action": torch.zeros(batch_size, n, 2),
        "mode_mask": torch.ones(batch_size, n, dtype=torch.bool),
    }


def _finite_gradients(module: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return bool(gradients) and all(bool(torch.all(torch.isfinite(value))) for value in gradients)


def _finite_nonzero_objective_gradients(
    objective: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> bool:
    """Check one named objective is connected to finite, nonzero head gradients."""

    if not bool(torch.isfinite(objective)) or not parameters:
        return False
    gradients = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    active = [value for value in gradients if value is not None]
    return bool(active) and all(
        bool(torch.all(torch.isfinite(value))) for value in active
    ) and any(bool(torch.any(value != 0.0)) for value in active)


def run_track_a_smoke(config: TrackAConfig, *, device: str = "cpu") -> Dict[str, Any]:
    config.validate()
    selected_device = torch.device(device)
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError("Track A engineering smoke supports cpu or cuda")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    authority = TrackAAuthority()
    authority.validate()

    physics_rng = np.random.Generator(np.random.Philox(config.data.global_seed))
    k1 = random_fixed_k_state(4, 1, physics_rng, max_sparse_terms=4)
    k2 = random_fixed_k_state(6, 2, physics_rng, max_sparse_terms=15)
    k3_large = random_fixed_k_state(64, 3, physics_rng, max_sparse_terms=64)
    k2_matrix = k2.coefficient_matrix_k2()
    collision = {
        arm.value: build_collision_arm(k2, collision_mass=0.1, arm=arm)
        for arm in CollisionArm
    }
    dense_large_rejected = False
    try:
        k3_large.dense_state_vector(dense_n_limit=config.physics.dense_n_limit)
    except ValueError:
        dense_large_rejected = True

    stream_rank0 = TrackAStream(config, stage="D2", rank=0, world_size=2)
    stream_rank1 = TrackAStream(config, stage="D2", rank=1, world_size=2)
    key0 = stream_rank0.key(global_step=3, microbatch=1, slot=2)
    key0_repeat = stream_rank0.key(global_step=3, microbatch=1, slot=2)
    key1 = stream_rank1.key(global_step=3, microbatch=1, slot=2)
    sample0 = stream_rank0.render(key0)
    sample0_repeat = stream_rank0.render(key0_repeat)
    deterministic_stream = (
        sample0.truth["state_sha"] == sample0_repeat.truth["state_sha"]
        and np.array_equal(
            sample0.observation["mode_counts"],
            sample0_repeat.observation["mode_counts"],
        )
    )
    rank_disjoint = key0.sample_id != key1.sample_id

    runtime_adapter = TrackARuntimeAdapter(config, device=str(selected_device))
    runtime_adapter.seed_all()
    runtime_contract = runtime_adapter.contract_report()
    suite = TrackAModelSuite.build_mainline(config)
    suite.classifier.to(selected_device)
    suite.feedback.to(selected_device)
    classifier_batch = {
        key: value.to(selected_device)
        for key, value in _classifier_batch().items()
    }
    classifier_output = suite.classifier(classifier_batch, N=4, fine_identifiable=False)
    classifier_objectives = {
        "shared": classifier_output["semantic_latent"].square().mean(),
        "family_head": classifier_output["fine_family_logits"].square().mean(),
        "structure_heads": (
            classifier_output["pair_entanglement_logits"].square().mean()
            + classifier_output["certificate_status_logits"].square().mean()
            + classifier_output["hyperedge_entanglement_probs"].square().mean()
        ),
        "w_head": (
            classifier_output["W_populations"].square().mean()
            + classifier_output["W_coherence_re"].square().mean()
            + classifier_output["joint_control_covariance"].square().mean()
        ),
        "denoise_head": (
            classifier_output["p_clean"].square().mean()
            + classifier_output["covariance_clean_diag"].square().mean()
            + classifier_output["nuisance_posterior_mean"].square().mean()
        ),
        "noise_head": (
            classifier_output["OOD_probability"].square().mean()
            + classifier_output["operational_noise_logits"].square().mean()
        ),
    }
    classifier_parameter_groups = suite.classifier.audit_parameter_groups()
    classifier_head_gradients = {
        name: _finite_nonzero_objective_gradients(
            objective, classifier_parameter_groups[name])
        for name, objective in classifier_objectives.items()
    }
    classifier_loss = torch.stack(tuple(classifier_objectives.values())).sum()
    classifier_loss.backward()
    classifier_gradients_finite = _finite_gradients(suite.classifier)

    feedback_inputs = {
        key: value.to(selected_device)
        for key, value in _feedback_inputs().items()
    }
    feedback_output = suite.feedback(feedback_inputs)
    feedback_objectives = {
        "control_moe": (
            feedback_output["delta_log_amp"].square().mean()
            + feedback_output["delta_phase"].square().mean()
            + feedback_output["moe_load_balance_loss"]
        ),
        "global_decision_head": (
            feedback_output["recoverability"].square().mean()
            + feedback_output["gain_quantiles"].square().mean()
            + feedback_output["harmful_probability"].square().mean()
            + feedback_output["stop_probability"].square().mean()
            + feedback_output["noop_gain"].square().mean()
            + feedback_output["history_gain"].square().mean()
        ),
        "recoverability_class_head": (
            feedback_output["recoverability_class_logits"].square().mean()
        ),
        "next_measurement_head": feedback_output["next_measurement"].square().mean(),
    }
    feedback_parameter_groups = {
        "control_moe": list(suite.feedback.moe.parameters()),
        "global_decision_head": list(suite.feedback.global_head.parameters()),
        "recoverability_class_head": list(
            suite.feedback.recoverability_class_head.parameters()),
        "next_measurement_head": list(suite.feedback.next_measurement_head.parameters()),
    }
    feedback_head_gradients = {
        name: _finite_nonzero_objective_gradients(
            objective, feedback_parameter_groups[name])
        for name, objective in feedback_objectives.items()
    }
    feedback_loss = torch.stack(tuple(feedback_objectives.values())).sum()
    feedback_loss.backward()
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    feedback_gradients_finite = _finite_gradients(suite.feedback)

    checkpoint_buffer = io.BytesIO()
    torch.save({
        "classifier": suite.classifier.state_dict(),
        "feedback": suite.feedback.state_dict(),
    }, checkpoint_buffer)
    checkpoint_buffer.seek(0)
    reloaded_state = torch.load(
        checkpoint_buffer, map_location="cpu", weights_only=True)
    reloaded_suite = TrackAModelSuite.build_mainline(config)
    reloaded_suite.classifier.load_state_dict(reloaded_state["classifier"], strict=True)
    reloaded_suite.feedback.load_state_dict(reloaded_state["feedback"], strict=True)
    checkpoint_roundtrip_exact = (
        state_dict_sha(suite.classifier.state_dict())
        == state_dict_sha(reloaded_suite.classifier.state_dict())
        and state_dict_sha(suite.feedback.state_dict())
        == state_dict_sha(reloaded_suite.feedback.state_dict())
    )

    observation_dim = 12
    logical_dim = 6
    jacobian = physics_rng.normal(size=(observation_dim, logical_dim))
    covariance = np.eye(observation_dim, dtype=np.float64)
    residual = physics_rng.normal(size=observation_dim)
    recoverability = build_recoverability_bundle(
        sample_id=sample0.key.sample_id,
        residual=residual,
        logical_jacobian=jacobian,
        covariance=covariance,
        direction_stability=0.5,
    )

    checks = {
        "authority_fail_closed": True,
        "fixed_k1_normalized": bool(np.isclose(np.vdot(k1.amplitudes, k1.amplitudes), 1.0)),
        "fixed_k2_symmetric_zero_diagonal": bool(
            np.allclose(k2_matrix, k2_matrix.T) and np.allclose(np.diag(k2_matrix), 0.0)
        ),
        "fixed_k_one_rdm_trace": bool(np.isclose(np.trace(k2.one_rdm()).real, 2.0)),
        "large_n_dense_path_rejected": dense_large_rejected,
        "collision_three_arm_separation": bool(
            collision[CollisionArm.COHERENT_EXACT.value].cross_sector_coherence_norm > 0.0
            and collision[CollisionArm.SECTOR_DEPHASED.value].cross_sector_coherence_norm == 0.0
            and collision[CollisionArm.CLASSICAL_MIXTURE.value].approximation
        ),
        "counter_stream_deterministic": deterministic_stream,
        "rank_keys_disjoint": rank_disjoint,
        "classifier_gradients_finite": classifier_gradients_finite,
        "feedback_gradients_finite": feedback_gradients_finite,
        "classifier_head_gradients_finite_nonzero": all(
            classifier_head_gradients.values()),
        "feedback_head_gradients_finite_nonzero": all(
            feedback_head_gradients.values()),
        "canonical_feedback_gate_abi": bool(
            suite.gate_input_dim == FEEDBACK_GATE_DIM
            and feedback_inputs["gate_features"].shape[-1] == FEEDBACK_GATE_DIM),
        "checkpoint_state_roundtrip_exact": checkpoint_roundtrip_exact,
        "runtime_adapter_reuses_v10_trainer": bool(
            runtime_contract["base_runtime"]
            == "ml_recon.physics_hybrid_v10.track_a_n8.training_runtime.V10Trainer"
            and runtime_contract["run_mode"] == "smoke"
            and runtime_contract["training_authorized"] is False
            and runtime_contract["ten_k_training_authorized"] is False),
        "recoverability_bundle_valid": recoverability.reason_code != "",
    }
    passed = all(checks.values())
    evidence = evidence_sha(checks, config.sha256(), sample0.key.sha256())
    ledger = TrackAGateLedger(authority)
    report = {
        "schema": "V10TrackAEngineeringSmokeV1",
        "status": "PASS_ENGINEERING_SMOKE" if passed else "FAIL_ENGINEERING_SMOKE",
        "source_bound": False,
        "config_sha": config.sha256(),
        "checks": checks,
        "gradient_audit": {
            "classifier": classifier_head_gradients,
            "feedback": feedback_head_gradients,
        },
        "runtime_adapter": runtime_contract,
        "model_parameter_counts": suite.parameter_counts(),
        "sample_id": sample0.key.sample_id,
        "recoverability": {
            "rank_fraction": recoverability.rank_fraction,
            "kappa_geometry": recoverability.kappa_geometry,
            "epsilon_irrecoverable": recoverability.epsilon_irrecoverable,
            "reason_code": recoverability.reason_code,
        },
        "gate_ledger": ledger.report(),
        "gate_assessment": {
            "status": "PARTIAL_LOCAL_EVIDENCE" if passed else "LOCAL_SMOKE_FAILED",
            "engineering_evidence_sha": evidence,
            "gate_decisions_written": False,
            "external_gate_artifact_required": True,
        },
        "authority": authority.__dict__,
        "runtime": {
            "device": str(selected_device),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda or "",
            "cuda_device_name": (
                torch.cuda.get_device_name(selected_device)
                if selected_device.type == "cuda" else ""
            ),
        },
        "limitations": [
            "Local smoke evidence does not close P0, E0, or any later gate.",
            "P0/E0/E1/E2/E3 and S1-S5 remain NOT_RUN pending external artifacts.",
            "The in-memory state roundtrip is not a V10Trainer resume proof.",
            "No 10k training, model selection, promotion, or hardware action is authorized.",
        ],
    }
    report["report_sha"] = sha256_of(report)
    return report
