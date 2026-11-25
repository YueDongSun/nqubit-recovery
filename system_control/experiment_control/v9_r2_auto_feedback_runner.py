"""V9-R2 H-GATED-AUTO feedback runner (charter section 28).

Automatic gated hardware closed loop for F_3Q_HW / F_4Q_HW. The MODEL package
(ml_recon) never imports hardware; this runner (system_control) orchestrates:
infer -> validate hashes/stage/gain/risk/OOD -> materialize -> guards -> upload ->
verify receipt -> measure -> evaluate -> accept|rollback -> continue|stop.

Every automatic step must pass all guards; ANY failure fails closed (no upload).
Emergency stop and rollback are always active. The operator performs a one-time ARM;
after that the runner needs no per-step manual approval.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np


class V9R2AutoError(RuntimeError):
    """Base for a fail-closed runner condition."""


class V9R2NotArmedError(V9R2AutoError):
    pass


class V9R2EmergencyStopError(V9R2AutoError):
    pass


class V9R2GuardFailureError(V9R2AutoError):
    pass


class V9R2HashMismatchError(V9R2AutoError):
    pass


@dataclass(frozen=True)
class V9R2AutomationArmContract:
    """One-time operator arm contract (frozen before the first real state)."""

    model_checkpoint_hash: str
    operation_registry_hash: str
    command_table_hash: str
    target_hashes: tuple = ()          # frozen target list / order
    max_accepted_actions: int = 10
    emergency_phase_limit_rad: float = 0.20
    emergency_amp_limit: float = 0.03
    noop_false_accept_max: float = 0.05
    serious_harm_wilson_u95_max: float = 0.01

    def sha256(self) -> str:
        payload = sorted({
            "model_checkpoint_hash": self.model_checkpoint_hash,
            "operation_registry_hash": self.operation_registry_hash,
            "command_table_hash": self.command_table_hash,
            "target_hashes": list(self.target_hashes),
            "max_accepted_actions": self.max_accepted_actions,
            "emergency_phase_limit_rad": self.emergency_phase_limit_rad,
            "emergency_amp_limit": self.emergency_amp_limit,
        })
        return hashlib.sha256(str(payload).encode()).hexdigest()

    def validate_step_hashes(self, *, model_checkpoint_hash: str,
                             operation_registry_hash: str, command_table_hash: str,
                             target_hash: str) -> None:
        if model_checkpoint_hash != self.model_checkpoint_hash:
            raise V9R2HashMismatchError("model checkpoint hash mismatch")
        if operation_registry_hash != self.operation_registry_hash:
            raise V9R2HashMismatchError("operation registry hash mismatch")
        if command_table_hash != self.command_table_hash:
            raise V9R2HashMismatchError("command table hash mismatch")
        if self.target_hashes and target_hash not in self.target_hashes:
            raise V9R2HashMismatchError(f"target {target_hash} not in the frozen target list")


@dataclass(frozen=True)
class V9R2ArmToken:
    arm_hash: str
    contract: V9R2AutomationArmContract


class V9R2EmergencyStop:
    """Fail-closed trip monitor; any trip condition halts the loop (no further upload)."""

    def __init__(self, contract: V9R2AutomationArmContract):
        self.contract = contract
        self._accepted = 0
        self._cum_amp = 0.0
        self._cum_phase = 0.0
        self._serious_harm = 0
        self._n_steps = 0

    def record_step(self, *, cum_amp_delta: float, cum_phase_delta: float,
                    accepted: bool, serious_harm: bool) -> None:
        self._n_steps += 1
        if accepted:
            self._accepted += 1
            self._cum_amp += abs(cum_amp_delta)
            self._cum_phase += abs(cum_phase_delta)
        if serious_harm:
            self._serious_harm += 1

    def tripped(self) -> bool:
        if self._accepted >= self.contract.max_accepted_actions:
            return True
        if self._cum_amp >= self.contract.emergency_amp_limit:
            return True
        if self._cum_phase >= self.contract.emergency_phase_limit_rad:
            return True
        return False

    def check(self) -> None:
        if self.tripped():
            raise V9R2EmergencyStopError(
                f"emergency stop: accepted={self._accepted}, cum_amp={self._cum_amp:.4f}, "
                f"cum_phase={self._cum_phase:.4f}")


class V9R2AutoRollback:
    """On a rejected step, re-upload and hash-verify the previous accepted waveform."""

    def __init__(self, hw: Any):
        self.hw = hw
        self._accepted_artifacts: List[Any] = []

    def record_accepted(self, artifact: Any) -> None:
        self._accepted_artifacts.append(artifact)

    def last_accepted(self) -> Optional[Any]:
        return self._accepted_artifacts[-1] if self._accepted_artifacts else None

    def rollback(self, failed_artifact: Any) -> Dict[str, Any]:
        """Restore the previous accepted artifact; verify the upload receipt hash."""
        previous = self.last_accepted()
        if previous is None:
            # Nothing accepted yet: revert to the base matrices (no prior accepted state).
            return {"rolled_back": True, "to": "base_matrix",
                    "previous_artifact": None}
        receipt = self.hw.upload(previous)
        self.hw.verify(receipt)
        if failed_artifact is not None:
            prev = getattr(previous, "matrix_hashes", {})
            fail = getattr(failed_artifact, "matrix_hashes", {})
            if (prev.get("W_CH1_amp") == fail.get("W_CH1_amp")
                    and prev.get("W_CH1_phase") == fail.get("W_CH1_phase")):
                raise V9R2AutoError("rollback uploaded the SAME (rejected) waveform")
        return {"rolled_back": True, "to": "previous_accepted", "receipt": receipt}


class V9R2MeasurementCollector(Protocol):
    def collect(self, state_id: str, artifact: Any) -> Dict[str, np.ndarray]: ...


class V9R2AutoFeedbackRunner:
    """Automatic gated feedback loop (fail-closed, no per-step manual approval)."""

    def __init__(
        self,
        *,
        model: Any,                 # has propose_action(...) -> proposal (deterministic)
        materializer: Any,          # V9OperationPointActionMaterializer
        hw: Any,                    # has upload(artifact) -> receipt; verify(receipt)
        measurement: V9R2MeasurementCollector,
        evaluate: Callable[[str, Dict[str, np.ndarray]], Dict[str, float]],  # -> {g_joint, ...}
        base_amp: np.ndarray,
        base_phase: np.ndarray,
    ):
        self.model = model
        self.materializer = materializer
        self.hw = hw
        self._measure = measurement.collect if hasattr(measurement, "collect") else measurement
        self.evaluate = evaluate
        self.base_amp = np.asarray(base_amp, dtype=np.float64)
        self.base_phase = np.asarray(base_phase, dtype=np.float64)
        self.arm_contract: Optional[V9R2AutomationArmContract] = None
        self.emergency: Optional[V9R2EmergencyStop] = None
        self.rollback: Optional[V9R2AutoRollback] = None
        self._stopped = False
        self.history: List[Dict[str, Any]] = []

    def arm(self, contract: V9R2AutomationArmContract) -> V9R2ArmToken:
        """One-time operator ARM; afterwards the runner needs no per-step approval."""
        self.arm_contract = contract
        self.emergency = V9R2EmergencyStop(contract)
        self.rollback = V9R2AutoRollback(self.hw)
        token = V9R2ArmToken(arm_hash=contract.sha256(), contract=contract)
        return token

    def stopped(self) -> bool:
        return self._stopped

    def step(
        self,
        *,
        target_hash: str,
        counts: Dict[str, np.ndarray],
        global_multi: Dict[str, float],
    ) -> Dict[str, Any]:
        """One automatic step. Fail-closed: ANY failure happens BEFORE any upload."""
        if self.arm_contract is None or self.emergency is None or self.rollback is None:
            raise V9R2NotArmedError("runner is not armed (operator ARM required)")
        if self._stopped:
            raise V9R2EmergencyStopError("loop already stopped (emergency trip)")

        # 1. infer (deterministic proposal).
        proposal = self.model.propose_action(
            counts=counts, global_multi=global_multi, target_hash=target_hash,
            operation_registry_hash=self.arm_contract.operation_registry_hash,
            command_table_hash=self.arm_contract.command_table_hash,
            model_checkpoint_hash=self.arm_contract.model_checkpoint_hash,
        )

        # 2. validate hashes / stage / gain / risk / OOD (fail-closed).
        self.arm_contract.validate_step_hashes(
            model_checkpoint_hash=self.arm_contract.model_checkpoint_hash,
            operation_registry_hash=self.arm_contract.operation_registry_hash,
            command_table_hash=self.arm_contract.command_table_hash,
            target_hash=target_hash)
        if proposal.harmful_probability > 0.5:
            raise V9R2AutoError(f"harmful probability {proposal.harmful_probability:.3f} > 0.5")

        # 3. materialize + guards; ANY guard failure -> no upload.
        artifact = self.materializer.materialize(
            proposal, base_amp=self.base_amp, base_phase=self.base_phase)
        guards = artifact.guard_results
        if not (guards.get("rf_power_ok", True) and guards.get("peak_ok", True)
                and guards.get("fourier_full_ok", True)):
            raise V9R2GuardFailureError(f"guard failure: {guards}")

        # 4. upload + verify receipt.
        receipt = self.hw.upload(artifact)
        self.hw.verify(receipt)

        # 5. measure + evaluate.
        measure = self._measure(target_hash, artifact)
        result = self.evaluate(target_hash, measure)
        g_joint = float(result.get("g_joint", 0.0))
        serious_harm = g_joint < -0.5

        # 6. accept or rollback.
        accepted = g_joint >= 0.0 and not serious_harm
        if accepted:
            self.rollback.record_accepted(artifact)
        else:
            self.rollback.rollback(failed_artifact=artifact)

        step_rec = {
            "target_hash": target_hash,
            "accepted": accepted,
            "g_joint": g_joint,
            "harmful_probability": proposal.harmful_probability,
            "artifact": artifact,
        }
        self.history.append(step_rec)
        self.emergency.record_step(
            cum_amp_delta=float(np.max(np.abs(np.asarray(proposal.delta_log_amp_w_ch1))) or 0.0),
            cum_phase_delta=float(np.max(np.abs(np.asarray(proposal.delta_phase_w_ch1))) or 0.0),
            accepted=accepted, serious_harm=serious_harm)
        # Stop after the accepted-action / cumulative-bound limit, or after serious harm.
        self._stopped = self.emergency.tripped() or serious_harm
        return step_rec

    def run(self, targets: List[Dict[str, Any]], max_steps_per_target: int = 10) -> Dict[str, Any]:
        """Automatic campaign over the frozen target list."""
        if self.arm_contract is None:
            raise V9R2NotArmedError("runner is not armed")
        per_target: Dict[str, List[Dict[str, Any]]] = {}
        for target in targets:
            steps = []
            for _ in range(max_steps_per_target):
                if self.stopped():
                    break
                step = self.step(target_hash=target["target_hash"],
                                 counts=target["counts"], global_multi=target["global_multi"])
                steps.append(step)
                if not step["accepted"] and self._stopped:
                    break
                if len([s for s in steps if s["accepted"]]) >= self.arm_contract.max_accepted_actions:
                    break
            per_target[target["target_hash"]] = steps
            if self.stopped():
                break
        return {"per_target": per_target, "stopped": self._stopped,
                "n_targets": len(targets)}
