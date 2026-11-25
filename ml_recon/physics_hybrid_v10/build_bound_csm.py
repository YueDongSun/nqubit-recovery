"""Single scientific entry point for bound same-forward CSM construction."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .control_manifold import ControlManifold, get_control_manifold
from .csm_contract import CSMContractV1
from .csm_covariance import (
    BlockSupportWhitener,
    MultinomialBlockSupportWhitener,
    SupportWhitener,
)
from .csm_geometry import (
    CSMGeometryResult, _seal_bound_csm_geometry, compute_csm_geometry,
)
from .csm_observation_contract import CSMObservationContract
from .csm_validators import finite_difference_plateau_audit
from .forward_instance import CSMForwardInstanceContract, forward_callable_sha, object_sha


def build_bound_csm_from_forward(
    forward: Callable[[np.ndarray], np.ndarray],
    u0: np.ndarray,
    covariance: np.ndarray | Sequence[np.ndarray] | str,
    N: int,
    *,
    instance_contract: CSMForwardInstanceContract,
    contract: CSMContractV1,
    manifold: ControlManifold | None = None,
    whitener: (
        SupportWhitener | BlockSupportWhitener | MultinomialBlockSupportWhitener
        | None
    ) = None,
    observation_contract: CSMObservationContract,
    require_plateau: bool = True,
) -> CSMGeometryResult:
    """Build CSM-G only after all forward-instance identities are verified."""
    if not require_plateau:
        raise ValueError("formal same-forward CSM cannot disable the FD plateau gate")
    if np.iscomplexobj(u0):
        raise ValueError("bound CSM base point must be real")
    base_point = np.asarray(u0, dtype=np.float64)
    if (base_point.ndim != 1 or base_point.shape != (2 * N - 2,)
            or not np.all(np.isfinite(base_point))):
        raise ValueError("bound CSM base point must be finite with logical dimension 2N-2")
    instance_contract.validate()
    if forward_callable_sha(forward) != instance_contract.forward_code_sha:
        raise ValueError("supplied forward callable does not match the forward instance")
    if object_sha(np.asarray(u0, dtype=np.float64)) != instance_contract.u0_sha:
        raise ValueError("supplied u0 does not match the forward instance")
    contract.validate(require_bound=True)
    observation_contract.validate()
    if (contract.feedback_observation_contract_sha
            != observation_contract.sha256()):
        raise ValueError(
            "CSM observation contract does not match the bound contract")
    if contract.observation_contract_kind != observation_contract.kind:
        raise ValueError(
            "CSM observation kind does not match the bound contract")
    if contract.forward_instance_sha != instance_contract.sha256():
        raise ValueError("CSM contract is not bound to the supplied forward instance")
    if contract.forward_sha != instance_contract.forward_code_sha:
        raise ValueError("CSM forward code SHA does not match the forward instance")
    if (contract.finite_difference.forward_sha != instance_contract.forward_code_sha
            or contract.finite_difference.manifold_sha != instance_contract.manifold_sha):
        raise ValueError("finite-difference contract is not bound to the same forward/manifold")
    if contract.manifold_contract_sha != instance_contract.manifold_sha:
        raise ValueError("CSM manifold SHA does not match the forward instance")
    if contract.feedback_measurement_schedule_sha \
            != instance_contract.feedback_measurement_schedule_sha:
        raise ValueError("CSM feedback measurement schedule is not instance-bound")
    selected = manifold or get_control_manifold(contract.control_manifold_id, N)
    if selected.sha256() != instance_contract.manifold_sha:
        raise ValueError("selected manifold does not match the forward instance")
    if selected.N != N:
        raise ValueError("manifold N does not match bound CSM N")
    baseline = np.asarray(forward(base_point))
    if (baseline.ndim == 0 or baseline.size == 0 or np.iscomplexobj(baseline)
            or not np.all(np.isfinite(baseline))):
        raise ValueError("bound forward baseline must be a finite real observation array")

    audit = finite_difference_plateau_audit(
        forward, np.asarray(u0, dtype=np.float64),
        h_amplitude=contract.finite_difference.h_amplitude,
        h_phase=contract.finite_difference.h_phase,
        tolerance=contract.finite_difference.plateau_tolerance,
        forward_sha=instance_contract.forward_code_sha,
        manifold_sha=instance_contract.manifold_sha,
    )
    if require_plateau and not bool(audit["plateau_pass"]):
        raise ValueError("same-forward finite-difference plateau did not pass")
    geometry = compute_csm_geometry(
        audit["jacobian_h"], covariance, N, manifold=selected,
        contract=contract,
        whitener=whitener,
        forward_instance_sha=instance_contract.sha256(),
        finite_difference_audit=audit,
    )
    if geometry.covariance_sha != instance_contract.covariance_sha:
        raise ValueError("CSM covariance does not match the forward instance")
    geometry.forward_sample_id = instance_contract.sample_id
    _seal_bound_csm_geometry(geometry)
    return geometry


__all__ = ["build_bound_csm_from_forward"]
