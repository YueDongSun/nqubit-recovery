"""General partial transpose and optional PPT-mixture SDP certificates."""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .contracts import CertificateStatus


def _validate_density_shape(rho: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    state = np.asarray(rho, dtype=np.complex128)
    dimension = int(np.prod(dims))
    if state.shape != (dimension, dimension):
        raise ValueError("density shape does not match subsystem dimensions")
    return state


def partial_transpose(
    rho: np.ndarray,
    subsystems: int | Iterable[int],
    dims: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Partial transpose arbitrary subsystems of a multipartite density matrix."""
    state = np.asarray(rho, dtype=np.complex128)
    if dims is None:
        N = int(round(np.log2(state.shape[0])))
        if state.shape != (2 ** N, 2 ** N):
            raise ValueError("qubit density dimension must be a power of two")
        dims = (2,) * N
    dims = tuple(int(value) for value in dims)
    state = _validate_density_shape(state, dims)
    selected = (subsystems,) if isinstance(subsystems, int) else tuple(subsystems)
    if not selected or len(selected) != len(set(selected)) \
            or min(selected) < 0 or max(selected) >= len(dims):
        raise ValueError("partial-transpose subsystem selection is invalid")
    tensor = state.reshape(*dims, *dims)
    permutation = list(range(2 * len(dims)))
    for subsystem in selected:
        permutation[subsystem], permutation[len(dims) + subsystem] = (
            permutation[len(dims) + subsystem], permutation[subsystem])
    return tensor.transpose(permutation).reshape(state.shape)


def partial_transpose_two_qubit(rho: np.ndarray, subsystem: int = 1) -> np.ndarray:
    return partial_transpose(rho, subsystem, dims=(2, 2))


def negativity(
    rho: np.ndarray,
    subsystems: int | Iterable[int] = 1,
    dims: Optional[Sequence[int]] = None,
) -> float:
    eigenvalues = np.linalg.eigvalsh(partial_transpose(rho, subsystems, dims))
    return float(np.maximum(-eigenvalues, 0.0).sum())


def ppt_compatibility(
    rho: np.ndarray,
    subsystems: int | Iterable[int] = 1,
    dims: Optional[Sequence[int]] = None,
    tolerance: float = 1e-9,
) -> Dict[str, object]:
    value = negativity(rho, subsystems, dims)
    return {
        "negativity_lower_bound": value,
        "ppt": value <= tolerance,
        "status": (CertificateStatus.NPT_CERTIFIED.value if value > tolerance
                   else CertificateStatus.PPT_MIXTURE_COMPATIBLE.value),
        "separability_certified": False,
    }


def _flat_index(bits: Sequence[int], dims: Sequence[int]) -> int:
    index = 0
    for value, dimension in zip(bits, dims):
        index = index * dimension + int(value)
    return index


def _cvx_partial_transpose(variable, subsystems: Tuple[int, ...], dims: Tuple[int, ...]):
    import cvxpy as cp

    basis = tuple(itertools.product(*(range(dimension) for dimension in dims)))
    rows = []
    for row_bits in basis:
        row = []
        for column_bits in basis:
            source_row = list(row_bits)
            source_column = list(column_bits)
            for subsystem in subsystems:
                source_row[subsystem], source_column[subsystem] = (
                    source_column[subsystem], source_row[subsystem])
            row.append(variable[_flat_index(source_row, dims),
                                _flat_index(source_column, dims)])
        rows.append(row)
    return cp.bmat(rows)


def ppt_mixture_sdp(
    rho: np.ndarray,
    bipartitions: Sequence[Sequence[int]],
    *,
    dims: Optional[Sequence[int]] = None,
    solver: Optional[str] = None,
    tolerance: float = 1e-7,
) -> Dict[str, object]:
    """Test membership in the convex PPT-mixture relaxation.

    Feasibility is a compatibility result, not a separability certificate.
    Infeasibility certifies GME only when the supplied cuts are complete.
    """
    state = np.asarray(rho, dtype=np.complex128)
    if dims is None:
        N = int(round(np.log2(state.shape[0])))
        dims = (2,) * N
    dims = tuple(int(value) for value in dims)
    state = _validate_density_shape(state, dims)
    raw_cuts = tuple(tuple(sorted(set(int(q) for q in cut))) for cut in bipartitions)
    universe = frozenset(range(len(dims)))
    def canonical_cut(cut: Tuple[int, ...]) -> Tuple[int, ...]:
        selected = frozenset(cut)
        complement = universe - selected
        left = tuple(sorted(selected))
        right = tuple(sorted(complement))
        return min(left, right, key=lambda value: (len(value), value))
    cuts = tuple(dict.fromkeys(canonical_cut(cut) for cut in raw_cuts))
    if not cuts or any(not cut or max(cut) >= len(dims) for cut in cuts):
        raise ValueError("at least one valid non-empty bipartition is required")
    if state.shape[0] > 16:
        return {
            "status": CertificateStatus.UNAVAILABLE.value,
            "reason": "ppt_mixture_sdp_dimension_guard",
            "separability_certified": False,
        }
    try:
        import cvxpy as cp
    except ImportError:
        return {
            "status": CertificateStatus.UNAVAILABLE.value,
            "reason": "cvxpy_not_installed",
            "separability_certified": False,
        }

    components = [cp.Variable(state.shape, hermitian=True) for _ in cuts]
    constraints = [sum(components) == state]
    for component, cut in zip(components, cuts):
        constraints.extend((component >> 0,
                            _cvx_partial_transpose(component, cut, dims) >> 0))
    problem = cp.Problem(cp.Minimize(0.0), constraints)
    try:
        solve_kwargs = {"verbose": False}
        if solver is not None:
            solve_kwargs["solver"] = solver
        value = problem.solve(**solve_kwargs)
    except Exception as error:  # solver availability/failure must be explicit and fail closed.
        return {
            "status": CertificateStatus.NUMERICAL_FAILURE.value,
            "reason": type(error).__name__,
            "solver_status": getattr(problem, "status", None),
            "separability_certified": False,
        }

    expected_cuts = {
        canonical_cut(tuple(cut))
        for size in range(1, len(dims))
        for cut in itertools.combinations(range(len(dims)), size)
    }
    cuts_complete = set(cuts) == expected_cuts
    if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        status = CertificateStatus.PPT_MIXTURE_COMPATIBLE
    elif problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE) and cuts_complete:
        status = CertificateStatus.CERTIFIED_GME
    elif problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        status = CertificateStatus.INCONCLUSIVE
    else:
        status = CertificateStatus.NUMERICAL_FAILURE
    return {
        "status": status.value,
        "solver_status": problem.status,
        "objective": None if value is None else float(value),
        "cuts_complete": cuts_complete,
        "tolerance": tolerance,
        "separability_certified": False,
    }


def ppt_mixture_measurement_feasibility(
    observables: Sequence[np.ndarray],
    measured_values: Sequence[float],
    deltas: Sequence[float],
    bipartitions: Sequence[Sequence[int]],
    *,
    dims: Sequence[int],
    solver: Optional[str] = None,
) -> Dict[str, object]:
    """PPT-mixture feasibility under incomplete measurement intervals."""
    dims = tuple(int(value) for value in dims)
    dimension = int(np.prod(dims))
    operators = tuple(np.asarray(value, dtype=np.complex128) for value in observables)
    values = np.asarray(measured_values, dtype=np.float64)
    tolerances = np.asarray(deltas, dtype=np.float64)
    if not operators or len(operators) != len(values) or len(values) != len(tolerances) \
            or any(operator.shape != (dimension, dimension) for operator in operators) \
            or np.any(tolerances < 0.0):
        raise ValueError("invalid incomplete-measurement feasibility contract")
    if dimension > 16:
        return {"status": CertificateStatus.UNAVAILABLE.value,
                "reason": "ppt_mixture_sdp_dimension_guard",
                "separability_certified": False}
    try:
        import cvxpy as cp
    except ImportError:
        return {"status": CertificateStatus.UNAVAILABLE.value,
                "reason": "cvxpy_not_installed", "separability_certified": False}
    universe = frozenset(range(len(dims)))

    def canonical(cut: Sequence[int]) -> Tuple[int, ...]:
        selected = frozenset(int(value) for value in cut)
        complement = universe - selected
        return min(tuple(sorted(selected)), tuple(sorted(complement)),
                   key=lambda value: (len(value), value))

    cuts = tuple(dict.fromkeys(canonical(cut) for cut in bipartitions))
    expected = {canonical(cut) for size in range(1, len(dims))
                for cut in itertools.combinations(range(len(dims)), size)}
    if not cuts or any(not cut for cut in cuts):
        raise ValueError("at least one valid bipartition is required")
    state = cp.Variable((dimension, dimension), hermitian=True)
    components = [cp.Variable((dimension, dimension), hermitian=True) for _ in cuts]
    constraints = [state >> 0, cp.trace(state) == 1.0, sum(components) == state]
    for component, cut in zip(components, cuts):
        constraints.extend((component >> 0, _cvx_partial_transpose(component, cut, dims) >> 0))
    for operator, value, delta in zip(operators, values, tolerances):
        prediction = cp.real(cp.trace(state @ operator))
        constraints.extend((prediction - float(value) <= float(delta),
                            float(value) - prediction <= float(delta)))
    problem = cp.Problem(cp.Minimize(0.0), constraints)
    try:
        kwargs = {"verbose": False}
        if solver is not None:
            kwargs["solver"] = solver
        objective = problem.solve(**kwargs)
    except Exception as error:
        return {"status": CertificateStatus.NUMERICAL_FAILURE.value,
                "reason": type(error).__name__, "separability_certified": False}
    complete = set(cuts) == expected
    if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        status = CertificateStatus.PPT_MIXTURE_COMPATIBLE
    elif problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE) and complete:
        status = CertificateStatus.CERTIFIED_GME
    elif problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        status = CertificateStatus.INCONCLUSIVE
    else:
        status = CertificateStatus.NUMERICAL_FAILURE
    return {"status": status.value, "solver_status": problem.status,
            "objective": None if objective is None else float(objective),
            "cuts_complete": complete, "measurement_constraints": len(operators),
            "separability_certified": False}
