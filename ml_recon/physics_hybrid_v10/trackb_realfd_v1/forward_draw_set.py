"""Sealed same-forward draw sets for Track B CSM construction."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..forward_instance import CSMForwardInstanceContract
from .contracts import canonical_json_bytes, canonical_sha256, require_sha256
from .source_artifacts import (
    FORWARD_SOURCE_ARTIFACT_NAMES,
    TrackBActionfulSourceArtifactBundleV1,
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    # Keep this byte ABI identical to ``real_csm.array_identity_sha256`` so a
    # sealed draw set can be checked against the arrays actually consumed.
    digest.update(b"TrackBArrayIdentityV1\x00")
    digest.update(str(tuple(int(item) for item in contiguous.shape)).encode("ascii"))
    digest.update(b"\x00float64-le\x00")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _readonly_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite rank-2 array")
    copied = np.array(array, dtype="<f8", copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _stable_value_identity(value: object, *, depth: int = 0) -> object:
    """Return a serializable identity for closures and referenced globals.

    Callable source alone misses mutable module globals.  The producer needs a
    bounded structural identity rather than a process-specific object address,
    so arrays, simple containers, module source files, and public object state
    are represented explicitly.  Unknown leaf objects retain their type and
    repr as a last-resort diagnostic, never as the sole identity for supported
    numeric/container inputs.
    """
    if depth > 3:
        return {"kind": "depth_limited", "type": f"{type(value).__module__}.{type(value).__qualname__}"}
    if isinstance(value, np.ndarray):
        return {"kind": "ndarray", "sha256": _array_sha256(value)}
    if inspect.iscode(value):
        return {
            "kind": "code",
            "co_code": value.co_code.hex(),
            "co_names": list(value.co_names),
            "co_consts": [_stable_value_identity(item, depth=depth + 1) for item in value.co_consts],
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "type": type(value).__name__, "value": value}
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [_stable_value_identity(item, depth=depth + 1) for item in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {
                str(key): _stable_value_identity(item, depth=depth + 1)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            },
        }
    if inspect.ismodule(value):
        module_path = getattr(value, "__file__", None)
        source_sha = None
        if isinstance(module_path, str) and Path(module_path).is_file():
            try:
                source_sha = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
            except OSError:
                source_sha = None
        return {
            "kind": "module", "name": getattr(value, "__name__", type(value).__name__),
            "file_sha256": source_sha,
            "version": getattr(value, "__version__", None),
        }
    if callable(value):
        code = getattr(value, "__code__", None)
        return {
            "kind": "callable",
            "module": getattr(value, "__module__", type(value).__module__),
            "qualname": getattr(value, "__qualname__", type(value).__qualname__),
            "code_sha256": None if code is None else hashlib.sha256(code.co_code).hexdigest(),
        }
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            "kind": "object",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "attributes": {
                str(key): _stable_value_identity(item, depth=depth + 1)
                for key, item in sorted(attributes.items(), key=lambda item: str(item[0]))
                if not str(key).startswith("__")
            },
        }
    return {
        "kind": "opaque",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _stable_callable_sha(callable_object: object) -> str:
    """Bind code, defaults, closure values, and callable identity."""
    if not callable(callable_object):
        raise TypeError("forward producer requires a callable")
    try:
        source = inspect.getsource(callable_object)
    except (OSError, TypeError):
        source = ""
    code = getattr(callable_object, "__code__", None)
    code_payload = None if code is None else {
        "co_code": code.co_code.hex(),
        "co_consts": [_stable_value_identity(value) for value in code.co_consts],
        "co_names": list(code.co_names),
    }
    closure = getattr(callable_object, "__closure__", None) or ()
    closure_values = []
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            value = "<empty-cell>"
        if isinstance(value, np.ndarray):
            closure_values.append(_stable_value_identity(value))
        else:
            closure_values.append(_stable_value_identity(value))
    global_values = []
    namespace = getattr(callable_object, "__globals__", {})
    if code is not None and isinstance(namespace, dict):
        for name in sorted(set(code.co_names)):
            if name in namespace:
                global_values.append({"name": name, "value": _stable_value_identity(namespace[name])})
    return canonical_sha256({
        "schema": "TrackBForwardCallableIdentityV1",
        "module": getattr(callable_object, "__module__", type(callable_object).__module__),
        "qualname": getattr(callable_object, "__qualname__", type(callable_object).__qualname__),
        "source": source,
        "code": code_payload,
        "defaults": _stable_value_identity(getattr(callable_object, "__defaults__", None)),
        "kwdefaults": _stable_value_identity(getattr(callable_object, "__kwdefaults__", None)),
        "closure": closure_values,
        "referenced_globals": global_values,
    })


forward_callable_identity_sha = _stable_callable_sha


def _callable_source_file_sha(callable_object: object) -> str:
    source_file = inspect.getsourcefile(callable_object)
    if source_file is None:
        raise ValueError("actionful Track B forward producer requires a file-backed callable")
    path = Path(source_file).resolve()
    if not path.is_file():
        raise ValueError("actionful Track B forward callable source file is missing")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forward_source_payloads(
    *,
    operation_descriptor: object,
    rf_descriptor: object,
    session_context: object,
    exposure_context: object,
    draw_ids: Sequence[str],
    draw_parameters: Sequence[object],
    base_point: np.ndarray,
    nuisance_point: np.ndarray,
) -> dict[str, bytes]:
    if any(isinstance(value, str) and len(value) == 64 for value in (
        operation_descriptor, rf_descriptor, session_context, exposure_context,
    )):
        raise ValueError("actionful Track B forward inputs cannot be pre-hashed strings")
    return {
        "operation_descriptor": canonical_json_bytes(operation_descriptor),
        "rf_descriptor": canonical_json_bytes(rf_descriptor),
        "session_context": canonical_json_bytes(session_context),
        "exposure_context": canonical_json_bytes(exposure_context),
        "draw_parameters": canonical_json_bytes({
            "schema": "TrackBDrawParametersV1",
            "draw_ids": list(draw_ids),
            "draw_parameters": list(draw_parameters),
        }),
        "base_point": canonical_json_bytes([float(value) for value in base_point]),
        "nuisance_point": canonical_json_bytes([float(value) for value in nuisance_point]),
    }


def _validate_forward_source_bundle(
    *,
    source_artifact_bundle: TrackBActionfulSourceArtifactBundleV1,
    forward_callable: Callable[[np.ndarray, np.ndarray, Any], Sequence[float]],
    covariance_callable: Callable[[np.ndarray, Any], Sequence[Sequence[float]]],
    operation_descriptor: object,
    rf_descriptor: object,
    session_context: object,
    exposure_context: object,
    draw_ids: Sequence[str],
    draw_parameters: Sequence[object],
    base_point: np.ndarray,
    nuisance_point: np.ndarray,
) -> str:
    if type(source_artifact_bundle) is not TrackBActionfulSourceArtifactBundleV1:
        raise TypeError("actionful Track B forward producer requires an exact source artifact bundle")
    source_artifact_bundle.assert_current(FORWARD_SOURCE_ARTIFACT_NAMES)
    payloads = _forward_source_payloads(
        operation_descriptor=operation_descriptor,
        rf_descriptor=rf_descriptor,
        session_context=session_context,
        exposure_context=exposure_context,
        draw_ids=draw_ids,
        draw_parameters=draw_parameters,
        base_point=base_point,
        nuisance_point=nuisance_point,
    )
    for name, payload in payloads.items():
        if source_artifact_bundle.read_bytes(name) != payload:
            raise ValueError("actionful Track B forward source artifact bytes do not match live input")
    if source_artifact_bundle.artifact_sha256("forward_callable_source") \
            != _callable_source_file_sha(forward_callable) \
            or source_artifact_bundle.artifact_sha256("covariance_callable_source") \
            != _callable_source_file_sha(covariance_callable):
        raise ValueError("actionful Track B forward callable source files do not match the source bundle")
    return source_artifact_bundle.forward_binding_sha256


@dataclass(frozen=True)
class BoundTrackBForwardDrawSetV1:
    """One source-bound forward instance plus every draw consumed by CSM.

    The CSM may only claim verified same-forward provenance when it receives
    this object.  A bare equal ``forward_sha`` string is intentionally
    insufficient.
    """

    forward_instance: CSMForwardInstanceContract
    draw_ids: tuple[str, ...]
    jacobian_draws: tuple[np.ndarray, ...]
    covariance_draws: tuple[np.ndarray, ...]
    nuisance_jacobian_draws: tuple[np.ndarray, ...]
    producer_status: str = "UNVERIFIED_ASSOCIATION_ONLY"
    forward_callable_sha: str | None = None
    covariance_builder_sha: str | None = None
    base_point_sha: str | None = None
    nuisance_point_sha: str | None = None
    operation_descriptor_sha: str | None = None
    rf_descriptor_sha: str | None = None
    session_sha: str | None = None
    exposure_sha: str | None = None
    finite_difference_step: float | None = None
    draw_parameter_shas: tuple[str, ...] = ()
    baseline_output_shas: tuple[str, ...] = ()
    covariance_output_shas: tuple[str, ...] = ()
    action_parameter_abi_sha: str | None = None
    nuisance_parameter_abi_sha: str | None = None
    covariance_output_recipe_sha: str | None = None
    source_artifact_bundle: TrackBActionfulSourceArtifactBundleV1 | None = None
    forward_source_binding_sha: str | None = None
    _replay_forward_callable: Callable[[np.ndarray, np.ndarray, Any], Sequence[float]] | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    _replay_covariance_callable: Callable[[np.ndarray, Any], Sequence[Sequence[float]]] | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    _replay_base_point: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _replay_nuisance_point: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _replay_draw_parameters: tuple[Any, ...] = field(default=(), init=False, repr=False, compare=False)
    _replay_baseline_outputs: tuple[np.ndarray, ...] = field(default=(), init=False, repr=False, compare=False)
    schema: str = field(default="BoundTrackBForwardDrawSetV1", init=False)

    def __post_init__(self) -> None:
        if type(self.forward_instance) is not CSMForwardInstanceContract:
            raise TypeError("Track B forward draw set requires a CSMForwardInstanceContract")
        self.forward_instance.validate()
        if self.producer_status not in {"UNVERIFIED_ASSOCIATION_ONLY", "AUTHORITATIVE_COMPUTED"}:
            raise ValueError("forward draw producer status is not registered")
        for name in (
            "forward_callable_sha", "covariance_builder_sha", "base_point_sha",
            "nuisance_point_sha", "operation_descriptor_sha", "rf_descriptor_sha",
            "session_sha", "exposure_sha", "action_parameter_abi_sha",
            "nuisance_parameter_abi_sha", "covariance_output_recipe_sha",
            "forward_source_binding_sha",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name)
        if self.producer_status == "AUTHORITATIVE_COMPUTED":
            if any(getattr(self, name) is None for name in (
                "forward_callable_sha", "covariance_builder_sha", "base_point_sha",
                "nuisance_point_sha", "operation_descriptor_sha", "rf_descriptor_sha",
                "session_sha", "exposure_sha",
            )):
                raise ValueError("authoritative forward draws require complete producer identities")
            if self.forward_callable_sha != self.forward_instance.forward_code_sha:
                raise ValueError("authoritative draw callable identity differs from forward instance")
            if self.finite_difference_step is None or not math.isfinite(float(self.finite_difference_step)) \
                    or float(self.finite_difference_step) <= 0.0:
                raise ValueError("authoritative forward draws require a finite-difference step")
            if self._replay_forward_callable is None or self._replay_covariance_callable is None \
                    or self._replay_base_point is None or self._replay_nuisance_point is None \
                    or len(self._replay_draw_parameters) != len(self.draw_ids) \
                or len(self._replay_baseline_outputs) != len(self.draw_ids):
                raise ValueError("authoritative forward draws require an attached replay producer")
            if self.source_artifact_bundle is not None:
                if type(self.source_artifact_bundle) is not TrackBActionfulSourceArtifactBundleV1 \
                        or self.forward_source_binding_sha is None \
                        or self.source_artifact_bundle.forward_binding_sha256 \
                        != self.forward_source_binding_sha:
                    raise ValueError("authoritative forward source-artifact binding is invalid")
        ids = tuple(self.draw_ids)
        if not ids or len(set(ids)) != len(ids) or any(
            type(value) is not str or not value for value in ids
        ):
            raise ValueError("forward draw IDs must be unique nonempty strings")
        jacobians = tuple(_readonly_array(value, "jacobian draw") for value in self.jacobian_draws)
        covariances = tuple(_readonly_array(value, "covariance draw") for value in self.covariance_draws)
        nuisances = tuple(
            _readonly_array(value, "nuisance Jacobian draw")
            for value in self.nuisance_jacobian_draws
        )
        if len(ids) != len(jacobians) or len(ids) != len(covariances) \
                or len(ids) != len(nuisances):
            raise ValueError("forward draw arrays and IDs must have equal length")
        if self.producer_status == "AUTHORITATIVE_COMPUTED":
            identity_fields = (
                ("draw_parameter_shas", self.draw_parameter_shas),
                ("baseline_output_shas", self.baseline_output_shas),
                ("covariance_output_shas", self.covariance_output_shas),
            )
            for name, values in identity_fields:
                if len(values) != len(ids) or any(
                    type(value) is not str or len(value) != 64 for value in values
                ):
                    raise ValueError(f"authoritative forward draws require one {name} per draw")
                for value in values:
                    require_sha256(value, name)
        if any(value.shape != jacobians[0].shape for value in jacobians):
            raise ValueError("Jacobian draws must share one shape")
        if any(value.shape != covariances[0].shape for value in covariances):
            raise ValueError("covariance draws must share one shape")
        if any(value.shape != nuisances[0].shape for value in nuisances):
            raise ValueError("nuisance Jacobian draws must share one shape")
        observation_count = jacobians[0].shape[0]
        action_dimension = jacobians[0].shape[1]
        if any(value.shape != (observation_count, observation_count) for value in covariances):
            raise ValueError("covariance draws must be square over the Jacobian rows")
        if any(value.shape[0] != observation_count for value in nuisances):
            raise ValueError("nuisance Jacobian draws must align to Jacobian rows")
        if any(not np.allclose(value, value.T, atol=1.0e-10, rtol=0.0)
               for value in covariances):
            raise ValueError("covariance draws must be symmetric")
        if action_dimension < 1:
            raise ValueError("forward draw action dimension must be positive")
        object.__setattr__(self, "draw_ids", ids)
        object.__setattr__(self, "jacobian_draws", jacobians)
        object.__setattr__(self, "covariance_draws", covariances)
        object.__setattr__(self, "nuisance_jacobian_draws", nuisances)
        object.__setattr__(self, "draw_parameter_shas", tuple(self.draw_parameter_shas))
        object.__setattr__(self, "baseline_output_shas", tuple(self.baseline_output_shas))
        object.__setattr__(self, "covariance_output_shas", tuple(self.covariance_output_shas))

    @classmethod
    def from_draws(
        cls,
        *,
        forward_instance: CSMForwardInstanceContract,
        draw_ids: Sequence[str],
        jacobian_draws: Sequence[np.ndarray],
        covariance_draws: Sequence[np.ndarray],
        nuisance_jacobian_draws: Sequence[np.ndarray],
    ) -> "BoundTrackBForwardDrawSetV1":
        draw_set = cls(
            forward_instance=forward_instance,
            draw_ids=tuple(draw_ids),
            jacobian_draws=tuple(jacobian_draws),
            covariance_draws=tuple(covariance_draws),
            nuisance_jacobian_draws=tuple(nuisance_jacobian_draws),
        )
        return draw_set

    @classmethod
    def build_bound_trackb_forward_draw_set(
        cls,
        *,
        forward_instance: CSMForwardInstanceContract,
        forward_callable: Callable[[np.ndarray, np.ndarray, Any], Sequence[float]],
        covariance_callable: Callable[[np.ndarray, Any], Sequence[Sequence[float]]],
        base_point: Sequence[float],
        nuisance_point: Sequence[float],
        draw_ids: Sequence[str],
        draw_parameters: Sequence[Any],
        operation_descriptor: object,
        rf_descriptor: object,
        session_context: object,
        exposure_context: object,
        source_artifact_bundle: TrackBActionfulSourceArtifactBundleV1 | None = None,
        finite_difference_step: float = 1.0e-6,
    ) -> "BoundTrackBForwardDrawSetV1":
        """Authoritatively compute all draws from one bound forward producer."""
        if len(draw_ids) != len(draw_parameters) or not draw_ids:
            raise ValueError("forward producer draw IDs and parameters must align and be nonempty")
        step = float(finite_difference_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("forward producer finite-difference step must be positive")
        action = np.asarray(tuple(base_point), dtype=np.float64)
        nuisance = np.asarray(tuple(nuisance_point), dtype=np.float64)
        if action.ndim != 1 or nuisance.ndim != 1 or not np.all(np.isfinite(action)) \
                or not np.all(np.isfinite(nuisance)):
            raise ValueError("forward producer base/nuisance points must be finite vectors")
        forward_source_binding_sha = None
        if source_artifact_bundle is not None:
            forward_source_binding_sha = _validate_forward_source_bundle(
                source_artifact_bundle=source_artifact_bundle,
                forward_callable=forward_callable,
                covariance_callable=covariance_callable,
                operation_descriptor=operation_descriptor,
                rf_descriptor=rf_descriptor,
                session_context=session_context,
                exposure_context=exposure_context,
                draw_ids=draw_ids,
                draw_parameters=draw_parameters,
                base_point=action,
                nuisance_point=nuisance,
            )
        forward_code_sha = _stable_callable_sha(forward_callable)
        covariance_code_sha = _stable_callable_sha(covariance_callable)
        if forward_instance.forward_code_sha != forward_code_sha:
            raise ValueError("forward instance code identity does not match the authoritative callable")
        operation_sha = operation_descriptor if isinstance(operation_descriptor, str) and len(operation_descriptor) == 64 \
            else canonical_sha256(operation_descriptor)
        rf_sha = rf_descriptor if isinstance(rf_descriptor, str) and len(rf_descriptor) == 64 \
            else canonical_sha256(rf_descriptor)
        require_sha256(operation_sha, "operation_descriptor_sha")
        require_sha256(rf_sha, "rf_descriptor_sha")
        session_sha = session_context if isinstance(session_context, str) and len(session_context) == 64 \
            else canonical_sha256(session_context)
        exposure_sha = exposure_context if isinstance(exposure_context, str) and len(exposure_context) == 64 \
            else canonical_sha256(exposure_context)
        require_sha256(session_sha, "session_sha")
        require_sha256(exposure_sha, "exposure_sha")
        jacobians = []
        covariances = []
        nuisances = []
        baseline_shas = []
        covariance_shas = []
        draw_parameter_shas = []

        def parameter_sha(index: int, value: Any) -> str:
            try:
                payload: object = {"schema": "TrackBDrawParameterPayloadV1", "index": index, "value": value}
                return canonical_sha256(payload)
            except (TypeError, ValueError):
                return canonical_sha256({
                    "schema": "TrackBDrawParameterPayloadReprFallbackV1",
                    "index": index,
                    "value_repr": repr(value),
                })

        for draw_parameter in draw_parameters:
            baseline = np.asarray(forward_callable(action, nuisance, draw_parameter), dtype=np.float64).reshape(-1)
            if baseline.size == 0 or not np.all(np.isfinite(baseline)):
                raise ValueError("forward producer returned an invalid baseline")
            baseline_shas.append(_array_sha256(baseline.reshape(1, -1)))
            draw_parameter_shas.append(parameter_sha(len(draw_parameter_shas), draw_parameter))
            jacobian = np.empty((baseline.size, action.size), dtype=np.float64)
            for index in range(action.size):
                plus = action.copy(); plus[index] += step
                minus = action.copy(); minus[index] -= step
                jacobian[:, index] = (
                    np.asarray(forward_callable(plus, nuisance, draw_parameter), dtype=np.float64).reshape(-1)
                    - np.asarray(forward_callable(minus, nuisance, draw_parameter), dtype=np.float64).reshape(-1)
                ) / (2.0 * step)
            nuisance_jacobian = np.empty((baseline.size, nuisance.size), dtype=np.float64)
            for index in range(nuisance.size):
                plus = nuisance.copy(); plus[index] += step
                minus = nuisance.copy(); minus[index] -= step
                nuisance_jacobian[:, index] = (
                    np.asarray(forward_callable(action, plus, draw_parameter), dtype=np.float64).reshape(-1)
                    - np.asarray(forward_callable(action, minus, draw_parameter), dtype=np.float64).reshape(-1)
                ) / (2.0 * step)
            covariance = np.asarray(covariance_callable(baseline, draw_parameter), dtype=np.float64)
            if covariance.ndim != 2 or covariance.shape != (baseline.size, baseline.size) \
                    or not np.all(np.isfinite(covariance)):
                raise ValueError("forward producer covariance output is invalid")
            covariance_shas.append(_array_sha256(covariance))
            jacobians.append(jacobian)
            covariances.append(covariance)
            nuisances.append(nuisance_jacobian)
        draw_set = cls(
            forward_instance=forward_instance,
            draw_ids=tuple(draw_ids),
            jacobian_draws=tuple(jacobians),
            covariance_draws=tuple(covariances),
            nuisance_jacobian_draws=tuple(nuisances),
            producer_status="UNVERIFIED_ASSOCIATION_ONLY",
            forward_callable_sha=forward_code_sha,
            covariance_builder_sha=covariance_code_sha,
            base_point_sha=_array_sha256(action),
            nuisance_point_sha=_array_sha256(nuisance),
            operation_descriptor_sha=operation_sha,
            rf_descriptor_sha=rf_sha,
            session_sha=session_sha,
            exposure_sha=exposure_sha,
            finite_difference_step=step,
            draw_parameter_shas=tuple(draw_parameter_shas),
            baseline_output_shas=tuple(baseline_shas),
            covariance_output_shas=tuple(covariance_shas),
            action_parameter_abi_sha=canonical_sha256({
                "schema": "TrackBForwardActionParameterABIV1",
                "dimension": int(action.size),
                "ordering": "base_point positional vector",
            }),
            nuisance_parameter_abi_sha=canonical_sha256({
                "schema": "TrackBForwardNuisanceParameterABIV1",
                "dimension": int(nuisance.size),
                "ordering": "nuisance_point positional vector",
            }),
            covariance_output_recipe_sha=canonical_sha256({
                "schema": "TrackBCovarianceOutputRecipeV1",
                "builder_sha": covariance_code_sha,
                "shape": ["observation_count", "observation_count"],
                "input": "baseline_output",
            }),
            source_artifact_bundle=source_artifact_bundle,
            forward_source_binding_sha=forward_source_binding_sha,
        )
        object.__setattr__(draw_set, "_replay_forward_callable", forward_callable)
        object.__setattr__(draw_set, "_replay_covariance_callable", covariance_callable)
        object.__setattr__(draw_set, "_replay_base_point", np.array(action, dtype="<f8", copy=True))
        object.__setattr__(draw_set, "_replay_nuisance_point", np.array(nuisance, dtype="<f8", copy=True))
        object.__setattr__(draw_set, "_replay_draw_parameters", tuple(draw_parameters))
        object.__setattr__(draw_set, "_replay_baseline_outputs", tuple(
            np.array(value, dtype="<f8", copy=True) for value in (
                np.asarray(forward_callable(action, nuisance, draw_parameter), dtype=np.float64).reshape(-1)
                for draw_parameter in draw_parameters
            )
        ))
        for value in draw_set._replay_baseline_outputs:
            value.setflags(write=False)
        object.__setattr__(draw_set, "producer_status", "AUTHORITATIVE_COMPUTED")
        draw_set.assert_authoritative()
        return draw_set

    @property
    def forward_instance_sha(self) -> str:
        return self.forward_instance.sha256()

    @property
    def forward_sha(self) -> str:
        return self.forward_instance.forward_code_sha

    @property
    def jacobian_draw_shas(self) -> tuple[str, ...]:
        return tuple(_array_sha256(value) for value in self.jacobian_draws)

    @property
    def covariance_draw_shas(self) -> tuple[str, ...]:
        return tuple(_array_sha256(value) for value in self.covariance_draws)

    @property
    def nuisance_jacobian_draw_shas(self) -> tuple[str, ...]:
        return tuple(_array_sha256(value) for value in self.nuisance_jacobian_draws)

    @property
    def baseline_outputs(self) -> tuple[np.ndarray, ...]:
        """Read-only baseline observations replayed by the authoritative producer."""
        self.assert_authoritative()
        return self._replay_baseline_outputs

    def assert_authoritative(self) -> None:
        """Recompute the producer path before trusting any stored draw arrays.

        The replay attachment is intentionally ``init=False``.  A
        ``dataclasses.replace`` copy therefore loses it and cannot retain
        authoritative status; array replacement is additionally detected by
        this exact replay comparison.
        """
        if self.producer_status != "AUTHORITATIVE_COMPUTED":
            raise ValueError("forward draw set is not an authoritative computed producer result")
        forward_callable = self._replay_forward_callable
        covariance_callable = self._replay_covariance_callable
        action = self._replay_base_point
        nuisance = self._replay_nuisance_point
        if forward_callable is None or covariance_callable is None or action is None or nuisance is None:
            raise ValueError("authoritative forward draw set has no replay producer")
        if _stable_callable_sha(forward_callable) != self.forward_callable_sha \
                or _stable_callable_sha(covariance_callable) != self.covariance_builder_sha:
            raise ValueError("authoritative forward replay callable identity changed")
        if _array_sha256(action) != self.base_point_sha or _array_sha256(nuisance) != self.nuisance_point_sha:
            raise ValueError("authoritative forward replay point identity changed")
        if len(self._replay_draw_parameters) != len(self.draw_ids):
            raise ValueError("authoritative forward replay draw parameters are incomplete")
        step = float(self.finite_difference_step or 0.0)
        for index, draw_parameter in enumerate(self._replay_draw_parameters):
            baseline = np.asarray(forward_callable(action, nuisance, draw_parameter), dtype=np.float64).reshape(-1)
            if _array_sha256(baseline.reshape(1, -1)) != self.baseline_output_shas[index]:
                raise ValueError("authoritative forward replay baseline output changed")
            if not np.array_equal(baseline, self._replay_baseline_outputs[index]):
                raise ValueError("authoritative forward replay baseline bytes changed")
            jacobian = np.empty_like(self.jacobian_draws[index])
            for coordinate in range(action.size):
                plus = action.copy(); plus[coordinate] += step
                minus = action.copy(); minus[coordinate] -= step
                jacobian[:, coordinate] = (
                    np.asarray(forward_callable(plus, nuisance, draw_parameter), dtype=np.float64).reshape(-1)
                    - np.asarray(forward_callable(minus, nuisance, draw_parameter), dtype=np.float64).reshape(-1)
                ) / (2.0 * step)
            nuisance_jacobian = np.empty_like(self.nuisance_jacobian_draws[index])
            for coordinate in range(nuisance.size):
                plus = nuisance.copy(); plus[coordinate] += step
                minus = nuisance.copy(); minus[coordinate] -= step
                nuisance_jacobian[:, coordinate] = (
                    np.asarray(forward_callable(action, plus, draw_parameter), dtype=np.float64).reshape(-1)
                    - np.asarray(forward_callable(action, minus, draw_parameter), dtype=np.float64).reshape(-1)
                ) / (2.0 * step)
            covariance = np.asarray(covariance_callable(baseline, draw_parameter), dtype=np.float64)
            if not np.array_equal(jacobian, self.jacobian_draws[index]) \
                    or not np.array_equal(nuisance_jacobian, self.nuisance_jacobian_draws[index]) \
                    or not np.array_equal(covariance, self.covariance_draws[index]):
                raise ValueError("authoritative forward replay arrays differ from the sealed producer output")

    def assert_actionful_source_bound(self) -> None:
        """Require rehashable producer inputs before the FD bridge can act."""
        self.assert_authoritative()
        if self.source_artifact_bundle is None or self.forward_source_binding_sha is None \
                or self._replay_forward_callable is None \
                or self._replay_covariance_callable is None \
                or self._replay_base_point is None \
                or self._replay_nuisance_point is None:
            raise ValueError("actionful Track B forward draw set lacks file-backed source artifacts")
        binding_sha = _validate_forward_source_bundle(
            source_artifact_bundle=self.source_artifact_bundle,
            forward_callable=self._replay_forward_callable,
            covariance_callable=self._replay_covariance_callable,
            operation_descriptor=self.source_artifact_bundle.read_canonical_json_object("operation_descriptor"),
            rf_descriptor=self.source_artifact_bundle.read_canonical_json_object("rf_descriptor"),
            session_context=self.source_artifact_bundle.read_canonical_json_object("session_context"),
            exposure_context=self.source_artifact_bundle.read_canonical_json_object("exposure_context"),
            draw_ids=self.draw_ids,
            draw_parameters=self._replay_draw_parameters,
            base_point=self._replay_base_point,
            nuisance_point=self._replay_nuisance_point,
        )
        if binding_sha != self.forward_source_binding_sha:
            raise ValueError("actionful Track B forward source binding changed after admission")

    def validate(self) -> None:
        self.forward_instance.validate()
        if not self.draw_ids:
            raise ValueError("forward draw set cannot be empty")
        if len(self.draw_ids) != len(self.jacobian_draws) \
                or len(self.draw_ids) != len(self.covariance_draws) \
                or len(self.draw_ids) != len(self.nuisance_jacobian_draws):
            raise ValueError("forward draw set arrays are not aligned")
        for name in ("forward_instance_sha", "forward_sha"):
            require_sha256(getattr(self, name), name)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "forward_instance": self.forward_instance.to_dict(),
            "forward_instance_sha": self.forward_instance_sha,
            "forward_sha": self.forward_sha,
            "draw_ids": list(self.draw_ids),
            "jacobian_draw_shas": list(self.jacobian_draw_shas),
            "covariance_draw_shas": list(self.covariance_draw_shas),
            "nuisance_jacobian_draw_shas": list(self.nuisance_jacobian_draw_shas),
            "producer_status": self.producer_status,
            "forward_callable_sha": self.forward_callable_sha,
            "covariance_builder_sha": self.covariance_builder_sha,
            "base_point_sha": self.base_point_sha,
            "nuisance_point_sha": self.nuisance_point_sha,
            "operation_descriptor_sha": self.operation_descriptor_sha,
            "rf_descriptor_sha": self.rf_descriptor_sha,
            "session_sha": self.session_sha,
            "exposure_sha": self.exposure_sha,
            "finite_difference_step": self.finite_difference_step,
            "draw_parameter_shas": list(self.draw_parameter_shas),
            "baseline_output_shas": list(self.baseline_output_shas),
            "covariance_output_shas": list(self.covariance_output_shas),
            "action_parameter_abi_sha": self.action_parameter_abi_sha,
            "nuisance_parameter_abi_sha": self.nuisance_parameter_abi_sha,
            "covariance_output_recipe_sha": self.covariance_output_recipe_sha,
            "forward_source_binding_sha": self.forward_source_binding_sha,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def build_bound_trackb_forward_draw_set(**kwargs: Any) -> BoundTrackBForwardDrawSetV1:
    """Public authoritative producer for a computed Track B draw set."""
    return BoundTrackBForwardDrawSetV1.build_bound_trackb_forward_draw_set(**kwargs)


__all__ = [
    "BoundTrackBForwardDrawSetV1", "build_bound_trackb_forward_draw_set",
    "forward_callable_identity_sha",
]
