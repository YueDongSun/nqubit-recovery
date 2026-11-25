"""Scaled, range-checked arithmetic for the opt-in balanced gradient composer."""

from __future__ import annotations

import math

import torch


Gradient_tuple = tuple[torch.Tensor | None, ...]
minimum_normal_float64 = torch.finfo(torch.float64).tiny


def _has_subnormal(value: torch.Tensor) -> bool:
    """Subnormal normalized intermediates can lose precision before rescaling."""
    magnitude = value.abs()
    return bool(((magnitude > 0) & (magnitude < minimum_normal_float64)).any())


def _scaled_norm_parts(gradients: Gradient_tuple) -> tuple[float, float, Gradient_tuple]:
    """Return max magnitude, norm relative to that magnitude, and scaled values.

    Squaring is performed only after scaling, so a nonzero vector cannot be
    misclassified as having zero norm. Unrepresentable intermediate directions
    fail explicitly instead of changing the balanced update silently.
    """
    for gradient in gradients:
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("balanced task gradient is non-finite")
    scale = max((float(gradient.detach().abs().amax()) for gradient in gradients
                 if gradient is not None and gradient.numel()), default=0.0)
    if scale == 0.0:
        return 0.0, 0.0, gradients
    scaled = tuple(None if gradient is None else gradient / scale for gradient in gradients)
    for original, value in zip(gradients, scaled, strict=True):
        if original is not None and (
                not bool(torch.isfinite(value).all())
                or bool(((original != 0) & (value == 0)).any()) or _has_subnormal(value)):
            raise FloatingPointError("balanced gradient direction exceeds the supported numeric range")
    norm = math.sqrt(math.fsum(float(value.detach().square().sum())
                              for value in scaled if value is not None))
    if not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError("balanced scaled norm is invalid")
    return scale, norm, scaled


def _match_gain_norm(action: Gradient_tuple, gain: Gradient_tuple, weight: float) -> Gradient_tuple:
    """Cancel the strictly positive weight algebraically when both tasks exist."""
    action_scale, action_norm, _ = _scaled_norm_parts(action)
    gain_scale, gain_norm, scaled_gain = _scaled_norm_parts(gain)
    if action_scale and gain_scale:
        # ||a|| / ||weight*g|| * weight*g == ||a|| / ||g|| * g.
        # Cancelling a positive weight before arithmetic avoids weight-induced
        # overflow/underflow; no epsilon, clamp or alternate matching is used.
        ratio = action_norm / gain_norm
        matched = []
        for value in scaled_gain:
            if value is None:
                matched.append(None)
                continue
            relative = value * ratio
            if bool(((value != 0) & (relative == 0)).any()) or _has_subnormal(relative):
                raise FloatingPointError("balanced norm matching underflowed its direction")
            actual = relative * action_scale
            if (not bool(torch.isfinite(actual).all())
                    or bool(((relative != 0) & (actual == 0)).any())):
                raise FloatingPointError("balanced norm matching exceeds the supported numeric range")
            matched.append(actual)
        return tuple(matched)
    weighted = tuple(None if value is None else value * weight for value in gain)
    for original, value in zip(gain, weighted, strict=True):
        if original is not None and (
                not bool(torch.isfinite(value).all())
                or bool(((original != 0) & (value == 0)).any())):
            raise FloatingPointError("balanced zero-task weighted gradient exceeds the supported numeric range")
    return weighted


def compose_balanced_shared_gradients(
    parameters: tuple[torch.nn.Parameter, ...],
    action_gradients: Gradient_tuple,
    gain_gradients: Gradient_tuple,
    *,
    shared_indices: tuple[int, ...],
    gain_loss_weight: float,
) -> tuple[Gradient_tuple, bool]:
    """Match global task norms, project shared conflict, then restore all dtypes."""
    if not math.isfinite(gain_loss_weight) or gain_loss_weight <= 0:
        raise ValueError("balanced gain weight must be finite and positive")
    if not (len(parameters) == len(action_gradients) == len(gain_gradients)):
        raise ValueError("balanced parameter and task gradient layouts differ")
    if len(set(shared_indices)) != len(shared_indices) or any(
            type(index) is not int or not 0 <= index < len(parameters) for index in shared_indices):
        raise ValueError("balanced shared parameter indices are invalid")
    for parameter, action, gain in zip(parameters, action_gradients, gain_gradients, strict=True):
        for value in (action, gain):
            if value is not None and (not value.is_floating_point() or value.shape != parameter.shape
                                      or value.device != parameter.device):
                raise ValueError("balanced task gradient shape, dtype or device is invalid")
    action = tuple(None if value is None else value.to(dtype=torch.float64) for value in action_gradients)
    gain = tuple(None if value is None else value.to(dtype=torch.float64) for value in gain_gradients)
    matched_gain = _match_gain_norm(action, gain, float(gain_loss_weight))
    action_scale, action_norm, shared_action = _scaled_norm_parts(tuple(action[index] for index in shared_indices))
    gain_scale, _, shared_gain = _scaled_norm_parts(tuple(matched_gain[index] for index in shared_indices))
    products = []
    for left, right in zip(shared_action, shared_gain, strict=True):
        if left is not None and right is not None:
            product = left * right
            if bool(((left != 0) & (right != 0) & (product == 0)).any()):
                raise FloatingPointError("balanced shared inner product underflowed")
            products.append(float(product.detach().sum()))
    inner_product = math.fsum(products) if action_scale and gain_scale else 0.0
    projection_applied = inner_product < 0
    if projection_applied:
        coefficient = inner_product / (action_norm * action_norm)
        if not math.isfinite(coefficient) or abs(coefficient) < minimum_normal_float64:
            raise FloatingPointError("balanced projection coefficient exceeds the supported numeric range")
        projected_gain = list(matched_gain)
        for index, left, right in zip(shared_indices, shared_action, shared_gain, strict=True):
            if left is None and right is None:
                continue
            relative = torch.zeros_like(parameters[index], dtype=torch.float64)
            if right is not None:
                relative = relative + right
            if left is not None:
                correction = coefficient * left
                if bool(((left != 0) & (correction == 0)).any()) or _has_subnormal(correction):
                    raise FloatingPointError("balanced projection correction exceeds the supported numeric range")
                relative = relative - correction
            if _has_subnormal(relative):
                raise FloatingPointError("balanced projected direction exceeds the supported numeric range")
            projected = relative * gain_scale
            if not bool(torch.isfinite(projected).all()) or bool(((relative != 0) & (projected == 0)).any()):
                raise FloatingPointError("balanced shared projection exceeds the supported numeric range")
            projected_gain[index] = projected
        matched_gain = tuple(projected_gain)
    combined = []
    for parameter, left, right in zip(parameters, action, matched_gain, strict=True):
        if left is None and right is None:
            combined.append(None)
            continue
        value = torch.zeros_like(parameter, dtype=torch.float64)
        if left is not None:
            value = value + left
        if right is not None:
            value = value + right
        restored = value.to(dtype=parameter.dtype, device=parameter.device)
        if not bool(torch.isfinite(restored).all()):
            raise FloatingPointError("balanced Track B gradient became non-finite after dtype restoration")
        if bool(((value != 0) & (restored == 0)).any()):
            raise FloatingPointError("balanced Track B gradient underflow after dtype restoration")
        combined.append(restored)
    return tuple(combined), projection_applied
