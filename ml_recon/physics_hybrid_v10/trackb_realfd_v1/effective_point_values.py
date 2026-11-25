"""Typed effective point values and native per-operation moment consistency.

The PSD is an effective normalized state, not identified microscopic truth.
The four count cells are joint signal/sticky-idler outcomes, not four SPAD
channels or W-state modes. Their conditional sampling
covariance is neither cross-operation covariance nor state-posterior error.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from .detected_count_moments import nested_four_cell_moments
from .native_read_trials import native_read_trials


point_values_schema = 'TrackBEffectivePointValuesV1'
state_round_trip_tolerance = 1e-5
moment_absolute_tolerance = 1e-8
moment_relative_tolerance = 1e-12
probability_absolute_tolerance = 1e-15
maximum_exact_read_integer = 2**53
probability_names = ('signal_probability', 'coin_given_signal_probability', 'idler_given_no_signal_probability')
point_value_fields = frozenset({'schema', 'native_read_count', 'prepared_prior_available',
    'effective_psd_real', 'effective_psd_imag', 'prepared_prior_real', 'prepared_prior_imag',
    *probability_names, 'four_cell_probability', 'four_cell_mean', 'four_cell_covariance',
    'four_cell_concentration', 'effective_offsets', 'effective_contrasts', 'effective_family_correction',
    'global_multi_mean', 'global_multi_dispersion', 'global_pair_probability'})


def numerical_array(value: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
    """Reject coercion of booleans, strings and malformed nested JSON arrays."""
    def check(entry, dimensions):
        if dimensions:
            if type(entry) is not list or len(entry) != dimensions[0]:
                raise ValueError(f'{name} has an invalid shape')
            for part in entry:
                check(part, dimensions[1:])
        elif type(entry) not in (int, float) or not math.isfinite(entry):
            raise ValueError(f'{name} must contain finite numbers without coercion')
    check(value, shape)
    return torch.tensor(value, dtype=torch.float64)


def _validate_psd(values, prefix, mode_count):
    shape = (mode_count, mode_count)
    state = torch.complex(numerical_array(values[prefix + '_real'], shape, prefix + '_real'),
        numerical_array(values[prefix + '_imag'], shape, prefix + '_imag'))
    if not torch.allclose(state, state.mH, rtol=0, atol=state_round_trip_tolerance) \
            or abs(complex(state.trace()) - 1) > state_round_trip_tolerance \
            or bool((torch.linalg.eigvalsh(state) < -state_round_trip_tolerance).any()):
        raise ValueError(f'{prefix} must be a normalized Hermitian PSD point value')


def validate_effective_point_values(values: Mapping[str, Any], *, mode_count: int, operation_count: int) -> None:
    if type(values) is not dict or set(values) != point_value_fields or values['schema'] != point_values_schema:
        raise ValueError('Effective point value fields or schema differ')
    if type(mode_count) is not int or mode_count not in (3, 4) \
            or type(operation_count) is not int or operation_count <= 0:
        raise ValueError('Effective point values require a registered real system and operation count')
    read = values['native_read_count']
    if type(read) is not int or not 0 <= read <= maximum_exact_read_integer \
            or type(values['prepared_prior_available']) is not bool:
        raise ValueError('Native READ or prepared-prior availability has an invalid type/domain')
    _validate_psd(values, 'effective_psd', mode_count)
    _validate_psd(values, 'prepared_prior', mode_count)
    probabilities = {name: numerical_array(values[name], (operation_count,), name) for name in probability_names}
    if any(bool(((value <= 0) | (value >= 1)).any()) for value in probabilities.values()):
        raise ValueError('Effective conditional probabilities must remain strictly inside native support')
    concentration = numerical_array(values['four_cell_concentration'], (), 'four_cell_concentration')
    if float(concentration) <= 0:
        raise ValueError('Native shared concentration must be positive')
    signal, coin, idler = (probabilities[name] for name in probability_names)
    probability = torch.stack(((1-signal)*(1-idler), signal*(1-coin), (1-signal)*idler, signal*coin), -1)
    actual_probability = numerical_array(values['four_cell_probability'], (operation_count, 4), 'four_cell_probability')
    if not torch.allclose(probability, actual_probability, rtol=moment_relative_tolerance, atol=probability_absolute_tolerance):
        raise ValueError('Four-cell probability does not match the native branch factorization')
    expected = nested_four_cell_moments(read_trials=torch.tensor(read, dtype=torch.int64),
        **probabilities, concentration=concentration)
    for name, shape in (('four_cell_mean', (operation_count, 4)), ('four_cell_covariance', (operation_count, 4, 4))):
        actual = numerical_array(values[name], shape, name)
        if not torch.allclose(actual, expected[name], rtol=moment_relative_tolerance, atol=moment_absolute_tolerance):
            raise ValueError(f'{name} does not reproduce native conditional point-parameter moments')
    numerical_array(values['effective_offsets'], (3,), 'effective_offsets')
    numerical_array(values['effective_contrasts'], (2,), 'effective_contrasts')
    if values['effective_family_correction'] is not None:
        numerical_array(values['effective_family_correction'], (operation_count, 3), 'effective_family_correction')
    for name in ('global_multi_mean', 'global_multi_dispersion'):
        if bool((numerical_array(values[name], (3,), name) <= 0).any()):
            raise ValueError(f'{name} must contain positive native head parameters')
    pair = numerical_array(values['global_pair_probability'], (6,), 'global_pair_probability')
    if bool(((pair < 0) | (pair > 1)).any()) or abs(float(pair.sum()) - 1) > 1e-12:
        raise ValueError('Global pair probabilities must form a probability vector')


def extract_effective_point_values(prediction: Mapping[str, Any], batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Serialize actual forward tensors; never invent unavailable model fields."""
    state = prediction['latent_state'][index]
    prior = prediction['prepared_prior'][index]
    native_read = prediction['native_read_trials']
    admitted_read = native_read_trials(batch)
    if not isinstance(native_read, torch.Tensor) or native_read.dtype != torch.int64 \
            or native_read.shape != admitted_read.shape or not torch.equal(native_read, admitted_read):
        raise ValueError('Actual forward READ trials differ from the admitted native scalar')
    values = {'schema': point_values_schema, 'native_read_count': int(batch['native_read_count_scalar'][index]),
        'prepared_prior_available': bool(prediction['prepared_prior_available'][index]),
        'effective_psd_real': state.real.double().tolist(), 'effective_psd_imag': state.imag.double().tolist(),
        'prepared_prior_real': prior.real.double().tolist(), 'prepared_prior_imag': prior.imag.double().tolist(),
        **{name: prediction[name][index].double().tolist() for name in (*probability_names,
            'four_cell_probability', 'four_cell_mean', 'four_cell_covariance', 'four_cell_concentration',
            'effective_offsets', 'effective_contrasts')},
        'effective_family_correction': None if 'effective_family_correction' not in prediction else
            prediction['effective_family_correction'][index].double().tolist(),
        'global_multi_mean': prediction['multi_means'][index].double().tolist(),
        'global_multi_dispersion': prediction['multi_dispersion'][index].double().tolist(),
        'global_pair_probability': prediction['pair_logits'][index].double().softmax(-1).tolist()}
    validate_effective_point_values(values, mode_count=state.shape[-1], operation_count=batch['operation_mask'].shape[-1])
    return values
