"""
AOD Nonlinear Calibration Model Base Class
Defines basic interface and data structures for AOD calibration
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
import json
import os


@dataclass
class FrequencyPair:
    """Frequency pair data point"""
    freq1: float      # Channel 1 set frequency (MHz)
    freq2: float      # Channel 2 set frequency (MHz)
    actual1: float    # Channel 1 actual frequency (MHz)
    actual2: float    # Channel 2 actual frequency (MHz)
    uncertainty1: float = 0.0  # Frequency uncertainty
    uncertainty2: float = 0.0
    set_freqs: Optional[List[float]] = None
    actual_freqs: Optional[List[float]] = None
    uncertainties: Optional[List[float]] = None
    amplitudes: Optional[List[float]] = None
    powers: Optional[List[float]] = None


@dataclass
class AODCalibrationData:
    """AOD calibration dataset"""
    # Channel configuration
    num_channels: int = 4
    freq_range: Tuple[float, float] = (84.0, 121.5)  # MHz
    freq_resolution: float = 2.5  # MHz step size

    # Measurement data
    calibration_points: List[FrequencyPair] = field(default_factory=list)

    # Metadata
    experiment_date: Optional[str] = None
    temperature: Optional[float] = None
    notes: str = ""

    def add_point(self, freq1: float, freq2: float, actual1: float, actual2: float,
                  uncertainty1: float = 0.0, uncertainty2: float = 0.0):
        """Add calibration point"""
        self.calibration_points.append(FrequencyPair(
            freq1=freq1, freq2=freq2,
            actual1=actual1, actual2=actual2,
            uncertainty1=uncertainty1, uncertainty2=uncertainty2,
            set_freqs=[float(freq1), float(freq2)],
            actual_freqs=[float(actual1), float(actual2)],
            uncertainties=[float(uncertainty1), float(uncertainty2)],
        ))

    def add_vector_point(
        self,
        set_freqs: List[float] | np.ndarray,
        actual_freqs: List[float] | np.ndarray,
        uncertainties: Optional[List[float] | np.ndarray] = None,
        amplitudes: Optional[List[float] | np.ndarray] = None,
        powers: Optional[List[float] | np.ndarray] = None,
    ):
        """Add one simultaneous multi-channel calibration point."""

        set_arr = np.asarray(set_freqs, dtype=float).reshape(-1)
        actual_arr = np.asarray(actual_freqs, dtype=float).reshape(-1)
        if set_arr.size != actual_arr.size:
            raise ValueError("set_freqs and actual_freqs must have the same length")
        if set_arr.size == 0:
            raise ValueError("at least one channel is required")
        self.num_channels = max(int(self.num_channels), int(set_arr.size))
        unc_arr = (
            np.zeros_like(set_arr, dtype=float)
            if uncertainties is None
            else np.asarray(uncertainties, dtype=float).reshape(-1)
        )
        if unc_arr.size != set_arr.size:
            raise ValueError("uncertainties must match set_freqs length")
        amp_list = None if amplitudes is None else np.asarray(amplitudes, dtype=float).reshape(-1).tolist()
        power_list = None if powers is None else np.asarray(powers, dtype=float).reshape(-1).tolist()
        self.calibration_points.append(FrequencyPair(
            freq1=float(set_arr[0]),
            freq2=float(set_arr[1] if set_arr.size > 1 else set_arr[0]),
            actual1=float(actual_arr[0]),
            actual2=float(actual_arr[1] if actual_arr.size > 1 else actual_arr[0]),
            uncertainty1=float(unc_arr[0]),
            uncertainty2=float(unc_arr[1] if unc_arr.size > 1 else unc_arr[0]),
            set_freqs=set_arr.tolist(),
            actual_freqs=actual_arr.tolist(),
            uncertainties=unc_arr.tolist(),
            amplitudes=amp_list,
            powers=power_list,
        ))

    def channel_arrays(self, channel_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return set, actual, and uncertainty arrays for one channel.

        Legacy two-channel records are expanded by using channel 2 values for
        channels beyond index 0. New vector records preserve full MIMO samples.
        """

        if channel_idx < 0 or channel_idx >= int(self.num_channels):
            raise IndexError(f"channel_idx {channel_idx} out of range for {self.num_channels} channels")
        set_values = []
        actual_values = []
        uncertainties = []
        for point in self.calibration_points:
            if point.set_freqs is not None and channel_idx < len(point.set_freqs):
                set_values.append(float(point.set_freqs[channel_idx]))
                actual_values.append(float(point.actual_freqs[channel_idx]))
                if point.uncertainties is not None and channel_idx < len(point.uncertainties):
                    uncertainties.append(float(point.uncertainties[channel_idx]))
                else:
                    uncertainties.append(0.0)
            else:
                set_values.append(float(point.freq1 if channel_idx == 0 else point.freq2))
                actual_values.append(float(point.actual1 if channel_idx == 0 else point.actual2))
                uncertainties.append(float(point.uncertainty1 if channel_idx == 0 else point.uncertainty2))
        if not set_values:
            raise ValueError("calibration_data contains no calibration points")
        return (
            np.asarray(set_values, dtype=float),
            np.asarray(actual_values, dtype=float),
            np.asarray(uncertainties, dtype=float),
        )

    def save(self, filepath: str):
        """Save calibration data"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        data = {
            'num_channels': self.num_channels,
            'freq_range': self.freq_range,
            'freq_resolution': self.freq_resolution,
            'experiment_date': self.experiment_date,
            'temperature': self.temperature,
            'notes': self.notes,
            'points': [
                {
                    'freq1': p.freq1, 'freq2': p.freq2,
                    'actual1': p.actual1, 'actual2': p.actual2,
                    'uncertainty1': p.uncertainty1, 'uncertainty2': p.uncertainty2,
                    'set_freqs': p.set_freqs,
                    'actual_freqs': p.actual_freqs,
                    'uncertainties': p.uncertainties,
                    'amplitudes': p.amplitudes,
                    'powers': p.powers,
                }
                for p in self.calibration_points
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'AODCalibrationData':
        """Load calibration data"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        cal = cls()
        cal.num_channels = data['num_channels']
        cal.freq_range = tuple(data['freq_range'])
        cal.freq_resolution = data['freq_resolution']
        cal.experiment_date = data.get('experiment_date')
        cal.temperature = data.get('temperature')
        cal.notes = data.get('notes', '')
        cal.calibration_points = [
            FrequencyPair(
                freq1=p['freq1'], freq2=p['freq2'],
                actual1=p['actual1'], actual2=p['actual2'],
                uncertainty1=p.get('uncertainty1', 0),
                uncertainty2=p.get('uncertainty2', 0),
                set_freqs=p.get('set_freqs'),
                actual_freqs=p.get('actual_freqs'),
                uncertainties=p.get('uncertainties'),
                amplitudes=p.get('amplitudes'),
                powers=p.get('powers'),
            )
            for p in data['points']
        ]
        return cal


class AODNonlinearModel:
    """AOD nonlinear calibration base class"""

    def __init__(self, freq_range: Tuple[float, float] = (84.0, 121.5)):
        """
        Args:
            freq_range: Frequency range (min_MHz, max_MHz)
        """
        self.freq_range = freq_range
        self.freq_min, self.freq_max = freq_range
        self.is_trained = False

    def train(self, calibration_data: AODCalibrationData):
        """Train model

        Args:
            calibration_data: AOD calibration dataset
        """
        raise NotImplementedError

    def correct(self, set_freqs: np.ndarray) -> np.ndarray:
        """
        Calculate corrected actual frequency from set frequency

        Args:
            set_freqs: Set frequency array [f1, f2, f3, f4] (MHz)

        Returns:
            Corrected frequency estimate [actual_f1, actual_f2, actual_f3, actual_f4] (MHz)
        """
        raise NotImplementedError

    def invert(self, target_freqs: np.ndarray) -> np.ndarray:
        """
        Calculate required set frequency from target actual frequency

        Args:
            target_freqs: Target actual frequency array (MHz)

        Returns:
            Required set frequency array (MHz)
        """
        raise NotImplementedError

    def estimate_uncertainty(self, set_freqs: np.ndarray) -> np.ndarray:
        """
        Estimate frequency correction uncertainty

        Args:
            set_freqs: Set frequency array

        Returns:
            Frequency estimate uncertainty array
        """
        raise NotImplementedError


class PolynomialAODModel(AODNonlinearModel):
    """Polynomial fitting based AOD nonlinear model"""

    def __init__(self, freq_range: Tuple[float, float] = (84.0, 121.5), degree: int = 3):
        """
        Args:
            freq_range: Frequency range
            degree: Polynomial fitting degree
        """
        super().__init__(freq_range)
        self.degree = degree
        self.poly_coeffs: Dict[int, np.ndarray] = {}
        self.x_range = np.array(freq_range)

    def train(self, calibration_data: AODCalibrationData):
        """Use polynomial fitting for AOD frequency response"""
        for ch_idx in range(calibration_data.num_channels):
            set_freqs, actual_freqs, _ = calibration_data.channel_arrays(ch_idx)
            order = np.argsort(set_freqs)
            set_freqs = set_freqs[order]
            actual_freqs = actual_freqs[order]
            unique_count = int(np.unique(set_freqs).size)
            degree = min(int(self.degree), max(1, unique_count - 1))
            coeffs = np.polyfit(set_freqs, actual_freqs, degree)
            self.poly_coeffs[ch_idx] = coeffs

        self.is_trained = True
        print(f"Polynomial model trained: degree={self.degree}")

    def correct(self, set_freqs: np.ndarray) -> np.ndarray:
        """Calculate corrected actual frequency"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        values = np.asarray(set_freqs, dtype=float).reshape(-1)
        corrected = np.zeros_like(values, dtype=float)
        for ch_idx in range(values.size):
            predicted = np.polyval(self.poly_coeffs[ch_idx], values[ch_idx])
            corrected[ch_idx] = predicted

        return corrected

    def invert(self, target_freqs: np.ndarray) -> np.ndarray:
        """Calculate required set frequency"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        targets = np.asarray(target_freqs, dtype=float).reshape(-1)
        inverted = np.zeros_like(targets, dtype=float)
        grid = np.linspace(float(self.freq_min), float(self.freq_max), 1024)
        for ch_idx, target in enumerate(targets):
            coeffs = self.poly_coeffs[ch_idx]
            actual_grid = np.polyval(coeffs, grid)
            order = np.argsort(actual_grid)
            actual_sorted = actual_grid[order]
            grid_sorted = grid[order]
            unique_actual, unique_idx = np.unique(actual_sorted, return_index=True)
            unique_grid = grid_sorted[unique_idx]
            clipped_target = float(np.clip(target, unique_actual[0], unique_actual[-1]))
            inverted[ch_idx] = np.interp(clipped_target, unique_actual, unique_grid)
        return np.clip(inverted, self.freq_min, self.freq_max)

    def estimate_uncertainty(self, set_freqs: np.ndarray) -> np.ndarray:
        """Estimate uncertainty"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Simplified: assume 0.1 MHz basic uncertainty
        return np.full_like(set_freqs, 0.1)


class LookupTableAODModel(AODNonlinearModel):
    """Lookup table based AOD model"""

    def __init__(self, freq_range: Tuple[float, float] = (84.0, 121.5), resolution: float = 0.5):
        """
        Args:
            freq_range: Frequency range
            resolution: Lookup table resolution (MHz)
        """
        super().__init__(freq_range)
        self.resolution = resolution
        self.lookup_tables: Dict[int, Dict] = {}

    def _build_grid(self):
        """Build frequency grid"""
        freqs = np.arange(self.freq_min, self.freq_max + self.resolution, self.resolution)
        return freqs

    def train(self, calibration_data: AODCalibrationData):
        """Build lookup table"""
        freqs = self._build_grid()

        for ch_idx in range(calibration_data.num_channels):
            set_vals, actual_vals, _ = calibration_data.channel_arrays(ch_idx)
            order = np.argsort(set_vals)
            set_vals = set_vals[order]
            actual_vals = actual_vals[order]
            actual_grid = np.interp(freqs, set_vals, actual_vals)

            self.lookup_tables[ch_idx] = {
                'freqs': freqs,
                'actuals': actual_grid,
                'resolution': self.resolution
            }

        self.is_trained = True
        print(f"Lookup table model trained: resolution={self.resolution} MHz")

    def correct(self, set_freqs: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        values = np.asarray(set_freqs, dtype=float).reshape(-1)
        corrected = np.zeros_like(values)
        for ch_idx in range(len(values)):
            freq_data = self.lookup_tables[ch_idx]
            corrected[ch_idx] = np.interp(
                values[ch_idx],
                freq_data['freqs'],
                freq_data['actuals']
            )

        return corrected

    def invert(self, target_freqs: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        targets = np.asarray(target_freqs, dtype=float).reshape(-1)
        inverted = np.zeros_like(targets)
        for ch_idx in range(len(targets)):
            freq_data = self.lookup_tables[ch_idx]
            order = np.argsort(freq_data['actuals'])
            inverted[ch_idx] = np.interp(
                targets[ch_idx],
                np.asarray(freq_data['actuals'])[order],
                np.asarray(freq_data['freqs'])[order]
            )

        return inverted

    def estimate_uncertainty(self, set_freqs: np.ndarray) -> np.ndarray:
        return np.full_like(set_freqs, self.resolution / np.sqrt(12))
