"""
Gaussian Process Regression AOD Lookup Table
Suitable for small-sample calibration scenarios
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import pickle
import os

from .nonlinear_model import AODNonlinearModel, AODCalibrationData


class GPLookupTable:
    """Gaussian Process Lookup Table"""

    def __init__(self, length_scale: float = 5.0, noise_variance: float = 0.1):
        """
        Args:
            length_scale: Length scale parameter (MHz)
            noise_variance: Noise variance (MHz^2)
        """
        self.length_scale = length_scale
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.X_train_raw = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Fit Gaussian Process

        Args:
            x_train: Set frequencies
            y_train: Actual frequencies
        """
        self.X_train = x_train.reshape(-1, 1)
        self.y_train = y_train
        self.X_train_raw = np.copy(x_train)

    def predict(self, x: np.ndarray):
        """
        Predict actual frequency for given set frequency

        Args:
            x: Set frequency

        Returns:
            (mean, variance) Predicted mean and variance
        """
        if self.X_train is None:
            raise ValueError("Model not fitted")

        x = np.atleast_1d(x).reshape(-1, 1)

        # Compute RBF kernel matrix
        K = self._rbf_kernel(self.X_train, self.X_train)
        K_inv = np.linalg.inv(K + self.noise_variance * np.eye(len(self.X_train)))

        K_star = self._rbf_kernel(self.X_train, x)

        # Posterior mean and variance
        y_pred = (K_star.T @ K_inv @ self.y_train).flatten()
        var_pred = np.diag(self._rbf_kernel(x, x)) - (K_star.T @ K_inv @ K_star).diagonal()

        return y_pred, np.maximum(var_pred, 0)

    def inverse_predict(self, y_target: np.ndarray, x_initial: Optional[np.ndarray] = None):
        """
        Inverse predict set frequency from target actual frequency

        Args:
            y_target: Target actual frequency
            x_initial: Initial guess

        Returns:
            Set frequency value
        """
        if self.X_train is None:
            raise ValueError("Model not fitted")

        y_target = np.atleast_1d(y_target)
        if x_initial is None:
            x_initial = y_target  # Assume linear

        x_current = np.atleast_1d(x_initial).copy()

        # Newton iteration
        for _ in range(50):
            y_pred, var_pred = self.predict(x_current)
            residual = y_target - y_pred

            # Numerical derivative
            h = 0.001
            derivative = np.zeros_like(x_current)
            for i in range(len(x_current)):
                x_perturb = x_current.copy()
                x_perturb[i] += h
                y_perturb, _ = self.predict(x_perturb)
                derivative[i] = (y_perturb - y_pred[i]) / h

            # Newton step
            x_current += derivative / (derivative ** 2 + 1e-10) * residual

            if np.max(np.abs(residual)) < 0.001:
                break

        return x_current

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray):
        """Compute RBF kernel matrix"""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        sq_dist = (
            np.sum(X1 ** 2, axis=1, keepdims=True) +
            np.sum(X2 ** 2, axis=1) -
            2 * X1 @ X2.T
        )

        return np.exp(-0.5 * sq_dist / (self.length_scale ** 2))


class GPRegressionAODModel(AODNonlinearModel):
    """Gaussian Regression AOD Calibration Model"""

    def __init__(self, freq_range: Tuple[float, float] = (84.0, 121.5),
                 length_scale: float = 5.0, noise_variance: float = 0.1):
        """
        Args:
            freq_range: Frequency range
            length_scale: Length scale
            noise_variance: Noise variance
        """
        super().__init__(freq_range)

        self.length_scale = length_scale
        self.noise_variance = noise_variance
        self.gp_models: Dict[int, GPLookupTable] = {}

    def train(self, calibration_data: AODCalibrationData):
        """Train Gaussian Process model"""
        for ch_idx in range(calibration_data.num_channels):
            gp = GPLookupTable(
                length_scale=self.length_scale,
                noise_variance=self.noise_variance
            )

            set_freqs, actual_freqs, _ = calibration_data.channel_arrays(ch_idx)

            gp.fit(set_freqs, actual_freqs)
            self.gp_models[ch_idx] = gp

        self.is_trained = True
        print(f"GP model trained with length_scale={self.length_scale}, noise_var={self.noise_variance}")

    def correct(self, set_freqs: np.ndarray) -> np.ndarray:
        """Calculate corrected actual frequency"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        corrected = np.zeros_like(set_freqs)
        uncertainties = np.zeros_like(set_freqs)

        for ch_idx in range(len(set_freqs)):
            gp = self.gp_models[ch_idx]
            mean, var = gp.predict(set_freqs[ch_idx])
            corrected[ch_idx] = mean[0]
            uncertainties[ch_idx] = np.sqrt(var[0])

        return corrected

    def invert(self, target_freqs: np.ndarray) -> np.ndarray:
        """Calculate set frequency from target frequency"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        inverted = np.zeros_like(target_freqs)

        for ch_idx in range(len(target_freqs)):
            gp = self.gp_models[ch_idx]
            inverted[ch_idx] = gp.inverse_predict(target_freqs[ch_idx])[0]

        return inverted

    def estimate_uncertainty(self, set_freqs: np.ndarray) -> np.ndarray:
        """Estimate prediction uncertainty"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        uncertainties = np.zeros_like(set_freqs)

        for ch_idx in range(len(set_freqs)):
            gp = self.gp_models[ch_idx]
            _, var = gp.predict(set_freqs[ch_idx])
            uncertainties[ch_idx] = np.sqrt(var[0])

        return uncertainties

    def save(self, filepath: str):
        """Save model"""
        save_data = {
            'freq_range': self.freq_range,
            'length_scale': self.length_scale,
            'noise_variance': self.noise_variance,
            'models': {
                str(ch_idx): {
                    'length_scale': gp.length_scale,
                    'noise_variance': gp.noise_variance,
                    'X_train': gp.X_train_raw,
                    'y_train': gp.y_train
                }
                for ch_idx, gp in self.gp_models.items()
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, filepath: str) -> 'GPRegressionAODModel':
        """Load model"""
        with open(filepath, 'rb') as f:
            load_data = pickle.load(f)

        model = cls(
            freq_range=tuple(load_data['freq_range']),
            length_scale=load_data['length_scale'],
            noise_variance=load_data['noise_variance']
        )

        model.gp_models = {}
        for ch_idx_str, gp_data in load_data['models'].items():
            ch_idx = int(ch_idx_str)
            gp = GPLookupTable(
                length_scale=gp_data['length_scale'],
                noise_variance=gp_data['noise_variance']
            )
            gp.X_train = gp_data['X_train'].reshape(-1, 1)
            gp.y_train = gp_data['y_train']
            gp.X_train_raw = np.copy(gp.X_train.flatten())
            model.gp_models[ch_idx] = gp

        model.is_trained = True
        return model
