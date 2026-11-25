"""
W-state projection measurements
Used for fast W-state fidelity measurement in experiments
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union


class WStateProjections:
    """W-state projection measurement class"""

    def __init__(self, num_qubits: int = 4):
        """
        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

        # Define basis vectors for W-state
        self.w_state_indices = self._get_w_state_indices()

    def _get_w_state_indices(self) -> List[int]:
        """Get computational basis indices contained in W-state"""
        indices = []
        for i in range(self.num_qubits):
            # i-th qubit is 1
            bitstring = [0] * self.num_qubits
            bitstring[i] = 1
            idx = sum(b * (2 ** j) for j, b in enumerate(reversed(bitstring)))
            indices.append(idx)
        return indices

    def compute_ideal_expectation(self, operator_name: str,
                                  rotation_angles: Optional[List[Tuple[float, float, float]]] = None
                                  ) -> float:
        """
        Compute expectation value for ideal W-state

        Args:
            operator_name: Operator name ('W_proj', 'parity', 'correlation_X', 'correlation_Z')
            rotation_angles: Rotation angles for each qubit [(theta1, phi1, gamma1), ...]

        Returns:
            Expectation value
        """
        # Ideal W-state
        w_state = self._create_w_state()
        ideal_rho = np.outer(w_state, np.conj(w_state))

        if operator_name == 'W_proj':
            projector = self._w_state_projector()
            return np.real(np.trace(projector @ ideal_rho))

        elif operator_name == 'parity':
            parity_op = self._parity_operator()
            return np.real(np.trace(parity_op @ ideal_rho))

        elif operator_name == 'correlation_X':
            corr_op = self._correlation_operator('X')
            return np.real(np.trace(corr_op @ ideal_rho))

        elif operator_name == 'correlation_Z':
            corr_op = self._correlation_operator('Z')
            return np.real(np.trace(corr_op @ ideal_rho))

        else:
            raise ValueError(f"Unknown operator: {operator_name}")

    def _create_w_state(self) -> np.ndarray:
        """Create ideal W-state"""
        w_state = np.zeros(self.dim, dtype=complex)
        for i in range(self.num_qubits):
            bitstring = [0] * self.num_qubits
            bitstring[i] = 1
            idx = sum(b * (2 ** j) for j, b in enumerate(reversed(bitstring)))
            w_state[idx] = 1 / np.sqrt(self.num_qubits)
        return w_state

    def _w_state_projector(self) -> np.ndarray:
        """Build W-state projector |W><W|"""
        w_state = self._create_w_state()
        return np.outer(w_state, np.conj(w_state))

    def _parity_operator(self) -> np.ndarray:
        """Build parity operator (for W-state verification)"""
        # P = sum |i><i| for all i with odd parity
        # All basis vectors in W-state have odd number of 1s
        parity_op = np.zeros((self.dim, self.dim), dtype=complex)

        for i in range(self.dim):
            # Count number of 1s in binary representation
            num_ones = bin(i).count('1')
            if num_ones % 2 == 1:
                parity_op[i, i] = 1

        return parity_op

    def _correlation_operator(self, basis: str) -> np.ndarray:
        """Build correlation operator"""
        if basis == 'Z':
            # Compute ZᵢZⱼ correlation
            result = np.zeros((self.dim, self.dim), dtype=complex)
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    result += self._z_z_correlation(i, j)
            return result
        elif basis == 'X':
            # Compute XX correlation
            result = np.zeros((self.dim, self.dim), dtype=complex)
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    result += self._x_x_correlation(i, j)
            return result
        else:
            raise ValueError(f"Unknown basis: {basis}")

    def _z_z_correlation(self, i: int, j: int) -> np.ndarray:
        """ZᵢZⱼ correlation operator"""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.array([[1, 0], [0, 1]], dtype=complex)

        result = I
        for k in range(self.num_qubits):
            if k == i or k == j:
                result = np.kron(Z, result)
            else:
                result = np.kron(I, result)

        return result

    def _x_x_correlation(self, i: int, j: int) -> np.ndarray:
        """XᵢXⱼ correlation operator"""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        I = np.array([[1, 0], [0, 1]], dtype=complex)

        result = I
        for k in range(self.num_qubits):
            if k == i or k == j:
                result = np.kron(X, result)
            else:
                result = np.kron(I, result)

        return result

    def simulate_measurements(self, experimental_params: Dict[str, float],
                              n_shots: int = 1000) -> Dict[int, int]:
        """
        Simulate experimental measurements

        Args:
            experimental_params: Experimental parameters
                - freq_offsets: Frequency offsets [Δf1, Δf2, Δf3, Δf4]
                - phase_offsets: Phase offsets [φ1, φ2, φ3, φ4]
                - amplitude_errors: Amplitude errors [ε1, ε2, ε3, ε4]
            n_shots: Number of measurements

        Returns:
            Measurement counts {bitstring_index: count}
        """
        # Apply parameter offsets to W-state
        w_state = self._create_w_state()

        # Compute perturbed W-state
        disturbed_state = np.zeros(self.dim, dtype=complex)
        for i in range(self.num_qubits):
            # Amplitude for i-th qubit being 1
            amplitude = w_state[i] * (1 + experimental_params.get('amplitude_errors', [0]*4)[i])
            # Phase offset
            phase = np.exp(1j * experimental_params.get('phase_offsets', [0]*4)[i])
            disturbed_state[i] = amplitude * phase

        # Normalize
        disturbed_state = disturbed_state / np.linalg.norm(disturbed_state)

        # Compute probability distribution
        probabilities = np.abs(disturbed_state) ** 2

        # Sample
        indices = np.arange(self.dim)
        counts = np.bincount(
            np.random.choice(indices, size=n_shots, p=probabilities),
            minlength=self.dim
        )

        return {int(i): int(count) for i, count in enumerate(counts) if count > 0}

    def estimate_fidelity(self, measured_counts: Dict[int, int],
                          normalization: str = 'total') -> float:
        """
        Estimate W-state fidelity from measurement counts

        Args:
            measured_counts: Measurement counts
            normalization: Normalization method
                - 'total': Use total counts
                - 'w_subspace': Use only W-subspace counts

        Returns:
            Fidelity estimate
        """
        # Compute fidelity
        if normalization == 'total':
            total = sum(measured_counts.values())
            w_count = sum(measured_counts.get(idx, 0) for idx in self.w_state_indices)

            if total == 0:
                return 0.0

            # F = N * P(W) / N(W)
            fidelity = w_count / total * self.num_qubits

        elif normalization == 'w_subspace':
            w_count = sum(measured_counts.get(idx, 0) for idx in self.w_state_indices)
            total_w = sum(measured_counts.values())

            if total_w == 0:
                return 0.0

            fidelity = w_count / total_w

        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        return min(max(fidelity, 0.0), 1.0)


class MeasurementSchedules:
    """Define measurement sequences"""

    # Full tomography measurement basis combinations
    FULL_TOMOGRAPHY = [
        'IIII', 'IXIX', 'IYIY', 'IZIZ',
        'XIXI', 'XXXX', 'XYXY', 'XZ XZ',
        'YIYI', 'YXYX', 'YYYY', 'YZ YZ',
        'ZIZI', 'ZXZX', 'ZYZY', 'ZZZZ'
    ]

    # Minimal measurement set for fast W-state verification
    MINIMAL_W_VERIFICATION = [
        'ZZZZ',  # Z-basis projection
    ]

    # Fidelity measurement sequences
    FIDELITY_MEASUREMENTS = [
        'IZIZ', 'IYIY', 'IXIX',  # Single-qubit expectation values
        'ZZZZ', 'XXXX',  # Correlation functions
    ]