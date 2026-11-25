"""
Quantum State Tomography Module
For state reconstruction and fidelity calculation of W-states
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import json
import os
from enum import Enum


class MeasurementBasis(Enum):
    """Measurement basis type"""
    X = 'X'      # σ_x basis
    Y = 'Y'      # σ_y basis
    Z = 'Z'      # σ_z basis


@dataclass
class TomographyResult:
    """State tomography result"""
    # Density matrix (4 qubits: 16×16)
    density_matrix: np.ndarray

    # State vector assuming purity
    state_vector: Optional[np.ndarray] = None

    # Experimental statistics
    measurement_counts: Dict[str, int] = None

    # Tomography quality metrics
    fidelity_to_ideal: float = 0.0
    purity: float = 0.0
    trace: float = 1.0

    # Error estimate
    uncertainty: Optional[np.ndarray] = None


class QuantumStateTomography:
    """Quantum state tomography class - for 4-qubit systems"""

    def __init__(self, num_qubits: int = 4):
        """
        Args:
            num_qubits: Number of qubits (default 4)
        """
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits  # Hilbert space dimension

        # Pauli matrices
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)

        self.paulis = {'I': self.I, 'X': self.X, 'Y': self.Y, 'Z': self.Z}

        # Ideal W-state
        self.ideal_w_state = self._create_w_state()

    def _create_w_state(self) -> np.ndarray:
        """Create 4-qubit ideal W-state"""
        # |W_4> = 1/2(|1000> + |0100> + |0010> + |0001>)
        w_state = np.zeros(self.dim, dtype=complex)
        for i in range(self.num_qubits):
            # i-th qubit is 1, others are 0
            bitstring = [0] * self.num_qubits
            bitstring[i] = 1
            idx = sum(b * (2 ** j) for j, b in enumerate(reversed(bitstring)))
            w_state[idx] = 1 / np.sqrt(self.num_qubits)

        return w_state

    def get_ideal_density_matrix(self) -> np.ndarray:
        """Get ideal W-state density matrix"""
        return np.outer(self.ideal_w_state, np.conj(self.ideal_w_state))

    def reconstruct_from_counts(self, measurement_counts: Dict[str, int],
                                num_measurements: int) -> TomographyResult:
        """
        Reconstruct quantum state from measurement counts

        Args:
            measurement_counts: Measurement counts {basis_string: count}
                               e.g.: {'XXXI': 150, 'XXZZ': 200, ...}
            num_measurements: Total number of measurements

        Returns:
            TomographyResult object
        """
        # Compute probabilities
        probabilities = {k: v / num_measurements for k, v in measurement_counts.items()}

        # Use maximum likelihood estimation or linear inversion to reconstruct density matrix
        rho = self._linear_inversion(probabilities)

        # Ensure physicality of density matrix
        rho = self._make_physical(rho)

        # Compute purity Tr(ρ²)
        purity = np.real(np.trace(rho @ rho))

        # Normalize
        rho = rho / np.trace(rho)

        # Compute fidelity to ideal W-state
        ideal_rho = self.get_ideal_density_matrix()
        fidelity = np.real(np.trace(np.sqrt(np.sqrt(rho) @ ideal_rho @ np.sqrt(rho)))) ** 2

        # Try to extract state vector (if nearly pure)
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        dominant_eigenvalue = eigenvalues[-1]
        state_vector = None
        if dominant_eigenvalue > 0.9:
            state_vector = eigenvectors[:, -1]

        return TomographyResult(
            density_matrix=rho,
            state_vector=state_vector,
            measurement_counts=measurement_counts,
            fidelity_to_ideal=float(fidelity),
            purity=float(purity),
            trace=float(np.trace(rho))
        )

    def _linear_inversion(self, probabilities: Dict[str, int]) -> np.ndarray:
        """
        Reconstruct density matrix using linear inversion

        Args:
            probabilities: Measurement probability dictionary

        Returns:
            Reconstructed density matrix
        """
        rho = np.zeros((self.dim, self.dim), dtype=complex)

        # Iterate through all possible measurement basis combinations
        pauli_basis = ['I', 'X', 'Y', 'Z']

        for combo in probabilities.keys():
            if len(combo) != self.num_qubits:
                continue

            # Build measurement operator
            measurement_op = self._build_pauli_operator(combo)

            # Compute expectation value
            exp_value = 2 ** self.num_qubits * probabilities[combo] - 1

            # Accumulate to density matrix
            rho += (exp_value / 2 ** self.num_qubits) * measurement_op

        return rho

    def _build_pauli_operator(self, combo: str) -> np.ndarray:
        """Build multi-qubit Pauli operator"""
        result = self.I

        for pauli in reversed(combo):
            if pauli == 'I':
                result = np.kron(self.I, result)
            elif pauli == 'X':
                result = np.kron(self.X, result)
            elif pauli == 'Y':
                result = np.kron(self.Y, result)
            elif pauli == 'Z':
                result = np.kron(self.Z, result)

        return result

    def _make_physical(self, rho: np.ndarray) -> np.ndarray:
        """
        Project density matrix to physical space (positive semi-definite, trace = 1)
        Using modified Reichl method
        """
        # Diagonalize
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        eigenvalues = np.real(eigenvalues)

        # Set negative eigenvalues to small positive value
        eigenvalues[eigenvalues < 0] = 1e-10

        # Reconstruct density matrix
        rho_physical = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T

        # Normalize
        rho_physical = rho_physical / np.trace(rho_physical)

        return rho_physical

    def compute_fidelity(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """
        Compute fidelity between two density matrices

        F(ρ₁, ρ₂) = Tr(√(√ρ₁ ρ₂ √ρ₁))²
        """
        sqrt_rho1 = self._matrix_sqrt(rho1)
        fidelity = np.real(np.trace(np.sqrt(sqrt_rho1 @ rho2 @ sqrt_rho1))) ** 2
        return float(fidelity)

    def _matrix_sqrt(self, rho: np.ndarray) -> np.ndarray:
        """Compute matrix square root"""
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        eigenvalues = np.maximum(np.real(eigenvalues), 0)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T

    def estimate_uncertainty(self, measurement_counts: Dict[str, int],
                             n_bootstrap: int = 100) -> np.ndarray:
        """
        Estimate density matrix uncertainty using bootstrap

        Args:
            measurement_counts: Measurement counts
            n_bootstrap: Number of bootstrap samples

        Returns:
            Density matrix uncertainty estimate
        """
        num_measurements = sum(measurement_counts.values())
        if num_measurements == 0:
            return np.zeros((self.dim, self.dim), dtype=float)

        pauli_basis = ['I', 'X', 'Y', 'Z']
        all_densities = []

        for _ in range(n_bootstrap):
            # Bootstrap sampling
            sampled_counts = {}
            for key, count in measurement_counts.items():
                # Sample from multinomial distribution
                n = np.random.multinomial(1, np.ones(len(pauli_basis)) / len(pauli_basis), count)
                sampled_counts[key] = int(np.sum(n))

            rho = self.reconstruct_from_counts(sampled_counts, num_measurements)
            all_densities.append(rho.density_matrix)

        # Compute standard deviation
        all_densities = np.array(all_densities)
        uncertainty = np.std(all_densities, axis=0)

        return uncertainty

    def save_result(self, result: TomographyResult, filepath: str):
        """Save tomography result"""
        data = {
            'density_matrix': result.density_matrix.tolist(),
            'fidelity_to_ideal': result.fidelity_to_ideal,
            'purity': result.purity,
            'trace': result.trace,
            'measurement_counts': result.measurement_counts
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class ProjectiveMeasurementTomography:
    """Projection measurement tomography - measure specific basis only, suitable for fast fidelity estimation"""

    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.ideal_w_state = self._create_w_state()

    def _create_w_state(self) -> np.ndarray:
        """Create 4-qubit ideal W-state"""
        w_state = np.zeros(self.dim, dtype=complex)
        for i in range(self.num_qubits):
            bitstring = [0] * self.num_qubits
            bitstring[i] = 1
            idx = sum(b * (2 ** j) for j, b in enumerate(reversed(bitstring)))
            w_state[idx] = 1 / np.sqrt(self.num_qubits)
        return w_state

    def compute_fidelity_from_counts(self, counts: Dict[int, int]) -> float:
        """
        Estimate W-state fidelity from counts

        Using only Z-basis projection measurement: expectation value of |W><W| in Z-basis
        """
        total_count = sum(counts.values())
        if total_count == 0:
            return 0.0

        # W-state has only one qubit as 1 in Z-basis
        # |1000>, |0100>, |0010>, |0001>
        w_state_indices = [
            2 ** (self.num_qubits - 1 - i)  # i-th qubit is 1
            for i in range(self.num_qubits)
        ]

        w_count = sum(counts.get(idx, 0) for idx in w_state_indices)

        # Normalized fidelity
        fidelity = w_count / total_count * self.num_qubits

        return min(fidelity, 1.0)