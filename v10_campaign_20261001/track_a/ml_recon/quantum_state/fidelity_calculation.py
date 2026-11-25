"""
Fidelity calculation module
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union


class FidelityCalculator:
    """Quantum state fidelity calculator"""

    def __init__(self, num_qubits: int = 4):
        """
        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

    def fidelity_density_matrices(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """
        Compute fidelity between two density matrices

        F(ρ₁, ρ₂) = [Tr(√(√ρ₁ ρ₂ √ρ₁))]²

        Args:
            rho1: First density matrix (dim × dim)
            rho2: Second density matrix (dim × dim)

        Returns:
            Fidelity value [0, 1]
        """
        # Ensure matrices are complex
        rho1 = np.array(rho1, dtype=complex)
        rho2 = np.array(rho2, dtype=complex)

        if rho1.shape != rho2.shape:
            raise ValueError("Matrices must have the same shape")

        # Compute √ρ₁
        sqrt_rho1 = self._matrix_sqrt(rho1)

        # Compute √ρ₁ ρ₂ √ρ₁
        intermediate = sqrt_rho1 @ rho2 @ sqrt_rho1

        # Compute √(√ρ₁ ρ₂ √ρ₁)
        sqrt_intermediate = self._matrix_sqrt(intermediate)

        # Compute Tr and square
        fidelity = np.real(np.trace(sqrt_intermediate)) ** 2

        # Numerical stability handling
        fidelity = max(0.0, min(1.0, fidelity))

        return float(fidelity)

    def fidelity_state_vectors(self, psi1: np.ndarray, psi2: np.ndarray) -> float:
        """
        Compute fidelity between two state vectors (pure states)

        F(|ψ₁⟩, |ψ₂⟩) = |⟨ψ₁|ψ₂⟩|²

        Args:
            psi1: First state vector
            psi2: Second state vector

        Returns:
            Fidelity value [0, 1]
        """
        psi1 = np.array(psi1, dtype=complex)
        psi2 = np.array(psi2, dtype=complex)

        # Normalize
        psi1 = psi1 / np.linalg.norm(psi1)
        psi2 = psi2 / np.linalg.norm(psi2)

        # Compute magnitude squared of inner product
        fidelity = np.abs(np.vdot(psi1, psi2)) ** 2

        return float(fidelity)

    def fidelity_mixed_pure(self, rho: np.ndarray, psi: np.ndarray) -> float:
        """
        Fidelity between mixed state and pure state

        F(ρ, |ψ⟩) = ⟨ψ|ρ|ψ⟩

        Args:
            rho: Density matrix
            psi: State vector

        Returns:
            Fidelity value
        """
        rho = np.array(rho, dtype=complex)
        psi = np.array(psi, dtype=complex)

        # Normalize state vector
        psi = psi / np.linalg.norm(psi)

        fidelity = np.real(np.vdot(psi, rho @ psi))

        return float(max(0.0, fidelity))

    def _matrix_sqrt(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute square root of positive semi-definite matrix

        Through eigen decomposition: ρ = U D U† => √ρ = U √D U†
        """
        eigenvalues, eigenvectors = np.linalg.eigh(rho)

        # Set negative eigenvalues to 0 (numerical error)
        eigenvalues = np.maximum(np.real(eigenvalues), 0)

        # Compute square root
        sqrt_d = np.diag(np.sqrt(eigenvalues))

        # Reconstruct √ρ
        sqrt_rho = eigenvectors @ sqrt_d @ eigenvectors.conj().T

        return sqrt_rho

    def process_fidelity(self, u1: np.ndarray, u2: np.ndarray) -> float:
        """
        Compute process fidelity between two quantum gates (unitary matrices)

        F_process = (|Tr(U₁†U₂)|² + dim) / dim²

        Args:
            u1: First unitary matrix
            u2: Second unitary matrix

        Returns:
            Process fidelity
        """
        u1 = np.array(u1, dtype=complex)
        u2 = np.array(u2, dtype=complex)

        dim = u1.shape[0]

        # Compute
        trace = np.trace(np.conj(u1.T) @ u2)
        fidelity = (np.abs(trace) ** 2 + dim) / (dim ** 2)

        return float(fidelity)

    def gate_fidelity(self, u1: np.ndarray, u2: np.ndarray) -> float:
        """
        Compute average gate fidelity

        F_gate = F_process + 1/(dim+1) * (1 - F_process)

        Args:
            u1: First unitary matrix
            u2: Second unitary matrix

        Returns:
            Average gate fidelity
        """
        dim = u1.shape[0]
        process_fid = self.process_fidelity(u1, u2)

        gate_fid = process_fid + (1 - process_fid) / (dim + 1)

        return float(gate_fid)

    def estimate_experimentally(self, measured_counts: Dict[int, int],
                                ideal_w_state_indices: List[int],
                                total_shots: int) -> Tuple[float, float]:
        """
        Estimate fidelity and statistical error from experimental measurements

        Args:
            measured_counts: Measurement counts
            ideal_w_state_indices: Basis vector indices for ideal W-state
            total_shots: Total number of measurements

        Returns:
            (fidelity, fidelity_std)
        """
        # Compute fidelity
        w_count = sum(measured_counts.get(idx, 0) for idx in ideal_w_state_indices)

        if total_shots == 0:
            return 0.0, 0.0

        fidelity = (w_count / total_shots) * self.num_qubits
        fidelity = min(max(fidelity, 0.0), 1.0)

        # Binomial distribution error estimation
        if total_shots > 0:
            p = w_count / total_shots
            variance = p * (1 - p) / total_shots
            fidelity_std = np.sqrt(variance) * self.num_qubits
        else:
            fidelity_std = 0.0

        return fidelity, fidelity_std


def compare_states(state1: Union[np.ndarray, Tuple],
                  state2: Union[np.ndarray, Tuple],
                  is_pure: bool = True) -> Dict[str, float]:
    """
    General function for comparing two quantum states

    Args:
        state1: First state (density matrix or state vector)
        state2: Second state
        is_pure: Whether it's a pure state

    Returns:
        Fidelity comparison results
    """
    calc = FidelityCalculator()

    if is_pure:
        if isinstance(state1, tuple) or isinstance(state1, list):
            state1 = np.array(state1, dtype=complex)
        if isinstance(state2, tuple) or isinstance(state2, list):
            state2 = np.array(state2, dtype=complex)
        fidelity = calc.fidelity_state_vectors(state1, state2)
    else:
        state1 = np.array(state1, dtype=complex)
        state2 = np.array(state2, dtype=complex)
        fidelity = calc.fidelity_density_matrices(state1, state2)

    return {'fidelity': fidelity}