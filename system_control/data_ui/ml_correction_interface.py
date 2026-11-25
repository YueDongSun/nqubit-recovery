#!/usr/bin/env python3
"""
ML Correction Interface for UI
Simplified interface to ML framework for density matrix correction suggestions
"""

import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional

# Add repo root to path so the top-level ml_recon package is importable from system_control/data_ui.
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ml_recon_path = os.path.join(repo_root, 'ml_recon')
if os.path.exists(ml_recon_path):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    ML_AVAILABLE = True
else:
    ML_AVAILABLE = False
    print("Warning: ml_recon directory not found")

try:
    if ML_AVAILABLE:
        from ml_recon.quantum_state.state_tomography import QuantumStateTomography
        from ml_recon.quantum_state.fidelity_calculation import FidelityCalculator
        from ml_recon.quantum_representation import (
            QuantumStateDiagnosisModel,
            QuantumStateFamilyTorchClassifier,
            RhoPairCovarianceDiagnosisModel,
            RhoPairTorchDiagnosisModel,
        )
        PYTORCH_AVAILABLE = True
    else:
        PYTORCH_AVAILABLE = False
except ImportError as e:
    print(f"Warning: Could not import ML modules: {e}")
    PYTORCH_AVAILABLE = False
    QuantumStateDiagnosisModel = None
    QuantumStateFamilyTorchClassifier = None
    RhoPairTorchDiagnosisModel = None
    RhoPairCovarianceDiagnosisModel = None

if ML_AVAILABLE and QuantumStateDiagnosisModel is None:
    try:
        from ml_recon.quantum_representation import (
            QuantumStateDiagnosisModel,
            QuantumStateFamilyTorchClassifier,
            RhoPairCovarianceDiagnosisModel,
            RhoPairTorchDiagnosisModel,
        )
    except Exception as e:
        print(f"Warning: Could not import quantum diagnosis model: {e}")
        QuantumStateDiagnosisModel = None
        QuantumStateFamilyTorchClassifier = None
        RhoPairCovarianceDiagnosisModel = None
        RhoPairTorchDiagnosisModel = None


class MLCorrectionInterface:
    """Simplified interface for ML-based density matrix correction"""

    def __init__(self, num_qubits: int = 4):
        """
        Args:
            num_qubits: Number of qubits (3 or 4)
        """
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

        # Initialize ML components if available
        self.tomography = None
        self.fidelity_calc = None
        self.model = None
        self.config = None
        self.diagnosis_model = None
        self.diagnosis_model_path = None
        self.diagnosis_model_backend = None
        self.state_family_model = None
        self.state_family_model_path = None
        # Deep physics classifier (60K-sample, ~1.9M params, outputs 26-dim physics_aux + 320 latent).
        # Used as pretrain knowledge source for the physics_transfer_gnn feedback head.
        self.deep_classifier_model = None
        self.deep_classifier_path = None
        self.deep_classifier_label_maps = None
        self.deep_classifier_config = None
        self.deep_classifier_deployment_eligible = False

        if ML_AVAILABLE and PYTORCH_AVAILABLE:
            try:
                self.tomography = QuantumStateTomography(num_qubits=num_qubits)
                self.fidelity_calc = FidelityCalculator(num_qubits=num_qubits)
                print(f"ML Correction Interface initialized for {num_qubits} qubits")
            except Exception as e:
                print(f"Warning: Failed to initialize ML components: {e}")

        self.load_diagnosis_model()
        self.load_state_family_model()

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Deprecated VAE correction loader.

        The old VAE calibration scripts now live under ``script_old/``. Runtime
        diagnosis uses ``load_diagnosis_model`` and feedback deltas use
        ``ml_recon.feedback_model.TrainedFeedbackModel``.
        """
        print("Legacy VAE correction model moved to script_old; use feedback_model instead.")
        return False

    def load_state_family_model(self, model_path: Optional[str] = None) -> bool:
        """Load optional fine-grained 3Q/4Q state-family classifier."""
        if QuantumStateFamilyTorchClassifier is None:
            return False
        if model_path is None:
            suffix = "3q" if self.num_qubits == 3 else "4q"
            candidates = [
                os.path.join(repo_root, 'ml_recon', 'models', f'state_family_classifier_{suffix}.pt'),
                os.path.join(repo_root, 'ml_recon', 'models', 'state_family_classifier.pt'),
            ]
        else:
            candidates = [model_path]
        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            try:
                self.state_family_model = QuantumStateFamilyTorchClassifier.load(candidate)
                self.state_family_model_path = candidate
                print(f"State-family classifier loaded from: {candidate}")
                return True
            except Exception as e:
                print(f"Warning: Failed to load state-family classifier {candidate}: {e}")
        return False

    def load_diagnosis_model(self, model_path: Optional[str] = None) -> bool:
        """Load the density-matrix diagnosis model for mode/error probabilities."""
        if QuantumStateDiagnosisModel is None and RhoPairTorchDiagnosisModel is None:
            return False
        if model_path is None:
            suffix = "3q" if self.num_qubits == 3 else "4q"
            candidates = [
                os.path.join(repo_root, 'ml_recon', 'models', f'quantum_rho_covariance_{suffix}.json'),
                os.path.join(repo_root, 'ml_recon', 'models', f'quantum_rho_encoder_{suffix}.pt'),
                os.path.join(repo_root, 'ml_recon', 'models', 'quantum_rho_encoder.pt'),
                os.path.join(repo_root, 'ml_recon', 'models', f'quantum_rho_encoder_{suffix}.json'),
                os.path.join(repo_root, 'ml_recon', 'models', 'quantum_rho_encoder.json'),
                os.path.join(repo_root, 'ml_recon', 'models', 'quantum_rho_encoder_smoke.json'),
            ]
        else:
            candidates = [model_path]
        for candidate in candidates:
            if os.path.exists(candidate):
                try:
                    if candidate.lower().endswith('.pt'):
                        if RhoPairTorchDiagnosisModel is None:
                            continue
                        self.diagnosis_model = RhoPairTorchDiagnosisModel.load(candidate)
                        self.diagnosis_model_backend = 'torch'
                    else:
                        try:
                            if RhoPairCovarianceDiagnosisModel is None:
                                raise ValueError("covariance model loader unavailable")
                            self.diagnosis_model = RhoPairCovarianceDiagnosisModel.load(candidate)
                            self.diagnosis_model_backend = 'covariance'
                        except Exception:
                            if QuantumStateDiagnosisModel is None:
                                continue
                            self.diagnosis_model = QuantumStateDiagnosisModel.load(candidate)
                            self.diagnosis_model_backend = 'numpy'
                    self.diagnosis_model_path = candidate
                    print(f"Diagnosis model loaded from: {candidate}")
                    return True
                except Exception as e:
                    print(f"Warning: Failed to load diagnosis model {candidate}: {e}")
        return False

    def load_deep_physics_classifier(self, model_path: Optional[str] = None) -> bool:
        """Load the DeepPhysicsClassifier (60K-sample physics classifier).

        Outputs 26-dim physics_aux + 320-dim latent that the production
        physics_transfer_gnn feedback head consumes via context.classifier_diagnostics.
        """
        if not (ML_AVAILABLE and PYTORCH_AVAILABLE):
            return False
        if model_path is None:
            # Production checkpoint typically lives in the main repo (not a worktree).
            # Try worktree-local first, then known absolute fallback.
            candidates = [
                os.path.join(repo_root, 'ml_recon', 'deep_models', 'runs',
                             'classifier_auto_target95_round5', 'classifier_precision_trial02',
                             'checkpoints', 'best.pt'),
                os.path.join(repo_root, 'ml_recon', 'models', 'deep_physics_classifier.pt'),
                # Absolute fallback for worktrees (main repo location)
                r'H:\N_qubit\ml_recon\deep_models\runs\classifier_auto_target95_round5\classifier_precision_trial02\checkpoints\best.pt',
            ]
        else:
            candidates = [model_path]

        try:
            import torch  # noqa: PLC0415
            from ml_recon.deep_models.models import DeepPhysicsClassifier  # noqa: PLC0415
        except ImportError:
            return False

        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            try:
                sd = torch.load(candidate, map_location='cpu', weights_only=False)
                cfg = sd.get('config', {})
                deployment_eligible = cfg.get('inference_contract') == 'measured_only_v1'
                if not deployment_eligible:
                    print(
                        f"Warning: deep classifier {candidate} is not measured-only; "
                        "it is loaded for research diagnostics only and will not feed hardware feedback."
                    )
                state = sd['model_state']
                inferred_physics_dim = int(
                    state.get('fusion.physics.0.weight').shape[1]
                    if state.get('fusion.physics.0.weight') is not None
                    else cfg.get('physics_feature_dim', cfg.get('physics_dim', 64))
                )
                model = DeepPhysicsClassifier(
                    physics_dim=inferred_physics_dim,
                    hidden_dim=int(cfg.get('hidden_dim', 192)),
                    base_channels=int(cfg.get('base_channels', 32)),
                    dropout=float(cfg.get('dropout', 0.08)),
                )
                model.load_state_dict(state)
                model.eval()
                self.deep_classifier_model = model
                self.deep_classifier_path = candidate
                self.deep_classifier_label_maps = sd.get('label_maps')
                self.deep_classifier_config = {
                    **dict(cfg),
                    'physics_feature_dim': inferred_physics_dim,
                    'legacy_feature_adapter': None if deployment_eligible else 'explicit_prefix_v1',
                }
                self.deep_classifier_deployment_eligible = bool(deployment_eligible)
                print(f"Deep physics classifier loaded from: {candidate}")
                return True
            except Exception as e:
                print(f"Warning: Failed to load deep classifier {candidate}: {e}")
        return False

    def diagnose_density_matrix(self, rho: np.ndarray, rho_target: Optional[np.ndarray] = None) -> Dict:
        """Return excitation-mode and error-source probabilities for a rho.

        When deep_classifier_model is loaded, also injects 26-dim physics_aux
        and 320-dim latent for downstream physics_transfer_gnn feedback head.
        """
        if (
            self.diagnosis_model is None
            and self.deep_classifier_model is None
            and self.state_family_model is None
        ):
            return {
                'available': False,
                'message': 'Quantum diagnosis model not loaded.'
            }
        try:
            if self.diagnosis_model is not None:
                result = self.diagnosis_model.predict(rho, rho_target)
                result['available'] = True
                result.setdefault('backend', self.diagnosis_model_backend or 'unknown')
                result.setdefault('model_path', self.diagnosis_model_path)
            else:
                result = {
                    'available': True,
                    'backend': (
                        'deep_physics_classifier_only'
                        if self.deep_classifier_model is not None
                        else 'state_family_classifier_only'
                    ),
                }

            if self.state_family_model is not None:
                family = self.state_family_model.predict(rho)
                result['state_family_probabilities'] = family.get('state_family_probabilities', {})
                result['state_family_groups'] = family.get('state_family_groups', {})
                result['state_family_group_probabilities'] = family.get('state_family_group_probabilities', {})
                result['state_family_backend'] = family.get('backend')
                result['state_family_model_path'] = self.state_family_model_path
                result['state_family_latent_vector'] = family.get('latent_vector', [])

            if self.deep_classifier_model is not None:
                deep_out = self._run_deep_classifier(rho, rho_target)
                if deep_out:
                    result['physics_aux'] = deep_out['physics_aux']
                    result['deep_latent'] = deep_out['latent']
                    result['deep_classifier_path'] = self.deep_classifier_path
                    result['deep_classifier_deployment_eligible'] = bool(
                        self.deep_classifier_deployment_eligible
                    )
                    result['classifier_feature_contract'] = (
                        'measured_only_v1'
                        if self.deep_classifier_deployment_eligible
                        else 'research_target_conditioned_legacy'
                    )
                    if 'severity' in deep_out:
                        result['deep_severity'] = deep_out['severity']
                        result['deep_fidelity'] = deep_out['fidelity']
                        result['deep_purity'] = deep_out['purity']

            return result
        except Exception as e:
            return {
                'available': False,
                'message': f'Quantum diagnosis failed: {e}'
            }

    def _run_deep_classifier(
        self, rho: np.ndarray, rho_target: Optional[np.ndarray]
    ) -> Optional[Dict]:
        """Build inputs and run DeepPhysicsClassifier; return physics_aux + latent."""
        if self.deep_classifier_model is None:
            return None
        try:
            import torch  # noqa: PLC0415
            from ml_recon.deep_models.synthetic_physics import (  # noqa: PLC0415
                rho_pair_image as build_rho_pair_image,
                physics_features as build_physics_features,
                get_system_config,
            )

            sys_id = 0 if self.num_qubits == 3 else 1
            sys_cfg = get_system_config('3qubit' if self.num_qubits == 3 else '4qubit')

            rm = np.asarray(rho, dtype=complex)
            # Measured-only checkpoints are trained with this same contract.
            # Target rho is deliberately ignored to prevent target leakage.
            rt = rm
            expected_physics_dim = int(
                (self.deep_classifier_config or {}).get('physics_feature_dim', sys_cfg.physics_feature_dim)
            )
            image = build_rho_pair_image(rt, rm, sys_cfg)

            # At inference time we don't know the true phase_map / freq_delta / crosstalk.
            # Use zeros as placeholder; the classifier should still produce reasonable
            # latent + physics_aux from the ρ pair alone.
            phase_map = np.zeros(sys_cfg.phase_map_shape, dtype=np.float32)
            freq_delta = np.zeros(sys_cfg.freq_grid_shape, dtype=np.float32)
            crosstalk = np.zeros(sys_cfg.crosstalk_shape, dtype=np.float32)
            phys = build_physics_features(sys_cfg, rt, rm, phase_map, freq_delta, crosstalk)
            if phys.size != expected_physics_dim:
                adapter = (self.deep_classifier_config or {}).get('legacy_feature_adapter')
                if adapter == 'explicit_prefix_v1' and phys.size > expected_physics_dim:
                    phys = phys[:expected_physics_dim]
                else:
                    raise ValueError(
                        f"deep classifier physics schema width {expected_physics_dim} "
                        f"does not match measured-only feature width {phys.size}"
                    )

            with torch.no_grad():
                img_t = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
                phys_t = torch.tensor(phys, dtype=torch.float32).unsqueeze(0)
                sid_t = torch.tensor([sys_id], dtype=torch.long)
                out = self.deep_classifier_model(img_t, phys_t, sid_t)

            return {
                'physics_aux': out['physics_aux'][0].cpu().numpy().tolist(),
                'latent': out['latent'][0].cpu().numpy().tolist(),
                'severity': float(out['severity'][0].cpu()),
                'fidelity': float(out['fidelity'][0].cpu()),
                'purity': float(out['purity'][0].cpu()),
            }
        except Exception as e:  # noqa: BLE001
            print(f"Warning: deep classifier inference failed: {e}")
            return None

    def get_ideal_w_state(self) -> np.ndarray:
        """Get ideal W state density matrix for comparison"""
        if self.tomography is None:
            # Fallback: create ideal W state manually
            rho_ideal = np.zeros((self.dim, self.dim), dtype=complex)
            for i in range(self.num_qubits):
                bitstring = [0] * self.num_qubits
                bitstring[i] = 1
                idx = sum(b * (2 ** j) for j, b in enumerate(reversed(bitstring)))
                rho_ideal[idx, idx] = 1.0 / self.num_qubits
            return rho_ideal
        else:
            return self.tomography.get_ideal_density_matrix()

    def analyze_density_matrix(self, rho: np.ndarray) -> Dict:
        """
        Analyze current density matrix against ideal W state

        Args:
            rho: Current density matrix (dim x dim)

        Returns:
            Analysis results including:
            - fidelity: Fidelity to ideal W state
            - purity: Purity of the state
            - trace: Trace of the matrix
            - top_eigenstates: Top eigenstates
            - suggestions: List of correction suggestions
        """
        if not isinstance(rho, np.ndarray) or rho.shape != (self.dim, self.dim):
            return {
                'error': f'Invalid density matrix shape. Expected ({self.dim}, {self.dim})',
                'fidelity': 0.0,
                'purity': 0.0
            }

        rho_ideal = self.get_ideal_w_state()

        # Calculate fidelity
        try:
            if self.fidelity_calc is not None:
                fidelity = self.fidelity_calc.fidelity_density_matrices(rho, rho_ideal)
            else:
                # Fallback calculation
                sqrt_rho = np.sqrt(np.abs(rho))
                intermediate = sqrt_rho @ rho_ideal @ sqrt_rho
                sqrt_intermediate = np.sqrt(np.abs(intermediate))
                fidelity = float(np.abs(np.trace(sqrt_intermediate)) ** 2)
                fidelity = max(0.0, min(1.0, fidelity))
        except Exception as e:
            print(f"Error calculating fidelity: {e}")
            fidelity = 0.0

        # Calculate purity
        try:
            purity = float(np.real(np.trace(rho @ rho)))
        except Exception as e:
            print(f"Error calculating purity: {e}")
            purity = 0.0

        # Calculate trace
        trace = float(np.real(np.trace(rho)))

        # Extract top eigenstates
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(rho)
            eigenvalues = np.real(eigenvalues)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            top_eigenstates = []
            for i in range(min(3, len(eigenvalues))):
                idx = sorted_indices[i]
                top_eigenstates.append((idx, float(eigenvalues[idx])))
        except Exception as e:
            print(f"Error extracting eigenstates: {e}")
            top_eigenstates = []

        # Generate suggestions
        suggestions = []

        if fidelity < 0.9:
            suggestions.append(f"Fidelity is low ({fidelity:.3f}). Consider adjusting phase and amplitude parameters.")
        if fidelity < 0.7:
            suggestions.append("Fidelity is very low. Major parameter recalibration may be needed.")

        if purity < 0.9:
            suggestions.append(f"Purity is low ({purity:.3f}). May indicate noise or decoherence.")

        if trace < 0.99:
            suggestions.append("Trace is less than 1. Check normalization.")

        # Check dominant eigenstate
        if top_eigenstates:
            dominant_idx, dominant_val = top_eigenstates[0]

            # For W state, check if any single qubit state dominates
            if dominant_val > 0.5:
                # Calculate which qubit is in |1> state for the dominant eigenstate
                qubit_states = []
                for q in range(self.num_qubits):
                    bit_val = (dominant_idx >> (self.num_qubits - 1 - q)) & 1
                    qubit_states.append(bit_val)

                # For ideal W state, we expect roughly equal weight on 4 states
                if dominant_val > 0.8:
                    suggestions.append("State appears to be close to a product state, not entangled W state.")

        return {
            'fidelity': fidelity,
            'purity': purity,
            'trace': trace,
            'top_eigenstates': top_eigenstates,
            'suggestions': suggestions,
            'num_qubits': self.num_qubits
        }

    def get_correction_suggestions(self, current_params: Dict[str, float]) -> Dict:
        """
        Get ML-based correction suggestions for experimental parameters

        Args:
            current_params: Current experimental parameters
                           e.g., {'freq_0': 90.0, 'phase_0': 0.0, 'amp_0': 1.0}

        Returns:
            Correction suggestions including phase and amplitude adjustments
        """
        return {
            'error': 'Legacy VAE correction model is archived under script_old.',
            'suggestions': [
                'Use real-time rho diagnosis here, then run feedback_optimize from the main experiment UI for matrix deltas.'
            ],
            'current_params': dict(current_params),
        }

    def get_quick_fidelity(self, measurement_counts: Dict[str, int]) -> float:
        """
        Quick fidelity estimation from measurement counts

        Args:
            measurement_counts: Dictionary of measurement counts

        Returns:
            Estimated fidelity [0, 1]
        """
        if not measurement_counts:
            return 0.0

        total = sum(measurement_counts.values())
        if total == 0:
            return 0.0

        # For W state: sum counts for states with single |1>
        w_indices = []
        for i in range(self.num_qubits):
            idx = 2 ** (self.num_qubits - 1 - i)
            w_indices.append(idx)

        w_count = sum(measurement_counts.get(idx, 0) for idx in w_indices)
        fidelity = (w_count / total) * self.num_qubits

        return min(fidelity, 1.0)
