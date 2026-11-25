"""
Data visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class ExperimentalDataVisualizer:
    """Experimental data visualizer"""

    def __init__(self, save_dir: str = "data/plots"):
        """
        Args:
            save_dir: Image save directory
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_optimization_results(self, results: Dict, save_name: str = "optimization"):
        """
        Plot optimization results

        Args:
            results: Optimization result dictionary
            save_name: Save file name
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Fidelity vs iteration
        all_fidelities = [r['fidelity'] for r in results.get('all_results', [])]
        axes[0, 0].plot(all_fidelities)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Fidelity')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].axhline(y=results.get('best_fidelity', 0), color='r', linestyle='--',
                         label=f'Best: {results.get("best_fidelity", 0):.4f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Parameter space visualization (if multi-dimensional parameters)
        all_params = [r['parameters'] for r in results.get('all_results', [])]
        if all_params:
            param_keys = list(all_params[0].keys())
            for i, key in enumerate(param_keys[:2]):  # Show first two parameters
                param_values = [r['parameters'].get(key, 0) for r in results['all_results']]
                fidelities = [r['fidelity'] for r in results['all_results']]

                axes[0, 1].scatter(param_values, fidelities, alpha=0.5)
                axes[0, 1].set_xlabel(key)
                axes[0, 1].set_ylabel('Fidelity')
                axes[0, 1].set_title(f'{key} vs Fidelity')
                axes[0, 1].grid(True, alpha=0.3)

        # 3. Best parameters pie chart
        best_params = results.get('best_parameters', {})
        if best_params:
            axes[1, 0].pie(best_params.values(), labels=best_params.keys(), autopct='%1.1f%%')
            axes[1, 0].set_title('Best Parameters Distribution')

        # 4. Optimization history histogram
        axes[1, 1].hist(all_fidelities, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Fidelity')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Fidelity Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.save_dir / f"{save_name}_optimization.png"
        plt.savefig(filepath, dpi=150)
        plt.close()

        print(f"Plot saved to {filepath}")

    def plot_density_matrix(self, rho: np.ndarray, title: str = "Density Matrix",
                           save_name: str = "density_matrix"):
        """
        Plot density matrix

        Args:
            rho: Density matrix
            title: Title
            save_name: Save file name
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Real part
        im1 = axes[0].imshow(np.real(rho), cmap='RdBu', vmin=-1, vmax=1)
        axes[0].set_xlabel('State')
        axes[0].set_ylabel('State')
        axes[0].set_title('Real Part')
        plt.colorbar(im1, ax=axes[0])

        # Imaginary part
        im2 = axes[1].imshow(np.imag(rho), cmap='RdBu', vmin=-1, vmax=1)
        axes[1].set_xlabel('State')
        axes[1].set_ylabel('State')
        axes[1].set_title('Imaginary Part')
        plt.colorbar(im2, ax=axes[1])

        plt.suptitle(title)
        plt.tight_layout()

        filepath = self.save_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=150)
        plt.close()

    def plot_fidelity_comparison(self, fidelities: List[float],
                                labels: List[str],
                                save_name: str = "fidelity_comparison"):
        """
        Compare fidelity of different methods

        Args:
            fidelities: Fidelity list
            labels: Label list
            save_name: Save file name
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        positions = range(len(fidelities))
        ax.bar(positions, fidelities, tick_label=labels, alpha=0.7,
              edgecolor='black', color=['skyblue'] * len(fidelities))

        ax.set_ylabel('Fidelity')
        ax.set_title('Fidelity Comparison')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ideal')

        # Add value labels
        for i, (pos, fid) in enumerate(zip(positions, fidelities)):
            ax.text(pos, fid + 0.02, f'{fid:.3f}', ha='center', va='bottom')

        plt.legend()
        plt.tight_layout()

        filepath = self.save_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=150)
        plt.close()

    def plot_aod_correction_curve(self, set_freqs: np.ndarray,
                                 actual_freqs: np.ndarray,
                                 ideal_line: bool = True,
                                 save_name: str = "aod_correction"):
        """
        Plot AOD correction curve

        Args:
            set_freqs: Set frequencies
            actual_freqs: Actual frequencies
            ideal_line: Whether to plot ideal line
            save_name: Save file name
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(set_freqs, actual_freqs, alpha=0.6, edgecolors='black',
                  label='Measured data')

        if ideal_line:
            ax.plot(set_freqs, set_freqs, 'r--', alpha=0.5, label='Ideal (1:1)')

        ax.set_xlabel('Set Frequency (MHz)')
        ax.set_ylabel('Actual Frequency (MHz)')
        ax.set_title('AOD Frequency Response')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(set_freqs, actual_freqs, 1)
        p = np.poly1d(z)
        ax.plot(set_freqs, p(set_freqs), "b-", alpha=0.8, label='Linear fit')

        plt.tight_layout()

        filepath = self.save_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=150)
        plt.close()

    def plot_wstate_components(self, state_vector: np.ndarray,
                              save_name: str = "wstate_components"):
        """
        Plot W-state component amplitudes

        Args:
            state_vector: State vector
            save_name: Save file name
        """
        n_qubits = int(np.log2(len(state_vector)))

        fig, axes = plt.subplots(1, n_qubits, figsize=(4 * n_qubits, 4))

        for i in range(n_qubits):
            # Find states where the i-th qubit is 1
            indices = [j for j in range(len(state_vector))
                      if (j >> (n_qubits - 1 - i)) & 1]

            amplitudes = [np.abs(state_vector[idx])**2 for idx in indices]
            phases = [np.angle(state_vector[idx]) for idx in indices]

            if n_qubits == 1:
                ax = axes
            else:
                ax = axes[i]

            ax.bar(range(len(amplitudes)), amplitudes, alpha=0.7,
                  edgecolor='black', color='skyblue')
            ax.set_xlabel(f'Index (i=th qubit = 1)')
            ax.set_ylabel('Probability |α|²')
            ax.set_title(f'{i}th Qubit')
            ax.set_ylim(0, 1)

        plt.tight_layout()

        filepath = self.save_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=150)
        plt.close()