"""
Data loading and processing utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import os


class ExperimentDataLoader:
    """Experimental data loader"""

    def __init__(self, data_dir: str = "data/training_sets"):
        """
        Args:
            data_dir: Data storage directory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment_data(self, experiment_id: str,
                            data: Dict) -> str:
        """
        Save experimental data

        Args:
            experiment_id: Experiment ID
            data: Data dictionary (parameters, results, etc.)

        Returns:
            Saved file path
        """
        filepath = self.data_dir / f"{experiment_id}.json"

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return str(filepath)

    def load_experiment_data(self, experiment_id: str) -> Optional[Dict]:
        """
        Load experimental data

        Args:
            experiment_id: Experiment ID

        Returns:
            Experimental data dictionary or None
        """
        filepath = self.data_dir / f"{experiment_id}.json"

        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            return json.load(f)

    def load_all_experiments(self, prefix: str = "") -> List[Dict]:
        """
        Load all experimental data

        Args:
            prefix: Experiment ID prefix filter

        Returns:
            List of experimental data
        """
        experiments = []

        for filepath in self.data_dir.glob("*.json"):
            if prefix and not filepath.stem.startswith(prefix):
                continue

            with open(filepath, 'r') as f:
                experiments.append(json.load(f))

        return experiments

    def create_training_dataset(self, experiments: List[Dict],
                                target_key: str = 'fidelity') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training dataset

        Args:
            experiments: List of experimental data
            target_key: Target key name

        Returns:
            (X, y) dataset
        """
        X_list = []
        y_list = []

        for exp in experiments:
            params = exp.get('best_parameters', exp.get('parameters', {}))
            fidelity = exp.get(target_key, 0)

            # Ensure parameters are a dictionary
            if isinstance(params, dict):
                X = np.array([params.get(f'freq_offsets_{i}', 0) for i in range(4)] +
                            [params.get(f'phase_offsets_{i}', 0) for i in range(4)])
            elif isinstance(params, (list, np.ndarray)):
                X = np.array(params)
            else:
                continue

            X_list.append(X)
            y_list.append(fidelity)

        return np.array(X_list), np.array(y_list)


class AWGSequenceLoader:
    """AWG sequence loader"""

    @staticmethod
    def load_awg_file(filepath: str) -> Dict[str, np.ndarray]:
        """
        Load AWG sequence file

        Args:
            filepath: File path

        Returns:
            Dictionary containing channel_data
        """
        data = {}

        # Read file
        header = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    header.append(line.strip('#').strip())
                else:
                    break

        # Parse data
        raw_data = np.loadtxt(filepath, skiprows=len(header) + 1, delimiter=',')

        if raw_data.ndim == 2 and raw_data.shape[1] >= 3:
            # Assume format: analog, digital1, digital2
            data['analog'] = raw_data[:, 0]
            data['digital1'] = raw_data[:, 1]
            data['digital2'] = raw_data[:, 2]

        return data

    @staticmethod
    def load_channel_data(channel_dir: str, channel_name: str) -> Optional[np.ndarray]:
        """
        Load data for specified channel

        Args:
            channel_dir: Channel directory
            channel_name: Channel name

        Returns:
            Channel data (analog values)
        """
        filepath = os.path.join(channel_dir, f"{channel_name}_T*.txt")

        import glob
        files = glob.glob(filepath)

        if not files:
            return None

        # Load latest file
        latest_file = max(files, key=os.path.getctime)
        return AWGSequenceLoader.load_awg_file(latest_file)['analog']


class FrequencyCalibrationLoader:
    """Frequency calibration data loader"""

    @staticmethod
    def load_calibration_data(filepath: str) -> Dict[str, List[Tuple[float, float]]]:
        """
        Load AOD calibration data

        Expected CSV format:
        set_freq_1, actual_freq_1, set_freq_2, actual_freq_2, ...

        Args:
            filepath: Calibration data file path

        Returns:
            Calibration data dictionary {channel: [(set_freq, actual_freq), ...]}
        """
        df = pd.read_csv(filepath)

        # Extract data for each channel
        calibration = {}
        n_channels = len([c for c in df.columns if c.startswith('set_freq')])

        for i in range(n_channels):
            set_key = f'set_freq_{i+1}'
            actual_key = f'actual_freq_{i+1}'

            if set_key in df.columns and actual_key in df.columns:
                calibration[f'channel_{i+1}'] = list(zip(
                    df[set_key], df[actual_key]
                ))

        return calibration

    @staticmethod
    def generate_calibration_grid(freq_range: Tuple[float, float],
                                  n_points: int = 20) -> pd.DataFrame:
        """
        Generate frequency scan grid data

        Args:
            freq_range: Frequency range (min, max)
            n_points: Number of scan points per frequency

        Returns:
            DataFrame containing frequency grid
        """
        freqs = np.linspace(freq_range[0], freq_range[1], n_points)

        # Generate grid
        freq_grid = np.meshgrid(freqs, freqs)

        data = {
            'set_freq_1': freq_grid[0].flatten(),
            'set_freq_2': freq_grid[1].flatten(),
        }

        return pd.DataFrame(data)