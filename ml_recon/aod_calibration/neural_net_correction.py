"""
Neural Network Based AOD Nonlinear Correction
Use deep learning model to learn AOD frequency mapping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import pickle
import json
import os

from .nonlinear_model import AODNonlinearModel, AODCalibrationData


class AODCorrectionNN(nn.Module):
    """AOD frequency correction neural network"""

    def __init__(self, input_dim: int = 1, hidden_dims: List[int] = [64, 128, 64],
                 output_dim: int = 1):
        """
        Args:
            input_dim: Input dimension (set frequency)
            hidden_dims: Hidden layer dimension list
            output_dim: Output dimension (corrected frequency)
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AODCorrectionNNModel(AODNonlinearModel):
    """Neural network AOD calibration model"""

    def __init__(self, freq_range: Tuple[float, float] = (84.0, 121.5),
                 hidden_dims: List[int] = [64, 128, 64],
                 learning_rate: float = 0.001, device: str = None):
        """
        Args:
            freq_range: Frequency range (min, max) in MHz
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            device: Training device ('cpu' or 'cuda')
        """
        super().__init__(freq_range)

        self.freq_min, self.freq_max = freq_range
        self.freq_range_val = freq_range
        self.hidden_dims = hidden_dims
        self.lr = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model for each channel
        self.models: Dict[int, Tuple[AODCorrectionNN, optim.Optim, np.ndarray]] = {}

        # Normalization parameters
        self.freq_min_train = freq_range[0]
        self.freq_max_train = freq_range[1]

    def _normalize(self, freq: np.ndarray) -> np.ndarray:
        """Normalize frequency to [0, 1]"""
        return (freq - self.freq_min_train) / (self.freq_max_train - self.freq_min_train)

    def _denormalize(self, freq: np.ndarray) -> np.ndarray:
        """Denormalize from normalized"""
        return freq * (self.freq_max_train - self.freq_min_train) + self.freq_min_train

    def _create_channel_model(self, channel_idx: int):
        """Create neural network for single channel"""
        model = AODCorrectionNN(
            input_dim=1,
            hidden_dims=self.hidden_dims,
            output_dim=1
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        self.models[channel_idx] = (model, optimizer, None)
        return model

    def train(self, calibration_data: AODCalibrationData, epochs: int = 500,
              batch_size: int = 32, validation_split: float = 0.2,
              early_stop_patience: int = 50):
        """
        Train neural network model

        Args:
            calibration_data: Calibration dataset
            epochs: Maximum training epochs
            batch_size: Batch size
            validation_split: Validation set ratio
            early_stop_patience: Early stopping patience
        """
        # Collect training data for each channel
        all_train_data = {}
        for ch_idx in range(calibration_data.num_channels):
            set_freqs, actual_freqs, _ = calibration_data.channel_arrays(ch_idx)

            # Normalize
            set_freqs_norm = self._normalize(set_freqs).reshape(-1, 1)
            actual_freqs_norm = self._normalize(actual_freqs).reshape(-1, 1)

            # Split train/validation sets
            n = len(set_freqs_norm)
            indices = np.random.permutation(n)
            split = int(n * (1 - validation_split))
            split = min(max(split, 1), max(n - 1, 1))

            train_idx = indices[:split]
            val_idx = indices[split:] if split < n else indices[:split]

            X_train = set_freqs_norm[train_idx]
            y_train = actual_freqs_norm[train_idx]
            X_val = set_freqs_norm[val_idx]
            y_val = actual_freqs_norm[val_idx]

            all_train_data[ch_idx] = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val
            }

            # Create model for this channel
            self._create_channel_model(ch_idx)

        # Train each channel
        for ch_idx in range(calibration_data.num_channels):
            model, optimizer, _ = self.models[ch_idx]
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(all_train_data[ch_idx]['X_train']),
                torch.FloatTensor(all_train_data[ch_idx]['y_train'])
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=max(2, min(int(batch_size), len(train_dataset))), shuffle=True
            )

            X_val = torch.FloatTensor(all_train_data[ch_idx]['X_val'])
            y_val = torch.FloatTensor(all_train_data[ch_idx]['y_val'])

            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            print(f"\nTraining channel {ch_idx}...")
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = nn.MSELoss()(output, y_batch)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_output = model(X_val)
                    val_loss = nn.MSELoss()(val_output, y_val)

                scheduler.step(val_loss)

                if (epoch + 1) % 50 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.6e}, "
                          f"Val Loss: {val_loss:.6e}")

                # Early stopping check
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # Restore best model
            if best_model_state:
                model.load_state_dict(best_model_state)

            self.models[ch_idx] = (model, optimizer, best_model_state)

        self.is_trained = True
        print("\nTraining completed!")

    def correct(self, set_freqs: np.ndarray) -> np.ndarray:
        """Calculate corrected actual frequency"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        set_freqs_norm = self._normalize(set_freqs).reshape(-1, 1)

        corrected_norm = []
        for ch_idx in range(len(set_freqs)):
            model, _, _ = self.models[ch_idx]
            model.eval()
            X = torch.FloatTensor([[float(set_freqs_norm[ch_idx, 0])]]).to(self.device)
            with torch.no_grad():
                output = model(X)
            corrected_norm.append(float(output.detach().cpu().item()))

        # Denormalize
        corrected = self._denormalize(np.array(corrected_norm, dtype=float))
        return np.clip(corrected, self.freq_min_train, self.freq_max_train)

    def invert(self, target_freqs: np.ndarray) -> np.ndarray:
        """Calculate required set frequency from target frequency (using optimization)"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        target_arr = np.asarray(target_freqs, dtype=float).reshape(-1)
        set_freqs = np.copy(target_arr)
        set_freqs_norm = self._normalize(set_freqs).reshape(-1, 1)

        for ch_idx in range(len(target_arr)):
            # Optimize for each channel
            target_norm = float(self._normalize(np.array([target_arr[ch_idx]], dtype=float))[0])

            # Use gradient descent to search for best input
            X_opt = nn.Parameter(torch.FloatTensor([[float(set_freqs_norm[ch_idx, 0])]]).to(self.device))
            target_tensor = torch.FloatTensor([[target_norm]]).to(self.device)
            optimizer_opt = optim.Adam([X_opt], lr=0.1)
            model, _, _ = self.models[ch_idx]
            model.eval()

            for _ in range(200):
                optimizer_opt.zero_grad()
                output = model(X_opt)
                loss = nn.MSELoss()(output, target_tensor)
                loss.backward()
                optimizer_opt.step()
                with torch.no_grad():
                    X_opt.clamp_(0.0, 1.0)

            set_freqs[ch_idx] = float(X_opt.detach().cpu().item())

        return np.clip(self._denormalize(set_freqs), self.freq_min_train, self.freq_max_train)

    def estimate_uncertainty(self, set_freqs: np.ndarray) -> np.ndarray:
        """Estimate uncertainty using Monte Carlo Dropout"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        set_freqs_norm = self._normalize(set_freqs).reshape(-1, 1)

        uncertainties = []
        for ch_idx in range(len(set_freqs)):
            model, _, _ = self.models[ch_idx]
            X = torch.FloatTensor([[float(set_freqs_norm[ch_idx, 0])]]).to(self.device)

            # Monte Carlo dropout sampling
            MC_samples = 50
            predictions = []
            for _ in range(MC_samples):
                model.train()  # Enable dropout
                with torch.no_grad():
                    output = model(X)
                predictions.append(float(output.detach().cpu().item()))

            # Compute standard deviation
            std = np.std(predictions)
            # Convert to actual frequency unit
            uncertainties.append(std * (self.freq_max_train - self.freq_min_train))

        return np.array(uncertainties)

    def save(self, filepath: str):
        """Save trained model"""
        save_dir = os.path.dirname(filepath)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        save_data = {
            'freq_range': self.freq_range_val,
            'hidden_dims': self.hidden_dims,
            'lr': self.lr,
            'device': self.device,
            'models_state_dicts': {
                str(k): v[2] if v[2] else v[0].state_dict()
                for k, v in self.models.items()
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: str = None) -> 'AODCorrectionNNModel':
        """Load trained model"""
        with open(filepath, 'rb') as f:
            load_data = pickle.load(f)

        model = cls(
            freq_range=tuple(load_data['freq_range']),
            hidden_dims=load_data['hidden_dims'],
            learning_rate=load_data['lr'],
            device=device or load_data['device']
        )

        model.models = {}
        for ch_idx, state_dict in load_data['models_state_dicts'].items():
            ch_idx = int(ch_idx)
            channel_model = model._create_channel_model(ch_idx)
            channel_model.load_state_dict(state_dict)
            channel_model = channel_model.to(model.device)
            optimizer = optim.Adam(channel_model.parameters(), lr=model.lr, weight_decay=1e-4)
            model.models[ch_idx] = (channel_model, optimizer, state_dict)

        model.is_trained = True
        print(f"Model loaded from {filepath}")
        return model


def train_from_csv(filepath: str, output_model: str, freq_range: Tuple[float, float] = (84.0, 121.5),
                   epochs: int = 500, hidden_dims: List[int] = [64, 128, 64]):
    """
    Train model from CSV file

    CSV format requirements:
    set_freq_1, set_freq_2, actual_freq_1, actual_freq_2, ...
    """
    import pandas as pd

    df = pd.read_csv(filepath)

    # Create calibration data
    cal_data = AODCalibrationData()
    cal_data.freq_range = freq_range

    n_channels = len([c for c in df.columns if c.startswith('set_freq')])
    cal_data.num_channels = n_channels

    for _, row in df.iterrows():
        set_freqs = [row[f'set_freq_{i+1}'] for i in range(n_channels)]
        actual_freqs = [row[f'actual_freq_{i+1}'] for i in range(n_channels)]
        cal_data.add_vector_point(set_freqs, actual_freqs)

    # Train model
    model = AODCorrectionNNModel(freq_range=freq_range, hidden_dims=hidden_dims)
    model.train(cal_data, epochs=epochs)

    # Save model
    model.save(output_model)

    return model
