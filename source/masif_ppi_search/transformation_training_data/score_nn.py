import os
import numpy as np
import torch
import torch.nn as nn

"""
score_nn.py: Class to score protein complex alignments based on a pre-trained
neural network (used for MaSIF-search's second stage protocol).
Freyr Sverrisson and Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""


class ScoreNN(nn.Module):
    """1D conv network for scoring surface patch alignments.

    Architecture mirrors the original Keras model:
    6x Conv1D(kernels 3→8→16→32→64→128→256, kernel_size=1) with BN + ReLU,
    GlobalAveragePooling, then 6x Dense layers down to 2 outputs (softmax).

    Input:  (batch, seq_len, 3)  — 3 geometric features per surface point.
    Output: (batch, 2)           — [negative_prob, positive_prob].
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)
        self.restore_model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 3) → (batch, 3, seq_len) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.mean(dim=2)  # GlobalAveragePooling1D
        return torch.softmax(self.fc(x), dim=1)

    def restore_model(self):
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'nn_score', 'trained_model.pt')
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location=self._device, weights_only=True))
        else:
            print(f'Warning: ScoreNN weights not found at {model_path}. '
                  'Running with random weights.')

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on a batch of patch feature arrays.

        Args:
            features: numpy array of shape (batch, seq_len, 3).
        Returns:
            numpy array of shape (batch, 2) — softmax [neg, pos] probabilities.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self._device)
            return self.forward(x).cpu().numpy()

    def train_model(self, features: np.ndarray, labels: np.ndarray,
                    n_negatives: int, n_positives: int):
        """Fine-tune the model on provided features / labels."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [1.0 / n_negatives, 1.0 / n_positives], dtype=torch.float32
            ).to(self._device)
        )
        self.train()
        x = torch.tensor(features, dtype=torch.float32).to(self._device)
        y = torch.tensor(labels.ravel(), dtype=torch.long).to(self._device)
        for epoch in range(50):
            optimizer.zero_grad()
            loss = loss_fn(self.forward(x), y)
            loss.backward()
            optimizer.step()
        save_dir = os.path.join(os.path.dirname(__file__), 'models', 'nn_score')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, 'trained_model.pt'))
