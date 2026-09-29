import numpy as np
from tqdm.auto import trange

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin


class DeepRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
        dropout=0.05,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        device=None,
        verbose=False
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.model_ = None
        self.in_features_ = None
        self.out_features_ = None

    def _build_model(self, in_features, out_features):
        layers = []

        # Input layer
        layers.extend([
            nn.Linear(in_features, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            # nn.Dropout(self.dropout)
        ])

        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                # nn.Dropout(self.dropout)
            ])

        # Output layer
        layers.append(
            nn.Linear(self.hidden_dim, out_features)
        )

        return nn.Sequential(*layers)

    def fit(self, X, y):
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Handle shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.in_features_ = X.shape[1]
        self.out_features_ = y.shape[1]

        # Build model
        self.model_ = self._build_model(self.in_features_, self.out_features_).to(self.device)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Training setup
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Training loop
        self.model_.train()
        iterator = (trange if self.verbose else range)(self.epochs)
        for epoch in iterator:
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model_(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if self.verbose:
                iterator.set_description(f"Loss: {epoch_loss / len(dataloader):.4f}")

        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        X = np.array(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(X_tensor).cpu().numpy()

        # Return 1D array if single output
        if self.out_features_ == 1:
            predictions = predictions.ravel()

        return predictions
