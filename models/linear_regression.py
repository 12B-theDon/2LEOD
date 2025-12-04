import os
from typing import Optional

import numpy as np


class LinearRegressor:
    def __init__(self, regularization: float = 1e-5):
        self.regularization = regularization
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        n_samples, n_features = X.shape
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        reg_matrix = self.regularization * np.eye(n_features + 1)
        reg_matrix[-1, -1] = 0.0
        lhs = X_aug.T.dot(X_aug) + reg_matrix
        rhs = X_aug.T.dot(y)
        coeffs = np.linalg.solve(lhs, rhs)
        self.weights = coeffs[:-1]
        self.bias = coeffs[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not trained")
        X = np.asarray(X, dtype=float)
        return X.dot(self.weights) + self.bias

    def save(self, checkpoint_dir: str) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if self.weights is None:
            raise ValueError("Cannot save an untrained regressor")
        np.save(os.path.join(checkpoint_dir, "linear_weights.npy"), self.weights)
        np.save(os.path.join(checkpoint_dir, "linear_bias.npy"), np.array([self.bias]))

    def load(self, checkpoint_dir: str) -> None:
        w_path = os.path.join(checkpoint_dir, "linear_weights.npy")
        b_path = os.path.join(checkpoint_dir, "linear_bias.npy")
        if not os.path.exists(w_path) or not os.path.exists(b_path):
            raise FileNotFoundError("Saved linear parameters missing")
        self.weights = np.load(w_path)
        self.bias = float(np.load(b_path)[0])
