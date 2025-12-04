import os
from typing import Optional

import numpy as np


class LogisticClassifier:
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 200,
        regularization: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        random_seed: int = 42,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.random_seed = random_seed
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def _initialize_weights(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.weights = rng.normal(scale=0.01, size=n_features)
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        n_samples, n_features = X.shape
        if self.weights is None or self.weights.shape[0] != n_features:
            self._initialize_weights(n_features)
        m = np.zeros(n_features, dtype=float)
        v = np.zeros(n_features, dtype=float)
        m_b = 0.0
        v_b = 0.0
        for epoch in range(1, self.epochs + 1):
            logits = X.dot(self.weights) + self.bias
            probs = self._sigmoid(logits)
            error = probs - y
            grad_w = X.T.dot(error) / n_samples + self.regularization * self.weights
            grad_b = np.mean(error)
            m = self.beta1 * m + (1 - self.beta1) * grad_w
            v = self.beta2 * v + (1 - self.beta2) * (grad_w ** 2)
            m_hat = m / (1 - self.beta1**epoch)
            v_hat = v / (1 - self.beta2**epoch)
            self.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            m_b = self.beta1 * m_b + (1 - self.beta1) * grad_b
            v_b = self.beta2 * v_b + (1 - self.beta2) * (grad_b**2)
            m_b_hat = m_b / (1 - self.beta1**epoch)
            v_b_hat = v_b / (1 - self.beta2**epoch)
            self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not trained")
        logits = X.dot(self.weights) + self.bias
        return self._sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, checkpoint_dir: str) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if self.weights is None:
            raise ValueError("Cannot save an untrained classifier")
        np.save(os.path.join(checkpoint_dir, "logistic_weights.npy"), self.weights)
        np.save(os.path.join(checkpoint_dir, "logistic_bias.npy"), np.array([self.bias]))

    def load(self, checkpoint_dir: str) -> None:
        weight_path = os.path.join(checkpoint_dir, "logistic_weights.npy")
        bias_path = os.path.join(checkpoint_dir, "logistic_bias.npy")
        if not os.path.exists(weight_path) or not os.path.exists(bias_path):
            raise FileNotFoundError("Saved logistic parameters missing")
        self.weights = np.load(weight_path)
        self.bias = float(np.load(bias_path)[0])
