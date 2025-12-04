import os
from typing import Optional

import numpy as np


class PCAEncoder:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        centered = X - self.mean_
        # Use SVD for numerical stability
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[: self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCAEncoder is not fitted")
        X = np.asarray(X, dtype=float)
        centered = X - self.mean_
        return centered.dot(self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, checkpoint_dir: str) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if self.mean_ is None or self.components_ is None:
            raise ValueError("Cannot save PCA before fitting")
        np.save(os.path.join(checkpoint_dir, "pca_mean.npy"), self.mean_)
        np.save(os.path.join(checkpoint_dir, "pca_components.npy"), self.components_)

    def load(self, checkpoint_dir: str) -> None:
        mean_path = os.path.join(checkpoint_dir, "pca_mean.npy")
        comp_path = os.path.join(checkpoint_dir, "pca_components.npy")
        if not os.path.exists(mean_path) or not os.path.exists(comp_path):
            raise FileNotFoundError("PCA checkpoint files missing")
        self.mean_ = np.load(mean_path)
        self.components_ = np.load(comp_path)
