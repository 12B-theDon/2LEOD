from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from config import LOGREG_DECISION_THRESHOLD, MODEL_SAVE_PATH


class RealtimeEncoder:
    """
    Encoder helper that exposes the scaler output latent, the opponent probability,
    and a running cosine similarity against the previous timestep latent.
    """

    def __init__(self, model_path: str | Path = MODEL_SAVE_PATH) -> None:
        bundle = joblib.load(model_path)
        self.clf_pipe = bundle["clf_pipe"]
        self.threshold = bundle.get("threshold", LOGREG_DECISION_THRESHOLD)
        self._prev_latent: np.ndarray | None = None

    def encode(self, feature_vector: np.ndarray) -> dict:
        """
        Transform a raw observation into:
         * standardized latent used for cosine similarity
         * classifier probability for opponent vs. non-opponent.
        """
        feature_array = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        scaler = self.clf_pipe.named_steps["scaler"]
        clf = self.clf_pipe.named_steps["clf"]
        latent = scaler.transform(feature_array)
        probability = float(clf.predict_proba(latent)[0, 1])
        is_opponent = probability >= self.threshold

        cosine_similarity = None
        if self._prev_latent is not None:
            # Cosine similarity compares the current latent to the previous latent for tracking.
            numer = float(np.dot(latent, self._prev_latent.T))
            denom = np.linalg.norm(latent) * np.linalg.norm(self._prev_latent)
            if denom > 0:
                cosine_similarity = numer / denom

        self._prev_latent = latent.copy()

        return {
            "latent": latent.flatten(),
            "opponent_probability": probability,
            "cosine_similarity": cosine_similarity,
            "decision_threshold": self.threshold,
            "is_opponent": is_opponent,
        }
