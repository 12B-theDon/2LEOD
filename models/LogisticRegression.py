#!/usr/bin/env python3
"""Lightweight softmax logistic regression for multi-class scan labeling."""

from __future__ import annotations

import math
from typing import List, Sequence


class LogisticRegression:
    """Simple softmax logistic regression without extra dependencies."""

    def __init__(
        self,
        num_features: int,
        num_classes: int = 4,
        learning_rate: float = 0.01,
    ) -> None:
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weights: List[List[float]] = [
            [0.0 for _ in range(num_features)] for _ in range(num_classes)
        ]
        self.bias: List[float] = [0.0] * num_classes

    def _softmax(self, logits: Sequence[float]) -> List[float]:
        max_logit = max(logits)
        exps = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exps)
        if total == 0.0:
            return [1.0 / len(logits)] * len(logits)
        return [value / total for value in exps]

    def _dot(self, features: Sequence[float], class_idx: int) -> float:
        weight_row = self.weights[class_idx]
        return sum(w * feat for w, feat in zip(weight_row, features)) + self.bias[
            class_idx
        ]

    def predict_proba(self, features: Sequence[float]) -> List[float]:
        logits = [self._dot(features, class_idx) for class_idx in range(self.num_classes)]
        return self._softmax(logits)

    def predict(self, features: Sequence[float]) -> int:
        probs = self.predict_proba(features)
        return max(range(len(probs)), key=lambda idx: probs[idx])

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
        epochs: int = 40,
        verbose: bool = False,
    ) -> List[float]:
        history: List[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for vector, label in zip(features, labels):
                probs = self.predict_proba(vector)
                loss = -math.log(max(probs[label], 1e-12))
                epoch_loss += loss
                for class_idx in range(self.num_classes):
                    gradient = probs[class_idx] - (1.0 if class_idx == label else 0.0)
                    for fid in range(self.num_features):
                        self.weights[class_idx][fid] -= (
                            self.learning_rate * gradient * vector[fid]
                        )
                    self.bias[class_idx] -= self.learning_rate * gradient
            epoch_loss /= max(1, len(features))
            history.append(epoch_loss)
            if verbose:
                print(f"[Logistic] epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")
        return history

    def to_dict(self) -> dict:
        return {
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LogisticRegression":
        instance = cls(
            num_features=int(payload["num_features"]),
            num_classes=int(payload["num_classes"]),
            learning_rate=float(payload.get("learning_rate", 0.01)),
        )
        instance.weights = [list(map(float, row)) for row in payload["weights"]]
        instance.bias = [float(value) for value in payload["bias"]]
        return instance
