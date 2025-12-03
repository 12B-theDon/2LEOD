#!/usr/bin/env python3
"""Evaluation helper for logistic/ensemble accuracy."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

from models.ensemble import ensemble_prediction
from models.LogisticRegression import LogisticRegression


def evaluate_split(
    name: str,
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    meta: Sequence[Mapping[str, int]],
    logistic_model: LogisticRegression,
    linear_map: Mapping[Tuple[int, int], Mapping[str, float]],
    threshold: float,
) -> Mapping[str, float | None]:
    logistic_correct = 0
    ensemble_correct = 0
    for vector, label, info in zip(features, labels, meta):
        logistic_pred = logistic_model.predict(vector)
        logistic_correct += int(logistic_pred == label)
        probs = logistic_model.predict_proba(vector)
        linear_info = linear_map.get((info["frame_index"], info["scan_index"]))
        ensemble_pred = ensemble_prediction(probs, linear_info, threshold)
        ensemble_correct += int(ensemble_pred == label)
    total = len(labels)
    return {
        "name": name,
        "total_points": total,
        "logistic_accuracy": logistic_correct / total if total else None,
        "ensemble_accuracy": ensemble_correct / total if total else None,
    }
