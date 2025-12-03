#!/usr/bin/env python3
"""Helpers that combine logistic probabilities with linear residuals."""

from __future__ import annotations

from typing import Mapping, Sequence


def wall_confidence(distance: float, threshold: float) -> float:
    """Normalize the distance to the closest fitted wall to [0, 1]."""
    if threshold <= 0:
        return 0.0
    clipped = max(0.0, min(distance, threshold))
    return 1.0 - clipped / threshold


def ensemble_prediction(
    probs: Sequence[float],
    linear_info: Mapping[str, float] | None,
    threshold: float,
    wall_weight: float = 0.5,
) -> int:
    """Blend logistic probabilities with the linear wall score."""
    scores = list(probs)
    if linear_info is not None:
        distance = float(linear_info.get("distance_to_line", float("inf")))
        scores[0] += wall_confidence(distance, threshold) * wall_weight
    return max(range(len(scores)), key=lambda idx: scores[idx])
