#!/usr/bin/env python3
"""Linear-regression utility that fits two wall candidates per frame."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _normalize_angle_deg(angle: float) -> float:
    return angle % 360.0


def _circular_diff(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff


def _circular_mean(angles: Iterable[float]) -> float:
    total_x = 0.0
    total_y = 0.0
    count = 0
    for angle in angles:
        rad = math.radians(angle)
        total_x += math.cos(rad)
        total_y += math.sin(rad)
        count += 1
    if count == 0:
        return 0.0
    if abs(total_x) < 1e-6 and abs(total_y) < 1e-6:
        return 0.0
    return _normalize_angle_deg(math.degrees(math.atan2(total_y, total_x)))


def _cluster_angles(angle_deg: Sequence[float], k: int = 2, max_iter: int = 10) -> List[int]:
    if not angle_deg:
        return []
    k = max(1, min(k, len(angle_deg)))
    centers = [_normalize_angle_deg(angle_deg[0])]
    for idx in range(1, k):
        centers.append(_normalize_angle_deg(angle_deg[0] + 180.0 * idx))
    assignments = [0] * len(angle_deg)
    for _ in range(max_iter):
        groups: List[List[float]] = [[] for _ in centers]
        for index, angle in enumerate(angle_deg):
            diffs = [abs(_circular_diff(angle, center)) for center in centers]
            best = min(range(len(diffs)), key=lambda idx: diffs[idx])
            assignments[index] = best
            groups[best].append(angle)
        changed = False
        new_centers: List[float] = []
        for idx in range(len(centers)):
            group = groups[idx]
            if not group:
                new_centers.append(centers[idx])
                continue
            mean_angle = _circular_mean(group)
            if abs(_circular_diff(mean_angle, centers[idx])) > 1e-3:
                changed = True
            new_centers.append(mean_angle)
        centers = new_centers
        if not changed:
            break
    final_assignments: List[int] = []
    for angle in angle_deg:
        diffs = [abs(_circular_diff(angle, center)) for center in centers]
        final_assignments.append(min(range(len(diffs)), key=lambda idx: diffs[idx]))
    return final_assignments


def _fit_line(points: Sequence[Tuple[float, float]]) -> Mapping[str, Any] | None:
    if len(points) < 2:
        return None
    avg_x = sum(p[0] for p in points) / len(points)
    avg_y = sum(p[1] for p in points) / len(points)
    sxx = 0.0
    sxy = 0.0
    for x, y in points:
        dx = x - avg_x
        dy = y - avg_y
        sxx += dx * dx
        sxy += dx * dy
    # Determine if the best-fit line is vertical
    is_vertical = abs(sxx) < 1e-6
    if is_vertical:
        slope = None
        intercept = avg_x
        direction = (0.0, 1.0)
    else:
        slope = sxy / sxx
        intercept = avg_y - slope * avg_x
        direction = (1.0, slope)
        norm = math.hypot(direction[0], direction[1])
        direction = (direction[0] / norm, direction[1] / norm)
    angle_deg = _normalize_angle_deg(math.degrees(math.atan2(direction[1], direction[0])))
    return {
        "point": [avg_x, avg_y],
        "direction": [direction[0], direction[1]],
        "angle_deg": angle_deg,
        "slope": slope,
        "intercept": intercept,
        "is_vertical": is_vertical,
    }


def _distance_to_line(point: Tuple[float, float], model: Mapping[str, Any]) -> float:
    px, py = model["point"]
    dx, dy = model["direction"]
    rel_x = point[0] - px
    rel_y = point[1] - py
    return abs(rel_x * dy - rel_y * dx)


def fit_two_line_models(
    points: Sequence[Tuple[float, float]],
    angle_deg: Sequence[float],
    max_models: int = 2,
) -> Tuple[List[Mapping[str, Any]], List[int]]:
    if len(points) < 2:
        return [], []
    cluster_count = max(1, min(max_models, len(points)))
    assignments = _cluster_angles(angle_deg, k=cluster_count)
    models: List[Mapping[str, Any]] = []
    for line_id in range(cluster_count):
        indices = [idx for idx, cluster in enumerate(assignments) if cluster == line_id]
        if len(indices) < 2:
            continue
        model = _fit_line([points[idx] for idx in indices])
        if not model:
            continue
        model = dict(model)
        model["cluster_id"] = line_id
        model["support"] = len(indices)
        models.append(model)
    if not models:
        fallback = _fit_line(points)
        if fallback:
            fallback = dict(fallback)
            fallback["cluster_id"] = 0
            fallback["support"] = len(points)
            models.append(fallback)
    return models, assignments


def classify_points(
    points: Sequence[Tuple[float, float]],
    models: Sequence[Mapping[str, Any]],
    threshold: float,
) -> Tuple[List[int], List[float], List[bool]]:
    count = len(points)
    assignments = [-1] * count
    distances = [float("inf")] * count
    if not models:
        return assignments, distances, [False] * count
    for idx, point in enumerate(points):
        best_line = -1
        best_distance = float("inf")
        for model_idx, model in enumerate(models):
            distance = _distance_to_line(point, model)
            if distance < best_distance:
                best_line = model_idx
                best_distance = distance
        assignments[idx] = best_line
        distances[idx] = best_distance
    mask = [distances[i] <= threshold for i in range(count)]
    return assignments, distances, mask


def classify_frame(
    frame_entries: Sequence[Mapping[str, Any]], threshold: float = 0.25
) -> Mapping[str, Any]:
    if not frame_entries:
        return {
            "line_models": [],
            "wall_scan_indexes": [],
            "non_wall_scan_indexes": [],
            "total_points": 0,
            "wall_ratio": 0.0,
        }
    points: List[Tuple[float, float]] = []
    angle_deg: List[float] = []
    for entry in frame_entries:
        distance = entry["distance"]
        angle_rad = entry["angle_rad"]
        x = distance * math.cos(angle_rad)
        y = distance * math.sin(angle_rad)
        points.append((x, y))
        angle_deg.append(_normalize_angle_deg(math.degrees(angle_rad)))
    models, _ = fit_two_line_models(points, angle_deg, max_models=2)
    assignments, distances, mask = classify_points(points, models, threshold)
    wall_scan_indexes = [
        int(frame_entries[idx]["scan_index"]) for idx, is_wall in enumerate(mask) if is_wall
    ]
    non_wall_scan_indexes = [
        int(frame_entries[idx]["scan_index"]) for idx, is_wall in enumerate(mask) if not is_wall
    ]
    line_models: List[Mapping[str, Any]] = []
    for idx, model in enumerate(models):
        total_assigned = sum(1 for assignment in assignments if assignment == idx)
        wall_assigned = sum(
            1
            for point_idx, assignment in enumerate(assignments)
            if assignment == idx and mask[point_idx]
        )
        line_models.append(
            {
                "angle_deg": model["angle_deg"],
                "point": model["point"],
                "direction": model["direction"],
                "support_total": total_assigned,
                "support_wall": wall_assigned,
                "cluster_id": model.get("cluster_id", idx),
                "slope": model.get("slope"),
                "intercept": model.get("intercept"),
                "is_vertical": bool(model.get("is_vertical")),
            }
        )
    total_points = len(points)
    wall_ratio = (len(wall_scan_indexes) / total_points) if total_points else 0.0
    point_info: List[Mapping[str, Any]] = []
    for idx, entry in enumerate(frame_entries):
        point_info.append(
            {
                "scan_index": int(entry["scan_index"]),
                "distance": float(entry["distance"]),
                "angle_rad": float(entry["angle_rad"]),
                "line_id": int(assignments[idx]) if assignments[idx] >= 0 else -1,
                "distance_to_line": float(distances[idx]),
                "is_wall": bool(mask[idx]),
            }
        )
    return {
        "line_models": line_models,
        "walls": wall_scan_indexes,
        "non_walls": non_wall_scan_indexes,
        "total_points": total_points,
        "wall_ratio": round(wall_ratio, 3),
        "point_info": point_info,
    }
