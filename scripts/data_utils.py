#!/usr/bin/env python3
"""Shared helpers for reading LiDAR CSVs and building training vectors."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def normalize_column(column: str) -> str:
    if not column:
        return ""
    normalized = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in column.strip()
    ).strip("_")
    return normalized


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_bool(value: Any) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"true", "t", "1", "yes", "y"}


def load_feature_index(
    path: Path,
) -> Tuple[Mapping[int, Mapping[str, Any]], Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    vectors = payload.get("vectors", [])
    mapping: Dict[int, Mapping[str, Any]] = {}
    for entry in vectors:
        scan_idx = entry.get("scan_index")
        if scan_idx is None:
            continue
        try:
            idx = int(scan_idx)
        except (TypeError, ValueError):
            continue
        mapping[idx] = entry
    return mapping, payload


def read_scan_csv(
    csv_path: Path, feature_map: Mapping[int, Mapping[str, Any]]
) -> Tuple[
    Dict[int, List[Mapping[str, Any]]],
    Dict[int, Dict[int, float]],
    List[Mapping[str, Any]],
]:
    frames: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    frame_vectors: Dict[int, Dict[int, float]] = defaultdict(dict)
    entries: List[Mapping[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("CSV contains no header.")
        for row in reader:
            normalized_row = {
                normalize_column(key): value for key, value in row.items()
            }
            distance = _parse_float(normalized_row.get("distance_m"))
            scan_index = _parse_int(normalized_row.get("scan_index"))
            if distance is None or scan_index is None:
                continue
            frame_index = _parse_int(normalized_row.get("frame_index"))
            frame_index = 0 if frame_index is None else frame_index
            vector_meta = feature_map.get(scan_index, {})
            angle_rad = vector_meta.get("angle_rad")
            if angle_rad is None:
                angle_deg = vector_meta.get("angle_deg", 0.0)
                angle_rad = math.radians(float(angle_deg))
            frames[frame_index].append(
                {
                    "scan_index": scan_index,
                    "distance": distance,
                    "angle_rad": float(angle_rad),
                }
            )
            frame_vectors[frame_index][scan_index] = distance
            label_bundle = {
                "iswall": _parse_bool(normalized_row.get("iswall")),
                "isstatic": _parse_bool(normalized_row.get("isstatic")),
                "isopponent": _parse_bool(normalized_row.get("isopponent")),
            }
            entries.append(
                {
                    "frame_index": frame_index,
                    "scan_index": scan_index,
                    "distance": distance,
                    "labels": label_bundle,
                }
            )
    for frame_entries in frames.values():
        frame_entries.sort(key=lambda entry: int(entry["scan_index"]))
    return frames, frame_vectors, entries


def build_frame_vector(
    scan_map: Mapping[int, float], width: int, fill_value: float = 0.0
) -> List[float]:
    vector = [fill_value] * width
    for scan_idx, distance in scan_map.items():
        if 0 <= scan_idx < width:
            vector[scan_idx] = float(distance)
    return vector


def map_label(label_bundle: Mapping[str, bool]) -> int:
    if label_bundle.get("iswall"):
        return 0
    if label_bundle.get("isstatic"):
        return 1
    if label_bundle.get("isopponent"):
        return 2
    return 3


def split_frames(
    frame_indices: Sequence[int], split_ratio: float, rng: random.Random
) -> Tuple[List[int], List[int]]:
    if split_ratio <= 0 or split_ratio >= 1:
        raise ValueError("split_ratio must be between 0 and 1")
    indices = list(frame_indices)
    rng.shuffle(indices)
    if len(indices) < 2:
        return indices, []
    split_at = max(1, min(len(indices) - 1, int(len(indices) * split_ratio)))
    return indices[:split_at], indices[split_at:]


def build_dataset(
    entries: Sequence[Mapping[str, Any]],
    frame_vectors: Mapping[int, Mapping[int, float]],
    width: int,
    selected_frames: Sequence[int],
) -> Tuple[List[List[float]], List[int], List[Mapping[str, Any]]]:
    selected = set(selected_frames)
    features: List[List[float]] = []
    labels: List[int] = []
    meta: List[Mapping[str, Any]] = []
    for entry in entries:
        frame_index = int(entry["frame_index"])
        if frame_index not in selected:
            continue
        vector = build_frame_vector(frame_vectors[frame_index], width)
        label = map_label(entry["labels"])
        features.append(vector)
        labels.append(label)
        meta.append(
            {
                "frame_index": frame_index,
                "scan_index": int(entry["scan_index"]),
                "label": label,
            }
        )
    return features, labels, meta
