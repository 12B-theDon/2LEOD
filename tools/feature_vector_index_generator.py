#!/usr/bin/env python3
"""Create auxiliary JSON guides that explain each scan vector position."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore


DEFAULT_ANGLE_MIN = -135.0
DEFAULT_ANGLE_MAX = 135.0
DEFAULT_VECTOR_WIDTH = 360


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce a JSON map that explains which scan_index matches which real-world angle."
    )
    parser.add_argument(
        "--csv_filefolder_path",
        "-p",
        type=Path,
        required=True,
        help="Folder that contains the CSV scans (e.g. `dataFiles`)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Where to write the generated JSON file(s) (defaults to CSV folder).",
    )
    parser.add_argument(
        "--default-width",
        "-w",
        type=int,
        default=DEFAULT_VECTOR_WIDTH,
        help="Fallback vector length when the YAML metadata is absent.",
    )
    parser.add_argument(
        "--odom-columns",
        nargs="+",
        default=("x_car", "y_car", "heading_car", "x_linear_vel", "y_linear_vel", "w_angular_vel"),
        help="Column names that should be collected per frame as per-odom data.",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> Mapping:
    if path is None or not path.exists():
        return {}
    if yaml is not None:
        try:
            with path.open("r", encoding="utf-8") as stream:
                return yaml.safe_load(stream) or {}
        except (OSError, Exception):
            return {}
    return parse_simple_yaml(path)


def _parse_scalar(value: str):
    trimmed = value.strip()
    if not trimmed:
        return ""
    if (trimmed[0] == trimmed[-1]) and trimmed.startswith(("'", '"')):
        return trimmed[1:-1]
    lowered = trimmed.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if "." in trimmed:
            return float(trimmed)
        return int(trimmed)
    except ValueError:
        return trimmed


def parse_simple_yaml(path: Path) -> Mapping:
    root: dict = {}
    stack: List = [(-1, root)]
    try:
        with path.open("r", encoding="utf-8") as stream:
            for raw_line in stream:
                line = raw_line.split("#", 1)[0].rstrip()
                if not line.strip():
                    continue
                indent = len(line) - len(line.lstrip(" \t"))
                stripped = line.lstrip(" \t")
                if stripped.startswith("-"):
                    continue
                if ":" not in stripped:
                    continue
                key, _, value = stripped.partition(":")
                key = key.strip()
                value = value.strip()
                while stack and indent <= stack[-1][0]:
                    stack.pop()
                parent = stack[-1][1]
                if not value:
                    container = {}
                    if isinstance(parent, dict):
                        parent[key] = container
                    stack.append((indent, container))
                else:
                    if isinstance(parent, dict):
                        parent[key] = _parse_scalar(value)
    except OSError:
        return {}
    return root


def find_yaml_for_csv(csv_path: Path) -> Optional[Path]:
    yaml_candidates = sorted(csv_path.parent.glob("*.yaml")) + sorted(csv_path.parent.glob("*.yml"))
    for candidate in yaml_candidates:
        try:
            config = load_yaml_config(candidate)
        except yaml.YAMLError:
            continue
        csv_reference = config.get("csv_file", {}).get("name")
        if csv_reference == csv_path.name:
            return candidate
    for ext in ("yaml", "yml"):
        fallback = csv_path.with_suffix(f".{ext}")
        if fallback.exists():
            return fallback
    return None


def gather_scan_indexes(csv_path: Path) -> List[int]:
    scan_indexes = set()
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "scan_index" not in reader.fieldnames:
                return []
            for row in reader:
                value = row.get("scan_index")
                if value is None or value == "":
                    continue
                try:
                    scan_index = int(float(value))
                except (TypeError, ValueError):
                    continue
                scan_indexes.add(scan_index)
    except (FileNotFoundError, PermissionError):
        return []
    return sorted(scan_indexes)


def categorize_angle(angle_deg: float) -> str:
    if angle_deg >= 315 or angle_deg < 45:
        return "front"
    if angle_deg < 135:
        return "left"
    if angle_deg < 225:
        return "rear"
    return "right"


def build_vector_entries(width: int, angle_min: float, angle_max: float) -> List[Mapping]:
    divisor = max(width - 1, 1)
    span = angle_max - angle_min
    entries = []
    for idx in range(width):
        angle = angle_min + (span * idx / divisor)
        angle = (angle + 360.0) % 360.0
        entries.append(
            {
                "scan_index": idx,
                "angle_deg": round(angle, 3),
                "angle_rad": round(math.radians(angle), 5),
                "direction": categorize_angle(angle),
                "description": f"scan {idx}: {angle:.1f}°",
            }
        )
    return entries


def build_frame_records(csv_path: Path, vector_length: int, odom_columns: Sequence[str]) -> List[Mapping]:
    frames: Dict[int, Dict] = {}
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                try:
                    frame_index = int(row.get("frame_index", -1))
                    scan_index = int(row.get("scan_index", -1))
                except ValueError:
                    continue
                frame = frames.setdefault(
                    frame_index,
                    {
                        "distances": [0.0] * vector_length,
                        "opponent_count": 0,
                        "wall_count": 0,
                        "free_count": 0,
                        "last_distance": 0.0,
                        "odom": [0.0] * len(odom_columns),
                    },
                )
                distance_raw = row.get("distance")
                if distance_raw is None:
                    distance_raw = row.get("distance[m]")
                try:
                    distance_val = float(distance_raw)
                except (TypeError, ValueError):
                    distance_val = frame["last_distance"]
                frame["last_distance"] = distance_val
                if 0 <= scan_index < vector_length:
                    frame["distances"][scan_index] = distance_val
                if odom_columns:
                    odom_vec = frame.get("odom", [0.0] * len(odom_columns))
                    for idx, col in enumerate(odom_columns):
                        value = row.get(col)
                        try:
                            odom_val = float(value)
                        except (TypeError, ValueError):
                            odom_val = odom_vec[idx]
                        odom_vec[idx] = odom_val
                    frame["odom"] = odom_vec
                if _parse_boolean(row.get("isOpponent")) or _parse_boolean(row.get("isObstacle")):
                    frame["opponent_count"] += 1
                if _parse_boolean(row.get("isWall")):
                    frame["wall_count"] += 1
                if _parse_boolean(row.get("isFree")):
                    frame["free_count"] += 1
    except OSError:
        return []
    dataset: List[Mapping] = []
    for frame_index in sorted(frames.keys()):
        frame = frames[frame_index]
        dataset.append(
            {
                "frame_index": frame_index,
                "distances": frame["distances"],
                "label": 1 if frame["opponent_count"] > 0 else 0,
                "odom": frame["odom"],
                "counts": {
                    "opponent": frame["opponent_count"],
                    "wall": frame["wall_count"],
                    "free": frame["free_count"],
                },
            }
        )
    return dataset


def _parse_boolean(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def prepare_feature_map(
    csv_path: Path,
    yaml_path: Optional[Path],
    default_width: int,
    odom_columns: Sequence[str],
) -> Mapping:
    config = load_yaml_config(yaml_path) if yaml_path else {}
    laser_cfg = config.get("laser", {})
    angle_min = float(laser_cfg.get("angle_min_deg", DEFAULT_ANGLE_MIN))
    angle_max = float(laser_cfg.get("angle_max_deg", DEFAULT_ANGLE_MAX))
    angle_model = laser_cfg.get("angle_model", {})
    width_candidate = angle_model.get("total_beams")
    observed_indexes = gather_scan_indexes(csv_path)
    observed_width = (max(observed_indexes) + 1) if observed_indexes else 0
    width = 0
    if width_candidate is not None:
        try:
            width_candidate_num = int(float(width_candidate))
        except (TypeError, ValueError):
            width_candidate_num = 0
        if width_candidate_num > 0:
            width = width_candidate_num
    width = max(width, observed_width, default_width)
    if width <= 0:
        width = default_width
    vector_entries = build_vector_entries(width, angle_min, angle_max)
    column_meta = config.get("columns", {})
    return {
        "csv_file": csv_path.name,
        "yaml_file": yaml_path.name if yaml_path else None,
        "vector_length": width,
        "angle_model": {
            "angle_min_deg": angle_min,
            "angle_max_deg": angle_max,
            "total_beams": width,
            "angle_step_deg": (
                round((angle_max - angle_min) / max(width - 1, 1), 5)
            ),
            "description": angle_model.get("formula"),
        },
        "observed_scan_indexes": observed_indexes,
        "columns": column_meta,
        "vectors": vector_entries,
        "frames": build_frame_records(csv_path, width, odom_columns),
    }


def write_json(output_path: Path, data: Mapping) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    folder = args.csv_filefolder_path.expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a directory.")
    output_dir = (args.output_dir or folder).expanduser().resolve()
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"no CSV files under {folder}")
        return
    for csv_path in csv_files:
        yaml_path = find_yaml_for_csv(csv_path)
        feature_map = prepare_feature_map(
            csv_path, yaml_path, args.default_width, tuple(args.odom_columns)
        )
        output_path = output_dir / f"{csv_path.stem}_feature_index.json"
        write_json(output_path, feature_map)
        print(f"generated {output_path} (width={feature_map['vector_length']})")


if __name__ == "__main__":
    main()
