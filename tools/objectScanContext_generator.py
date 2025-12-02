#!/usr/bin/env python3
"""Create stacked ScanContext-like tensors from CSV scan exports."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


DEFAULT_CSV = Path("dataFiles/example_csv_format.csv")
DEFAULT_YAML = Path("dataFiles/example_yaml_format.yaml")
DEFAULT_OUTPUT = Path("dataFiles/scan_context_images")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Turn a single laser CSV into ScanContext-friendly images."
    )
    parser.add_argument(
        "--csv",
        "-c",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV file holding the scan data (default: example).",
    )
    parser.add_argument(
        "--yaml",
        "-y",
        type=Path,
        default=DEFAULT_YAML,
        help="YAML metadata (angle model / range limits).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory to emit one png per timestamp.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=360,
        help="Horizontal bins (typically degrees 0-359).",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=20.0,
        help="Vertical resolution that maps meters to pixels.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=None,
        help="Clamp distances below this meter (defaults to laser range min).",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Clamp distances above this meter (defaults to laser range max).",
    )
    parser.add_argument(
        "--delta-clip",
        type=float,
        default=2.0,
        help="Meters to clip delta channel before mapping to 0-255.",
    )
    return parser.parse_args()


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def normalize_columns(columns):
    normalized = {}
    for col in columns:
        clean = re.sub(r"[^0-9A-Za-z_]+", "_", col.strip()).strip("_")
        normalized[col] = clean.lower()
    return normalized


def to_boolean_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    truthy = {"true", "1", "t", "yes"}
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .isin(truthy)
    )


def prepare_scan_dataframe(csv_path: Path, yaml_config: dict):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    column_map = normalize_columns(df.columns)
    df = df.rename(columns=column_map)
    distance_column = "distance_m"
    if distance_column not in df.columns:
        raise ValueError("CSV must provide a `distance[m]` column.")
    df[distance_column] = pd.to_numeric(df[distance_column], errors="coerce")
    df["scan_index"] = pd.to_numeric(df.get("scan_index", None), errors="coerce")
    timestamp_col = "timestamp"
    if timestamp_col not in df.columns:
        raise ValueError("CSV must contain a `Timestamp` column.")
    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors="coerce")
    df["frame_index"] = pd.to_numeric(df.get("frame_index", None), errors="coerce")
    df = df.dropna(subset=[distance_column, "scan_index", timestamp_col])
    label_cols = []
    for label in ("iswall", "isopponent", "isobstacle", "isfree", "isstatic"):
        if label in df.columns:
            df[label] = to_boolean_series(df[label])
            label_cols.append(label)
    angle_cfg = yaml_config.get("laser", {})
    angle_min = float(angle_cfg.get("angle_min_deg", -135.0))
    angle_max = float(angle_cfg.get("angle_max_deg", 135.0))
    total_beams = int(angle_cfg.get("angle_model", {}).get("total_beams", df["scan_index"].max() + 1))
    total_beams = max(total_beams, int(df["scan_index"].max() + 1))
    divisor = max(total_beams - 1, 1)
    df["angle_deg"] = angle_min + df["scan_index"] * (angle_max - angle_min) / divisor
    df["angle_deg"] = (df["angle_deg"] + 360.0) % 360.0
    return df, label_cols


def simulate_scan_images(args, config, df, label_columns):
    laser_cfg = config.get("laser", {})
    if args.min_distance is not None:
        range_min = args.min_distance
    else:
        range_min = laser_cfg.get("range_min_m")
        range_min = float(range_min) if range_min is not None else 0.0
    if args.max_distance is not None:
        range_max = args.max_distance
    else:
        max_cfg = laser_cfg.get("range_max_m")
        if max_cfg is None:
            max_cfg = df["distance_m"].max()
        range_max = float(max_cfg)
    height_px = max(32, math.ceil(range_max * args.pixels_per_meter))
    width = args.width
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = args.output_dir / "metadata"
    labels_dir.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["timestamp", "frame_index"], dropna=False, sort=True)
    last_distances = np.full(width, np.nan)
    range_span = max(range_max - range_min, 1e-3)
    delta_clip = max(1e-3, args.delta_clip)
    for (timestamp, frame_index), group in grouped:
        binned = group.assign(angle_bin=np.floor(group["angle_deg"] * width / 360.0).astype(int))
        binned["angle_bin"] = binned["angle_bin"].mod(width)
        bin_groups = binned.groupby("angle_bin")
        distance_series = bin_groups["distance_m"].min()
        if distance_series.empty:
            continue
        count_series = bin_groups.size()
        max_count = max(1, int(count_series.max()))

        label_data = {}
        if label_columns:
            label_frame = bin_groups[label_columns].any()
            label_data = {
                int(idx): row.to_dict() for idx, row in label_frame.iterrows()
            }

        canvas_layers = np.zeros((height_px, width, 3), dtype=np.uint8)
        bin_labels = ["none"] * width
        for col, stats in label_data.items():
            if stats.get("iswall"):
                bin_labels[col] = "wall"
            elif stats.get("isopponent"):
                bin_labels[col] = "opponent"
            elif stats.get("isobstacle"):
                bin_labels[col] = "obstacle"
            elif stats.get("isstatic"):
                bin_labels[col] = "static"
            elif stats.get("isfree"):
                bin_labels[col] = "free"
            else:
                bin_labels[col] = "unknown"

        for angle_bin, dist in distance_series.items():
            col = int(angle_bin) % width
            dist = float(np.clip(dist, range_min, range_max))
            distance_norm = (dist - range_min) / range_span
            distance_norm = max(0.0, min(1.0, distance_norm))
            fill_height = int(distance_norm * height_px)
            if distance_norm > 0 and fill_height == 0:
                fill_height = 1
            if fill_height <= 0:
                continue
            y_start = height_px - fill_height
            distance_value = int(distance_norm * 255)
            distance_values = (
                np.linspace(0, distance_value, fill_height, dtype=np.uint8)
                if fill_height > 1
                else np.array([distance_value], dtype=np.uint8)
            )

            count_val = int(count_series.get(angle_bin, 0))
            density_norm = min(float(count_val) / max_count, 1.0)
            density_value = int(density_norm * 255)

            previous = last_distances[col]
            delta_val = dist - previous if not math.isnan(previous) else 0.0
            last_distances[col] = dist
            delta_norm = (np.clip(delta_val, -delta_clip, delta_clip) + delta_clip) / (2 * delta_clip)
            delta_value = int(delta_norm * 255)

            canvas_layers[y_start:, col, 0] = delta_value
            canvas_layers[y_start:, col, 1] = density_value
            canvas_layers[y_start:, col, 2] = distance_values
            if bin_labels[col] == "none":
                bin_labels[col] = "unknown"

        img = Image.fromarray(canvas_layers, mode="RGB")
        ts_str = f"{int(timestamp):012d}"
        frame_tag = (
            f"frame{int(frame_index):03d}" if not pd.isna(frame_index) else "frame_unknown"
        )
        output_path = args.output_dir / f"scan_{ts_str}_{frame_tag}.png"
        img.save(output_path)

        metadata = {
            "timestamp": int(timestamp),
            "frame_index": None if pd.isna(frame_index) else int(frame_index),
            "labels": bin_labels,
            "range_min": range_min,
            "range_max": range_max,
            "delta_clip": delta_clip,
        }
        meta_path = labels_dir / f"scan_{ts_str}_{frame_tag}.json"
        with meta_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False, indent=2)

        print(
            f"wrote {output_path} ({distance_series.size} columns, "
            f"height {height_px}px, width {width}px)"
        )


def main():
    args = parse_args()
    config = load_config(args.yaml)
    df, label_columns = prepare_scan_dataframe(args.csv, config)
    simulate_scan_images(args, config, df, label_columns)


if __name__ == "__main__":
    main()
