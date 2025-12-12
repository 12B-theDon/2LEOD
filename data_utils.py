from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import logging

from config import (
    CLUSTER_FEATURE_COLUMNS,
    CLUSTER_GAP_THRESHOLD,
    DISTANCE_COLUMN,
    FEATURE_COLUMNS,
    FRAME_ID_COLUMN,
    LABEL_CLASS_COLUMN,
    MIN_CLUSTER_POINTS,
    ODOM_SOURCE_COLUMNS,
    ODOM_TARGET_COLUMNS,
    RANDOM_SEED,
    SCAN_INDEX_COLUMN,
    TARGET_BG_RATIO,
    TEST_SPLIT_RATIO,
    WORLD_MATCH_DIST,
    DATASET_PAIRS,
)

GT_RADIUS = 2.2
OPP_RATIO_THRESHOLD = 0.10


OPPONENT_CLUSTER_COLUMN = "is_opponent_cluster"
DATASET_COLUMN = "dataset_id"


def _ensure_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _prepare_odometry(odom_path: Path) -> pd.DataFrame:
    df = pd.read_csv(odom_path)
    if df.empty:
        raise ValueError(f"Odom file is empty at {odom_path}")

    df = df.sort_values([FRAME_ID_COLUMN, "stamp_sec", "stamp_nsec"]).reset_index(drop=True)
    df["stamp_sec"] = pd.to_numeric(df["stamp_sec"], errors="raise")
    df["stamp_nsec"] = pd.to_numeric(df["stamp_nsec"], errors="raise")

    ego_x = df["base_link_x"].to_numpy(dtype=float)
    ego_y = df["base_link_y"].to_numpy(dtype=float)
    ego_yaw = df["base_link_yaw"].to_numpy(dtype=float)

    timestamps = df["stamp_sec"].to_numpy(dtype=float) + df["stamp_nsec"].to_numpy(dtype=float) * 1e-9
    delta = np.empty_like(timestamps)
    delta[0] = 1e-3
    if len(timestamps) > 1:
        delta[1:] = timestamps[1:] - timestamps[:-1]
    delta = np.clip(delta, 1e-4, None)

    dx = np.zeros_like(ego_x)
    dy = np.zeros_like(ego_y)
    if len(ego_x) > 1:
        dx[1:] = np.diff(ego_x)
        dy[1:] = np.diff(ego_y)

    dyaw = np.zeros_like(ego_yaw)
    if len(ego_yaw) > 1:
        raw = np.diff(ego_yaw)
        wrapped = (raw + np.pi) % (2 * np.pi) - np.pi
        dyaw[1:] = wrapped

    df["ego_v"] = np.sqrt(dx**2 + dy**2) / delta
    df["ego_w"] = dyaw / delta
    df["ego_vx"] = dx / delta
    df["ego_vy"] = dy / delta

    df = df.rename(columns=dict(zip(ODOM_SOURCE_COLUMNS, ODOM_TARGET_COLUMNS)))

    return df


def _normalize_boolean(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.astype(int).to_numpy()
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
    )
    if normalized.isna().any():
        raise ValueError(f"Unable to normalize boolean column '{series.name}'")
    return normalized.astype(int).to_numpy()


def _load_single_dataset(scan_path: Path, odom_path: Path) -> pd.DataFrame:
    scan_df = pd.read_csv(scan_path)
    odom_df = _prepare_odometry(odom_path)

    scan_df["stamp_sec"] = pd.to_numeric(scan_df["stamp_sec"], errors="raise")
    scan_df["stamp_nsec"] = pd.to_numeric(scan_df["stamp_nsec"], errors="raise")

    merged = pd.merge(
        scan_df,
        odom_df,
        on=[FRAME_ID_COLUMN, "stamp_sec", "stamp_nsec"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError(
            f"No matching frames found between scan {scan_path} and odom {odom_path} data."
        )

    merged[DATASET_COLUMN] = scan_path.stem
    _ensure_required_columns(
        merged,
        [
            FRAME_ID_COLUMN,
            LABEL_CLASS_COLUMN,
            *FEATURE_COLUMNS,
            *ODOM_TARGET_COLUMNS,
        ],
    )
    return merged


def load_dataset_pair(scan_path: Path, odom_path: Path) -> pd.DataFrame:
    """Load a single scan/odom pair without reading every entry in DATASET_PAIRS."""
    return _load_single_dataset(scan_path, odom_path)


def _load_merged_data() -> pd.DataFrame:
    merged_frames: list[pd.DataFrame] = []
    for scan_path, odom_path in DATASET_PAIRS:
        merged_frames.append(_load_single_dataset(scan_path, odom_path))
    if not merged_frames:
        raise ValueError("No dataset pairs defined in DATASET_PAIRS.")
    return pd.concat(merged_frames, ignore_index=True)


def _extract_cluster_features(
    rows: list[pd.Series], prev_clusters: list[dict], frame_info: pd.Series
) -> tuple[dict, dict]:
    cluster_df = pd.DataFrame(rows)
    start_idx = int(cluster_df[SCAN_INDEX_COLUMN].min())
    end_idx = int(cluster_df[SCAN_INDEX_COLUMN].max())

    distance_mean = float(cluster_df[DISTANCE_COLUMN].mean())
    delta_range = 0.0
    for prev in prev_clusters:
        if not (prev["end"] < start_idx or prev["start"] > end_idx):
            delta_range = float(distance_mean - prev["mean_distance"])
            break

    cx_local = float(cluster_df["local_x"].mean())
    cy_local = float(cluster_df["local_y"].mean())
    cx_global = float(cluster_df["global_x"].mean())
    cy_global = float(cluster_df["global_y"].mean())
    radius = float(np.hypot(cx_local, cy_local))
    frame_stamp = float(frame_info["stamp_sec"]) + float(frame_info["stamp_nsec"]) * 1e-9
    local_points = cluster_df[["local_x", "local_y"]].to_numpy(dtype=float)
    center = np.array([cx_local, cy_local])
    spread = float(np.mean(np.linalg.norm(local_points - center, axis=1)))
    ranges = cluster_df[DISTANCE_COLUMN].to_numpy(dtype=float)
    angle_vals = cluster_df[SCAN_INDEX_COLUMN].to_numpy(dtype=float)
    entropy = 0.0
    if len(angle_vals) > 1:
        bins = min(len(angle_vals), 8)
        counts, _ = np.histogram(angle_vals, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs)))
    opponent_ratio = float(cluster_df[LABEL_CLASS_COLUMN].mean())
    global_r = float(np.hypot(cx_global, cy_global))
    op_x = frame_info.get("opp_x", np.nan)
    op_y = frame_info.get("opp_y", np.nan)
    gt_label = 0
    gt_valid = 0
    if not np.isnan(op_x) and not np.isnan(op_y):
        dist = math.hypot(cx_local - float(op_x), cy_local - float(op_y))
        gt_valid = 1
        gt_label = int(dist < GT_RADIUS)
    range_span = float(ranges.max() - ranges.min()) if len(ranges) > 1 else 0.0
    angle_span = float(angle_vals.max() - angle_vals.min()) if len(angle_vals) > 1 else 0.0
    features = {
        "center_local_x": cx_local,
        "center_local_y": cy_local,
        "global_x": cx_global,
        "global_y": cy_global,
        "cluster_size": len(cluster_df),
        "distance_mean": distance_mean,
        "distance_std": float(cluster_df[DISTANCE_COLUMN].std(ddof=0)),
        "range_extent": range_span,
        "angle_index": float(angle_vals.mean()),
        "angle_extent": angle_span,
        "delta_range": delta_range,
        "ego_v": float(frame_info["ego_v"]),
        "ego_w": float(frame_info["ego_w"]),
        "ego_vx": float(frame_info["ego_vx"]),
        "ego_vy": float(frame_info["ego_vy"]),
        "local_x_extent": float(cluster_df["local_x"].max() - cluster_df["local_x"].min()),
        "local_y_extent": float(cluster_df["local_y"].max() - cluster_df["local_y"].min()),
        "local_x_std": float(cluster_df["local_x"].std(ddof=0)),
        "local_y_std": float(cluster_df["local_y"].std(ddof=0)),
        "cluster_cx": cx_local,
        "cluster_cy": cy_local,
        "cluster_radius": radius,
        "cluster_vx": 0.0,
        "cluster_vy": 0.0,
        "cluster_speed": 0.0,
        "cluster_vr": 0.0,
        "cluster_vx_world": 0.0,
        "cluster_vy_world": 0.0,
        "cluster_speed_world": 0.0,
        "cluster_rel_vx": 0.0,
        "cluster_rel_vy": 0.0,
        "cluster_rel_speed": 0.0,
        "cluster_spread": spread,
        "cluster_r_span": range_span,
        "cluster_angle_entropy": entropy,
        "distance_to_ego": global_r,
        "bearing": math.atan2(cy_global, cx_global),
        "cluster_opponent_ratio": opponent_ratio,
        "gt_label": gt_label,
        "gt_valid": gt_valid,
        LABEL_CLASS_COLUMN: int(cluster_df[LABEL_CLASS_COLUMN].any()),
        OPPONENT_CLUSTER_COLUMN: int(cluster_df[LABEL_CLASS_COLUMN].any()),
        "frame_index": int(frame_info[FRAME_ID_COLUMN]),
        "frame_stamp": frame_stamp,
    }
    features[DATASET_COLUMN] = frame_info[DATASET_COLUMN]
    for target in ODOM_TARGET_COLUMNS:
        features[target] = float(frame_info[target])

    cluster_info = {
        "start": start_idx,
        "end": end_idx,
        "mean_distance": distance_mean,
        "timestamp": frame_stamp,
    }
    return features, cluster_info


def _build_clusters(df: pd.DataFrame) -> pd.DataFrame:
    clusters: list[dict] = []
    for _, dataset_df in df.groupby(DATASET_COLUMN, sort=True):
        prev_clusters: list[dict] = []
        for _, frame in dataset_df.groupby(FRAME_ID_COLUMN, sort=True):
            frame = frame.sort_values(SCAN_INDEX_COLUMN).reset_index(drop=True)
            if frame.empty:
                continue
            frame_info = frame.iloc[0]
            cluster_rows: list[pd.Series] = []

            def flush_cluster_rows() -> None:
                nonlocal cluster_rows, prev_idx, prev_distance
                if len(cluster_rows) >= MIN_CLUSTER_POINTS:
                    features, info = _extract_cluster_features(
                        cluster_rows, prev_clusters, frame_info
                    )
                    clusters.append(features)
                    current_frame_clusters.append(info)
                cluster_rows = []
                prev_idx = None
                prev_distance = None

            prev_idx = None
            prev_distance = None
            current_frame_clusters: list[dict] = []
            for _, row in frame.iterrows():
                scan_idx = int(row[SCAN_INDEX_COLUMN])
                distance = float(row[DISTANCE_COLUMN])
                gap_cond = (
                    prev_distance is not None
                    and abs(distance - prev_distance) > CLUSTER_GAP_THRESHOLD
                )
                if prev_idx is not None and scan_idx != prev_idx + 1:
                    flush_cluster_rows()
                elif gap_cond:
                    flush_cluster_rows()
                cluster_rows.append(row)
                prev_idx = scan_idx
                prev_distance = distance
            if cluster_rows:
                flush_cluster_rows()
            prev_clusters = current_frame_clusters
    if not clusters:
        raise ValueError("No clusters extracted from scan data.")
    cluster_df = pd.DataFrame(clusters)
    return _attach_motion_features(cluster_df)


def _attach_motion_features(clusters_df: pd.DataFrame) -> pd.DataFrame:
    clusters_df = clusters_df.sort_values(
        [DATASET_COLUMN, "frame_index", "cluster_cx", "cluster_cy"]
    ).reset_index(drop=True)
    local_motion_cols = ["cluster_vx", "cluster_vy", "cluster_speed", "cluster_vr"]
    world_motion_cols = ["cluster_vx_world", "cluster_vy_world", "cluster_speed_world"]
    relative_motion_cols = ["cluster_rel_vx", "cluster_rel_vy", "cluster_rel_speed"]

    for col in local_motion_cols + world_motion_cols + relative_motion_cols:
        clusters_df[col] = 0.0

    prev_frame_df: pd.DataFrame | None = None
    prev_stamp: float | None = None
    prev_dataset: str | None = None
    for (dataset_id, _), frame_group in clusters_df.groupby(
        [DATASET_COLUMN, "frame_index"], sort=True
    ):
        frame_stamp = float(frame_group["frame_stamp"].iloc[0])
        if prev_dataset == dataset_id and prev_frame_df is not None and prev_stamp is not None:
            dt = frame_stamp - prev_stamp
            if dt <= 0:
                dt = 1e-3

            prev_coords = prev_frame_df[["global_x", "global_y"]].to_numpy(dtype=float)
            prev_radius = prev_frame_df["cluster_radius"].to_numpy(dtype=float)

            for idx, row in frame_group.iterrows():
                current = np.array([row["global_x"], row["global_y"]], dtype=float)
                diffs = prev_coords - current
                dists = np.linalg.norm(diffs, axis=1)
                best_idx = int(np.argmin(dists))
                if dists[best_idx] < WORLD_MATCH_DIST:
                    prev_row = prev_frame_df.iloc[best_idx]
                    dX = row["global_x"] - prev_row["global_x"]
                    dY = row["global_y"] - prev_row["global_y"]
                    vX = dX / dt
                    vY = dY / dt
                    speed_world = np.hypot(vX, vY)
                    clusters_df.at[idx, "cluster_vx_world"] = vX
                    clusters_df.at[idx, "cluster_vy_world"] = vY
                    clusters_df.at[idx, "cluster_speed_world"] = speed_world
                    dx_local = row["cluster_cx"] - prev_row["cluster_cx"]
                    dy_local = row["cluster_cy"] - prev_row["cluster_cy"]
                    vx_local = dx_local / dt
                    vy_local = dy_local / dt
                    clusters_df.at[idx, "cluster_vx"] = vx_local
                    clusters_df.at[idx, "cluster_vy"] = vy_local
                    clusters_df.at[idx, "cluster_speed"] = np.hypot(vx_local, vy_local)
                    dr = row["cluster_radius"] - prev_row["cluster_radius"]
                    clusters_df.at[idx, "cluster_vr"] = dr / dt
                    ego_vx = row.get("ego_vx", 0.0)
                    ego_vy = row.get("ego_vy", 0.0)
                    rel_vx = vX - ego_vx
                    rel_vy = vY - ego_vy
                    clusters_df.at[idx, "cluster_rel_vx"] = rel_vx
                    clusters_df.at[idx, "cluster_rel_vy"] = rel_vy
                    clusters_df.at[idx, "cluster_rel_speed"] = np.hypot(rel_vx, rel_vy)

        prev_stamp = frame_stamp
        prev_frame_df = frame_group.copy()
        prev_dataset = dataset_id

    return clusters_df


def build_clusters_dataframe(merged: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    if merged is None:
        merged = _load_merged_data()
    clusters_df = _build_clusters(merged)
    clusters_df = _attach_motion_features(clusters_df)

    opponent_ratio = clusters_df["cluster_opponent_ratio"]
    label_orig = (opponent_ratio >= OPP_RATIO_THRESHOLD).astype(int)
    label_new = ((clusters_df["gt_label"] == 1) | (opponent_ratio >= OPP_RATIO_THRESHOLD)).astype(int)
    clusters_df["label_orig"] = label_orig
    clusters_df["label_new"] = label_new

    relabel_count = int(np.sum(label_new != label_orig))
    relabel_stats = {
        "total": len(clusters_df),
        "relabeled": relabel_count,
        "ratio": relabel_count / len(clusters_df) if len(clusters_df) > 0 else 0.0,
    }
    return clusters_df, relabel_stats


def split_by_frame_indices(
    frame_indices: np.ndarray, test_ratio: float, random_state: int
) -> tuple[np.ndarray, np.ndarray, set[int], set[int]]:
    logger = logging.getLogger(__name__)
    groups = np.asarray(frame_indices).flatten()
    dummy = np.zeros((len(groups), 1))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    train_idx, test_idx = next(splitter.split(dummy, groups=groups))
    train_frames = set(groups[train_idx])
    test_frames = set(groups[test_idx])
    overlap = train_frames & test_frames
    logger.info(
        f"[data][split] train_frames={len(train_frames)}, test_frames={len(test_frames)}, overlap={len(overlap)}"
    )
    if overlap:
        raise RuntimeError(f"Frame overlap detected ({len(overlap)}) between train and test.")
    return train_idx, test_idx, train_frames, test_frames


def _downsample_background(
    X: np.ndarray, y: np.ndarray, y_reg: np.ndarray, *extras: np.ndarray
) -> tuple:
    rng = np.random.default_rng(RANDOM_SEED)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_count = len(pos_idx)
    neg_count = len(neg_idx)

    if pos_count == 0:
        size = min(neg_count, 10) if neg_count > 0 else 0
        sampled_neg = rng.choice(neg_idx, size=size, replace=False) if size > 0 else np.array([], dtype=int)
        target_ratio = 0.0
    else:
        desired_bg = max(1, int(pos_count * TARGET_BG_RATIO))
        target_ratio = TARGET_BG_RATIO
        if desired_bg >= neg_count:
            sampled_neg = neg_idx
        else:
            sampled_neg = rng.choice(neg_idx, size=desired_bg, replace=False)

    extras_arrays = [np.asarray(arr) for arr in extras]
    combined_idx = []
    if pos_idx.size:
        combined_idx.append(pos_idx)
    if sampled_neg.size:
        combined_idx.append(sampled_neg)
    stats = {
        "pos_count": pos_count,
        "neg_count": neg_count,
        "bg_kept": len(sampled_neg),
        "target_ratio": target_ratio,
    }
    if not combined_idx:
        return X, y, y_reg, *extras_arrays, stats
    keep_idx = np.concatenate(combined_idx).astype(int)
    permuted = rng.permutation(len(keep_idx))
    final_idx = keep_idx[permuted]

    result = [X[final_idx], y[final_idx], y_reg[final_idx]]
    result.extend(arr[final_idx] for arr in extras_arrays)
    return (*result, stats)


def load_dataset() -> dict:
    clusters_df, relabel_stats = build_clusters_dataframe()
    world_cols = ["cluster_vx_world", "cluster_vy_world", "cluster_speed_world"]
    relative_cols = ["cluster_rel_vx", "cluster_rel_vy", "cluster_rel_speed"]
    motion_stats = {
        f"{col}_mean": float(clusters_df[col].mean()) for col in world_cols
    }
    motion_stats.update(
        {f"{col}_std": float(clusters_df[col].std(ddof=0)) for col in world_cols}
    )
    relative_stats = {
        f"{col}_mean": float(clusters_df[col].mean()) for col in relative_cols
    }
    relative_stats.update(
        {f"{col}_std": float(clusters_df[col].std(ddof=0)) for col in relative_cols}
    )

    X = clusters_df[CLUSTER_FEATURE_COLUMNS].astype(float).to_numpy()
    y_class = clusters_df["label_new"].to_numpy()
    y_reg = clusters_df[ODOM_TARGET_COLUMNS].astype(float).to_numpy()
    gt_label = clusters_df["gt_label"].astype(int).to_numpy()
    gt_valid = clusters_df["gt_valid"].astype(int).to_numpy()
    label_orig = clusters_df["label_orig"].to_numpy()

    frame_indices = clusters_df[FRAME_ID_COLUMN].to_numpy()
    train_idx, test_idx, train_frames, test_frames = split_by_frame_indices(
        frame_indices, TEST_SPLIT_RATIO, RANDOM_SEED
    )
    logger = logging.getLogger(__name__)
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_class_train = y_class[train_idx]
    y_class_test = y_class[test_idx]
    y_reg_train = y_reg[train_idx]
    y_reg_test = y_reg[test_idx]
    gt_label_train = gt_label[train_idx]
    gt_label_test = gt_label[test_idx]
    gt_valid_train = gt_valid[train_idx]
    gt_valid_test = gt_valid[test_idx]
    label_orig_train = label_orig[train_idx]
    label_orig_test = label_orig[test_idx]
    pos_train = int(np.sum(y_class_train == 1))
    neg_train = int(np.sum(y_class_train == 0))
    pos_test = int(np.sum(y_class_test == 1))
    neg_test = int(np.sum(y_class_test == 0))
    logger.info(
        f"[data][split] Train frames={len(train_frames)}, Test frames={len(test_frames)}"
    )
    logger.info(
        f"[data][split] Train class dist pos={pos_train}, neg={neg_train}; "
        f"Test class dist pos={pos_test}, neg={neg_test}"
    )

    (
        X_train,
        y_class_train,
        y_reg_train,
        gt_label_train,
        gt_valid_train,
        label_orig_train,
        stats,
    ) = _downsample_background(
        X_train,
        y_class_train,
        y_reg_train,
        gt_label_train,
        gt_valid_train,
        label_orig_train,
    )

    train_positive_mask = y_class_train == 1
    test_positive_mask = y_class_test == 1

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_class_train": y_class_train,
        "y_class_test": y_class_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "train_positive_mask": train_positive_mask,
        "test_positive_mask": test_positive_mask,
        "y_reg_train_visible": y_reg_train[train_positive_mask],
        "y_reg_test_visible": y_reg_test[test_positive_mask],
        "train_sample_stats": stats,
        "motion_stats": motion_stats,
        "relative_stats": relative_stats,
        "relabel_stats": relabel_stats,
        "gt_label_train": gt_label_train,
        "gt_label_test": gt_label_test,
        "gt_valid_train": gt_valid_train.astype(bool),
        "gt_valid_test": gt_valid_test.astype(bool),
        "label_orig_train": label_orig_train,
        "label_orig_test": label_orig_test,
    }
