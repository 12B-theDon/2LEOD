import argparse
import os

import joblib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

from config import (
    CLUSTER_FEATURE_COLUMNS,
    FRAME_ID_COLUMN,
    LOGREG_DECISION_THRESHOLD,
    MODEL_SAVE_PATH,
)
from data_utils import build_clusters_dataframe, _load_merged_data
from logger import get_logger

logger = get_logger("viz_logger")


def bool_mask(series) -> pd.Series:
    return pd.Series(series).astype(str).str.strip().str.lower().isin({"true", "1"})


def visualize_frame(frame_index: int | None = None) -> None:
    merged = _load_merged_data()
    clusters_df, _ = build_clusters_dataframe(merged)

    if frame_index is None:
        candidates = clusters_df[clusters_df["gt_label"] == 1]
        if candidates.empty:
            raise ValueError("No GT opponent clusters available to visualize.")
        frame_index = int(candidates["frame_index"].iloc[0])

    logger.info(f"[viz] Visualizing frame {frame_index}")

    frame_clusters = clusters_df[clusters_df["frame_index"] == frame_index]
    if frame_clusters.empty:
        raise ValueError(f"No clusters found for frame {frame_index}.")

    frame_points = merged[merged[FRAME_ID_COLUMN] == frame_index]
    if frame_points.empty:
        raise ValueError(f"No scan points for frame {frame_index}.")

    gt_row = frame_points.iloc[0]
    if "opp_x" in frame_points and "opp_y" in frame_points:
        gt_x = float(gt_row["opp_x"])
        gt_y = float(gt_row["opp_y"])
    else:
        gt_x = float(gt_row["base_link_op_x"])
        gt_y = float(gt_row["base_link_op_y"])

    ego_x, ego_y = None, None
    if "base_link_x" in frame_points and "base_link_y" in frame_points:
        ego_x = float(gt_row["base_link_x"])
        ego_y = float(gt_row["base_link_y"])

    bundle = joblib.load(MODEL_SAVE_PATH)
    clf_pipe = bundle["clf_pipe"]
    threshold = bundle.get("threshold", LOGREG_DECISION_THRESHOLD)

    feats = frame_clusters[CLUSTER_FEATURE_COLUMNS].astype(float).to_numpy()
    probs = clf_pipe.predict_proba(feats)[:, 1]
    frame_clusters = frame_clusters.copy()
    frame_clusters["pred_prob"] = probs
    frame_clusters["pred_label"] = (probs >= threshold).astype(int)

    logger.info(
        "[viz][frame_stats] "
        f"clusters={len(frame_clusters)}, "
        f"opponent_candidates={int(frame_clusters['label_new'].sum())}, "
        f"gt_clusters={int(frame_clusters['gt_label'].sum())}"
    )

    opp_rows = frame_clusters[frame_clusters["label_new"] == 1]
    if not opp_rows.empty:
        logger.info(
            "[viz][opponent_clusters]\n"
            f"{opp_rows[['cluster_cx','cluster_cy','cluster_opponent_ratio','pred_prob','gt_label']].to_string(index=False)}"
        )

    debug_dir = "figs"
    os.makedirs(debug_dir, exist_ok=True)
    frame_clusters[
        [
            "cluster_cx",
            "cluster_cy",
            "global_x",
            "global_y",
            "label_new",
            "gt_label",
            "pred_prob",
            "pred_label",
            "cluster_opponent_ratio",
        ]
    ].to_csv(os.path.join(debug_dir, f"cluster_debug_frame_{frame_index}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {
        "isWall": "gray",
        "isStatic": "blue",
        "isFree": "lightgray",
        "isOpponent": "red",
    }
    for label, color in colors.items():
        mask = bool_mask(frame_points[label]) if label in frame_points else np.full(
            len(frame_points), False
        )
        pts = frame_points[mask]
        if not pts.empty:
            ax.scatter(
                pts["global_x"],
                pts["global_y"],
                c=color,
                s=12,
                label=label,
                alpha=0.7,
            )

    for _, row in frame_clusters.iterrows():
        cx = row["global_x"]
        cy = row["global_y"]
        label_new = int(row["label_new"])
        marker_kwargs = {
            "edgecolor": "gold" if row["gt_label"] == 1 else "black",
            "facecolor": "red" if label_new else "black",
            "s": 120 if row["gt_label"] == 1 else 70,
            "alpha": 0.9,
        }
        ax.scatter(cx, cy, marker="o", **marker_kwargs)
        ax.text(
            cx,
            cy,
            f"{row['pred_prob']:.2f}",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
            path_effects=[pe.withStroke(linewidth=1, foreground="black")],
        )

    ax.scatter(gt_x, gt_y, marker="*", c="yellow", s=200, label="GT opponent")
    if ego_x is not None and ego_y is not None:
        ax.scatter(ego_x, ego_y, marker="^", c="cyan", s=150, label="Ego pose")
    ax.set_title(f"Frame {frame_index} clusters")
    ax.set_xlabel("global_x")
    ax.set_ylabel("global_y")
    ax.legend()
    ax.set_aspect("equal", "box")

    filepath = os.path.join(debug_dir, f"cluster_debug_frame_{frame_index}.png")
    fig.savefig(filepath)
    logger.info(f"[viz] Saved cluster debug figure to {filepath}")
    if os.environ.get("VISUALIZE_SHOW", "1") == "1":
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize LiDAR cluster labels.")
    parser.add_argument("--frame-index", type=int, help="Frame index to plot.")
    args = parser.parse_args()
    visualize_frame(args.frame_index)


if __name__ == "__main__":
    main()
