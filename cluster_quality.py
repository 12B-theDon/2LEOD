import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from config import FRAME_ID_COLUMN
from data_utils import build_clusters_dataframe, _load_merged_data


def bool_mask(series) -> np.ndarray:
    return np.asarray(
        series.astype(str).str.strip().str.lower().isin({"true", "1"})
    )


def render_cluster_frame(
    frame_points,
    frame_clusters,
    output_path,
    title="Cluster overview",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {
        "isWall": "gray",
        "isStatic": "blue",
        "isFree": "lightgray",
        "isOpponent": "orange",
    }
    for label, color in colors.items():
        if label not in frame_points:
            continue
        mask = bool_mask(frame_points[label])
        pts = frame_points[mask]
        if pts.empty:
            continue
        ax.scatter(pts["global_x"], pts["global_y"], c=color, s=12, alpha=0.7, label=label)

    for _, row in frame_clusters.iterrows():
        edge = "gold" if int(row["gt_label"]) == 1 else "black"
        fill = "magenta" if row["label_new"] == 1 else "none"
        size = 150 if row["label_new"] == 1 else 80
        ax.scatter(
            row["global_x"],
            row["global_y"],
            facecolors=fill,
            edgecolors=edge,
            linewidths=1.4,
            s=size,
            alpha=0.9,
        )
        ax.text(
            row["global_x"],
            row["global_y"],
            str(int(row["cluster_size"])),
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            color="white" if row["label_new"] else "black",
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.1"),
        )

    ax.set_title(title)
    ax.set_xlabel("global_x")
    ax.set_ylabel("global_y")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LiDAR clustering per frame.")
    parser.add_argument("--frame-index", type=int, help="Frame index to visualize.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3,
        help="Maximum number of frames to render if no index is provided.",
    )
    parser.add_argument(
        "--focus-positives",
        action="store_true",
        help="Only render frames containing at least one label_new==1 cluster.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figs/clustering",
        help="Directory to save cluster overview figures.",
    )
    args = parser.parse_args()

    merged = _load_merged_data()
    clusters_df, _ = build_clusters_dataframe(merged)

    frames = []
    if args.frame_index is not None:
        frames = [args.frame_index]
    else:
        source = clusters_df
        if args.focus_positives:
            source = source[source["label_new"] == 1]
        frames = sorted(source[FRAME_ID_COLUMN].unique())[: args.max_frames]

    if not frames:
        raise SystemExit("No frames available for cluster visualization.")

    os.makedirs(args.output_dir, exist_ok=True)
    for frame_index in frames:
        frame_points = merged[merged[FRAME_ID_COLUMN] == frame_index]
        frame_clusters = clusters_df[clusters_df["frame_index"] == frame_index]
        if frame_points.empty or frame_clusters.empty:
            continue
        output_path = os.path.join(
            args.output_dir, f"cluster_quality_frame_{frame_index}.png"
        )
        render_cluster_frame(
            frame_points,
            frame_clusters,
            output_path,
            title=f"Frame {frame_index} clusters",
        )
        print(f"[cluster_quality] Saved {output_path}")


if __name__ == "__main__":
    main()
