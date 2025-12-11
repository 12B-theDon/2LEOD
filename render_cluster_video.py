import argparse
import os
from pathlib import Path

try:
    import imageio
except ModuleNotFoundError:
    imageio = None

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import (
    CLUSTER_FEATURE_COLUMNS,
    FRAME_ID_COLUMN,
    LOGREG_DECISION_THRESHOLD,
    MODEL_SAVE_PATH,
)
from data_utils import _load_merged_data, build_clusters_dataframe

VIDEO_DIR = Path("figs")


def bool_mask(series: pd.Series) -> np.ndarray:
    """Normalize text/bool columns into a boolean numpy mask."""
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1"})
    )
    return normalized.to_numpy(dtype=bool)


def select_frame_indices(indices: list[int], count: int) -> list[int]:
    """Return `count` frame indices sampled evenly across the sequence."""
    if not indices:
        raise ValueError("No valid frame indices available.")
    if count <= 0:
        raise ValueError("Frame count must be positive.")
    positions = np.linspace(0, len(indices) - 1, count)
    positions = np.clip(np.round(positions).astype(int), 0, len(indices) - 1)
    return [indices[pos] for pos in positions]


def load_classifier(bundle_path: Path):
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {bundle_path}")
    bundle = joblib.load(bundle_path)
    clf_pipe = bundle["clf_pipe"]
    threshold = bundle.get("cls_threshold")
    if threshold is None:
        threshold = bundle.get("threshold", LOGREG_DECISION_THRESHOLD)
    return clf_pipe, float(threshold)


def extract_ground_truth(frame_points: pd.DataFrame) -> tuple[float, float]:
    row = frame_points.iloc[0]
    if "opp_x" in frame_points and "opp_y" in frame_points:
        return float(row["opp_x"]), float(row["opp_y"])
    return float(row["base_link_op_x"]), float(row["base_link_op_y"])


def extract_ego_pose(frame_points: pd.DataFrame) -> tuple[float | None, float | None]:
    row = frame_points.iloc[0]
    if "base_link_x" in frame_points and "base_link_y" in frame_points:
        return float(row["base_link_x"]), float(row["base_link_y"])
    return None, None


def render_frame_image(
    frame_points: pd.DataFrame,
    frame_clusters: pd.DataFrame,
    gt_xy: tuple[float, float],
    ego_xy: tuple[float | None, float | None],
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(8, 8))
    cluster_colors = {
        "isWall": "gray",
        "isStatic": "blue",
        "isFree": "lightgray",
        "isOpponent": "magenta",
    }
    for label, color in cluster_colors.items():
        if label not in frame_points:
            continue
        mask = bool_mask(frame_points[label])
        pts = frame_points[mask]
        if pts.empty:
            continue
        ax.scatter(
            pts["global_x"],
            pts["global_y"],
            c=color,
            s=12,
            label=label,
            alpha=0.6,
        )
    # emphasize raw isOpponent points for readability
    if "isOpponent" in frame_points:
        mask = bool_mask(frame_points["isOpponent"])
        opp_pts = frame_points[mask]
        if not opp_pts.empty:
            ax.scatter(
                opp_pts["global_x"],
                opp_pts["global_y"],
                facecolors="none",
                edgecolors="magenta",
                s=80,
                linewidths=1.6,
                alpha=0.85,
                label=None,
                zorder=5,
            )

    cmap = plt.get_cmap("viridis")
    predicted_mask = frame_clusters["pred_label"] == 1
    top_pred_index = None
    if predicted_mask.any():
        top_prob = frame_clusters.loc[predicted_mask, "pred_prob"].astype(float).max()
        top_candidates = frame_clusters[
            predicted_mask & (frame_clusters["pred_prob"] == top_prob)
        ]
        if not top_candidates.empty:
            top_pred_index = top_candidates.index[0]

    for idx, cluster in frame_clusters.iterrows():
        cx = cluster["global_x"]
        cy = cluster["global_y"]
        prob = float(cluster.get("pred_prob", 0.0))
        facecolor = cmap(np.clip(prob, 0.0, 1.0))
        is_predicted = idx == top_pred_index
        is_gt = int(cluster.get("gt_label", 0)) == 1
        if is_predicted:
            edge_color = "red"
            size = 260
            linewidth = 2.5
            text_color = "yellow"
            marker_zorder = 10
            text_zorder = 11
        elif is_gt:
            edge_color = "gold"
            size = 150
            linewidth = 1.6
            text_color = "white"
            marker_zorder = 8
            text_zorder = 9
        else:
            edge_color = "black"
            size = 80
            linewidth = 1.0
            text_color = "white"
            marker_zorder = 6
            text_zorder = 7
        ax.scatter(
            cx,
            cy,
            marker="o",
            facecolor=facecolor,
            edgecolor=edge_color,
            s=size,
            linewidths=linewidth,
            alpha=0.9,
            zorder=marker_zorder,
        )
        ax.text(
            cx,
            cy,
            f"{prob:.2f}",
            fontsize=7,
            color=text_color,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.1"),
            zorder=text_zorder,
        )

    # Draw heading arrow for the highest-confidence predicted cluster
    if top_pred_index is not None:
        hero = frame_clusters.loc[top_pred_index]
        heading = float(hero.get("opp_theta", 0.0))
        speed = float(hero.get("cluster_speed_world", 0.0))
        arrow_length = np.clip(0.5 + speed, 0.5, 2.0)
        dx = np.cos(heading) * arrow_length
        dy = np.sin(heading) * arrow_length
        ax.arrow(
            hero["global_x"],
            hero["global_y"],
            dx,
            dy,
            head_width=0.3,
            head_length=0.3,
            fc="red",
            ec="red",
            linewidth=2.0,
            zorder=12,
        )

    gt_x, gt_y = gt_xy
    ax.scatter(
        gt_x,
        gt_y,
        marker="*",
        c="yellow",
        edgecolor="black",
        s=320,
        linewidths=1.5,
        label="GT opponent",
        zorder=15,
    )
    ego_x, ego_y = ego_xy
    if ego_x is not None and ego_y is not None:
        ax.scatter(
            ego_x,
            ego_y,
            marker="^",
            c="cyan",
            edgecolor="black",
            s=220,
            label="Ego pose",
            zorder=14,
        )

    ax.set_title(f"Frame {int(frame_points.iloc[0][FRAME_ID_COLUMN])} clusters")
    ax.set_xlabel("global_x")
    ax.set_ylabel("global_y")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    fig.canvas.draw()
    buffer, (width, height) = fig.canvas.print_to_buffer()
    image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
    image = image[..., :3]
    plt.close(fig)
    return image


def _save_velocity_plot(
    stats: list[dict],
    output_path: Path,
) -> None:
    if not stats:
        return
    df = pd.DataFrame(stats).sort_values("frame_index")
    df["running_mean"] = df["mean_speed"].expanding().mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(
        df["frame_index"],
        df["mean_speed"],
        color="darkred",
        alpha=0.8,
        s=40,
        label="frame speed",
    )
    ax.plot(
        df["frame_index"],
        df["running_mean"],
        color="navy",
        linewidth=2.0,
        label="running mean",
    )
    ax.set_xlabel("frame index")
    ax.set_ylabel("mean cluster speed (world frame)")
    ax.set_title("Predicted cluster linear velocity (running mean)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[video] Saved velocity plot to {output_path}")


def build_frames(
    merged: pd.DataFrame,
    clusters_df: pd.DataFrame,
    clf_pipe,
    threshold: float,
    frame_indices: list[int],
    total_frames: int,
    fps: float,
    duration: float,
) -> tuple[list[np.ndarray], list[dict]]:
    frames: list[np.ndarray] = []
    stats: list[dict] = []
    selected = select_frame_indices(frame_indices, total_frames)
    print(
        f"[video] Planning {len(selected)} frames (duration={duration:.1f}s at {fps:.1f}fps)"
        if total_frames > 0
        else "[video] Planning no frames"
    )
    for position, frame_index in enumerate(selected, start=1):
        print(f"[video] Rendering frame {position}/{len(selected)} (frame_index={frame_index})")
        frame_points = merged[merged[FRAME_ID_COLUMN] == frame_index]
        if frame_points.empty:
            print(f"[video][warning] No scan points for frame {frame_index}; skipping")
            continue
        frame_clusters = clusters_df[clusters_df["frame_index"] == frame_index]
        if frame_clusters.empty:
            print(f"[video][warning] No clusters for frame {frame_index}; skipping")
            continue
        frame_clusters = frame_clusters.copy()
        features = frame_clusters[CLUSTER_FEATURE_COLUMNS].astype(float).to_numpy()
        try:
            probabilities = clf_pipe.predict_proba(features)[:, 1]
        except AttributeError as exc:
            raise RuntimeError("Classifier pipeline lacks predict_proba support.") from exc
        frame_clusters["pred_prob"] = probabilities
        frame_clusters["pred_label"] = (probabilities >= threshold).astype(int)
        gt_xy = extract_ground_truth(frame_points)
        ego_xy = extract_ego_pose(frame_points)
        print(
            "[video][frame_stats] "
            f"clusters={len(frame_clusters)}, "
            f"predicted_opponents={int(frame_clusters['pred_label'].sum())}"
        )
        frame_image = render_frame_image(frame_points, frame_clusters, gt_xy, ego_xy)
        frames.append(frame_image)
        stats.append(
            {
                "frame_index": frame_index,
                "mean_speed": float(frame_clusters["cluster_speed_world"].mean()),
                "predicted_opponents": int(frame_clusters["pred_label"].sum()),
            }
        )
    if not frames:
        raise ValueError("No frames were rendered for the video.")
    if len(frames) < len(selected):
        last_frame = frames[-1]
        while len(frames) < len(selected):
            frames.append(last_frame)
    return frames, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render clustered frames into a 30s video.")
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Target video duration in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Frames per second to emit.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_SAVE_PATH,
        help="Joblib bundle containing the trained classifier.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VIDEO_DIR / "clusters.mp4",
        help="Output mp4 path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if imageio is None:
        raise SystemExit("Install imageio (`pip install imageio`) to render videos.")
    if args.duration <= 0 or args.fps <= 0:
        raise SystemExit("Duration and fps must be positive numbers.")
    print(f"[video] Loading classifier from {args.model}")
    clf_pipe, threshold = load_classifier(args.model)
    print(f"[video] Loaded classifier (threshold={threshold:.2f})")
    print("[video] Loading merged scan/odom data")
    merged = _load_merged_data()
    clusters_df, _ = build_clusters_dataframe(merged)
    frame_indices = sorted(clusters_df[FRAME_ID_COLUMN].unique())
    if not frame_indices:
        raise SystemExit("No frames found in the dataset.")
    target_count = max(1, int(round(args.duration * args.fps)))
    frames, velocity_stats = build_frames(
        merged,
        clusters_df,
        clf_pipe,
        threshold,
        frame_indices,
        target_count,
        args.fps,
        args.duration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[video] Assembling {len(frames)} frames into {args.output}")
    imageio.mimsave(args.output, frames, fps=args.fps)
    _save_velocity_plot(
        velocity_stats, args.output.parent / "clusters_velocity.png"
    )
    print(f"[video] Saved video to {args.output}")


if __name__ == "__main__":
    main()
