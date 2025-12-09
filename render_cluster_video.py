import argparse
import os
import subprocess
import sys
from pathlib import Path

import imageio

from data_utils import build_clusters_dataframe

VIDEO_DIR = Path("figs")


def collect_frames(limit: int | None = None) -> list[int]:
    clusters_df, _ = build_clusters_dataframe()
    valid_frames = sorted(clusters_df["frame_index"].unique())
    if limit is not None and limit > 0:
        valid_frames = valid_frames[:limit]
    return valid_frames


def render_frame(frame_index: int) -> Path:
    env = os.environ.copy()
    env["VISUALIZE_SHOW"] = "0"
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(
        [sys.executable, "visualize_clusters.py", "--frame-index", str(frame_index)],
        check=True,
        env=env,
    )
    return VIDEO_DIR / f"cluster_debug_frame_{frame_index}.png"


def build_video(image_paths: list[Path], output: Path, fps: int) -> None:
    frames = []
    for path in image_paths:
        if path.exists():
            frames.append(imageio.imread(path))
    if not frames:
        raise ValueError("No visualization frames found to build video.")
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render cluster visualization video.")
    parser.add_argument("--limit", type=int, help="Max number of frames to include.")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second for the mp4.")
    parser.add_argument("--output", type=Path, default=VIDEO_DIR / "clusters.mp4")
    parser.add_argument("--all", action="store_true", help="Render all collected frames instead of only first/last.")
    args = parser.parse_args()

    frames = collect_frames(limit=args.limit)
    if not frames:
        raise SystemExit("No cluster frames available for visualization.")

    if args.all:
        selected_frames = frames
    else:
        selected_frames = frames if len(frames) <= 2 else [frames[0], frames[-1]]
    image_paths = []
    for frame_index in selected_frames:
        logger_msg = f"[video] Rendering frame {frame_index}"
        print(logger_msg)
        try:
            image_paths.append(render_frame(frame_index))
        except subprocess.CalledProcessError as exc:
            print(f"[video][warning] Skipping frame {frame_index}: {exc}")

    build_video(image_paths, args.output, args.fps)
    print(f"[video] Saved mp4 to {args.output}")


if __name__ == "__main__":
    main()
