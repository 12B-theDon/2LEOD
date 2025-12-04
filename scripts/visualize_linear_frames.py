import os
from typing import List, Optional

import numpy as np

_CHAR_MAP = " .:-=+*#%@"


def _ascii_visual(distances: List[float], width: int = 60, height: int = 8) -> str:
    if not distances:
        return "(no scan data)"
    arr = np.asarray(distances, dtype=float)
    arr = np.clip(arr, 0.0, np.nanmax(arr) if np.nanmax(arr) > 0 else 1.0)
    scaled = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    buckets = np.linspace(0, len(scaled) - 1, num=width, dtype=int)
    rows = []
    for level in range(height - 1, -1, -1):
        line = []
        threshold = level / max(1, height - 1)
        for offset in buckets:
            value = scaled[offset]
            char_idx = int(value * (len(_CHAR_MAP) - 1))
            if value >= threshold:
                line.append(_CHAR_MAP[char_idx])
            else:
                line.append(" ")
        rows.append("".join(line))
    return "\n".join(rows)


def visualize_frame_scan(
    frame_index: int,
    distances: List[float],
    label: int,
    prediction: int,
    save_dir: Optional[str] = None,
) -> None:
    title = f"frame={frame_index} label={label} pred={prediction}"
    try:
        import matplotlib.pyplot as plt

        angles = np.linspace(-np.pi, np.pi, len(distances))
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(angles, distances, label="distance")
        ax.set_title(title)
        ax.set_ylim(0, np.max(distances) * 1.1 if distances else 1)
        ax.grid(True)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"frame_{frame_index}.png")
            fig.savefig(path)
            print(f"Saved visualization to {path}")
        else:
            fig.canvas.draw()
        plt.close(fig)
    except ImportError:
        print("MATPLOTLIB NOT AVAILABLE: falling back to ASCII scan visualization")
        print(title)
        print(_ascii_visual(distances))
