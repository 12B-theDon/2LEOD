from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne(
    latent: Sequence[Sequence[float]],
    labels: Sequence[int],
    output_path: Path,
    title: str = "latent t-SNE",
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(latent, dtype=float)
    proj = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(matrix)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=np.asarray(labels), cmap="coolwarm", s=8, alpha=0.7)
    plt.title(title)
    plt.colorbar(scatter, label="label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
