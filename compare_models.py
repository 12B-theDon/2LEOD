"""Plot ROC & PR curves for multiple trained bundles."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from logger import get_logger


PRESET_BUNDLES: list[tuple[str, Path]] = [
    ("36_over_log", Path("models/36_over_opponent_bundle_logreg.joblib")),
    #("logreg_v2", Path("models/opponent_bundle_logreg_alt.joblib")),
    ("36_over_svm", Path("models/36_over_opponent_bundle_svm.joblib")),
    #("svm_poly", Path("models/poly_opponent_bundle_svm.joblib")),
    ("36_over_xgb", Path("models/36_over_opponent_bundle_xgb.joblib")),

    ("logreg", Path("models/opponent_bundle_logreg.joblib")),
    #("logreg_v2", Path("models/opponent_bundle_logreg_alt.joblib")),
    ("svm", Path("models/opponent_bundle_svm.joblib")),
    #("svm_poly", Path("models/poly_opponent_bundle_svm.joblib")),
    ("xgb", Path("models/opponent_bundle_xgb.joblib")),
]
"""Default bundles for quick plotting; add as many entries as needed."""


DEFAULT_OUTPUT_PATH = Path("figs/runtime_compare_curves.png")
"""Default figure path used when `--output` is omitted."""


def _apply_config_override(config_path: Path | None) -> None:
    if config_path is None:
        return
    spec = importlib.util.spec_from_file_location("compare_config_override", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import config override from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _load_dataset() -> dict[str, np.ndarray]:
    from data_utils import load_dataset

    return load_dataset()


def _score(bundle: dict, X: np.ndarray) -> np.ndarray:
    clf_pipe = bundle["clf_pipe"]
    if not hasattr(clf_pipe, "predict_proba"):
        raise ValueError("Classifier bundle lacks predict_proba.")
    return np.array(clf_pipe.predict_proba(X)[:, 1], dtype=float)


def _plot_curves(
    datasets: tuple[np.ndarray, np.ndarray],
    bundles: list[tuple[str, dict]],
    out_path: Path,
) -> None:
    X_test, y_test = datasets
    fig, axes = plt.subplots(ncols=3, figsize=(18, 5))
    roc_ax, pr_ax, score_ax = axes

    bins = np.linspace(0.0, 1.0, 51)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (label, bundle) in enumerate(bundles):
        scores = _score(bundle, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        prec, rec, _ = precision_recall_curve(y_test, scores)
        roc_auc = roc_auc_score(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)

        roc_ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
        pr_ax.plot(rec, prec, label=f"{label} (AP={pr_auc:.3f})")

        color = colors[idx % len(colors)]
        neg_scores = scores[y_test == 0]
        pos_scores = scores[y_test == 1]
        score_ax.hist(
            neg_scores,
            bins=bins,
            density=True,
            alpha=0.35,
            color=color,
            label=f"{label} background",
            edgecolor="none",
        )
        score_ax.hist(
            pos_scores,
            bins=bins,
            density=True,
            alpha=0.35,
            color=color,
            label=f"{label} opponent",
            edgecolor="none",
        )

    roc_ax.set_title("ROC curve")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1)
    roc_ax.grid(True)
    roc_ax.legend(loc="lower right")

    pr_ax.set_title("Precision-Recall curve")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_ylim(0, 1)
    pr_ax.grid(True)
    pr_ax.legend(loc="lower left")

    score_ax.set_xlim(0, 1)
    score_ax.set_ylim(bottom=0)
    score_ax.set_xlabel("Predicted probability P(y=1)")
    score_ax.set_ylabel("Density")
    score_ax.set_title("Test score distribution")
    score_ax.grid(True, alpha=0.3)
    score_ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple model bundles via ROC and PR curves."
    )
    parser.add_argument(
        "--bundles",
        nargs="+",
        type=Path,
        help="Override the preset bundles with custom joblib files.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Optional labels for each bundle (must match bundle count).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="File path for the output figure.",
    )
    parser.add_argument(
        "--config-override",
        type=Path,
        default=Path("36_config.py"),
        help="Optional config module to import before building the dataset (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.bundles and args.names and len(args.names) != len(args.bundles):
        parser.error("Number of --names must match number of bundles.")

    logger = get_logger("compare_models")
    _apply_config_override(args.config_override)
    logger.info("Loading dataset for model comparison...")
    dataset = _load_dataset()
    logger.info("Dataset ready; scaffolding comparison.")
    datasets = (dataset["X_test"], dataset["y_class_test"])

    loaded: list[tuple[str, dict]] = []
    if args.bundles:
        missing = [str(p) for p in args.bundles if not p.exists()]
        if missing:
            raise SystemExit(f"Missing bundle files: {', '.join(missing)}")
        for idx, bundle_path in enumerate(args.bundles):
            bundle = joblib.load(bundle_path)
            label = args.names[idx] if args.names else bundle_path.stem
            loaded.append((label, bundle))
    else:
        presets = [(label, path) for label, path in PRESET_BUNDLES if path.exists()]
        if not presets:
            raise SystemExit("No preset bundles are available; specify with --bundles.")
        for label, bundle_path in presets:
            bundle = joblib.load(bundle_path)
            loaded.append((label, bundle))
            logger.info(f"Loaded preset bundle '{label}' from {bundle_path}")

    logger.info("Plotting ROC and PR curves.")
    _plot_curves(datasets, loaded, args.output)
    logger.info(f"Wrote comparison figure to {args.output}")


if __name__ == "__main__":
    main()
