import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from config import get_default_test_config, update_config_from_args
from models.linear_regression import LinearRegressor
from models.logistic_regression import LogisticClassifier
from models.pca_encoder import PCAEncoder
from scripts.data_utils import load_scan_dataset
from scripts.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen encoder on held-out data")
    parser.add_argument("data_csv", help="Path to the labeled scan CSV file")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", help="Where checkpoints were stored")
    parser.add_argument("--split-indices", dest="split_indices", help="JSON file with train/test split ids")
    parser.add_argument("--threshold", type=float, help="Decision threshold for logistic classifier")
    return parser.parse_args()


def _feature_json_for_csv(csv_path: str) -> Path:
    csv_path_obj = Path(csv_path)
    return csv_path_obj.with_name(f"{csv_path_obj.stem}_feature_index.json")

def _classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _load_split_indices(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()
    config = get_default_test_config()
    update_config_from_args(config, args)

    dataset_path = Path(args.data_csv)
    checkpoint_base = Path(args.checkpoint_dir or config.checkpoint_dir)
    checkpoint_dir = checkpoint_base / dataset_path.stem
    split_indices_path = args.split_indices or os.path.join(checkpoint_dir, "split_indices.json")
    thresholds = args.threshold if args.threshold is not None else config.threshold
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = setup_logger("test", os.path.join(checkpoint_dir, "test.log"))
    logger.info("Evaluating dataset=%s split=%s", args.data_csv, split_indices_path)

    feature_json = _feature_json_for_csv(args.data_csv)
    dataset = load_scan_dataset(str(feature_json))
    if not dataset:
        logger.error("No data available for evaluation")
        raise RuntimeError("No data available for evaluation")

    logger.info("Loaded %d frames for evaluation", len(dataset))

    splits = _load_split_indices(split_indices_path)
    test_frame_set = set(splits.get("test_frames", []))
    test_data = [entry for entry in dataset if entry["frame_index"] in test_frame_set]
    if not test_data:
        logger.error("No test frames found in the provided CSV using split indices")
        raise RuntimeError("No test frames found in the provided CSV using split indices")

    X_test = np.asarray([entry["distances"] for entry in test_data], dtype=float)
    y_test = np.asarray([entry["label"] for entry in test_data], dtype=float)
    pca = PCAEncoder(n_components=X_test.shape[1])
    pca.load(checkpoint_dir)
    pca.n_components = pca.components_.shape[0]
    latent_test = pca.transform(X_test)
    odom_features = np.asarray([entry["odom"] for entry in test_data], dtype=float)
    feature_matrix = np.hstack([latent_test, odom_features])

    logistic = LogisticClassifier()
    logistic.load(checkpoint_dir)
    start_time = time.monotonic()
    predictions = logistic.predict(feature_matrix, threshold=thresholds)
    report = _classification_report(y_test, predictions)
    logger.info("Classification report: %s", {k: report[k] for k in ["accuracy", "precision", "recall", "f1"]})
    logger.info("Confusion counts: TP=%d TN=%d FP=%d FN=%d", report["tp"], report["tn"], report["fp"], report["fn"])
    duration = time.monotonic() - start_time
    logger.info("Test logistic accuracy: %.3f", report["accuracy"])
    logger.info("Test ensemble accuracy: %.3f", report["accuracy"])
    logger.info("Test evaluation duration: %.1f sec", duration)
    print("Classification report:")
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {key}: {report[key]:.3f}")
    print(f"  TP={report['tp']} TN={report['tn']} FP={report['fp']} FN={report['fn']}")

    linear = LinearRegressor()
    linear.load(checkpoint_dir)
    linear_output = linear.predict(feature_matrix)
    mse = float(np.mean((linear_output - y_test) ** 2))
    logger.info("Linear regression MSE on test frames: %.4f", mse)
    print(f"Linear regression MSE on test frames: {mse:.4f}")


if __name__ == "__main__":
    main()
