import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from config import get_default_train_config, update_config_from_args
from models.linear_regression import LinearRegressor
from models.logistic_regression import LogisticClassifier
from models.pca_encoder import PCAEncoder
from scripts.data_utils import load_scan_dataset, split_frames
from scripts.logging_utils import setup_logger
from scripts.visualize_linear_frames import visualize_frame_scan
from tools.tsne_visualizer import plot_tsne


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train encoder on labeled scan data")
    parser.add_argument("data_csv", help="Path to the labeled scan CSV file")
    parser.add_argument("--pca-components", type=int, help="Number of PCA components")
    parser.add_argument("--logistic-lr", type=float, help="Learning rate for logistic regression")
    parser.add_argument("--logistic-epochs", type=int, help="Epochs for logistic regression")
    parser.add_argument("--logistic-reg", type=float, help="L2 regularization for logistic regression")
    parser.add_argument("--linear-reg", type=float, help="Regularization for linear regression")
    parser.add_argument("--train-ratio", type=float, help="Train split ratio (0-1)")
    parser.add_argument("--visualize-samples", type=int, help="Number of training frames to visualize")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory where checkpoints are stored")
    parser.add_argument("--random-seed", type=int, help="Seed for reproducible splits")
    return parser.parse_args()


def _feature_json_for_csv(csv_path: str) -> Path:
    csv_path_obj = Path(csv_path)
    return csv_path_obj.with_name(f"{csv_path_obj.stem}_feature_index.json")


def _build_feature_matrix(data: list, latent: np.ndarray) -> np.ndarray:
    odom = np.asarray([entry["odom"] for entry in data], dtype=float)
    return np.hstack([latent, odom])


def _save_split_indices(path: str, train_frames: list, test_frames: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"train_frames": train_frames, "test_frames": test_frames}, fh)


def main() -> None:
    args = parse_args()
    config = get_default_train_config()
    update_config_from_args(config, args)

    dataset_path = Path(args.data_csv)
    checkpoint_base = Path(args.checkpoint_dir or config.checkpoint_dir)
    checkpoint_dir = checkpoint_base / dataset_path.stem
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = setup_logger("train", os.path.join(checkpoint_dir, "train.log"))
    logger.info("Starting training run with dataset=%s", args.data_csv)

    dataset = load_scan_dataset(str(_feature_json_for_csv(args.data_csv)))
    if not dataset:
        logger.error("No frames were read from the provided CSV")
        raise RuntimeError("No frames were read from the provided CSV")

    logger.info("Loaded %d frames", len(dataset))

    train_set, test_set = split_frames(dataset, config.train_ratio, config.random_seed)
    train_frame_ids = sorted(entry["frame_index"] for entry in train_set)
    test_frame_ids = sorted(entry["frame_index"] for entry in test_set)

    split_index_path = os.path.join(checkpoint_dir, "split_indices.json")
    _save_split_indices(split_index_path, train_frame_ids, test_frame_ids)
    logger.info("Saved split indices to %s", split_index_path)

    X_train = np.asarray([entry["distances"] for entry in train_set], dtype=float)
    y_train = np.asarray([entry["label"] for entry in train_set], dtype=float)

    pca = PCAEncoder(config.pca_components)
    latent_train = pca.fit_transform(X_train)
    feature_matrix = _build_feature_matrix(train_set, latent_train)

    logistic = LogisticClassifier(
        learning_rate=config.logistic_lr,
        epochs=config.logistic_epochs,
        regularization=config.logistic_reg,
        random_seed=config.random_seed,
    )
    logistic.fit(feature_matrix, y_train)

    linear = LinearRegressor(regularization=config.linear_reg_reg)
    linear.fit(feature_matrix, y_train)

    pca.save(checkpoint_dir)
    logistic.save(checkpoint_dir)
    linear.save(checkpoint_dir)
    np.save(os.path.join(checkpoint_dir, "train_latent.npy"), latent_train)
    np.save(
        os.path.join(checkpoint_dir, "train_frame_ids.npy"), np.array(train_frame_ids, dtype=int)
    )
    logger.info("Saved checkpoint artifacts to %s", checkpoint_dir)

    tsne_dir = Path("train_result_figs")
    tsne_dir.mkdir(parents=True, exist_ok=True)
    tsne_path = tsne_dir / f"{dataset_path.stem}_tsne.png"
    plot_tsne(latent_train, y_train, tsne_path)
    logger.info("Saved t-SNE plot to %s", tsne_path)

    predictions = logistic.predict(feature_matrix, threshold=config.similarity_threshold)
    accuracy = np.mean(predictions == y_train)
    logger.info("Training accuracy: %.3f", accuracy)
    logger.info("Train logistic accuracy: %.3f", accuracy)
    logger.info("Train ensemble accuracy: %.3f", accuracy)
    print(f"Training accuracy: {accuracy:.3f}")

    visualization_dir = os.path.join(checkpoint_dir, "visualizations")
    sample_count = min(config.visualize_samples, len(train_set))
    sample_indices = random.Random(config.random_seed).sample(range(len(train_set)), sample_count)
    for idx in sample_indices:
        entry = train_set[idx]
        sample_feature = feature_matrix[idx : idx + 1]
        pred = int(logistic.predict(sample_feature, threshold=config.similarity_threshold)[0])
        logger.info(
            "Visualizing frame %s (label=%s prediction=%s)",
            entry["frame_index"],
            entry["label"],
            pred,
        )
        visualize_frame_scan(
            frame_index=entry["frame_index"],
            distances=entry["distances"],
            label=entry["label"],
            prediction=pred,
            save_dir=visualization_dir,
        )

    logger.info("Checkpoint artifacts: %s", checkpoint_dir)
    logger.info("Split indices stored in %s", split_index_path)


if __name__ == "__main__":
    main()
