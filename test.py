import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import MODEL_SAVE_PATH, LOGREG_DECISION_THRESHOLD, ODOM_TARGET_COLUMNS
from data_utils import load_dataset
from logger import get_logger


def _log_classification(logger, prefix, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    logger.info(f"{prefix} accuracy: {acc:.4f}")
    logger.info(f"{prefix} balanced accuracy: {bal_acc:.4f}")
    logger.info(f"{prefix} confusion matrix:\n{confusion_matrix(y_true, y_pred)}")
    logger.info(f"{prefix} classification report:\n{classification_report(y_true, y_pred)}")


def main() -> None:
    logger = get_logger("test_logger")
    logger.info("Loading dataset for evaluation...")
    data = load_dataset()

    X_test = data["X_test"]
    y_class_test = data["y_class_test"]
    y_reg_test = data["y_reg_test"]
    test_visible_mask = data["test_positive_mask"]

    model_bundle = joblib.load(MODEL_SAVE_PATH)
    clf_pipe = model_bundle["clf_pipe"]
    regressor = model_bundle["regressor"]
    threshold = model_bundle.get("threshold", LOGREG_DECISION_THRESHOLD)

    y_pred = clf_pipe.predict(X_test)
    _log_classification(logger, "[test][clf]", y_class_test, y_pred)

    probs = clf_pipe.predict_proba(X_test)[:, 1]
    threshold_pred = (probs >= threshold).astype(int)
    logger.info(f"[test][clf] Thresholding at {threshold:.2f}: positives={threshold_pred.sum()}")
    _log_classification(logger, "[test][clf<threshold]", y_class_test, threshold_pred)

    X_test_visible = X_test[test_visible_mask]
    y_reg_test_visible = y_reg_test[test_visible_mask]

    if len(X_test_visible) == 0:
        logger.warning("[test][reg] No visible clusters for regression evaluation.")
        return

    y_reg_pred = regressor.predict(X_test_visible)
    rmse_per_dim = np.sqrt(np.mean((y_reg_test_visible - y_reg_pred) ** 2, axis=0))
    for name, rmse in zip(ODOM_TARGET_COLUMNS, rmse_per_dim):
        logger.info(f"[test][reg] RMSE ({name}): {rmse:.4f}")


if __name__ == "__main__":
    main()
