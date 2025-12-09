import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    CLASSIFIER_TYPE,
    LOGREG_C,
    LOGREG_DECISION_THRESHOLD,
    LOGREG_MAX_ITER,
    MODEL_SAVE_PATH,
    ODOM_TARGET_COLUMNS,
    TARGET_BG_RATIO,
    THRESHOLDS,
)
from data_utils import load_dataset
from logger import get_logger
from models import build_classifier_pipeline, build_regressor


def _evaluate_thresholds(
    thresholds: list[float], y_true: np.ndarray, y_prob: np.ndarray, logger
) -> dict:
    best = {
        "threshold": thresholds[0],
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "balanced": -1.0,
    }
    best_precision_candidate = None
    rows: list[tuple[float, float, float, float, float]] = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        balanced = balanced_accuracy_score(y_true, y_pred)

        logger.info(
            f"[train][threshold={threshold:.2f}] precision={precision:.4f}, recall={recall:.4f}, "
            f"f1={f1:.4f}, balanced={balanced:.4f}"
        )
        logger.info(
            f"[train][threshold={threshold:.2f}] confusion matrix:\n"
            + str(confusion_matrix(y_true, y_pred))
        )

        rows.append((threshold, precision, recall, f1, balanced))

        if balanced > best["balanced"]:
            best.update(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "balanced": balanced,
                }
            )

        if recall >= 0.7:
            if (
                best_precision_candidate is None
                or precision > best_precision_candidate["precision"]
            ):
                best_precision_candidate = {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "balanced": balanced,
                }

    if (
        best_precision_candidate
        and best["precision"] < best["recall"]
        and best_precision_candidate["precision"] >= best["precision"]
    ):
        selected = best_precision_candidate
    else:
        selected = best

    logger.info(
        f"[train][threshold] Selected threshold={selected['threshold']:.2f} "
        f"(precision={selected['precision']:.4f}, recall={selected['recall']:.4f}, "
        f"balanced={selected['balanced']:.4f})"
    )
    table_lines = ["threshold | precision | recall | f1 | balanced"]
    for threshold, precision, recall, f1, balanced in rows:
        table_lines.append(
            f"{threshold:>9.2f} | {precision:>9.4f} | {recall:>6.4f} | {f1:>4.4f} | {balanced:>8.4f}"
        )
    logger.info("[train][threshold] Summary:\n" + "\n".join(table_lines))
    return selected


def _log_relabel_metrics(
    logger,
    label_orig: np.ndarray,
    label_new: np.ndarray,
    gt_label: np.ndarray,
    gt_valid: np.ndarray,
):
    mask = gt_valid.astype(bool)
    if not mask.any():
        logger.info("[data][relabel] No GT-valid clusters available for evaluation.")
        return

    def _metrics(y_pred: np.ndarray, tag: str):
        acc = accuracy_score(gt_label[mask], y_pred[mask])
        prec = precision_score(gt_label[mask], y_pred[mask], zero_division=0)
        rec = recall_score(gt_label[mask], y_pred[mask], zero_division=0)
        cm = confusion_matrix(gt_label[mask], y_pred[mask])
        logger.info(
            f"[data][relabel][{tag}] acc={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}"
        )
        tn, fp, fn, tp = cm.ravel()
        logger.info(
            f"[data][relabel][cm_{tag}] TN={tn}, FP={fp}, FN={fn}, TP={tp}"
        )
        return cm

    _metrics(label_orig, "orig_vs_gt")
    _metrics(label_new, "new_vs_gt")
    changed_mask = mask & (label_orig != label_new)
    improved = int(
        np.sum(
            (label_orig != gt_label)
            & (label_new == gt_label)
            & changed_mask
        )
    )
    worsened = int(
        np.sum(
            (label_orig == gt_label)
            & (label_new != gt_label)
            & changed_mask
        )
    )
    unchanged = int(
        np.sum(
            (label_orig != gt_label)
            & (label_new != gt_label)
            & changed_mask
        )
    )
    logger.info(
        f"[data][relabel][changed_only] improved={improved}, worsened={worsened}, unchanged={unchanged}"
    )


def main() -> None:
    logger = get_logger("train_logger")
    logger.info("Loading dataset...")
    data = load_dataset()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_class_train = data["y_class_train"]
    y_class_test = data["y_class_test"]
    y_reg_train = data["y_reg_train"]
    y_reg_test = data["y_reg_test"]
    train_visible_mask = data["train_positive_mask"]
    test_visible_mask = data["test_positive_mask"]
    stats = data["train_sample_stats"]
    motion_stats = data["motion_stats"]
    relative_stats = data["relative_stats"]
    relabel_stats = data["relabel_stats"]

    total_raw = stats["pos_count"] + stats["neg_count"]
    ratio_percent = (stats["pos_count"] / total_raw * 100) if total_raw else 0.0
    logger.info(
        f"[data] Opponent samples before sampling: {stats['pos_count']} / {total_raw} ({ratio_percent:.2f}%)"
    )
    logger.info(
        f"[data][sampling] After sampling: {stats['pos_count']} opponent vs {stats['bg_kept']} background "
        f"(target ratio {TARGET_BG_RATIO:.1f}:1)"
    )
    logger.info(f"[data] Feature dimension: {X_train.shape[1]}")
    logger.info(
        f"[data][world motion] vx_mean={motion_stats['cluster_vx_world_mean']:.4f}, "
        f"vx_std={motion_stats['cluster_vx_world_std']:.4f}, "
        f"speed_mean={motion_stats['cluster_speed_world_mean']:.4f}, "
        f"speed_std={motion_stats['cluster_speed_world_std']:.4f}"
    )
    logger.info(
        f"[data][relative motion] rel_vx_mean={relative_stats['cluster_rel_vx_mean']:.4f}, "
        f"rel_vx_std={relative_stats['cluster_rel_vx_std']:.4f}, "
        f"rel_speed_mean={relative_stats['cluster_rel_speed_mean']:.4f}, "
        f"rel_speed_std={relative_stats['cluster_rel_speed_std']:.4f}"
    )
    logger.info(
        f"[data][relabel] {relabel_stats['relabeled']} / {relabel_stats['total']} clusters relabeled "
        f"({relabel_stats['ratio']:.2%})"
    )
    label_orig_all = np.concatenate(
        [data["label_orig_train"], data["label_orig_test"]]
    )
    label_new_all = np.concatenate(
        [data["y_class_train"], data["y_class_test"]]
    )
    gt_label_all = np.concatenate(
        [data["gt_label_train"], data["gt_label_test"]]
    )
    gt_valid_all = np.concatenate(
        [data["gt_valid_train"], data["gt_valid_test"]]
    )
    _log_relabel_metrics(
        logger, label_orig_all, label_new_all, gt_label_all, gt_valid_all
    )
    logger.info("[data] Using class_weight='balanced'")

    pos_train = int(np.sum(y_class_train == 1))
    neg_train = int(np.sum(y_class_train == 0))
    scale_pos_weight = neg_train / max(1, pos_train)
    logger.info(
        f"[train][data] Training positives={pos_train}, negatives={neg_train}, scale_pos_weight={scale_pos_weight:.2f}"
    )
    logger.info(f"[train][classifier] Type={CLASSIFIER_TYPE}")
    logger.info("Building classifier pipeline...")
    clf_pipe = build_classifier_pipeline(CLASSIFIER_TYPE, scale_pos_weight)
    if CLASSIFIER_TYPE == "logreg":
        logger.info(
            f"[train][logreg] Hyperparameters: C={LOGREG_C}, max_iter={LOGREG_MAX_ITER}, solver=lbfgs, n_jobs=-1"
        )
    elif CLASSIFIER_TYPE == "svm":
        logger.info(
            "[train][svm] Hyperparameters: kernel=rbf, C=1.0, gamma=scale, "
            "lda components=1, probability=True"
        )
    elif CLASSIFIER_TYPE == "xgb":
        logger.info(
            "[train][xgb] Hyperparameters: max_depth=4, lr=0.1, estimators=300, "
            "subsample=0.8, colsample_bytree=0.8"
        )

    logger.info("Training classifier...")
    clf_pipe.fit(X_train, y_class_train)

    y_pred_train = clf_pipe.predict(X_train)
    y_pred_test = clf_pipe.predict(X_test)

    train_acc = accuracy_score(y_class_train, y_pred_train)
    test_acc = accuracy_score(y_class_test, y_pred_test)
    bal_acc = balanced_accuracy_score(y_class_test, y_pred_test)

    prefix = CLASSIFIER_TYPE
    logger.info(f"[train][{prefix}] Train accuracy: {train_acc:.4f}")
    logger.info(f"[train][{prefix}] Test accuracy:  {test_acc:.4f}")
    logger.info(f"[train][{prefix}] Balanced accuracy: {bal_acc:.4f}")
    logger.info(
        f"[train][{prefix}] Confusion matrix:\n" + str(confusion_matrix(y_class_test, y_pred_test))
    )
    logger.info(
        f"[train][{prefix}] Classification report:\n"
        + classification_report(y_class_test, y_pred_test)
    )

    y_prob = clf_pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_class_test, y_prob)
    pr_auc = average_precision_score(y_class_test, y_prob)
    logger.info(f"[train][threshold] ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    _evaluate_thresholds(THRESHOLDS, y_class_test, y_prob, logger)
    fixed_threshold = LOGREG_DECISION_THRESHOLD
    logger.info(
        f"[train][threshold] Using fixed deployment threshold={fixed_threshold:.2f}"
    )

    if not np.any(train_visible_mask):
        raise RuntimeError("No opponent clusters available for regression training.")

    logger.info("Preparing regression data (visible clusters)...")
    X_train_visible = X_train[train_visible_mask]
    y_reg_train_visible = y_reg_train[train_visible_mask]
    X_test_visible = X_test[test_visible_mask]
    y_reg_test_visible = y_reg_test[test_visible_mask]

    logger.info(f"[train][reg] Visible train clusters: {len(X_train_visible)}")

    regressor = build_regressor()
    logger.info("Training regressor (visible frames only)...")
    regressor.fit(X_train_visible, y_reg_train_visible)

    if len(X_test_visible) > 0:
        y_reg_pred = regressor.predict(X_test_visible)
        rmse_per_dim = np.sqrt(np.mean((y_reg_test_visible - y_reg_pred) ** 2, axis=0))
        for name, rmse in zip(ODOM_TARGET_COLUMNS, rmse_per_dim):
            logger.info(f"[train][reg] RMSE ({name}): {rmse:.4f}")
    else:
        logger.warning("[train][reg] Skipping regression eval; no visible test clusters.")

    logger.info("Saving model bundle...")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "clf_pipe": clf_pipe,
            "regressor": regressor,
            "threshold": fixed_threshold,
        },
        MODEL_SAVE_PATH,
    )
    logger.info(f"Saved model to {MODEL_SAVE_PATH}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
