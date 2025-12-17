from __future__ import annotations

from pathlib import Path

import config as _base_config

# Remove the opponent-ratio feature to keep the vector 36-dimensional.
CLUSTER_FEATURE_COLUMNS = [
    column
    for column in _base_config.CLUSTER_FEATURE_COLUMNS
    if column != "cluster_opponent_ratio"
]
_base_config.CLUSTER_FEATURE_COLUMNS = CLUSTER_FEATURE_COLUMNS

# Logistic regression defaults (repeat base defaults but easily tweaked here).
LOGREG_C = 0.3
LOGREG_MAX_ITER = 2000
_base_config.LOGREG_C = LOGREG_C
_base_config.LOGREG_MAX_ITER = LOGREG_MAX_ITER

# SVM pipeline control knobs.
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"
SVM_LDA_COMPONENTS = 1
_base_config.SVM_KERNEL = SVM_KERNEL
_base_config.SVM_C = SVM_C
_base_config.SVM_GAMMA = SVM_GAMMA
_base_config.SVM_LDA_COMPONENTS = SVM_LDA_COMPONENTS

# XGBoost hyperparameters.
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.01
XGB_N_ESTIMATORS = 600
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_MIN_CHILD_WEIGHT = 10
XGB_USE_LABEL_ENCODER = False
XGB_EVAL_METRIC = "logloss"
_base_config.XGB_MAX_DEPTH = XGB_MAX_DEPTH
_base_config.XGB_LEARNING_RATE = XGB_LEARNING_RATE
_base_config.XGB_N_ESTIMATORS = XGB_N_ESTIMATORS
_base_config.XGB_SUBSAMPLE = XGB_SUBSAMPLE
_base_config.XGB_COLSAMPLE_BYTREE = XGB_COLSAMPLE_BYTREE
_base_config.XGB_MIN_CHILD_WEIGHT = XGB_MIN_CHILD_WEIGHT
_base_config.XGB_USE_LABEL_ENCODER = XGB_USE_LABEL_ENCODER
_base_config.XGB_EVAL_METRIC = XGB_EVAL_METRIC

TARGET_BG_RATIO = _base_config.TARGET_BG_RATIO
THRESHOLDS = _base_config.THRESHOLDS
ODOM_TARGET_COLUMNS = _base_config.ODOM_TARGET_COLUMNS

MODEL_SAVE_PATH = Path("models/opponent_bundle_36.joblib")
