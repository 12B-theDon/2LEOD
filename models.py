from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import LOGREG_C, LOGREG_MAX_ITER


def build_logistic_pipeline() -> Pipeline:
    """StandardScaler → LogisticRegression with strong regularization."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    C=LOGREG_C,
                    max_iter=LOGREG_MAX_ITER,
                    solver="lbfgs",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_svm_pipeline() -> Pipeline:
    """Pipeline that stacks LDA with an RBF SVM for richer decision surfaces."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(n_components=1)),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                ),
            ),
        ]
    )


def build_xgb_pipeline(scale_pos_weight: float) -> Pipeline:
    """XGBoost classifier with gradient boosting and imbalance handling."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    max_depth=4,
                    learning_rate=0.1,
                    n_estimators=300,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    use_label_encoder=False,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def build_classifier_pipeline(classifier_type: str, scale_pos_weight: float = 1.0) -> Pipeline:
    """Factory for classifier pipelines keyed by config.CLASSIFIER_TYPE."""
    if classifier_type == "logreg":
        return build_logistic_pipeline()
    if classifier_type == "svm":
        return build_svm_pipeline()
    if classifier_type == "xgb":
        return build_xgb_pipeline(scale_pos_weight)
    raise ValueError(f"Unknown classifier type '{classifier_type}'")


def build_regressor() -> LinearRegression:
    """Simple linear regression trained on visible cluster features."""
    return LinearRegression()
