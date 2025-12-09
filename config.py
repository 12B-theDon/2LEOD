from pathlib import Path

# File paths
SCAN_CSV_PATH = Path("dataFiles/output_1.csv")
ODOM_CSV_PATH = Path("dataFiles/odom_output_1.csv")

# Raw columns
FEATURE_COLUMNS = [
    "local_x",
    "local_y",
    "global_x",
    "global_y",
    "scan_index",
    "distance",
]
LABEL_CLASS_COLUMN = "isOpponent"
FRAME_ID_COLUMN = "frame_index"
SCAN_INDEX_COLUMN = "scan_index"
DISTANCE_COLUMN = "distance"

ODOM_SOURCE_COLUMNS = [
    "base_link_op_x",
    "base_link_op_y",
    "base_link_op_yaw",
]
ODOM_TARGET_COLUMNS = ["opp_x", "opp_y", "opp_theta"]

# Cluster-level features produced from contiguous sweep segments
CLUSTER_FEATURE_COLUMNS = [
    "center_local_x",
    "center_local_y",
    "global_x",
    "global_y",
    "cluster_size",
    "distance_mean",
    "distance_std",
    "range_extent",
    "angle_index",
    "angle_extent",
    "delta_range",
    "ego_v",
    "ego_w",
    "ego_vx",
    "ego_vy",
    "local_x_extent",
    "local_y_extent",
    "local_x_std",
    "local_y_std",
    "cluster_cx",
    "cluster_cy",
    "cluster_radius",
    "cluster_vx",
    "cluster_vy",
    "cluster_speed",
    "cluster_vr",
    "cluster_vx_world",
    "cluster_vy_world",
    "cluster_speed_world",
    "cluster_rel_vx",
    "cluster_rel_vy",
    "cluster_rel_speed",
    "cluster_spread",
    "cluster_r_span",
    "cluster_angle_entropy",
    "distance_to_ego",
    "bearing",
    "cluster_opponent_ratio",
]

# Encoder/model/config
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
LOGREG_DECISION_THRESHOLD = 0.55
MODEL_SAVE_PATH = Path("models/opponent_bundle.joblib")

# Sampling/thresholding enhancements
TARGET_BG_RATIO = 8.0
THRESHOLDS = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
LOGREG_C = 0.3
LOGREG_MAX_ITER = 2000
CLASSIFIER_TYPE = "logreg"
WORLD_MATCH_DIST = 1.0
