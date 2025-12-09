# Opponent Tracking Pipeline

## Overview
- **High-level goal**: Using only 2D LiDAR (`/scan`, 360 beams) and ego odometry (`/odom`, containing position, orientation, and velocities), the system classifies which LiDAR cluster corresponds to the opponent vehicle, regresses its pose (`x, y, yaw`) relative to the ego frame, and in ROS2 publishes `/opponent_odom` only when an opponent is likely visible.
- **Inputs**:
  - `/scan`: `sensor_msgs/LaserScan` produced by the onboard LiDAR.
  - `/odom`: ego vehicle odometry (`x, y, yaw, linear_velocity, angular_velocity`).
- **Output**: A classifier + regressor bundle that detects opponent clusters and produces a pose estimate; deployed ROS2 node publishes `/opponent_odom` when detection surpasses a calibrated threshold.

## Data Format
1. **Frame-level CSV**:
   - Columns: `frame_index, stamp_sec, stamp_nsec, base_link_x, base_link_y, base_link_yaw, base_link_op_x, base_link_op_y, base_link_op_yaw`.
   - Meaning: `base_link_*` is ego pose in odom/world coordinates and `base_link_op_*` is the ground-truth opponent pose for that timestamp.
2. **Scan point-level CSV**:
   - Columns: `frame_index, stamp_sec, stamp_nsec, scan_index, distance, local_x, local_y, global_x, global_y, isOpponent, isWall, isStatic, isFree`.
   - Meaning: Each LiDAR point is already projected into local/global coordinates; boolean labels indicate its type.
Each dataset therefore ships with one frame CSV and one point CSV that cover the same timestamps.

## Pipeline Overview (Classical ML Encoder)
- **Clustering**: Consecutive `scan_index` values form clusters of points representing rigid objects or walls. Each cluster stores metadata such as start/end indices, timestamps, and centroid positions.
- **Geometric features**:
  - Cluster size (number of points).
  - Centroid (`cx, cy`) in both local and global frames.
  - Radius and distance statistics (min/max/mean).
  - Angular span and range extents along `x`/`y`.
  - Shape descriptors (lengths, spread, entropy).
- **Motion features**:
  - Ego velocities computed from frame odometry (`vx, vy, speed`, derived from `/odom` deltas).
  - Cluster motion relative to ego when past frames are available (delta positions/velocities).
- **Label-based ratios (training only)**:
  - `opponent_ratio`, `isStatic_ratio`, `isWall_ratio`, `isFree_ratio`, etc., computed by counting labeled points inside a cluster.
  - These ratios help distinguish surfaces even though they are not available at inference time.
- **Feature vector**: At training time each cluster contributes a ~38-dimensional vector combining geometric, motion, and label-ratio statistics.

## Re-labeling Strategy (Point → Cluster)
- **Problem**: Point-level labels are noisy (mixed classes inside a cluster) and suffer from severe class imbalance.
- **GT-based cluster label**:
  - Given `base_link_op_x/y` per frame, compute cluster centroid `cx, cy`. Define
    ```
    dist = sqrt((cx - op_x)**2 + (cy - op_y)**2)
    GT_RADIUS = 2.2  # meters
    gt_label = 1 if dist < GT_RADIUS else 0
    ```
- **Opponent ratio filter**:
  - `OPP_RATIO_THRESHOLD = 0.10`. If `opponent_ratio >= threshold`, mark the cluster as an opponent candidate regardless of distance.
- **Final label**:
  - `label_new = 1` if `(gt_label == 1 or opponent_ratio >= OPP_RATIO_THRESHOLD)` else `0`.
- **Logging**: The code reports `orig_vs_gt` and `new_vs_gt` metrics (accuracy, precision, recall, confusion matrix) plus how many clusters improved or worsened in GT consistency after relabeling.

## Class Imbalance & Sampling
- **Cluster imbalance**: Opponent clusters become ~9% of the dataset after relabeling.
- **Undersampling**:
  - Preserve all positives (`label_new = 1`).
  - Randomly sample negatives to achieve `TARGET_BG_RATIO = 8.0` (e.g., if `n_pos = 264`, then `n_neg ≈ 8 · 264 = 2112`).
- **Classifier weight**: `class_weight='balanced'` further compensates for imbalance during training.

## Training Scripts & Artefacts
- **`data_utils.py`**:
  - Loads frame/point CSVs and builds frame-wise and cluster-wise data structures.
  - Computes all geometric/motion features, label ratios, GT labels, and sampling decisions.
  - Emits summary logs (`feature dimension`, `motion stats`, `relabel metrics`, `sampling stats`).
- **`train.py`**:
  - Uses `data_utils` to retrieve the sampled feature vectors and `label_new`.
  - Splits data (train/test) and trains a pipeline: `StandardScaler` → `LogisticRegression(C=0.3, max_iter=2000, solver='lbfgs', class_weight='balanced')`.
  - Evaluates training and test sets: accuracy, balanced accuracy, confusion matrix, classification report.
  - Runs threshold analysis over `[0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`, logging precision/recall/F1/balanced accuracy per threshold and selecting a recommended deployment threshold (currently `0.55`).
  - Trains a regressor (e.g., `LinearRegression`) on positive clusters only to predict `(opp_x, opp_y, opp_theta)` and logs RMSE per target.
  - Saves a bundled object via `joblib` at `models/opponent_bundle.joblib` containing the classifier pipeline, regressor, scaler/PCA artifacts (if any), and the chosen threshold.
- **`test.py`** (if present):
  - Loads the saved bundle and evaluates on the held-out split using the same metrics as training.
  - Useful to validate that retrained models behave like the originals.
- **`visualize_clusters.py`**:
  - Reconstructs clusters for a single frame and plots:
    - Raw LiDAR points colored by their `isWall/isStatic/isFree/isOpponent` label.
    - Cluster centroids colored by `label_new`.
    - Ground-truth opponent pose.
    - (Optional) classifier predictions per cluster, if a bundle is provided.
  - Saves plots under `figs/` and logs diagnostic tables (`[viz][frame_stats]`, `[viz][opponent_clusters]`).

## Training & Evaluation
1. Create & activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the models:
   ```bash
   python3 train.py
   ```
   Expect logs such as `[data] Opponent samples before sampling: ...`, `[data][sampling] After sampling: ...`, and `[train][logreg] Train/Test accuracy ...`.
4. (Optional) Evaluate saved bundle:
   ```bash
   python3 test.py
   ```
5. Visualize a frame:
   ```bash
   python3 visualize_clusters.py --frame-index 123
   ```
   Inspect the plot for cluster coloring, classifier candidate scores, and GT overlap.

## ROS2 Integration (Preview)
- A separate ROS2 package (documented elsewhere) loads `models/opponent_bundle.joblib` and subscribes to `/scan` and `/odom`.
- It builds the same cluster-level geometric/motion features (excluding label ratios), invocations the classifier & regressor, and publishes `/opponent_odom` (`nav_msgs/Odometry`) only when the classifier probability exceeds the stored deployment threshold.
- This keeps the perception stack lightweight and deterministic since it relies on classical ML rather than deep nets.

## Project Layout (Key Files)
```
final_project/
├── data_utils.py          # CSV loading, clustering, featurization, relabeling, sampling
├── train.py               # Training + threshold sweep logic, model bundle output
├── test.py                # (If present) evaluation script for saved bundle
├── visualize_clusters.py  # Debug/visualization per frame
├── realtime_encoder.py    # Runtime encoder that reuses the training pipeline
├── models/                # Serialized bundles (`opponent_bundle.joblib`)
├── dataFiles/             # CSV datasets (frame + scan files)
├── figs/                  # Saved visualizations
└── README.md
```

## Next Steps
1. Re-train with new datasets by swapping in alternate CSV pairs and rerunning `python3 train.py`.
2. Tune `GT_RADIUS`, `OPP_RATIO_THRESHOLD`, and `TARGET_BG_RATIO` in `config.py` if the environment changes.
3. Deploy the saved bundle to the ROS2 node and verify `/opponent_odom` on the vehicle.
