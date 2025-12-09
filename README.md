# Opponent Tracking Pipeline

## 1. Project Goal
LiDAR + ego odometry are fused frame-by-frame to detect and track an opposing 1/8-scale vehicle. We build a classical ML stack: cluster raw scan points, engineer geometry/motion features, classify opponent clusters, regress odometry (x, y, yaw) from the learned latent, and expose a realtime encoder that only needs LiDAR + ego odom at inference time.

## 2. Input CSV Formats
1. `frame` CSV: `frame_index, stamp_sec, stamp_nsec, base_link_x, base_link_y, base_link_yaw, base_link_op_x, base_link_op_y, base_link_op_yaw` – ego pose plus GT opponent pose per frame.
2. `point` CSV: `frame_index, scan_index, distance, local_x, local_y, global_x, global_y, isOpponent, isWall, isStatic, isFree` – 360° scan points with coordinates and point-level labels.

## 3. Data Preprocessing Pipeline
- Merge scan and odometry data (`data_utils._load_merged_data`) to align timestamps.
- Compute ego velocities (`ego_v`, `ego_w`) from pose deltas.
- Group contiguous `scan_index` runs into clusters (`_build_clusters`).
- Each cluster carries frame metadata, collision ratios, and per-point stats before feature extraction.

## 4. Clustering Logic
Contiguous `scan_index` segments form a cluster; we record start/end indices, mean distance, centroid (local/global), radius, timestamp, and link clusters across frames for motion estimation.

## 5. Feature Engineering
`CLUSTER_FEATURE_COLUMNS` (~38 dims) comprises:
- Geometric descriptors (centroids, radius stats, range/angle spans, spread, entropy).
- Ego telemetry (`ego_v`, `ego_w`, `ego_vx`, `ego_vy`).
- Motion derived between frames (local/world deltas and relative velocities).
- Shape & location cues (`cluster_spread`, `distance_to_ego`, `bearing`, `cluster_opponent_ratio`).
Motion features are computed in `_attach_motion_features`, including ego-compensated velocities.

## 6. Re-labeling (GT-Guided Refinement)
Point-level labels are noisy, so we re-label clusters:
- `GT_RADIUS = 2.2 m` ⇒ `gt_label = 1` if centroid is within range of GT opponent pose.
- `OPP_RATIO_THRESHOLD = 0.10` ⇒ `label_orig = 1` when point-level opponent ratio is high.
- `label_new = 1` if `gt_label == 1` OR `opp_ratio >= threshold`.
Training uses `label_new`; logs track `orig_vs_gt` and `new_vs_gt` confusion matrices plus how many labels improved/worsened.

## 7. Sampling Strategy (Undersampling)
Background dominates, so we keep all positives and sample at most `TARGET_BG_RATIO = 8.0` negatives per positive. `_downsample_background` returns sampled arrays plus stats (`pos`, `neg`, `target_ratio`) that are logged for traceability.

## 8. Classifier / Regressor Models
- Classifier: `Pipeline([StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, solver='lbfgs', class_weight='balanced', n_jobs=-1)])` trained on normalized cluster features.
- Regressor: `LinearRegression`, trained only on visible (positive) clusters to predict `opp_x`, `opp_y`, `opp_theta`.

## 9. Threshold Selection Logic
Sweep thresholds `[0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`, logging precision, recall, F1, balanced accuracy, confusion matrices, ROC-AUC, and PR-AUC. The best threshold (balanced accuracy + recall constraints) is saved, but deployment always enforces a fixed `0.55` for stability.

## 10. Runtime Encoder Usage
`realtime_encoder.py`: load the bundle (`clf_pipe`, `regressor`, `threshold`), compute cluster features, predict `prob`, compare with threshold, compute cosine similarity against the previous latent, and regress opponent odometry for confirmed clusters.

## 11. Visualization Tool
`visualize_clusters.py` recreates the clustering logic for a selected frame, plots global `x/y` for raw points (color-coded by label), shows cluster centroids with GT highlights, marks the ego pose derived from the odometry merge, annotates classifier probabilities, saves figures under `figs/`, and logs `[viz][frame_stats]` plus `[viz][opponent_clusters]` tables to help understand which clusters are candidate opponents.

## 12. Cluster Video Rendering
`render_cluster_video.py` wraps `visualize_clusters.py` to produce a short MP4 (`figs/clusters.mp4`). By default it renders just the first/last frames from the collected set; pass `--all` if you want every frame (you can also use `--limit` to cap how many frames are considered before slicing).

## 13. Hyperparameter Tuning
- `GT_RADIUS`: expand/contract to control GT coverage.
- `OPP_RATIO_THRESHOLD`: lower to be permissive, higher to avoid noisy positives.
- `TARGET_BG_RATIO`: adjust to show more or fewer negatives.
- Threshold: trade-off recall vs false positives.
- Logistic Regression `C`: stronger regularization (lower `C`) smooths decision boundary.

## 14. Interpreting Training Logs
- `[data]` logs show sampling ratios and relabel stats.
- `[train][logreg]` logs report train/test accuracy, balanced accuracy, confusion matrix, classification report.
- Threshold sweep logs detail per-threshold metrics.
- `[train][reg]` logs regression RMSE on visible clusters. Use recall to ensure opponents are detected, precision to control false positives, and balanced accuracy for class imbalance.

## 15. Current Performance (example log)
- Pre-sampling opponent ratio ≈ 9%; after sampling ≈ 264 positives vs 2112 negatives (target 8:1).
- Balanced accuracy ≈ 0.71, recall ≈ 0.68 (threshold 0.5).
- Regressor RMSE: x 0.72 m, y 1.16 m, yaw 0.33 rad.

## 16. Folder Structure
```
final_project/
├── train.py
├── test.py
├── data_utils.py
├── models.py
├── realtime_encoder.py
├── visualize_clusters.py
├── logger.py
├── config.py
├── models/opponent_bundle.joblib
├── dataFiles/
│   ├── output_1.csv
│   └── odom_output_1.csv
├── figs/
│   └── cluster_debug_frame_{frame}.png
└── README.md
```

## 17. Future Improvements
- Try stronger classifiers (SVM, LDA, XGBoost) via `CLASSIFIER_TYPE`.
- Add temporal smoothing / Kalman filtering across frames.
- Enrich motion features (velocity consistency, acceleration).
- Hard negative mining around walls/static clusters.
- Aggregate multi-frame features for richer decisions.
