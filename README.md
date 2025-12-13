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

## Classifier Hyperparameters

- **`LogisticRegression` pipeline (`models.py:8-24`)**
  - `StandardScaler` 재스케일 후 `LogisticRegression`을 쓰고, `n_jobs=-1`로 학습/예측에 CPU 코어를 최대한 활용합니다.
  - `C`는 규제 강도의 역수 (`C=0.3`)로, 작을수록 파라미터 크기를 억제해서 과적합과 극단적인 결정 경계를 완화합니다.
  - `max_iter=2000`은 `lbfgs` 최적화가 수렴할 충분한 반복을 허용하되 무한 루프를 방지합니다.
  - `class_weight='balanced'`는 샘플 비율(`TARGET_BG_RATIO`)로부터 유추한 클래스 비율을 반영하여 minority 클래스 손실에 더 큰 가중치를 줍니다.

- **`SVC` pipeline (`models.py:26-43`)**
  - `StandardScaler` → `LinearDiscriminantAnalysis` → `SVC(kernel='rbf', probability=True)` 흐름으로 거리 기반 SVM을 시도하면서 차원 축소(LDA)로 노이즈를 완화합니다.
  - `C=1.0`과 `gamma='scale'`은 기본값이지만, `C`는 잘못된 판별을 줄이기 위해 결정 경계의 여유를 조절합니다.
  - `class_weight='balanced'`가 불균형을 보정하고, `probability=True`로 ROC/PR 곡선 생성을 위한 확률 스코어를 허용합니다.
  - `n_jobs`는 `SVC`에 직접 지정하지 않지만, `LinearDiscriminantAnalysis`와 `StandardScaler`가 내부 연산에서 NumPy/BLAS의 스레드를 활용하며, 전반적으로 병렬화된 환경에서 빠르게 동작합니다.

- **`XGBClassifier` pipeline (`models.py:45-76`)**
  - `max_depth=4`로 각 트리의 복잡도를 제한하여 과적합을 억제하고 논리적인 움직임을 포착합니다.
  - `learning_rate=0.1`은 새 트리가 기존 예측을 얼마나 빠르게 수정할지를 정하며, 작게 둬서 학습을 안정화합니다.
  - `n_estimators=300`은 생성할 트리 개수로, 충분히 많은 트리로 복잡도를 확보하면서 `learning_rate`가 작으므로 과적합을 피합니다.
  - `subsample=0.8`과 `colsample_bytree=0.8`은 각각 샘플과 피처의 무작위 부분집합을 사용하여 과대적합 방지와 다양성 확보를 지원합니다.
  - `scale_pos_weight`는 `train.py`에서 `neg_train / max(1, pos_train)`으로 계산된 값으로, 클래스 불균형(다수 클래스에 비해 소수 클래스의 중요도)을 반영하여 다음 트리에서 positive 샘플의 영향력을 키웁니다.
  - `use_label_encoder=False`는 오래된 XGBoost label encoder를 비활성화해서 `eval_metric`이 정상적으로 로그/경고 없이 작동하게 합니다.
  - `eval_metric="logloss"`는 학습 과정에서 손실 로그를 추적하는 지표로, log-loss 값을 줄이는 방향으로 트리를 업데이트합니다.

각 파이프라인에는 `StandardScaler`가 앞단에 있어서 서로 다른 feature 스케일에 의한 모델 편향을 줄이고, 불균형 환경에서는 `class_weight`/`scale_pos_weight`를 통해 minority에 대한 감도를 높이는 방향으로 설계되어 있습니다. 필요시 위 파라미터들을 `config.py` 또는 `models.py`에서 튜닝하면서 `train.py`의 정확도/PR 로그로 영향도를 점검하세요.

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
6. Inspect cluster grouping:
   ```bash
   python3 cluster_quality.py --focus-positives --max-frames 5
   ```
   Saves plain cluster-overview PNGs under `figs/clustering/` so you can confirm the raw clustering aligns with the GT positions.

## Rendering Video
- Render clustered frames with `python3 render_cluster_video.py` and customize `--duration`/`--fps` as needed.
- By default the script consumes every scan/odom pair listed in `config.DATASET_PAIRS`, but you can target a single dataset by passing `--scan-csv <scan_file.csv> --odom-csv <odom_file.csv>`.
- Example: `python3 render_cluster_video.py --model models/iccas_opponent_bundle_logreg.joblib --duration 30 --fps 5 --scan-csv dataFiles/iccas_track_1.csv --odom-csv dataFiles/iccas_track_odom_output_1.csv`.

## ROS2 Integration (Preview)
- A separate ROS2 package (documented elsewhere) loads `models/opponent_bundle.joblib` and subscribes to `/scan` and `/odom`.
- It builds the same cluster-level geometric/motion features (excluding label ratios), invocations the classifier & regressor, and publishes `/opponent_odom` (`nav_msgs/Odometry`) only when the classifier probability exceeds the stored deployment threshold.
- This keeps the perception stack lightweight and deterministic since it relies on classical ML rather than deep nets.

## Evaluation & Visualization Tools
- `opponent_tracker/evaluate_opponent.py` reuses the cluster feature builder to:
  - subscribe to `/odom`, `/scan` and `/opponent_odom` (GT).
  - publish `/pred_opponent_odom`, `/pred_opponent_path`, `/gt_opponent_path`, and an optional `/pred_opponent_marker`.
  - compute RMSE between predictions and GT (position + yaw) while buffering the most recent GT samples, logging both incremental and final RMSE, and writing an optional CSV (`csv_path` parameter) at runtime.
  - monitor real-time inference delays by comparing `time.perf_counter()` to each scan timestamp, reporting avg/median/max delay every `log_interval` predictions, and keeping a max delay gauge.
  - emit a rainbow-colored PNG of the predicted trajectory where the colour encodes the instantaneous RMSE; the file path is controlled by `rmse_plot_path` (default `rmse_trajectory.png`) and the plot includes GT/pred lines, a colourbar legend, and axes labels.
- The `opponent_tracker/launch/eval_opponent_bag.launch.py` launch file:
  - plays a bag via `ros2 bag play <bag_path> --clock`.
  - runs the evaluation node with configurable `model_path`, `frame_id`, `csv_path`, and `rmse_plot_path`.
  - optionally starts RViz2 with `opponent_tracker/rviz/eval_opponent.rviz` to show `/scan`, ego odometry, `/pred_opponent_odom`, both paths, and prediction markers in the `odom` frame.

### Running the evaluation launch
1. Source your ROS2 Humble workspace and activate any Python virtualenv that has the dependencies installed (make sure `matplotlib` is available — it is listed in `opponent_tracker/setup.py` and `package.xml`).
2. Run:
   ```bash
   ros2 launch opponent_tracker eval_opponent_bag.launch.py \
     bag_path:=/path/to/your.bag \
     model_path:=models/opponent_bundle.joblib \
     frame_id:=odom \
     csv_path:=logs/predictions.csv \
     rmse_plot_path:=logs/rmse_trajectory.png
   ```
   *omit `csv_path` or `rmse_plot_path` if you only need the live ROS topics.*
   If you prefer to skip CLI arguments, update `opponent_tracker/config/eval_opponent.yaml` (it now contains both bag defaults and evaluate_opponent parameters) with your bag/model/CSV paths and the launch will pick those defaults so you can run `ros2 launch opponent_tracker eval_opponent_bag.launch.py` directly.
3. After the bag finishes, the log will show RMSE and real-time delay stats, and the PNG will be saved under the chosen `rmse_plot_path`. Use RViz to inspect the `/pred_opponent_*` topics and verify the marker+path alignment.

## Project Layout (Key Files)
```
   After rendering `clusters.mp4`, a scatter plot of the per-frame mean cluster speed plus its running mean is saved at `figs/clusters_velocity.png`.
final_project/
├── data_utils.py          # CSV loading, clustering, featurization, relabeling, sampling
├── train.py               # Training + threshold sweep logic, model bundle output
├── test.py                # (If present) evaluation script for saved bundle
├── visualize_clusters.py  # Debug/visualization per frame
├── cluster_quality.py     # Frame snapshots of raw cluster centroids
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
