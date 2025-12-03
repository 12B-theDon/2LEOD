# 2LEOD: 2D LiDAR Ensemble Opponent Detector
The repository for using 2D LiDAR scan data and estimates real-time object opponent odometry. We use 3 approaches which is linear regression(2 wall line fitting), logistic regression(naively using each points of scan data), and DBSCAN(rule-based). 
We check the DBSCAN is slower, fails on some scenarios, and less accurate. We use learning-based approach to overcome those limitations. 


### CSV format
- rule: one css file for one rosbag file. Must check the local frame scan position are corresponding to global scan position
- include:  timestamp, frame_index, scan_index, distance[m], x_global position, y_global_positon, x_local_position, y_local_positon
the file name is rosbagNumber_scenarioName.csv
- example format can be seen at /dataFiles/example_csv_format.csv and /dataFiles/example_yaml_format.yaml 


### YAML format
- rule: one yaml for one csv format. Must check the data in it.
- include: corresponding csv file name, laser min/max angle range, laser min/max distance range/whether laser coordinate is same with base_link frame, laser sensor frequency, opponent_odom publishing frequency
- If possible, check how long does it take to convert csv file 


### 2D scan heatmap conversion
Based on the image like at this site, https://www.nature.com/articles/s41467-019-12943-7/figures/4 we can visualize scan data and use that as a feature. Since it can be treated as an image, then we can use feature-descriptor method such as FAST, color histogram. So rather using csv file, we can make that as an image(just like occupangy grid map) per each frame. 
We chekced how long does it take to convert the given /scan data to the grid map. 

So we can say that if we have 360 degree distance data, then width will be 360(360 number of bins) and height will be distances. so \theta degree value is mapped to column vector, and the height of this column vector is maximum lidar scan distance(detection range). So when the lidar has a obstacle at 2m, then 0~2m value has increasing intensity, and no value is assigned above 2~maximum lidar scan distance. So black(0) at y = 0, and white(255) at y = detected distance. the value between is smootly increased(calculated automatically). Like an RGB channel, we can choose 3 channel values. (1) Distance, (2) Density(How many points are in that range), (3) Delta(current distance - previous distance). And then we can get a RGB color map. Since we have label(isWall, isOpponent, isFree, isStatic) each columns have a label. 

### Step A: Linear regression wall filtering
Each frame \(i\) is represented by a feature vector \(\mathbf{x}^{(i)} \in \mathbb{R}^{360}\) containing the range measurements for every scan angle. The \(k\)th range value \(r_{k}\) is converted to a Cartesian point according to
\[
    x_{k} = r_{k} \cos(\theta_{k}), \qquad y_{k} = r_{k} \sin(\theta_{k}), \qquad k = 1,\dots,360,
\]
where \(\theta_{k}\) is the scan angle mapped from the feature vector index. Step A fits the resulting points with two linear regression models, \(\ell_1\) and \(\ell_2\), each of which estimates a centroid \((\bar{x}_j, \bar{y}_j)\) and a unit direction vector \(\mathbf{d}_j\) by minimizing perpendicular residuals (similar to an SVD or PCA line fit).

After both line models are determined, the distance of each point \((x_k, y_k)\) to line \(\ell_j\) is calculated as
\[
    \text{dist}_j(k) = \left| (x_k - \bar{x}_j) d_{j,y} - (y_k - \bar{y}_j) d_{j,x} \right|.
\]
Each point picks the line with the smallest residual, and it is classified as `isWall` if that distance is below a pre-defined threshold \(\tau\):
\[
    \text{isWall}_{k} = \begin{cases}
        \text{True} & \text{if } \min_{j=1,2} \text{dist}_j(k) \le \tau, \\
        \text{False} & \text{otherwise}.
    \end{cases}
\]
The index set \(\mathbf{w}^{(i)} = \{\text{scan\_index} \mid \text{isWall}_{k} = \text{True}\}\) filters wall hits before Step B’s logistic regression. `models/LinearRegression.py` records each line’s angle, support size, and the wall scan indexes, and `scripts/train.py` saves this per-frame metadata in `checkpoints/linear_wall_checkpoint.json` so downstream stages can skip points already labeled as walls.

## Workflow overview
1. **Gather CSV/YAML pairs** that describe each LiDAR rosbag and run `tools/feature_vector_index_generator.py` to produce an angle-aware JSON guide per CSV.
2. **Create per-frame feature vectors** (360‑dim distance arrays plus angles) and feed them to `scripts/train.py`.
3. **Split frames into train/test** (default 80/20) inside `scripts/train.py`.
4. **Call `models.LinearRegression`** to fit two line models per frame and to surface per-scan residuals, and **call `models.LogisticRegression`** to train a softmax classifier over wall/static/opponent/free classes.
5. **Compute confidence scores** by combining the softmax probabilities with the linear distance metrics and store all metadata during training.
6. **Print logistic loss per epoch** while training and report accuracy after each epoch.
7. **Evaluate on the held-out test split** and compare logistic-only vs. ensemble accuracy before finalizing the checkpoint.
8. **Emit a checkpoint JSON** (`checkpoints/linear_wall_checkpoint.json`) carrying logistic weights, frame-wise linear geometry, and metrics so you can do zero-shot inference on real `/scan` streams.

## Python components
- **`tools/feature_vector_index_generator.py`**: Scans the given CSV folder, finds the associated YAML metadata, and emits one `{csv_stem}_feature_index.json` per file. Each JSON describes the angular model (min/max/step), scan-index-to-angle mapping, column descriptions, and a human-readable direction label so feature vectors can be interpreted downstream.

### Detailed regression modules
#### `models/LinearRegression.py`
- **Input**: For each frame we consider the point set \(\mathcal{P}^{(i)} = \{(x_k, y_k)\}_{k=1}^{N_i}\) obtained by mapping the 360-range vector \(\mathbf{x}^{(i)}\) into Cartesian coordinates using \((x_k, y_k) = (r_k \cos\theta_k, r_k \sin\theta_k)\).
- **Process**: We cluster the angles into two groups and fit two linear models \(y = a_j x + b_j\) (or \(x = c_j\) for near-vertical cases) for \(j=1,2\) by ordinary least squares. Each fit produces slope/intercept pairs \((a_j, b_j)\) plus directional vectors \(\mathbf{d}_j\) and line centroid \((\bar{x}_j, \bar{y}_j)\).
- **Output**: For every frame the module emits:
  1. Two line models \(\ell_1, \ell_2\) with \((a_j, b_j)\), direction, support counts, and whether the line is vertical.
  2. `point_info` for each scan containing `distance_to_line`, `line_id`, and a boolean `is_wall` based on the threshold \(\tau\): \(\text{isWall}_k = \min_j \text{dist}_j(k) \le \tau\), where \(\text{dist}_j(k)\) is the perpendicular residual.
  3. Diagnostics stored in `linear_regression_metadata/` that record the fitted parameters per frame for offline verification.

#### `models/LogisticRegression.py`
- **Input**: Every training sample is the original 360-dimensional vector \(\mathbf{x}^{(i)} \in \mathbb{R}^{360}\) fed directly to the classifier.
- **Process**: The module keeps weight matrix \(W \in \mathbb{R}^{4 \times 360}\) and bias vector \(b \in \mathbb{R}^{4}\), and computes logits \(z_j = W_j \cdot \mathbf{x}^{(i)} + b_j\). The softmax probabilities are \(p_j = \exp(z_j)/\sum_{k=1}^{4} \exp(z_k)\), and cross-entropy loss \(-\log p_{\text{label}}\) is minimized via SGD.
- **Output**: After training it exposes:
  1. `predict_proba(\mathbf{x})` yielding \(p_1,\dots,p_4\),
  2. `predict(\mathbf{x}) = \arg\max_j p_j\), and
  3. Serialized weights/biases through `to_dict()` for checkpointing.

#### `models/ensemble.py`
- Combines the softmax probabilities with the linear regression distance signal. The `wall_confidence(d, \tau) = 1 - \min(d, \tau)/\tau` rescales the closest residual \(d\) into a \([0,1]\) score, and `ensemble_prediction` boosts the wall class probability by a tunable weight before voting.

- **Usage**: `scripts/train.py` imports `ensemble_prediction` instead of duplicating the scoring logic, so both training and evaluation reuse the same blending strategy.

- **`scripts/train.py`**: Orchestrates the entire pipeline. It reads the CSV+JSON data, splits frames using the requested ratio/seed, builds 360‑dim feature vectors, trains the logistic softmax on the training split, computes linear regression outputs for every frame, assembles confidence scores via the ensemble helper, reports losses/accuracies (logistic-only vs. ensemble) for both train and test sets, and writes `checkpoints/linear_wall_checkpoint.json` that bundles the logistic model, the per-frame linear summaries (including \((a_j,b_j)\)), and evaluation metrics for rapid zero-shot inference.

### Component details
- `tools/feature_vector_index_generator.py` is executed first. Use it with `python3 tools/feature_vector_index_generator.py --csv_filefolder_path dataFiles` to find the YAML metadata, parse the column/angle specs, and emit `dataFiles/<csv_stem>_feature_index.json`. The generated JSON carries `vector_length`, angle ranges, column mapping, and a `vectors` array so downstream code can build 360-dimensional arrays without recalculating scan-index → angle logic.
- `models/LinearRegression.py` is imported by `scripts/train.py`. Call `classify_frame(frame_entries, threshold)` with the datapoints for a specific frame to get: the two fitted line models (`angle_deg`, `support_total`, etc.), wall/non-wall scan index lists, the total point count, and a `point_info` list containing per-scan residuals. Those residuals are the confidences used later for ensemble voting.
- `models/LogisticRegression.py` encapsulates softmax training. You instantiate it with `num_features` and `num_classes`, call `fit(features, labels, epochs, verbose=True)` to train, and then you can score new vectors via `predict_proba`. The model only depends on pure Python and logs cross-entropy loss per epoch for the CLI output.
- `models/ensemble.py` supplies the `wall_confidence` and `ensemble_prediction` helpers that `scripts/train.py` imports. The logistic output is boosted by `wall_confidence(distance, threshold)` before voting, so you can weight the wall hypothesis by how closely each scan follows one of the two fitted lines.
- `scripts/train.py` is the entry point. Provide `--csv`, `--feature-index`, and optional hyperparameters for epochs/learning rate/threshold. It performs the 80/20 split (`split_frames`), vectorizes frames (`build_frame_vector`), maps CSV labels to integers (`map_label`), trains `LogisticRegression`, obtains per-frame linear residuals (`classify_frame`), evaluates both logistic-only and ensemble accuracy via `evaluate_split`, prints losses/metrics, and writes everything to `checkpoints/linear_wall_checkpoint.json`. Use the resulting JSON for zero-shot inference by reloading the logistic weights and linear summaries.

### How to run
1. Generate the scan-angle metadata JSON (one shot per CSV folder):

```bash
python3 tools/feature_vector_index_generator.py --csv_filefolder_path dataFiles
```

2. Run training/evaluation (the default commands already split 80/20, train logistic regression, fit the two wall lines and blend via ensemble):

```bash
python3 scripts/train.py \
  --csv dataFiles/example_csv_format.csv \
  --feature-index dataFiles/example_csv_format_feature_index.json \
  --epochs 40 \
  --learning-rate 0.01 \
  --split-ratio 0.8 \
  --threshold 0.25
```

This prints epoch-level logistic loss plus train/test accuracy for both logistic-only and ensemble predictions. After the run you will have:

- `checkpoints/linear_wall_checkpoint.json` (logistic weights, frame-wise linear results, metrics).
- `linear_regression_metadata/<timestamp>.json` (slope/intercept summaries for each frame to monitor how the line fits evolve).
3. Re-run just the held-out test split using the saved checkpoint:

```bash
python3 scripts/test.py \
  --csv dataFiles/example_csv_format.csv \
  --feature-index dataFiles/example_csv_format_feature_index.json \
  --checkpoint checkpoints/linear_wall_checkpoint.json
```

`test.py` rebuilds the 360‑dim vectors for the recorded test frames, loads the saved logistic weights, and prints logistic-only plus ensemble accuracy without retraining.
