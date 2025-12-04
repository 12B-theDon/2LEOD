# Scan + Odometry Encoder Pipeline

This project ingests labeled LiDAR/odometry CSV recordings, learns a lightweight encoder that combines 360-beam scans with vehicle pose/speed, and evaluates the resulting binary opponent/obstacle classifier in a reproducible split.

## Workflow overview
1. **Prepare the environment.** Create a virtualenv (`python3 -m venv .venv`), activate it, and install `numpy` (only dependency).
2. **Generate feature JSONs.** Run
   ```sh
   PYTHONPATH=. tools/feature_vector_index_generator.py -p dataFiles
   ```
   (add `--odom-columns ...` if your odometry columns differ). This script now consumes the raw CSV/YAML, builds the frame-level vectors + odometry counts, and writes `*_feature_index.json` files that contain the angle metadata and the aggregated frames used by training.
3. **Run training once per CSV.** The command
   ```sh
   PYTHONPATH=. .venv/bin/python scripts/train.py dataFiles/output_1.csv --train-ratio 0.8 --pca-components 32
   ```
   reads the feature JSON produced above, shuffles/splits frames (80/20 default), fits a PCA, concatenates odometry, trains logistic/linear heads, saves checkpoints, plots five random frames, and stores a t-SNE scatter under `train_result_figs/<csv_stem>_tsne.png`.
4. **Inspect checkpoints.** After training you will find:
   - `checkpoints/pca_mean.npy`, `.npy` components, and `train_latent.npy` for the PCA encoder.
   - `checkpoints/logistic_weights.npy`/`logistic_bias.npy` and `checkpoints/linear_weights.npy`/`linear_bias.npy`.
   - `checkpoints/<csv_stem>/split_indices.json` for deterministic train/test splitting and `checkpoints/<csv_stem>/train_frame_ids.npy`.
   - `checkpoints/visualizations/frame_<frame_index>.png` (or ASCII output when matplotlib is missing) for five sample scans showing label + prediction.
5. **Evaluate using the frozen encoder.** Execute
   ```sh
   PYTHONPATH=. .venv/bin/python scripts/test.py dataFiles/output_1.csv --threshold 0.45
   ```
   The test script loads the PCA + classifiers saved under `checkpoints/`, rebuilds the latent+odom matrix for the held-out frames, prints accuracy/precision/recall/F1, and reports linear regression MSE as a diagnostic. Change `--threshold` to tune the binary output.
6. **Reuse checkpoints for deployment.** The PCA mean/components plus logistic weights form the frozen encoder. When a real-time system receives a new scan + odom sample, it needs to replicate `scripts/train.py`’s feature building (`PCAEncoder.transform + odom concatenation`) and then compare the resulting latent vector against the training checkpoints (e.g., via cosine similarity if you choose). The frozen encoder is primarily the logistic head that takes latent+odom input.

-## Directory layout
- `config.py`: dataclass-backed defaults for PCA size, logistic hyperparameters, regularization values, dataset split ratios, sample counts, and checkpoint/split directories. CLI arguments mirror dataclass fields so you can override any setting without touching the file.
- `models/`: custom numpy-only implementations of the PCA encoder, logistic classifier, and linear regressor with `fit`/`predict` and `save`/`load` helpers.
- `tools/feature_vector_index_generator.py`: processes the raw CSVs, reads the YAML to infer angles, aggregates each frame’s 360-beam vector/odometry, and writes out `{csv}_feature_index.json` files that include both metadata and the actual frame list.
- `scripts/data_utils.py`: loads the preprocessed `{csv}_feature_index.json` produced by `feature_vector_index_generator`, extracts each frame's 360-beam vector + odometry, and returns train/test splits (via `checkpoints/split_indices.json`) without parsing CSV again.
- `tools/tsne_visualizer.py`: produces a t-SNE scatter from `checkpoints/<csv_stem>/train_latent.npy` and saves the figure under `train_result_figs/<csv_stem>_tsne.png`.
- `scripts/train.py`: orchestrates the flow described above, uses `models.PCAEncoder` + `LogisticClassifier` + `LinearRegressor`, saves artifacts, reports accuracy, visualizes random training frames, and dumps a t-SNE plot via the new tool.
- `train_result_figs/`: stores t-SNE scatter plots (one per CSV) so you can inspect latent separation.
- `scripts/test.py`: loads checkpoints, rebuilds latent+odom matrices for the held-out split, evaluates classification metrics & regression MSE, and prints them to stdout.
- `scripts/visualize_linear_frames.py`: attempts to draw polar charts via matplotlib; when unavailable, it prints ASCII art plus the label/prediction pair.
- `checkpoints/`: generated at training time, stores numpy checkpoints and visualization artifacts.

## Data expectations
1. **Mandatory scan columns**: `frame_index`, `scan_index`, `distance` or `distance[m]` (in meters), `isWall`, `isFree`, and either `isOpponent` or `isObstacle`. Additional columns are ignored but tolerated.
2. **Inline odometry columns**: The CSV should include `x_car`, `y_car`, `heading_car`, `x_linear_vel`, `y_linear_vel`, and `w_angular_vel` for each beam row; `scripts.data_utils.load_scan_dataset` automatically collects these per frame so you no longer need a separate odometry file.
3. **Beam grouping**: Every row contributes one of 360 beams; `load_scan_dataset` builds a 360-element `distances` list per frame and remembers the last valid distance so that absent readings do not zero-out whole sectors.
4. **Labeling**: Frames are labeled `1` if any beam has `isOpponent=True` or `isObstacle=True`; otherwise they are `0`. `isWall`/`isFree` counts are stored for diagnostics but do not directly drive the final classification (you can experiment with them as features or additional thresholds).
5. **Class imbalance**: Expect most frames to look like walls. Logistic training still works because it sees sparse positives and can use regularization and threshold tuning to maintain low false positives.

## Model components
### `models.PCAEncoder`
- Implements PCA via SVD (numpy-only) and stores `mean_` + principal axes.
- Offers `fit`, `transform`, `fit_transform`, and `save/load` so the same components are available at test time.

### `models.LogisticClassifier`
- Trains via gradient descent with sigmoid activation.
- Clips logits to avoid overflow inside `_sigmoid`.
- Stores weights and bias in numpy arrays for reuse.
- Prediction threshold is configurable (`--threshold` in `scripts/test.py` and `config.TestConfig.threshold`).

### `models.LinearRegressor`
- Solves the regularized normal equations (closed form) with a bias term.
- Only used for regression diagnostics but stays saved alongside the logistic head so you can track MSE drift.

## script/train.py details
1. Parses CLI args for CSV path, PCA components, logistic hyperparameters, linear regularization, base checkpoint directory, and random seed.
2. Loads scan frames (including inline odometry columns) from `{csv}_feature_index.json`, shuffles/splits them (train/test), and writes `checkpoints/<csv_stem>/split_indices.json`.
3. Builds the feature matrix: latent = PCAEncoder(X), odometry matrix = `[entry['odom']]`, feature matrix = `[latent | odom]`.
4. Trains logistic + linear heads on the feature matrix + binary label, saves their weights/biases, and preserves PCA params into `checkpoints/<csv_stem>/`.
5. Uses the logistic classifier to predict on the training set (threshold defaults to `config.TrainConfig.similarity_threshold`) and prints accuracy.
6. Visualizes `config.TrainConfig.visualize_samples` random frames (default 5); uses matplotlib if present, otherwise prints ASCII art so you always get a sanity check, and emits a t-SNE plot to `train_result_figs/<csv_stem>_tsne.png`.

## script/test.py details
1. Accepts CLI arguments for CSV, checkpoint directory base, split indices file, and threshold.
2. Loads the same PCA/logistic/linear checkpoints from `checkpoints/<csv_stem>/` and applies them to the test frames determined by `checkpoints/<csv_stem>/split_indices.json`.
3. Computes accuracy, precision, recall, F1, TP/TN/FP/FN counts, and prints them.
4. Computes linear regression MSE on the test split for reference (the final loss view).

## Visualization and logging
- Visualizations are saved under `checkpoints/visualizations` and include the label/prediction as part of the title.
- ASCII fallback prints an 8-row chart spanning the scan; this is useful if matplotlib is not installed (and it will gracefully skip the plot while you still see the prediction label).

## Deployment pointers
1. Use the saved PCA mean/components + logistic weights as the frozen encoder available for runtime inference.
2. Real-time processing must replicate `feature_matrix = [PCAEncoder.transform(scan), odom_vector]` before applying logistic prediction.
3. Stored `train_latent.npy` and the latent outputs saved during training can serve as the checkpointed latent states for cosine similarity checks when matching incoming scans to known opponents.
4. Save the generated checkpoint files somewhere persistent so that inference logic can load them without rerunning training.

## Commands summary
- Environment setup:
  ```sh
  python3 -m venv .venv
  source .venv/bin/activate
  pip install numpy
  ```
- Generate feature JSONs:
  ```sh
  PYTHONPATH=. tools/feature_vector_index_generator.py -p dataFiles
  ```
- Train:
  ```sh
  PYTHONPATH=. .venv/bin/python scripts/train.py dataFiles/output_1.csv --train-ratio 0.8 --pca-components 32
  ```
- Evaluate:
  ```sh
  PYTHONPATH=. .venv/bin/python scripts/test.py dataFiles/output_1.csv --threshold 0.5
  ```
- Override defaults via CLI flags matching dataclass fields (e.g., `--logistic-lr 0.25`, `--linear-reg 1e-4`, `--visualize-samples 3`).

## Dependencies
- `numpy` (required for all PCA/logistic/linear math). Install with `pip install numpy` inside `.venv`.
- `matplotlib` (required for `tools/tsne_visualizer.py` and scan polar plots). Install it inside `.venv` as well.
- `scikit-learn` (needed for the t-SNE visualizer). Install it inside `.venv`.

## Additional implementation notes
- `tools/feature_vector_index_generator.py` still reports per-frame metadata (`distances`, `label`, `odom`, `counts`) alongside the angle map, so any post hoc analysis or zero-shot similarity check can refer to `frames` inside the generated `{csv}_feature_index.json`.
- `scripts/data_utils.py` now simply loads that JSON and hands the `frames` list to the trainer/tester instead of parsing CSV itself, which keeps preprocessing centralized in the generator.
- The logistic classifier is intentionally kept simple (single sigmoid output) so it can be deployed without neural frameworks; you already have `predict_proba`, `predict`, `save`, and `load` in one lightweight module.
- The linear regressor uses a closed-form solution with optional regularization; it ensures we have a sanity-check MSE and a different view on the latent geometry at training time.
- The PCA encoder relies on numpy’s SVD for numerical stability; once you call `fit`, both `mean_` and `components_` are stored and used for transformation so the frozen encoder is deterministic.

## Troubleshooting
1. **`ModuleNotFoundError: No module named 'config'`** – Always run scripts with `PYTHONPATH=.` or activate the repo root before executing to ensure `config.py` is importable.
2. **`numpy` installation errors** – Use the provided `.venv` and install dependencies inside it; avoid system-wide installations due to macOS-managed Python restrictions.
3. **Unbalanced labels** – If you see low recall, tweak `--threshold` or reweight positive frames manually (e.g., duplicate `isOpponent` frames before training, or adjust the training labels yourselves).
4. **No matplotlib?** – The visualizer catches `ImportError` and prints ASCII, so training continues. You can install matplotlib in the same `.venv` later with `pip install matplotlib`.

## What to add next
- Real-time inference: reuse the saved PCA/logistic checkpoints in a ROS node or other runtime by streaming scans through the same preprocessing pipeline and invoking `models.LogisticClassifier.predict`.
- Similarity scoring: you can compute cosine similarity between new latent vectors and `train_latent.npy`, returning odometry for frames whose cosine exceeds a precomputed threshold, which is useful for “frozen encoder + checkpoint similarity” verification.
- Additional labels: the dataset still exposes `isWall`, `isStatic`, and `isFree`; you can extend the logistic head to multi-class if future requirements need more than a binary opponent flag.

## Training detail reference
1. `scripts/train.py` always prints `Training accuracy:` followed by the float accuracy over the training split using the logistic classifier and configured threshold.
2. After training, `Checkpoint artifacts:` lists the directory containing saved numpy parameters and metadata files. Keep this directory, it is the frozen encoder.
3. `checkpoints/split_indices.json` records the train/test partitioning so you can share or reuse exactly the same frames without rerunning `train.py`.
4. Five sample frames (default) are visualized with label/prediction overlays; the filenames appear in the log so you can inspect whether the model misclassified them.
5. `checkpoints/train_latent.npy` and `checkpoints/train_frame_ids.npy` map latent vectors back to frame indices, making it easy to resolve which scan generated a given latent state when debugging.
6. Visualizations can be suppressed by setting `--visualize-samples 0`. If matplotlib is missing, the output explicitly states “MATPLOTLIB NOT AVAILABLE” and prints ASCII art.
7. PCA storage ensures deterministic inference: the same `pca_mean.npy` and `pca_components.npy` are loaded during testing and frozen deployments.
8. Regularization parameters (`--logistic-reg`, `--linear-reg`) help compress the wall-heavy data space; they act as L2 penalties when training the logistic head and the linear regressor.

## Test run checklist
1. Confirm `checkpoints/split_indices.json` exists and lists frame IDs for both partitions; missing file means `train.py` failed before saving the split indices.
2. Ensure `checkpoints/split_indices.json` lists the test frame indices you expect. This JSON is read by `scripts/test.py`.
3. Run `PYTHONPATH=. .venv/bin/python scripts/test.py dataFiles/output_1.csv --threshold 0.5`. It should print classification metrics and regression MSE. If the accuracy/recall seem wrong, verify the logistic weights were saved/loaded correctly.
4. Inspect `checkpoints/visualizations` to check for polar plots or ASCII outputs; they often reveal why a frame was mispredicted (e.g., a nearly empty scan with a single opponent beam).
5. Use the logistic classifier on the saved checkpoint in a REPL to experiment with thresholds: load `models.LogisticClassifier`, call `load`, build a feature matrix, and try values between 0.1 and 0.9 to balance precision vs. recall.

## Latent similarity usage idea
- Use the precomputed latent features in `checkpoints/train_latent.npy` as templates. When a new scan arrives at runtime, transform it via the saved PCA + odometry pipeline and compute cosine similarity against every saved latent vector (dot product / L2 norm). If the similarity exceeds a threshold you choose, consider it a match and return the corresponding odometry from `checkpoints/train_frame_ids.npy`.
- This provides a safety check before or alongside the logistic classifier; latent similarity ensures the scan geometry itself resembles a known opponent frame regardless of the logistic threshold.
- Store the PCA and latent artifacts together so the frozen encoder and similarity check share the exact feature spaces, eliminating drift and avoiding the need to recompute PCA at runtime.
