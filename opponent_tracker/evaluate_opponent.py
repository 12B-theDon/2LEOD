from __future__ import annotations

import csv
import math
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, Twist, Vector3, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from opponent_tracker.feature_builder import ClusterFeatureBuilder


class EvaluateOpponentNode(Node):
    def __init__(self):
        super().__init__('evaluate_opponent')
        self.declare_parameter('model_path', 'models/opponent_bundle.joblib')
        self.declare_parameter('decision_threshold', 0.55)
        self.declare_parameter('smoothing_alpha', 0.5)
        self.declare_parameter('min_cluster_points', 3)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('timestamp_tolerance', 0.05)
        self.declare_parameter('gt_history_size', 512)
        self.declare_parameter('path_length', 500)
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('publish_markers', True)
        self.declare_parameter('marker_ns', 'pred_opponent')
        self.declare_parameter('marker_scale', 0.5)
        self.declare_parameter('csv_path', '')
        self.declare_parameter('log_interval', 10)
        self.declare_parameter('rmse_plot_path', 'rmse_trajectory.png')

        model_path = Path(self.get_parameter('model_path').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.timestamp_tolerance = float(self.get_parameter('timestamp_tolerance').value)
        self.gt_history_size = int(self.get_parameter('gt_history_size').value)
        self.path_length = int(self.get_parameter('path_length').value)
        self.publish_markers = bool(self.get_parameter('publish_markers').value)
        self.marker_ns = self.get_parameter('marker_ns').value
        self.marker_scale = float(self.get_parameter('marker_scale').value)
        self.csv_path = self.get_parameter('csv_path').value
        self.log_interval = max(1, int(self.get_parameter('log_interval').value))
        rmse_plot_path = self.get_parameter('rmse_plot_path').value
        self.rmse_plot_path = Path(rmse_plot_path) if rmse_plot_path else None

        self.decision_threshold = float(self.get_parameter('decision_threshold').value)
        self.smoothing_alpha = float(self.get_parameter('smoothing_alpha').value)

        self.builder = ClusterFeatureBuilder(
            min_cluster_points=int(self.get_parameter('min_cluster_points').value),
            max_range=float(self.get_parameter('max_range').value),
        )

        bundle = joblib.load(str(model_path))
        self.classifier = bundle.get('clf_pipe') or bundle.get('classifier')
        self.regressor = bundle.get('regressor')
        bundle_cls_threshold = bundle.get('cls_threshold')
        self.cls_score_type = bundle.get('cls_score_type', 'proba')
        self.cls_threshold = float(bundle_cls_threshold if bundle_cls_threshold is not None else self.decision_threshold)

        if self.classifier is None or self.regressor is None:
            self.get_logger().error('Classifier or regressor missing in bundle.')
            raise RuntimeError('Invalid model bundle')

        self.get_logger().info(
            f'Loaded model from {model_path}; cls_score_type={self.cls_score_type}, '
            f'threshold={self.cls_threshold:.3f}'
        )

        self.prev_pose: Optional[Tuple[float, float, float]] = None
        self.latest_odom: Optional[Odometry] = None
        self.gt_buffer: Deque[Tuple[float, Odometry]] = deque(maxlen=self.gt_history_size)
        self.pred_path_queue: Deque[PoseStamped] = deque(maxlen=self.path_length)
        self.gt_path_queue: Deque[PoseStamped] = deque(maxlen=self.path_length)

        self.pred_path_msg = Path()
        self.pred_path_msg.header.frame_id = self.frame_id
        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = self.frame_id

        self.position_error_sum = 0.0
        self.yaw_error_sum = 0.0
        self.match_count = 0
        self.match_gt_positions: list[Tuple[float, float]] = []
        self.match_pred_positions: list[Tuple[float, float]] = []
        self.match_errors: list[float] = []

        self.delay_sum = 0.0
        self.delay_count = 0
        self.delay_history: Deque[float] = deque(maxlen=4096)
        self.max_delay = 0.0

        self.csv_file = None
        self.csv_writer = None
        if self.csv_path:
            csv_path = Path(self.csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'gt_x', 'gt_y', 'pred_x', 'pred_y', 'error'])

        self.pred_odom_pub = self.create_publisher(Odometry, '/pred_opponent_odom', 10)
        self.pred_path_pub = self.create_publisher(Path, '/pred_opponent_path', 10)
        self.gt_path_pub = self.create_publisher(Path, '/gt_opponent_path', 10)
        if self.publish_markers:
            self.marker_pub = self.create_publisher(Marker, '/pred_opponent_marker', 10)

        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_callback, 10)
        self.create_subscription(Odometry, '/opponent_odom', self._gt_opponent_callback, 10)

        self.marker_id = 0

    def _odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self.builder.update_odom(msg)

    def _scan_callback(self, msg: LaserScan) -> None:
        if self.latest_odom is None:
            return
        callback_start = time.perf_counter()
        scan_ts = self._stamp_to_seconds(msg.header.stamp)
        if scan_ts is not None:
            delay = callback_start - scan_ts
            self.delay_sum += delay
            self.delay_count += 1
            self.delay_history.append(delay)
            self.max_delay = max(self.max_delay, delay)

        clusters = self.builder.build_clusters_from_scan(msg)
        if not clusters:
            return

        features = self.builder.compute_features(clusters, self.latest_odom)
        if features.size == 0:
            return

        scores = self._compute_opponent_scores(features)
        mask = scores >= self.cls_threshold
        if not mask.any():
            return

        candidate_indices = np.where(mask)[0]
        selected_idx = int(candidate_indices[np.argmax(scores[candidate_indices])])
        latent = features[selected_idx].reshape(1, -1)
        prediction = self.regressor.predict(latent)[0]
        x, y, yaw = self._smooth_pose(prediction)

        pred_msg = Odometry()
        pred_msg.header.stamp = self.latest_odom.header.stamp
        pred_msg.header.frame_id = self.frame_id
        pred_msg.child_frame_id = 'pred_opponent_base_link'
        pred_msg.pose.pose.position.x = x
        pred_msg.pose.pose.position.y = y
        q = self._yaw_to_quaternion(yaw)
        pred_msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        pred_msg.twist.twist = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )

        self.pred_odom_pub.publish(pred_msg)
        self._publish_pred_path(pred_msg)
        if self.publish_markers:
            self._publish_marker(pred_msg)
        self._match_with_gt(pred_msg)

    def _gt_opponent_callback(self, msg: Odometry) -> None:
        stamp = self._stamp_to_seconds(msg.header.stamp)
        if stamp is None:
            return

        self.gt_buffer.append((stamp, msg))

        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.frame_id
        pose.pose = msg.pose.pose
        self.gt_path_queue.append(pose)

        self.gt_path_msg.header.stamp = msg.header.stamp
        self.gt_path_msg.poses = list(self.gt_path_queue)
        self.gt_path_pub.publish(self.gt_path_msg)

    def _match_with_gt(self, pred_msg: Odometry) -> None:
        pred_ts = self._stamp_to_seconds(pred_msg.header.stamp)
        if pred_ts is None or not self.gt_buffer:
            return

        best_match: Optional[Tuple[float, Odometry]] = None
        best_delta = self.timestamp_tolerance
        for gt_ts, gt_msg in self.gt_buffer:
            delta = abs(gt_ts - pred_ts)
            if delta <= best_delta:
                best_delta = delta
                best_match = (gt_ts, gt_msg)

        if best_match is None:
            return

        _, gt_msg = best_match
        self._record_match(pred_msg, gt_msg, pred_ts)

    def _record_match(self, pred_msg: Odometry, gt_msg: Odometry, timestamp: float) -> None:
        pred_x = pred_msg.pose.pose.position.x
        pred_y = pred_msg.pose.pose.position.y
        gt_x = gt_msg.pose.pose.position.x
        gt_y = gt_msg.pose.pose.position.y

        dx = pred_x - gt_x
        dy = pred_y - gt_y
        error = math.hypot(dx, dy)
        yaw_error = self._angle_difference(
            self._quaternion_to_yaw(pred_msg.pose.pose.orientation),
            self._quaternion_to_yaw(gt_msg.pose.pose.orientation),
        )

        self.position_error_sum += error ** 2
        self.yaw_error_sum += yaw_error ** 2
        self.match_count += 1
        self.match_gt_positions.append((gt_x, gt_y))
        self.match_pred_positions.append((pred_x, pred_y))
        self.match_errors.append(error)

        if self.csv_writer:
            self.csv_writer.writerow([f'{timestamp:.6f}', gt_x, gt_y, pred_x, pred_y, error])
            self.csv_file.flush()

        if self.match_count % self.log_interval == 0:
            self._log_rmse()

    def _log_rmse(self, final: bool = False) -> None:
        if self.match_count == 0:
            return
        pos_rmse = math.sqrt(self.position_error_sum / self.match_count)
        yaw_rmse = math.sqrt(self.yaw_error_sum / self.match_count)
        if final:
            self.get_logger().info(
                f'Final RMSE (pos/yaw) after {self.match_count} samples: {pos_rmse:.3f} m / {yaw_rmse:.3f} rad'
            )
        else:
            self.get_logger().info(
                f'[RMSE] pos={pos_rmse:.3f} m, yaw={yaw_rmse:.3f} rad, samples={self.match_count}'
            )
        self._log_delay_stats()

    def _publish_pred_path(self, odom: Odometry) -> None:
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = odom.header.stamp
        pose.header.frame_id = self.frame_id
        pose.pose = odom.pose.pose
        self.pred_path_queue.append(pose)

        self.pred_path_msg.header.stamp = odom.header.stamp
        self.pred_path_msg.poses = list(self.pred_path_queue)
        self.pred_path_pub.publish(self.pred_path_msg)

    def _publish_marker(self, odom: Odometry) -> None:
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = odom.header.stamp
        marker.header.frame_id = self.frame_id
        marker.ns = self.marker_ns
        marker.id = self.marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = odom.pose.pose
        marker.scale = Vector3(x=self.marker_scale, y=self.marker_scale, z=self.marker_scale)
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 0.9
        marker.lifetime = Duration(sec=0, nanosec=250_000_000)
        self.marker_pub.publish(marker)
        self.marker_id = (self.marker_id + 1) % 1024

    def _compute_opponent_scores(self, features: np.ndarray) -> np.ndarray:
        if self.cls_score_type == 'proba':
            proba = self.classifier.predict_proba(features)
            return proba[:, 1]
        if self.cls_score_type == 'decision_function':
            scores = self.classifier.decision_function(features)
            return np.ravel(scores)
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(features)[:, 1]
        scores = self.classifier.decision_function(features)
        return np.ravel(scores)

    def _smooth_pose(self, new_prediction: np.ndarray) -> Tuple[float, float, float]:
        alpha = self.smoothing_alpha
        x, y, yaw = float(new_prediction[0]), float(new_prediction[1]), float(new_prediction[2])
        if self.prev_pose is None:
            self.prev_pose = (x, y, yaw)
            return x, y, yaw

        prev_x, prev_y, prev_yaw = self.prev_pose
        smoothed_x = alpha * x + (1 - alpha) * prev_x
        smoothed_y = alpha * y + (1 - alpha) * prev_y
        delta_yaw = self._angle_difference(yaw, prev_yaw)
        smoothed_yaw = prev_yaw + alpha * delta_yaw
        self.prev_pose = (smoothed_x, smoothed_y, smoothed_yaw)
        return smoothed_x, smoothed_y, smoothed_yaw

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
        half = yaw / 2.0
        return (0.0, 0.0, math.sin(half), math.cos(half))

    @staticmethod
    def _angle_difference(a: float, b: float) -> float:
        diff = a - b
        return ((diff + math.pi) % (2 * math.pi)) - math.pi

    @staticmethod
    def _quaternion_to_yaw(orientation: Quaternion) -> float:
        siny = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy = 1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2)
        return math.atan2(siny, cosy)

    @staticmethod
    def _stamp_to_seconds(stamp) -> Optional[float]:
        if stamp.sec == 0 and stamp.nanosec == 0:
            return None
        return stamp.sec + stamp.nanosec * 1e-9

    def _log_delay_stats(self) -> None:
        if self.delay_count == 0:
            return
        avg_delay = self.delay_sum / self.delay_count
        median_delay = float(np.median(list(self.delay_history))) if self.delay_history else 0.0
        self.get_logger().info(
            f'[RT delay] avg={avg_delay*1000:.1f} ms, median={median_delay*1000:.1f} ms, max={self.max_delay*1000:.1f} ms'
        )

    def _plot_rmse_trajectory(self) -> None:
        if not self.match_errors or self.rmse_plot_path is None:
            return
        gt_xs, gt_ys = zip(*self.match_gt_positions)
        pred_xs, pred_ys = zip(*self.match_pred_positions)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(gt_xs, gt_ys, label='GT opponent', color='gold', linewidth=2)
        ax.plot(pred_xs, pred_ys, label='Predicted opponent', color='tab:red', linewidth=2)
        sc = ax.scatter(
            pred_xs,
            pred_ys,
            c=self.match_errors,
            cmap='rainbow',
            s=40,
            edgecolor='black',
            label='Error coloration',
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('position RMSE (m)')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Opponent prediction deviation (rainbow = RMSE)')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        self.rmse_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.rmse_plot_path, dpi=200)
        plt.close(fig)
        self.get_logger().info(f'Saved RMSE trajectory plot to {self.rmse_plot_path}')

    def destroy_node(self) -> None:
        self._log_rmse(final=True)
        if self.csv_file:
            self.csv_file.close()
        self._plot_rmse_trajectory()
        super().destroy_node()


def main():
    rclpy.init()
    node = EvaluateOpponentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
