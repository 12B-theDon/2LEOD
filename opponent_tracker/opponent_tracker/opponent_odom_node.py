#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from opponent_tracker.feature_builder import ClusterFeatureBuilder


class OpponentOdomNode(Node):
    def __init__(self):
        super().__init__('opponent_odom_node')
        self.declare_parameter('model_path', 'models/opponent_bundle.joblib')
        self.declare_parameter('decision_threshold', 0.55)
        self.declare_parameter('smoothing_alpha', 0.5)
        self.declare_parameter('classifier_type', 'auto')
        self.declare_parameter('min_cluster_points', 3)
        self.declare_parameter('max_range', 10.0)

        model_path = Path(self.get_parameter('model_path').get_parameter_value().string_value)
        self.decision_threshold = self.get_parameter('decision_threshold').get_parameter_value().double_value
        self.smoothing_alpha = self.get_parameter('smoothing_alpha').get_parameter_value().double_value
        declared_classifier_type = self.get_parameter('classifier_type').get_parameter_value().string_value

        self.builder = ClusterFeatureBuilder(
            min_cluster_points=self.get_parameter('min_cluster_points').get_parameter_value().integer_value,
            max_range=self.get_parameter('max_range').get_parameter_value().double_value,
        )

        bundle = joblib.load(str(model_path))
        self.classifier = bundle.get('clf_pipe') or bundle.get('classifier')
        self.regressor = bundle.get('regressor')
        bundle_classifier_type = bundle.get('classifier_type', 'logreg')
        self.classifier_type = (
            declared_classifier_type
            if declared_classifier_type in {'logreg', 'svm'}
            else bundle_classifier_type
        )
        self.cls_score_type = bundle.get('cls_score_type', 'proba')
        self.cls_threshold = float(bundle.get('cls_threshold', self.decision_threshold))

        if self.classifier is None or self.regressor is None:
            self.get_logger().error('Classifier or regressor missing in bundle.')
            raise RuntimeError('Invalid model bundle')

        self.get_logger().info(
            f'Loaded model from {model_path}; classifier_type={self.classifier_type}, '
            f'score_type={self.cls_score_type}, threshold={self.cls_threshold:.3f}'
        )

        self.prev_pose: Optional[Tuple[float, float, float]] = None
        self.latest_odom: Optional[Odometry] = None

        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_callback, 10)
        self.publisher = self.create_publisher(Odometry, '/opponent_odom', 10)

    def _odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self.builder.update_odom(msg)

    def _scan_callback(self, msg: LaserScan) -> None:
        if self.latest_odom is None:
            return

        callback_start = time.perf_counter()

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

        msg = Odometry()
        msg.header.stamp = self.latest_odom.header.stamp
        msg.header.frame_id = 'base_link'
        msg.child_frame_id = 'opponent_base_link'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        q = self._yaw_to_quaternion(yaw)
        msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        msg.twist.twist = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )

        self.publisher.publish(msg)
        self.get_logger().info(f'Published opponent @ ({x:.2f},{y:.2f})')
        elapsed = time.perf_counter() - callback_start
        self.get_logger().debug(
            f"[runtime] scan processed in {elapsed:.4f}s, features={features.shape[0]}, candidates={mask.sum()}"
        )

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

    def _smooth_pose(self, new_pose: np.ndarray) -> Tuple[float, float, float]:
        alpha = self.smoothing_alpha
        x, y, yaw = float(new_pose[0]), float(new_pose[1]), float(new_pose[2])
        if self.prev_pose is None:
            self.prev_pose = (x, y, yaw)
            return x, y, yaw

        prev_x, prev_y, prev_yaw = self.prev_pose
        smoothed_x = alpha * x + (1 - alpha) * prev_x
        smoothed_y = alpha * y + (1 - alpha) * prev_y
        delta_yaw = ((yaw - prev_yaw + math.pi) % (2 * math.pi)) - math.pi
        smoothed_yaw = prev_yaw + alpha * delta_yaw
        self.prev_pose = (smoothed_x, smoothed_y, smoothed_yaw)
        return smoothed_x, smoothed_y, smoothed_yaw

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
        half = yaw / 2.0
        return (0.0, 0.0, math.sin(half), math.cos(half))


def main():
    rclpy.init()
    node = OpponentOdomNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
