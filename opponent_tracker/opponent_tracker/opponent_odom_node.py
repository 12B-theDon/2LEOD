from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Deque, Optional, Tuple

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
        self.declare_parameter('smoothing_window_size', 5)
        self.declare_parameter('min_cluster_points', 3)
        self.declare_parameter('max_range', 10.0)

        model_path = Path(self.get_parameter('model_path').get_parameter_value().string_value)
        self.decision_threshold = self.get_parameter('decision_threshold').get_parameter_value().double_value
        smoothing_window = self.get_parameter('smoothing_window_size').get_parameter_value().integer_value

        self.builder = ClusterFeatureBuilder(
            min_cluster_points=self.get_parameter('min_cluster_points').get_parameter_value().integer_value,
            max_range=self.get_parameter('max_range').get_parameter_value().double_value,
        )

        bundle = joblib.load(str(model_path))
        self.classifier = bundle.get('clf_pipe') or bundle.get('classifier')
        self.regressor = bundle.get('regressor')
        self.threshold = bundle.get('threshold', self.decision_threshold)

        if self.classifier is None or self.regressor is None:
            self.get_logger().error('Classifier or regressor missing in bundle.')
            raise RuntimeError('Invalid model bundle')

        self.get_logger().info(f'Loaded model from {model_path}; threshold={self.threshold}')

        self.smoothing_buffer: Deque[np.ndarray] = collections.deque(maxlen=smoothing_window)
        self.latest_odom: Optional[Odometry] = None

        self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            10,
        )
        self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            10,
        )
        self.publisher = self.create_publisher(Odometry, '/opponent_odom', 10)

    def _odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self.builder.update_odom(msg)

    def _scan_callback(self, msg: LaserScan) -> None:
        if self.latest_odom is None:
            return

        clusters = self.builder.build_clusters_from_scan(msg)
        if not clusters:
            return

        features = self.builder.compute_features(clusters, self.latest_odom)
        if features.size == 0:
            return

        probs = self.classifier.predict_proba(features)[:, 1]
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        if best_prob < self.threshold:
            return

        latent = features[best_idx].reshape(1, -1)
        prediction = self.regressor.predict(latent)[0]
        self.smoothing_buffer.append(prediction)
        smooth = np.mean(np.stack(self.smoothing_buffer), axis=0)
        x, y, yaw = float(smooth[0]), float(smooth[1]), float(smooth[2])

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
        self.get_logger().info(f'Published opponent @ ({x:.2f},{y:.2f}) p={best_prob:.2f}')

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
