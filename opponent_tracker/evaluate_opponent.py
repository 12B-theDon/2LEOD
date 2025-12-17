#!/usr/bin/env python3
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
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import tf2_geometry_msgs  # noqa: F401
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
    TransformException,
)
from visualization_msgs.msg import Marker, MarkerArray

from opponent_tracker.feature_builder import ClusterFeatureBuilder


class EvaluateOpponentNode(Node):
    def __init__(self):
        super().__init__('evaluate_opponent')
        self.declare_parameter('model_path', 'models/opponent_bundle.joblib')
        self.declare_parameter('decision_threshold', 0.55)
        self.declare_parameter('use_model_cls_threshold', True)
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
        self.declare_parameter('marker_lifetime', 0.0)
        self.declare_parameter('use_tf_odom', False)
        self.declare_parameter('tf_odom_frame', 'base_link')
        self.declare_parameter('tf_world_frame', '')
        self.declare_parameter('tf_odom_poll_rate', 50.0)
        self.declare_parameter('csv_path', '')
        self.declare_parameter('log_interval', 10)
        self.declare_parameter('prediction_debug', False)
        self.declare_parameter('rmse_plot_path', 'rmse_trajectory.png')
        self.declare_parameter('speed_plot_path', 'speed_error.png')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('gt_topic', '/opponent_odom')
        self.declare_parameter('enable_scan_angle_filter', False)
        self.declare_parameter('scan_angle_min_deg', float('nan'))
        self.declare_parameter('scan_angle_max_deg', float('nan'))

        model_path = Path(self.get_parameter('model_path').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.timestamp_tolerance = float(self.get_parameter('timestamp_tolerance').value)
        self.gt_history_size = int(self.get_parameter('gt_history_size').value)
        self.path_length = int(self.get_parameter('path_length').value)
        self.publish_markers = bool(self.get_parameter('publish_markers').value)
        self.marker_ns = self.get_parameter('marker_ns').value
        self.marker_scale = float(self.get_parameter('marker_scale').value)
        self.marker_lifetime = float(self.get_parameter('marker_lifetime').value)
        self.csv_path = self.get_parameter('csv_path').value
        self.log_interval = max(1, int(self.get_parameter('log_interval').value))
        self.prediction_debug = bool(self.get_parameter('prediction_debug').value)
        rmse_plot_path = self.get_parameter('rmse_plot_path').value
        self.rmse_plot_path = Path(rmse_plot_path) if rmse_plot_path else None
        speed_plot_path = self.get_parameter('speed_plot_path').value
        self.speed_plot_path = Path(speed_plot_path) if speed_plot_path else None

        self.decision_threshold = float(self.get_parameter('decision_threshold').value)
        self.use_model_cls_threshold = bool(self.get_parameter('use_model_cls_threshold').value)
        self.smoothing_alpha = float(self.get_parameter('smoothing_alpha').value)
        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.gt_topic = self.get_parameter('gt_topic').value
        self.enable_scan_angle_filter = bool(self.get_parameter('enable_scan_angle_filter').value)
        scan_angle_min_deg = self.get_parameter('scan_angle_min_deg').value
        scan_angle_max_deg = self.get_parameter('scan_angle_max_deg').value
        self.scan_angle_min = self._deg_param_to_rad(scan_angle_min_deg) if self.enable_scan_angle_filter else None
        self.scan_angle_max = self._deg_param_to_rad(scan_angle_max_deg) if self.enable_scan_angle_filter else None
        self.use_tf_odom = bool(self.get_parameter('use_tf_odom').value)
        if not self.odom_topic:
            self.use_tf_odom = True
        self.tf_odom_frame = self.get_parameter('tf_odom_frame').value or 'base_link'
        tf_world_frame_param = self.get_parameter('tf_world_frame').value
        self.tf_world_frame = tf_world_frame_param if tf_world_frame_param else self.frame_id
        poll_rate = max(1.0, float(self.get_parameter('tf_odom_poll_rate').value))
        self.tf_odom_poll_period = 1.0 / poll_rate

        self.builder = ClusterFeatureBuilder(
            min_cluster_points=int(self.get_parameter('min_cluster_points').value),
            max_range=float(self.get_parameter('max_range').value),
            min_angle=self.scan_angle_min,
            max_angle=self.scan_angle_max,
        )

        bundle = joblib.load(str(model_path))
        self.classifier = bundle.get('clf_pipe') or bundle.get('classifier')
        self.regressor = bundle.get('regressor')
        bundle_cls_threshold = bundle.get('cls_threshold')
        self.cls_score_type = bundle.get('cls_score_type', 'proba')
        if self.use_model_cls_threshold and bundle_cls_threshold is not None:
            self.cls_threshold = float(bundle_cls_threshold)
            threshold_src = 'model bundle'
        else:
            self.cls_threshold = self.decision_threshold
            threshold_src = 'decision_threshold parameter'

        if self.classifier is None or self.regressor is None:
            self.get_logger().error('Classifier or regressor missing in bundle.')
            raise RuntimeError('Invalid model bundle')

        self.get_logger().info(
            f'Loaded model from {model_path}; cls_score_type={self.cls_score_type}, '
            f'threshold={self.cls_threshold:.3f} ({threshold_src})'
        )

        self.prev_pose: Optional[Tuple[float, float, float]] = None
        self.latest_odom: Optional[Odometry] = None
        self._last_aligned_state: Optional[dict] = None
        self.gt_buffer: Deque[Tuple[float, Odometry]] = deque(maxlen=self.gt_history_size)
        self.pred_path_queue: Deque[PoseStamped] = deque(maxlen=self.path_length)
        self.gt_path_queue: Deque[PoseStamped] = deque(maxlen=self.path_length)

        self.pred_path_msg = NavPath()
        self.pred_path_msg.header.frame_id = self.frame_id
        self.gt_path_msg = NavPath()
        self.gt_path_msg.header.frame_id = self.frame_id

        self.position_error_sum = 0.0
        self.yaw_error_sum = 0.0
        self.match_count = 0
        self.match_gt_positions: list[Tuple[float, float]] = []
        self.match_pred_positions: list[Tuple[float, float]] = []
        self.match_errors: list[float] = []
        self.match_timestamps: list[float] = []

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
            self.csv_file.flush()

        self.pred_odom_pub = self.create_publisher(Odometry, '/pred_opponent_odom', 10)
        self.pred_path_pub = self.create_publisher(NavPath, '/pred_opponent_path', 10)
        self.gt_path_pub = self.create_publisher(NavPath, '/gt_opponent_path', 10)
        if self.publish_markers:
            self.marker_pub = self.create_publisher(Marker, '/pred_opponent_marker', 10)
        self.cluster_candidates_pub = self.create_publisher(MarkerArray, '/cluster_candidates', 10)
        self.cluster_opponents_pub = self.create_publisher(MarkerArray, '/cluster_opponents', 10)

        if not self.use_tf_odom:
            self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        else:
            self.tf_odom_timer = self.create_timer(self.tf_odom_poll_period, self._poll_tf_odom)
            self._last_tf_odom = None
        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.create_subscription(Odometry, self.gt_topic, self._gt_opponent_callback, 10)

        self.marker_id = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self._scan_tf_fail_frames: set[str] = set()

    def _poll_tf_odom(self) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.tf_world_frame,
                self.tf_odom_frame,
                Time(),
                timeout=RclpyDuration(seconds=0.2),
            )
        except (LookupException, ConnectivityException, ExtrapolationException, TransformException) as exc:
            if getattr(self, '_tf_warned_odom', False) is False:
                self.get_logger().warn(
                    f'Failed to fetch TF odom transform {self.tf_world_frame}->{self.tf_odom_frame}: {exc}'
                )
                self._tf_warned_odom = True
            return
        self._tf_warned_odom = False

        odom = Odometry()
        odom.header = Header()
        odom.header.stamp = transform.header.stamp
        odom.header.frame_id = transform.header.frame_id
        odom.child_frame_id = transform.child_frame_id
        odom.pose.pose.position.x = transform.transform.translation.x
        odom.pose.pose.position.y = transform.transform.translation.y
        odom.pose.pose.position.z = transform.transform.translation.z
        odom.pose.pose.orientation = Quaternion(
            x=transform.transform.rotation.x,
            y=transform.transform.rotation.y,
            z=transform.transform.rotation.z,
            w=transform.transform.rotation.w,
        )

        twist = Twist()
        current_time = Time.from_msg(transform.header.stamp)
        current_yaw = self._quaternion_to_yaw(odom.pose.pose.orientation)
        if hasattr(self, '_last_tf_odom') and self._last_tf_odom:
            dt = (current_time - self._last_tf_odom['time']).nanoseconds / 1e9
            if dt > 1e-6:
                twist.linear.x = (odom.pose.pose.position.x - self._last_tf_odom['pos'].x) / dt
                twist.linear.y = (odom.pose.pose.position.y - self._last_tf_odom['pos'].y) / dt
                twist.linear.z = (odom.pose.pose.position.z - self._last_tf_odom['pos'].z) / dt
                yaw_delta = math.atan2(
                    math.sin(current_yaw - self._last_tf_odom['yaw']),
                    math.cos(current_yaw - self._last_tf_odom['yaw']),
                )
                twist.angular.z = yaw_delta / dt
        odom.twist.twist = twist

        self._last_tf_odom = {
            'time': current_time,
            'pos': Vector3(
                x=odom.pose.pose.position.x,
                y=odom.pose.pose.position.y,
                z=odom.pose.pose.position.z,
            ),
            'yaw': current_yaw,
        }
        self._odom_callback(odom)

    def _odom_callback(self, msg: Odometry) -> None:
        self._update_latest_odom(msg)

    def _update_latest_odom(self, msg: Odometry) -> None:
        aligned = self._align_odom(msg)
        if aligned is None:
            return
        self.latest_odom = aligned
        self.builder.update_odom(aligned)
        if self.prediction_debug:
            pos = aligned.pose.pose.position
            self.get_logger().debug(
                f'prediction_debug: aligned odom frame={aligned.header.frame_id} '
                f'pos=({pos.x:.2f}, {pos.y:.2f})'
            )

    def _scan_callback(self, msg: LaserScan) -> None:
        if self.latest_odom is None:
            if self.prediction_debug:
                self.get_logger().info('prediction_debug: skipping scan, odom not received yet')
            return
        callback_start = time.perf_counter()
        scan_ts = self._stamp_to_seconds(msg.header.stamp)
        if scan_ts is not None:
            delay = callback_start - scan_ts
            self.delay_sum += delay
            self.delay_count += 1
            self.delay_history.append(delay)
            self.max_delay = max(self.max_delay, delay)

        scan_transform = self._lookup_scan_transform(msg.header)
        clusters = self.builder.build_clusters_from_scan(msg, sensor_transform=scan_transform)
        if self.prediction_debug:
            stats = self.builder.get_latest_cluster_stats()
            if stats:
                accepted = stats.get('accepted_clusters', 0)
                rejected = stats.get('rejected_clusters', 0)
                invalid = stats.get('invalid_readings', 0)
                out_of_range = stats.get('out_of_range_readings', 0)
                below_min = stats.get('below_min_range_readings', 0)
                out_of_fov = stats.get('out_of_fov_readings', 0)
                self.get_logger().info(
                    f'prediction_debug: scan cluster stats accepted={accepted} rejected={rejected} '
                    f'invalid_readings={invalid} below_min={below_min} out_of_range={out_of_range} '
                    f'out_of_fov={out_of_fov} '
                    f'min_cluster_points={self.builder.min_cluster_points}'
                )
        if not clusters:
            if self.prediction_debug:
                self.get_logger().info('prediction_debug: scan produced no valid clusters')
            return

        features = self.builder.compute_features(clusters, self.latest_odom, msg)
        metadata = self.builder.get_latest_cluster_info()
        if features.size == 0:
            if self.prediction_debug:
                self.get_logger().info('prediction_debug: cluster features empty after computation')
            self._publish_cluster_markers(metadata)
            return

        scores = self._compute_opponent_scores(features)
        mask = scores >= self.cls_threshold
        if not mask.any():
            if self.prediction_debug:
                best_score = float(np.max(scores)) if scores.size > 0 else float('nan')
                self.get_logger().info(
                    f'prediction_debug: all scores below threshold {self.cls_threshold:.3f}, '
                    f'best={best_score:.3f}'
                )
            self._publish_cluster_markers(metadata, scores, mask)
            return
        self._publish_cluster_markers(metadata, scores, mask)

        candidate_indices = np.where(mask)[0]
        selected_idx = int(candidate_indices[np.argmax(scores[candidate_indices])])
        latent = features[selected_idx].reshape(1, -1)
        prediction = self.regressor.predict(latent)[0]
        rel_x, rel_y, rel_yaw = self._smooth_pose(prediction)
        ego_pose = self.latest_odom.pose.pose
        ego_x = ego_pose.position.x
        ego_y = ego_pose.position.y
        ego_yaw = self._quaternion_to_yaw(ego_pose.orientation)
        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)
        default_x = ego_x + rel_x * cos_yaw - rel_y * sin_yaw
        default_y = ego_y + rel_x * sin_yaw + rel_y * cos_yaw
        yaw = ego_yaw + rel_yaw
        x = default_x
        y = default_y
        if 0 <= selected_idx < len(metadata):
            cluster_info = metadata[selected_idx]
            cluster_x = cluster_info.get('global_x')
            cluster_y = cluster_info.get('global_y')
            if cluster_x is not None and cluster_y is not None:
                x = float(cluster_x)
                y = float(cluster_y)
        if self.prediction_debug:
            ego_pose = self.latest_odom.pose.pose.position
            ego_quat = self.latest_odom.pose.pose.orientation
            ego_yaw = self._quaternion_to_yaw(ego_quat)
            self.get_logger().info(
                f'prediction_debug: publishing prediction idx={selected_idx} '
                f'score={float(scores[selected_idx]):.3f} pos=({x:.2f}, {y:.2f}), '
                f'ego=({ego_pose.x:.2f}, {ego_pose.y:.2f}, yaw={ego_yaw:.2f})'
            )

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

        # Transform prediction into the configured frame before publishing.
        source_frame = self.latest_odom.header.frame_id or self.frame_id
        if source_frame != self.frame_id:
            pred_pose = PoseStamped()
            pred_pose.header.stamp = pred_msg.header.stamp
            pred_pose.header.frame_id = source_frame
            pred_pose.pose = pred_msg.pose.pose
            transformed = self._transform_pose(pred_pose, target_frame=self.frame_id)
            pred_msg.pose.pose = transformed.pose
            pred_msg.header.frame_id = transformed.header.frame_id

        self.pred_odom_pub.publish(pred_msg)
        self._publish_pred_path(pred_msg)
        if self.publish_markers and scores[selected_idx] >= self.cls_threshold:
            self._publish_marker(pred_msg)
        self._match_with_gt(pred_msg)

    def _lookup_scan_transform(self, header: Header) -> Optional[dict[str, float]]:
        scan_frame = header.frame_id
        if not scan_frame or scan_frame == self.tf_odom_frame:
            return None
        stamp = header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            tf_time = Time()
        else:
            tf_time = Time.from_msg(stamp)
        try:
            transform = self.tf_buffer.lookup_transform(
                self.tf_odom_frame,
                scan_frame,
                tf_time,
                timeout=RclpyDuration(seconds=0.05),
            )
        except (LookupException, ConnectivityException, ExtrapolationException, TransformException) as exc:
            if scan_frame not in self._scan_tf_fail_frames:
                self.get_logger().warn(
                    f'prediction_debug: unable to transform scan frame {scan_frame} to {self.tf_odom_frame}: {exc}'
                )
                self._scan_tf_fail_frames.add(scan_frame)
            return None
        self._scan_tf_fail_frames.discard(scan_frame)
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rot_xx, rot_xy, rot_yx, rot_yy = self._planar_rotation_matrix(rotation)
        return {
            'x': translation.x,
            'y': translation.y,
            'rot_xx': rot_xx,
            'rot_xy': rot_xy,
            'rot_yx': rot_yx,
            'rot_yy': rot_yy,
        }

    @staticmethod
    def _planar_rotation_matrix(quat: Quaternion) -> Tuple[float, float, float, float]:
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        # Rotation matrix derived from quaternion, projected onto XY plane.
        rot_xx = 1.0 - 2.0 * (y * y + z * z)
        rot_xy = 2.0 * (x * y - z * w)
        rot_yx = 2.0 * (x * y + z * w)
        rot_yy = 1.0 - 2.0 * (x * x + z * z)
        return rot_xx, rot_xy, rot_yx, rot_yy

    def _gt_opponent_callback(self, msg: Odometry) -> None:
        stamp = self._stamp_to_seconds(msg.header.stamp)
        if stamp is None:
            return

        gt_pose = PoseStamped()
        gt_pose.header = msg.header
        gt_pose.pose = msg.pose.pose
        transformed_pose = self._transform_pose(gt_pose)

        transformed_msg = Odometry()
        transformed_msg.header = Header()
        transformed_msg.header.stamp = msg.header.stamp
        transformed_msg.header.frame_id = self.frame_id
        transformed_msg.child_frame_id = msg.child_frame_id
        transformed_msg.pose.pose = transformed_pose.pose
        transformed_msg.twist = msg.twist

        self.gt_buffer.append((stamp, transformed_msg))

        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = transformed_pose.header.stamp
        pose.header.frame_id = self.frame_id
        pose.pose = transformed_pose.pose
        self.gt_path_queue.append(pose)

        self.gt_path_msg.header.stamp = transformed_pose.header.stamp
        self.gt_path_msg.poses = list(self.gt_path_queue)
        self.gt_path_pub.publish(self.gt_path_msg)

    def _align_odom(self, odom_msg: Odometry) -> Optional[Odometry]:
        target_frame = self.frame_id
        if not target_frame:
            return odom_msg

        pose = PoseStamped()
        pose.header = odom_msg.header
        pose.pose = odom_msg.pose.pose
        transformed = self._transform_pose(pose, target_frame=target_frame, fail_on_error=True)
        if transformed is None:
            if self.prediction_debug:
                self.get_logger().warn(
                    f'prediction_debug: unable to transform odom from {odom_msg.header.frame_id} to {target_frame}'
                )
            return None

        aligned = Odometry()
        aligned.header = Header()
        aligned.header.stamp = transformed.header.stamp
        aligned.header.frame_id = transformed.header.frame_id
        aligned.child_frame_id = odom_msg.child_frame_id or self.tf_odom_frame
        aligned.pose.pose = transformed.pose

        lin_vel, yaw_rate = self._derive_world_twist(transformed)
        aligned.twist.twist = Twist(
            linear=lin_vel,
            angular=Vector3(x=0.0, y=0.0, z=yaw_rate),
        )
        return aligned

    def _derive_world_twist(self, pose: PoseStamped) -> Tuple[Vector3, float]:
        stamp = self._stamp_to_seconds(pose.header.stamp)
        if stamp is None:
            return Vector3(x=0.0, y=0.0, z=0.0), 0.0
        x = pose.pose.position.x
        y = pose.pose.position.y
        yaw = self._quaternion_to_yaw(pose.pose.orientation)
        vx = vy = wz = 0.0
        if self._last_aligned_state:
            dt = stamp - self._last_aligned_state['time']
            if dt > 1e-5:
                vx = (x - self._last_aligned_state['x']) / dt
                vy = (y - self._last_aligned_state['y']) / dt
                yaw_delta = math.atan2(
                    math.sin(yaw - self._last_aligned_state['yaw']),
                    math.cos(yaw - self._last_aligned_state['yaw']),
                )
                wz = yaw_delta / dt
        self._last_aligned_state = {'time': stamp, 'x': x, 'y': y, 'yaw': yaw}
        return Vector3(x=vx, y=vy, z=0.0), wz

    def _transform_pose(
        self,
        pose: PoseStamped,
        target_frame: str | None = None,
        fail_on_error: bool = False,
    ) -> Optional[PoseStamped]:
        target = target_frame or self.frame_id
        if not pose.header.frame_id or pose.header.frame_id == target:
            pose.header.frame_id = target
            return pose
        timeout = RclpyDuration(seconds=0.2)
        try:
            return self.tf_buffer.transform(
                pose,
                target,
                timeout=timeout,
            )
        except ExtrapolationException:
            # Try again using the latest available transform instead of the stamp encoded on the message.
            try:
                latest_tf = self.tf_buffer.lookup_transform(
                    target,
                    pose.header.frame_id,
                    Time(),
                    timeout=timeout,
                )
                pose.header.stamp = latest_tf.header.stamp
                return self.tf_buffer.transform(
                    pose,
                    target,
                    timeout=timeout,
                )
            except (LookupException, ConnectivityException, ExtrapolationException, TransformException) as exc:
                self.get_logger().warn(
                    f'Failed to transform pose from {pose.header.frame_id} to {target}: {exc}'
                )
                if fail_on_error:
                    return None
                pose.header.frame_id = target
                return pose
        except (LookupException, ConnectivityException, TransformException) as exc:
            self.get_logger().warn(
                f'Failed to transform pose from {pose.header.frame_id} to {target}: {exc}'
            )
            if fail_on_error:
                return None
            pose.header.frame_id = target
            return pose

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
        self.match_timestamps.append(timestamp)

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
        lifetime = Duration()
        if self.marker_lifetime > 0.0:
            lifetime.sec = int(self.marker_lifetime)
            lifetime.nanosec = int((self.marker_lifetime - lifetime.sec) * 1e9)
        marker.lifetime = lifetime
        self.marker_pub.publish(marker)
        self.marker_id = (self.marker_id + 1) % 1024

    def _publish_cluster_markers(
        self,
        metadata: list[dict],
        scores: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        candidates = MarkerArray()
        opponents = MarkerArray()
        lifetime = Duration()
        lifetime.sec = 0
        lifetime.nanosec = int(0.2 * 1e9)
        for idx, info in enumerate(metadata):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = stamp
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(info.get('global_x', 0.0))
            marker.pose.position.y = float(info.get('global_y', 0.0))
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 0.2
            marker.color.g = 0.4
            marker.color.b = 1.0
            marker.color.a = 0.6
            marker.lifetime = lifetime
            candidates.markers.append(marker)
        if scores is not None and mask is not None:
            for idx, (info, keep) in enumerate(zip(metadata, mask)):
                if not keep:
                    continue
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.header.stamp = stamp
                marker.id = idx
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(info.get('global_x', 0.0))
                marker.pose.position.y = float(info.get('global_y', 0.0))
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.25
                marker.scale.y = 0.25
                marker.scale.z = 0.25
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.3
                marker.color.a = 0.8
                marker.lifetime = lifetime
                opponents.markers.append(marker)
        if candidates.markers:
            self.cluster_candidates_pub.publish(candidates)
        if opponents.markers:
            self.cluster_opponents_pub.publish(opponents)

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

    @staticmethod
    def _deg_param_to_rad(value) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return math.radians(numeric)

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

    def _plot_speed_error(self) -> None:
        if self.speed_plot_path is None or len(self.match_timestamps) < 2:
            return
        timestamps = np.array(self.match_timestamps)
        pred_positions = np.array(self.match_pred_positions)
        gt_positions = np.array(self.match_gt_positions)
        if len(pred_positions) < 2 or len(gt_positions) < 2:
            return
        dt = np.diff(timestamps)
        valid = dt > 1e-6
        if not valid.any():
            return
        pred_deltas = np.linalg.norm(np.diff(pred_positions, axis=0), axis=1)[valid]
        gt_deltas = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)[valid]
        dt_valid = dt[valid]
        times = timestamps[1:][valid]
        pred_speeds = pred_deltas / dt_valid
        gt_speeds = gt_deltas / dt_valid
        speed_errors = np.abs(pred_speeds - gt_speeds)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, gt_speeds, label='GT speed', color='tab:blue', linewidth=2)
        ax.plot(times, pred_speeds, label='Predicted speed', color='tab:orange', linewidth=2)
        sc = ax.scatter(
            times,
            pred_speeds,
            c=speed_errors,
            cmap='rainbow',
            edgecolor='black',
            s=40,
            label='Speed error (abs)',
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('speed difference (m/s)')
        ax.set_xlabel('timestamp (s)')
        ax.set_ylabel('speed (m/s)')
        ax.set_title('Opponent speed prediction (rainbow = error)')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        self.speed_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.speed_plot_path, dpi=200)
        plt.close(fig)
        self.get_logger().info(f'Saved speed error plot to {self.speed_plot_path}')

    def destroy_node(self) -> None:
        self._log_rmse(final=True)
        if self.csv_file:
            self.csv_file.close()
        self._plot_rmse_trajectory()
        self._plot_speed_error()
        super().destroy_node()


def main():
    rclpy.init()
    node = EvaluateOpponentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
