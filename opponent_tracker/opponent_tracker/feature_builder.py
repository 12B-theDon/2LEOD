from __future__ import annotations

import math
from collections import deque
from typing import List, Tuple

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from config import CLUSTER_FEATURE_COLUMNS


class Cluster:
    def __init__(self, start_idx: int):
        self.start = start_idx
        self.ranges: List[float] = []
        self.angles: List[float] = []
        self.local_points: List[Tuple[float, float]] = []

    def add_point(self, distance: float, angle: float, x: float, y: float):
        self.ranges.append(distance)
        self.angles.append(angle)
        self.local_points.append((x, y))

    def is_valid(self, min_points: int) -> bool:
        return len(self.local_points) >= min_points

    def get_mean_local(self) -> Tuple[float, float]:
        pts = np.array(self.local_points, dtype=float)
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

    def get_mean_distance(self) -> float:
        return float(np.mean(self.ranges)) if self.ranges else 0.0

    def get_distance_std(self) -> float:
        return float(np.std(self.ranges, ddof=0)) if len(self.ranges) > 1 else 0.0

    def get_angle_span(self) -> float:
        if not self.angles:
            return 0.0
        return float(max(self.angles) - min(self.angles))


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


class ClusterFeatureBuilder:
    def __init__(self, min_cluster_points: int = 3, max_range: float = 30.0):
        self.min_cluster_points = min_cluster_points
        self.max_range = max_range
        self.previous_odom: Odometry | None = None
        self.previous_clusters: List[Cluster] = []
        self.previous_centers: List[Tuple[float, float]] = []

    def update_odom(self, odom_msg: Odometry) -> None:
        self.previous_odom = odom_msg

    def build_clusters_from_scan(self, scan_msg: LaserScan) -> List[Cluster]:
        clusters: List[Cluster] = []
        current: Cluster | None = None
        for idx, distance in enumerate(scan_msg.ranges):
            if math.isinf(distance) or math.isnan(distance) or distance > self.max_range:
                if current and current.is_valid(self.min_cluster_points):
                    clusters.append(current)
                current = None
                continue

            angle = scan_msg.angle_min + idx * scan_msg.angle_increment
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            if current is None:
                current = Cluster(idx)
            current.add_point(distance, angle, x, y)
        if current and current.is_valid(self.min_cluster_points):
            clusters.append(current)
        return clusters

    def compute_features(self, clusters: List[Cluster], odom_msg: Odometry) -> np.ndarray:
        ego_x = odom_msg.pose.pose.position.x
        ego_y = odom_msg.pose.pose.position.y
        ego_yaw = self._quaternion_to_yaw(odom_msg.pose.pose.orientation)
        ego_v = math.hypot(
            odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y
        )
        ego_w = odom_msg.twist.twist.angular.z
        features = []
        for cluster in clusters:
            center_x, center_y = cluster.get_mean_local()
            distance_mean = cluster.get_mean_distance()
            values = {
                "center_local_x": center_x,
                "center_local_y": center_y,
                "global_x": ego_x + center_x * math.cos(ego_yaw) - center_y * math.sin(ego_yaw),
                "global_y": ego_y + center_x * math.sin(ego_yaw) + center_y * math.cos(ego_yaw),
                "cluster_size": len(cluster.local_points),
                "distance_mean": distance_mean,
                "distance_std": cluster.get_distance_std(),
                "range_extent": max(cluster.ranges) - min(cluster.ranges) if cluster.ranges else 0.0,
                "angle_index": float(np.mean(cluster.angles)) if cluster.angles else 0.0,
                "angle_extent": cluster.get_angle_span(),
                "delta_range": 0.0,
                "ego_v": ego_v,
                "ego_w": ego_w,
                "ego_vx": odom_msg.twist.twist.linear.x,
                "ego_vy": odom_msg.twist.twist.linear.y,
                "local_x_extent": max(p[0] for p in cluster.local_points) - min(p[0] for p in cluster.local_points),
                "local_y_extent": max(p[1] for p in cluster.local_points) - min(p[1] for p in cluster.local_points),
                "local_x_std": float(np.std([p[0] for p in cluster.local_points], ddof=0)),
                "local_y_std": float(np.std([p[1] for p in cluster.local_points], ddof=0)),
                "cluster_cx": center_x,
                "cluster_cy": center_y,
                "cluster_radius": math.hypot(center_x, center_y),
                "cluster_vx": 0.0,
                "cluster_vy": 0.0,
                "cluster_speed": 0.0,
                "cluster_vr": 0.0,
                "cluster_vx_world": 0.0,
                "cluster_vy_world": 0.0,
                "cluster_speed_world": 0.0,
                "cluster_rel_vx": 0.0,
                "cluster_rel_vy": 0.0,
                "cluster_rel_speed": 0.0,
                "cluster_spread": float(np.mean([math.hypot(p[0] - center_x, p[1] - center_y) for p in cluster.local_points])),
                "cluster_r_span": float(
                    max(cluster.ranges) - min(cluster.ranges) if cluster.ranges else 0.0
                ),
                "cluster_angle_entropy": 0.0,
                "distance_to_ego": math.hypot(center_x, center_y),
                "bearing": math.atan2(center_y, center_x),
                "cluster_opponent_ratio": 0.0,
            }
            feature_vector = [float(values.get(col, 0.0)) for col in CLUSTER_FEATURE_COLUMNS]
            features.append(feature_vector)
        return np.array(features, dtype=float)

    @staticmethod
    def _quaternion_to_yaw(orientation) -> float:
        siny = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny, cosy)
