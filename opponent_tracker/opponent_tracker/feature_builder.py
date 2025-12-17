from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# Feature column order used by the classifier/regressor bundle.
# Keep this list in sync with the training pipeline defaults.
CLUSTER_FEATURE_COLUMNS = [
    "center_local_x",
    "center_local_y",
    "global_x",
    "global_y",
    "cluster_size",
    "distance_mean",
    "distance_std",
    "range_extent",
    "angle_index",
    "angle_extent",
    "delta_range",
    "ego_v",
    "ego_w",
    "ego_vx",
    "ego_vy",
    "local_x_extent",
    "local_y_extent",
    "local_x_std",
    "local_y_std",
    "cluster_cx",
    "cluster_cy",
    "cluster_radius",
    "cluster_vx",
    "cluster_vy",
    "cluster_speed",
    "cluster_vr",
    "cluster_vx_world",
    "cluster_vy_world",
    "cluster_speed_world",
    "cluster_rel_vx",
    "cluster_rel_vy",
    "cluster_rel_speed",
    "cluster_spread",
    "cluster_r_span",
    "cluster_angle_entropy",
    "distance_to_ego",
    "bearing",
]


WORLD_MATCH_DIST = 1.0
HINT_MATCH_RADIUS = 2.5


class Cluster:
    def __init__(self, start_idx: int):
        self.start = start_idx
        self.end = start_idx
        self.ranges: List[float] = []
        self.angles: List[float] = []
        self.local_points: List[Tuple[float, float]] = []

    def add_point(self, distance: float, angle: float, x: float, y: float, scan_idx: int):
        self.ranges.append(distance)
        self.angles.append(angle)
        self.local_points.append((x, y))
        self.end = scan_idx

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
    def __init__(
        self,
        min_cluster_points: int = 3,
        max_range: float = 30.0,
        min_angle: float | None = None,
        max_angle: float | None = None,
        world_match_dist: float = WORLD_MATCH_DIST,
        opponent_match_radius: float = HINT_MATCH_RADIUS,
    ):
        self.min_cluster_points = min_cluster_points
        self.max_range = max_range
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.world_match_dist = world_match_dist
        self.opponent_match_radius = opponent_match_radius
        self.previous_odom: Odometry | None = None
        self._prev_cluster_states: list[dict] = []
        self._prev_stamp: float | None = None
        self._latest_cluster_info: list[dict] = []
        self._latest_cluster_stats: dict[str, int] = {}
        self.debug_logger = None

    def update_odom(self, odom_msg: Odometry) -> None:
        self.previous_odom = odom_msg

    def update_prediction_hint(self, position: Tuple[float, float] | None) -> None:
        return

    def build_clusters_from_scan(
        self,
        scan_msg: LaserScan,
        sensor_transform: dict[str, float] | None = None,
    ) -> List[Cluster]:
        clusters: List[Cluster] = []
        current: Cluster | None = None
        accepted_clusters = 0
        rejected_clusters = 0
        invalid_readings = 0
        out_of_range_readings = 0
        below_min_range_readings = 0
        out_of_fov_readings = 0
        range_min = scan_msg.range_min if scan_msg.range_min > 0.0 else 0.0
        range_min = max(range_min, 1e-3)
        range_max = scan_msg.range_max if scan_msg.range_max > 0.0 else float('inf')
        effective_max_range = min(self.max_range, range_max)
        for idx, distance in enumerate(scan_msg.ranges):
            angle = scan_msg.angle_min + idx * scan_msg.angle_increment
            invalid_measurement = False
            if math.isinf(distance) or math.isnan(distance):
                invalid_measurement = True
                invalid_readings += 1
            elif distance < range_min:
                invalid_measurement = True
                below_min_range_readings += 1
            elif distance >= effective_max_range:
                invalid_measurement = True
                out_of_range_readings += 1
            elif self.min_angle is not None and angle < self.min_angle:
                invalid_measurement = True
                out_of_fov_readings += 1
            elif self.max_angle is not None and angle > self.max_angle:
                invalid_measurement = True
                out_of_fov_readings += 1

            if invalid_measurement:
                if current and current.is_valid(self.min_cluster_points):
                    clusters.append(current)
                    accepted_clusters += 1
                elif current:
                    rejected_clusters += 1
                current = None
                continue

            angle = scan_msg.angle_min + idx * scan_msg.angle_increment
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            if sensor_transform:
                rot_x = sensor_transform["rot_xx"] * x + sensor_transform["rot_xy"] * y
                rot_y = sensor_transform["rot_yx"] * x + sensor_transform["rot_yy"] * y
                x = rot_x + sensor_transform["x"]
                y = rot_y + sensor_transform["y"]
            if current is None:
                current = Cluster(idx)
            current.add_point(distance, angle, x, y, idx)
        if current and current.is_valid(self.min_cluster_points):
            clusters.append(current)
            accepted_clusters += 1
        elif current:
            rejected_clusters += 1
        self._latest_cluster_stats = {
            "accepted_clusters": accepted_clusters,
            "rejected_clusters": rejected_clusters,
            "invalid_readings": invalid_readings,
            "out_of_range_readings": out_of_range_readings,
            "below_min_range_readings": below_min_range_readings,
            "out_of_fov_readings": out_of_fov_readings,
        }
        return clusters

    def get_latest_cluster_stats(self) -> dict[str, int]:
        return dict(self._latest_cluster_stats)

    def compute_features(
        self,
        clusters: List[Cluster],
        odom_msg: Odometry,
        scan_msg: LaserScan | None = None,
    ) -> np.ndarray:
        if odom_msg is None:
            return np.empty((0, len(CLUSTER_FEATURE_COLUMNS)), dtype=float)

        ego_x = odom_msg.pose.pose.position.x
        ego_y = odom_msg.pose.pose.position.y
        ego_yaw = self._quaternion_to_yaw(odom_msg.pose.pose.orientation)
        ego_v = math.hypot(odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y)
        ego_w = odom_msg.twist.twist.angular.z
        ego_vx_world = (
            odom_msg.twist.twist.linear.x * math.cos(ego_yaw)
            - odom_msg.twist.twist.linear.y * math.sin(ego_yaw)
        )
        ego_vy_world = (
            odom_msg.twist.twist.linear.x * math.sin(ego_yaw)
            + odom_msg.twist.twist.linear.y * math.cos(ego_yaw)
        )
        stamp = self._extract_stamp(scan_msg, odom_msg)
        prev_states = self._prev_cluster_states
        entries: list[dict] = []
        metadata: list[dict] = []
        for idx, cluster in enumerate(clusters):
            center_x, center_y = cluster.get_mean_local()
            distance_mean = cluster.get_mean_distance()
            spread = float(
                np.mean([math.hypot(px - center_x, py - center_y) for px, py in cluster.local_points])
            ) if cluster.local_points else 0.0
            entropy = self._compute_angle_entropy(cluster.angles)
            delta_range = self._compute_delta_range(cluster, prev_states, distance_mean)
            global_x = ego_x + center_x * math.cos(ego_yaw) - center_y * math.sin(ego_yaw)
            global_y = ego_y + center_x * math.sin(ego_yaw) + center_y * math.cos(ego_yaw)
            if self.debug_logger:
                self.debug_logger.debug(
                    f'cluster_debug[{idx}]: local=({center_x:.2f}, {center_y:.2f}) '
                    f'global=({global_x:.2f}, {global_y:.2f}) '
                    f'ego=({ego_x:.2f}, {ego_y:.2f}, yaw={ego_yaw:.2f}) '
                    f'radius={math.hypot(center_x, center_y):.2f}'
                )
            values = {
                "center_local_x": center_x,
                "center_local_y": center_y,
                "global_x": global_x,
                "global_y": global_y,
                "cluster_size": len(cluster.local_points),
                "distance_mean": distance_mean,
                "distance_std": cluster.get_distance_std(),
                "range_extent": max(cluster.ranges) - min(cluster.ranges) if cluster.ranges else 0.0,
                "angle_index": float(np.mean(cluster.angles)) if cluster.angles else 0.0,
                "angle_extent": cluster.get_angle_span(),
                "delta_range": delta_range,
                "ego_v": ego_v,
                "ego_w": ego_w,
                "ego_vx": ego_vx_world,
                "ego_vy": ego_vy_world,
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
                "cluster_spread": spread,
                "cluster_r_span": float(
                    max(cluster.ranges) - min(cluster.ranges) if cluster.ranges else 0.0
                ),
                "cluster_angle_entropy": entropy,
                "distance_to_ego": math.hypot(global_x, global_y),
                "bearing": math.atan2(global_y, global_x),
            }
            values["_start"] = cluster.start
            values["_end"] = cluster.end
            values["_mean_distance"] = distance_mean
            values["_global"] = (global_x, global_y)
            values["_local"] = (center_x, center_y)
            entries.append(values)
            metadata.append(
                {
                    "global_x": global_x,
                    "global_y": global_y,
                    "radius": values["cluster_radius"],
                }
            )

        dt = None
        if stamp is not None and self._prev_stamp is not None:
            dt = max(stamp - self._prev_stamp, 1e-3)
        self._apply_temporal_features(entries, prev_states, dt, ego_yaw, ego_vx_world, ego_vy_world)

        features = [
            [float(entry.get(col, 0.0)) for col in CLUSTER_FEATURE_COLUMNS] for entry in entries
        ]

        self._prev_cluster_states = [
            {
                "start": entry["_start"],
                "end": entry["_end"],
                "mean_distance": entry["_mean_distance"],
                "global_x": entry["_global"][0],
                "global_y": entry["_global"][1],
                "cluster_cx": entry["_local"][0],
                "cluster_cy": entry["_local"][1],
                "cluster_radius": entry["cluster_radius"],
            }
            for entry in entries
        ]
        self._prev_stamp = stamp
        self._latest_cluster_info = metadata
        return np.array(features, dtype=float)

    def _compute_delta_range(
        self,
        cluster: Cluster,
        prev_states: list[dict],
        distance_mean: float,
    ) -> float:
        for prev in prev_states:
            if cluster.end < prev["start"] or cluster.start > prev["end"]:
                continue
            return distance_mean - prev["mean_distance"]
        return 0.0

    def _apply_temporal_features(
        self,
        entries: list[dict],
        prev_states: list[dict],
        dt: float | None,
        ego_yaw: float,
        ego_vx_world: float,
        ego_vy_world: float,
    ) -> None:
        if not prev_states or dt is None or not entries:
            return

        prev_positions = np.array(
            [[state["global_x"], state["global_y"]] for state in prev_states], dtype=float
        )

        for entry in entries:
            current = np.array(entry["_global"], dtype=float)
            diffs = prev_positions - current
            dists = np.linalg.norm(diffs, axis=1)
            if dists.size == 0:
                break
            match_idx = int(np.argmin(dists))
            if dists[match_idx] > self.world_match_dist:
                continue
            prev_state = prev_states[match_idx]
            vx_world = (entry["_global"][0] - prev_state["global_x"]) / dt
            vy_world = (entry["_global"][1] - prev_state["global_y"]) / dt
            speed_world = math.hypot(vx_world, vy_world)
            entry["cluster_vx_world"] = vx_world
            entry["cluster_vy_world"] = vy_world
            entry["cluster_speed_world"] = speed_world

            # Transform world velocity into current ego frame for cluster_v*
            cos_yaw = math.cos(-ego_yaw)
            sin_yaw = math.sin(-ego_yaw)
            vx_local = vx_world * cos_yaw - vy_world * sin_yaw
            vy_local = vx_world * sin_yaw + vy_world * cos_yaw
            entry["cluster_vx"] = vx_local
            entry["cluster_vy"] = vy_local
            entry["cluster_speed"] = math.hypot(vx_local, vy_local)

            radius_prev = prev_state["cluster_radius"]
            entry["cluster_vr"] = (entry["cluster_radius"] - radius_prev) / dt

            rel_vx = vx_world - ego_vx_world
            rel_vy = vy_world - ego_vy_world
            entry["cluster_rel_vx"] = rel_vx
            entry["cluster_rel_vy"] = rel_vy
            entry["cluster_rel_speed"] = math.hypot(rel_vx, rel_vy)

    def get_latest_cluster_info(self) -> list[dict]:
        return list(self._latest_cluster_info)

    @staticmethod
    def _compute_angle_entropy(angles: List[float]) -> float:
        if len(angles) <= 1:
            return 0.0
        bins = min(len(angles), 8)
        counts, _ = np.histogram(angles, bins=bins)
        probs = counts / counts.sum() if counts.sum() > 0 else counts
        probs = probs[probs > 0]
        if probs.size == 0:
            return 0.0
        return float(-np.sum(probs * np.log(probs)))

    @staticmethod
    def _extract_stamp(scan_msg: LaserScan | None, odom_msg: Odometry | None) -> float | None:
        if scan_msg:
            return scan_msg.header.stamp.sec + scan_msg.header.stamp.nanosec * 1e-9
        if odom_msg:
            return odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
        return None

    @staticmethod
    def _quaternion_to_yaw(orientation) -> float:
        siny = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny, cosy)
