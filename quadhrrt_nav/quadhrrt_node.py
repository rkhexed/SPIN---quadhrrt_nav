#!/usr/bin/env python3
# coding=utf-8
"""
quadhrrt_node.py
----------------
STEP 1 node: run QuadHRRT* as a standalone Nav2-integrated planning node.

Pipeline (no semantics, no LLM yet):
    /global_costmap/costmap  (nav_msgs/OccupancyGrid)  -> CostmapAdapter
    /goal_pose               (geometry_msgs/PoseStamped)
    TF map -> base_link      -> start pose
    QuadRRTPlanner.plan(start, goal) -> list of (x, y)
    -> nav_msgs/Path (frame 'map')
    -> FollowPath action (nav2_msgs/action/FollowPath) -> Nav2 controller drives

This mirrors Abaza's ComputeRoute -> FollowPath structure. It is NOT a Nav2
global-planner plugin.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSHistoryPolicy, QoSDurabilityPolicy)

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath

import tf2_ros

from quadhrrt_nav.costmap_adapter import CostmapAdapter, LETHAL_THRESHOLD
from quadhrrt_nav.planner_core import QuadRRTPlanner


class QuadHRRTNode(Node):
    def __init__(self):
        super().__init__('quadhrrt_node')

        # ── parameters ──────────────────────────────────────────
        self.declare_parameter('costmap_topic', '/global_costmap/costmap')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('controller_id', '')
        self.declare_parameter('goal_checker_id', '')
        self.declare_parameter('lethal_threshold', LETHAL_THRESHOLD)
        # Bounded latency + evaluation logging (Step 2 wrap-up).
        self.declare_parameter('plan_time_budget', 1.0)   # seconds
        self.declare_parameter('metrics_csv', '')          # '' => disabled
        # ── Step 3: semantic cost injection ──
        self.declare_parameter('semantic_weight', 5.0)
        self.declare_parameter('semantic_sigma', 0.5)
        # force_anytime: deployment leaves this False (anytime auto-ON only when
        # detections are present). The evaluation harness sets it True to run
        # condition B (planner, no semantics) under the SAME optimization as
        # condition C (planner + semantics), isolating the semantic effect.
        self.declare_parameter('force_anytime', False)
        # Static test detections (no live YOLO yet): "class:x:y" strings in world
        # coords, e.g. ['person:1.0:2.0']. Empty => no semantics.
        self.declare_parameter('detections', [''])

        self.costmap_topic   = self.get_parameter('costmap_topic').value
        self.goal_topic      = self.get_parameter('goal_topic').value
        self.global_frame    = self.get_parameter('global_frame').value
        self.robot_frame     = self.get_parameter('robot_frame').value
        self.controller_id   = self.get_parameter('controller_id').value
        self.goal_checker_id = self.get_parameter('goal_checker_id').value
        lethal_threshold     = int(self.get_parameter('lethal_threshold').value)
        plan_time_budget     = float(self.get_parameter('plan_time_budget').value)
        metrics_csv          = self.get_parameter('metrics_csv').value or None
        semantic_weight      = float(self.get_parameter('semantic_weight').value)
        semantic_sigma       = float(self.get_parameter('semantic_sigma').value)
        force_anytime        = bool(self.get_parameter('force_anytime').value)

        # ── planner + costmap adapter ───────────────────────────
        self.adapter = CostmapAdapter(lethal_threshold=lethal_threshold)
        self.planner = QuadRRTPlanner(
            self.adapter,
            metrics_csv=metrics_csv,
            plan_time_budget=plan_time_budget,
            semantic_weight=semantic_weight,
            semantic_sigma=semantic_sigma,
            force_anytime=force_anytime)
        if metrics_csv:
            self.get_logger().info(f"Logging plan metrics to {metrics_csv}")

        # Load any static test detections (world coords).
        dets = self.parse_detections(self.get_parameter('detections').value)
        if dets:
            self.planner.set_detections(dets)
            self.get_logger().info(
                f"Semantic: {len(dets)} detection(s) loaded, "
                f"weight={semantic_weight}, sigma={semantic_sigma}.")

        # ── TF (for start pose) ─────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── costmap subscription (Nav2 latches it: transient_local) ─
        costmap_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, self.costmap_topic,
            self.on_costmap, costmap_qos)

        # ── goal subscription ───────────────────────────────────
        self.goal_sub = self.create_subscription(
            PoseStamped, self.goal_topic, self.on_goal, 10)

        # ── FollowPath action client ────────────────────────────
        self.follow_client = ActionClient(self, FollowPath, 'follow_path')

        # ── path publisher (for RViz inspection) ────────────────
        self.path_pub = self.create_publisher(Path, 'quadhrrt_path', 10)

        self.get_logger().info(
            f"quadhrrt_node up. costmap='{self.costmap_topic}', "
            f"goal='{self.goal_topic}', frames '{self.global_frame}'->"
            f"'{self.robot_frame}'. Waiting for costmap + goal.")

    # ── callbacks ───────────────────────────────────────────────
    def on_costmap(self, msg: OccupancyGrid):
        first = not self.adapter.ready()
        self.adapter.update_from_msg(msg)
        if first:
            self.get_logger().info(
                f"First costmap: {self.adapter.w}x{self.adapter.h} @ "
                f"{self.adapter.res:.3f} m/cell, "
                f"origin=({self.adapter.origin_x:.2f}, "
                f"{self.adapter.origin_y:.2f}).")

    def on_goal(self, msg: PoseStamped):
        if not self.adapter.ready():
            self.get_logger().warn("Goal received but no costmap yet — ignoring.")
            return

        start = self.lookup_start()
        if start is None:
            return

        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.get_logger().info(
            f"Planning: start=({start[0]:.2f}, {start[1]:.2f}) -> "
            f"goal=({gx:.2f}, {gy:.2f})")

        path_pts = self.planner.plan(start, (gx, gy))
        if not path_pts:
            self.get_logger().error("Planner returned no path.")
            return

        self.get_logger().info(
            f"Path found: {len(path_pts)} waypoints, "
            f"{self.planner.plan_time*1000:.0f} ms, "
            f"{self.planner.smoothed_path_length:.2f} m.")

        path_msg = self.build_path_msg(path_pts)
        self.path_pub.publish(path_msg)
        self.send_follow_path(path_msg)

    # ── helpers ─────────────────────────────────────────────────
    def parse_detections(self, raw):
        """Parse ['class:x:y', ...] param into [(class, x, y), ...] world coords.
        Empty/blank entries are ignored. Malformed entries are warned + skipped."""
        dets = []
        for item in (raw or []):
            if not item or not item.strip():
                continue
            parts = item.split(':')
            if len(parts) != 3:
                self.get_logger().warn(f"Bad detection '{item}' (want class:x:y).")
                continue
            try:
                dets.append((parts[0], float(parts[1]), float(parts[2])))
            except ValueError:
                self.get_logger().warn(f"Bad detection coords in '{item}'.")
        return dets

    def lookup_start(self):
        """Robot position in the global frame from TF, or None on failure."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame, self.robot_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(
                f"TF {self.global_frame}->{self.robot_frame} failed: {e}. "
                "Is AMCL localized?")
            return None
        t = tf.transform.translation
        return (t.x, t.y)

    def build_path_msg(self, path_pts):
        """list[(x, y)] -> nav_msgs/Path in the global frame, with yaws."""
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()

        n = len(path_pts)
        for i, (x, y) in enumerate(path_pts):
            # Yaw points toward the next waypoint (last inherits previous).
            if i < n - 1:
                nx, ny = path_pts[i + 1]
                yaw = math.atan2(ny - y, nx - x)
            elif n >= 2:
                px, py = path_pts[i - 1]
                yaw = math.atan2(y - py, x - px)
            else:
                yaw = 0.0

            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            ps.pose.orientation.w = math.cos(yaw / 2.0)
            path_msg.poses.append(ps)

        return path_msg

    def send_follow_path(self, path_msg):
        if not self.follow_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "FollowPath action server 'follow_path' not available.")
            return

        goal = FollowPath.Goal()
        goal.path = path_msg
        goal.controller_id = self.controller_id
        goal.goal_checker_id = self.goal_checker_id

        self.get_logger().info("Sending path to FollowPath...")
        future = self.follow_client.send_goal_async(
            goal, feedback_callback=self.on_follow_feedback)
        future.add_done_callback(self.on_follow_response)

    def on_follow_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("FollowPath goal REJECTED.")
            return
        self.get_logger().info("FollowPath goal accepted; robot following.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_follow_result)

    def on_follow_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f"FollowPath: {fb.distance_to_goal:.2f} m to goal, "
            f"speed {fb.speed:.2f} m/s", throttle_duration_sec=2.0)

    def on_follow_result(self, future):
        self.get_logger().info("FollowPath finished.")


def main(args=None):
    rclpy.init(args=args)
    node = QuadHRRTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
