#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.ultra_rl_sub = self.create_subscription(LaserScan, '/ultrasonic_rl/scan',
                                                     self.ultrasonic_rl_callback, 10)
        self.ultra_rr_sub = self.create_subscription(LaserScan, '/ultrasonic_rr/scan',
                                                     self.ultrasonic_rr_callback, 10)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Goal P1
        self.goal_x = 0.26
        self.goal_y = -1.95
        self.goal_yaw = 1.57

        # Tolerances
        self.pos_tolerance = 0.2
        self.yaw_tolerance = math.radians(10.0)

        # Speed control
        self.max_linear_speed = 0.5
        self.max_angular_speed = 0.7
        self.k_linear = 0.6
        self.k_angular = 1.2

        # Obstacle thresholds
        self.safe_front_dist = 0.6
        self.critical_front_dist = 0.35
        self.safe_ultra_dist = 0.35

        # Pose
        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        # States
        self.initial_yaw = None
        self.initial_rotate_done = False
        self.goal_reached = False
        self.orientation_aligned = False

        # Sensors
        self.have_scan = False
        self.min_front = self.min_left = self.min_right = float('inf')
        self.have_ultra_left = self.have_ultra_right = False
        self.min_ultra_left = self.min_ultra_right = float('inf')

        self.get_logger().info('Node Running: First rotate 90° CW then navigate.')

    # ========== Callbacks ==========

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        if self.initial_yaw is None:
            self.initial_yaw = self.current_yaw

    def scan_callback(self, msg):
        r = [msg.range_max if (math.isinf(x) or math.isnan(x) or x <= 0) else x for x in msg.ranges]
        n = len(r)
        if n < 3: return
        third = n // 3
        self.min_right = min(r[:third])
        self.min_front = min(r[third:2*third])
        self.min_left = min(r[2*third:])
        self.have_scan = True

    def ultrasonic_rl_callback(self, msg):
        r = [msg.range_max if (math.isinf(x) or math.isnan(x) or x <= 0) else x for x in msg.ranges]
        if r:
            self.min_ultra_left = min(r)
            self.have_ultra_left = True

    def ultrasonic_rr_callback(self, msg):
        r = [msg.range_max if (math.isinf(x) or math.isnan(x) or x <= 0) else x for x in msg.ranges]
        if r:
            self.min_ultra_right = min(r)
            self.have_ultra_right = True

    # ========== CONTROL LOOP ==========

    def control_loop(self):
        if self.current_yaw is None:
            self.get_logger().info_once("Waiting for /odom ...")
            return

        # ---------- PHASE-0: Rotation Only ----------
        if not self.initial_rotate_done:
            target_yaw = self.normalize_angle(self.initial_yaw - math.pi/2)
            yaw_error = self.normalize_angle(target_yaw - self.current_yaw)

            if abs(yaw_error) > math.radians(5):
                ang = self.k_angular * yaw_error
                ang = max(-self.max_angular_speed, min(self.max_angular_speed, ang))
                self.publish_cmd_vel(0.0, ang)
                self.get_logger().info(f"Initial rotate: yaw_err={math.degrees(yaw_error):.1f}°")
                return
            else:
                self.publish_cmd_vel(0.0, 0.0)
                self.initial_rotate_done = True
                self.get_logger().info("Initial 90° clockwise rotation done ✓")
                return

        # After rotation → original navigation logic

        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y
        distance = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        heading_error = self.normalize_angle(angle_to_goal - self.current_yaw)
        yaw_error = self.normalize_angle(self.goal_yaw - self.current_yaw)

        linear_cmd = angular_cmd = 0.0

        # ----- Move to goal -----
        if not self.goal_reached:
            if distance > self.pos_tolerance:
                if abs(heading_error) > 0.3:
                    angular_cmd = self.k_angular * heading_error
                else:
                    linear_cmd = self.k_linear * distance
                    angular_cmd = self.k_angular * heading_error
            else:
                self.goal_reached = True
                self.get_logger().info("Reached goal area")

        # ----- Align final yaw -----
        if self.goal_reached and not self.orientation_aligned:
            if abs(yaw_error) > self.yaw_tolerance:
                angular_cmd = self.k_angular * yaw_error
            else:
                self.orientation_aligned = True
                self.publish_cmd_vel(0, 0)
                self.get_logger().info(
                    "Goal fully achieved ✔\n"
                    f"final pos err={distance:.3f} m | yaw err={math.degrees(yaw_error):.1f}°"
                )

        # ----- Obstacle Avoidance -----
        if self.have_scan:
            left = max(self.min_left, self.min_ultra_left)
            right = max(self.min_right, self.min_ultra_right)

            if self.min_front < self.critical_front_dist:
                rear_blocked = ((self.have_ultra_left and self.min_ultra_left < self.safe_ultra_dist)
                                or (self.have_ultra_right and self.min_ultra_right < self.safe_ultra_dist))
                if rear_blocked:
                    linear_cmd = 0.0
                else:
                    linear_cmd = -0.05
                angular_cmd = +0.6 if left > right else -0.6

            elif self.min_front < self.safe_front_dist:
                linear_cmd = 0.0
                angular_cmd = +0.5 if left > right else -0.5

        # Limit speeds
        linear_cmd = max(-self.max_linear_speed, min(self.max_linear_speed, linear_cmd))
        angular_cmd = max(-self.max_angular_speed, min(self.max_angular_speed, angular_cmd))

        self.publish_cmd_vel(linear_cmd, angular_cmd)

    # ========== Utils ==========

    def publish_cmd_vel(self, x, z):
        m = Twist()
        m.linear.x = x
        m.angular.z = z
        self.cmd_pub.publish(m)

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    @staticmethod
    def normalize_angle(a):
        while a > math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd_vel(0,0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

