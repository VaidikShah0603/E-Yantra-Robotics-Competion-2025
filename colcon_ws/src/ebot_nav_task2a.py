#!/usr/bin/env python3
'''
# Team ID: 1924
# Theme: Krishi coBot
# Author List: Shah Vaidik Sanjaykumar
# Filename: ebot_nav_task2a.py
# Functions: init, odom_callback, scan_callback, control_loop, is_obstacle_in_sector, angle_diff, normalize_angle, main
# Global variables: None
'''
import rclpy
from rclpy.node import Node
import math
import numpy as np
import time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion
class EbotNavigator(Node):
    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the EbotNavigator node:
        - Sets up publishers/subscribers for cmd_vel, odometry, and laser scan.
        - Defines waypoints and internal states for waypoint navigation.
        Input Arguments:
        ---
        None
        Returns:
        ---
        None
        Example call:
        ---
        node = EbotNavigator()
        '''
        super().__init__('ebot_nav_task2a')
        # Publisher for velocity commands to robot
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # Subscriber to robot odometry for current position and orientation
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        # Subscriber to LIDAR scan data
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        # Publisher for detection status
        self.detection_pub = self.create_publisher(String, '/detection_status', 10)
        # Current robot pose: [x, y, yaw]
        self.pose = None
        # Current LIDAR ranges array
        self.lidar_ranges = []
        # Timer to run control loop periodically at 0.15s interval (~6.6 Hz)
        self.control_timer = self.create_timer(0.15, self.control_loop)
        # Waypoints (x, y, yaw) to navigate sequentially
        self.waypoints = [
            [0.26, -1.95, 1.57], # Waypoint 1
            [-1.48, -0.67, -1.57], # Waypoint 2
            [-1.53, -6.61, -1.57] # Waypoint 3
        ]
        # Index of current waypoint target in waypoints list
        self.current_waypoint_index = 0
        # Navigation phase/state variable
        self.phase = 'rotate_90_clockwise'
        # Initial yaw when starting rotation or movement phase
        self.initial_yaw = None
        # Target yaw for rotation phases
        self.target_yaw = None
        # Position where certain phases start (for distance tracking)
        self.initial_position = None
        # Traveled distance during linear motion phases
        self.distance_moved = 0.0
        # Sub-phase tracker for after waypoint 1 (movement and rotation)
        self.after_wp1_phase = 'not_started'
        # Sub-phase tracker for after rotation phase
        self.after_rotation_phase = 'not_started'
        # Flag indicating all waypoints have been reached
        self.reached_all_waypoints = False
        # Time when waypoint 1 is reached for stopping
        self.wp1_reach_time = None
    def odom_callback(self, msg):
        '''
        Purpose:
        ---
        Callback for processing Odometry messages, updating current robot pose.
        Input Arguments:
        ---
        msg : Odometry
            ROS Odometry message containing robot pose information.
        Returns:
        ---
        None
        Example call:
        ---
        # Called automatically by ROS subscription.
        '''
        # Extract position
        position = msg.pose.pose.position
        # Extract quaternion orientation
        orientation = msg.pose.pose.orientation
        # Convert quaternion to Euler yaw angle
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(quaternion)
        # Normalize yaw angle to [-pi, +pi]
        yaw = self.normalize_angle(yaw)
        # Update internal pose state
        self.pose = np.array([position.x, position.y, yaw])
        # Initialize initial yaw and position if null
        if self.initial_yaw is None:
            self.initial_yaw = yaw
            self.target_yaw = self.normalize_angle(self.initial_yaw - math.pi / 2) # Rotate 90 deg clockwise start
            self.initial_position = np.array([position.x, position.y])
    def scan_callback(self, msg):
        '''
        Purpose:
        ---
        Callback to update the internal LIDAR ranges with new data from /scan topic.
        Input Arguments:
        ---
        msg : LaserScan
            ROS LaserScan message containing range data.
        Returns:
        ---
        None
        Example call:
        ---
        # Called automatically by ROS subscription.
        '''
        # Convert scan ranges to numpy array for easy processing
        self.lidar_ranges = np.array(msg.ranges)
    def control_loop(self):
        '''
        Purpose:
        ---
        Main control loop run periodically.
        Executes navigation behaviour based on current phase and sensor data.
        Input Arguments:
        ---
        None
        Returns:
        ---
        None
        Example call:
        ---
        # Called automatically by ROS timer.
        '''
        # Return early if pose or lidar data not ready or navigation complete
        if (self.pose is None or len(self.lidar_ranges) == 0 or self.reached_all_waypoints):
            return
        # Cmd velocity message to publish
        twist = Twist()
        # Rotation and movement speeds
        rotate_speed = 3.5 # radians/sec
        move_speed = 2.8 # meters/sec
        # Phase: Initial 90 deg clockwise rotation
        if self.phase == 'rotate_90_clockwise':
            yaw_error = self.angle_diff(self.target_yaw, self.pose[2])
            if abs(yaw_error) > 0.05:
                twist.angular.z = rotate_speed * np.sign(yaw_error)
            else:
                twist.angular.z = 0.0
                self.phase = 'move_forward_0_8m'
                self.get_logger().info("Rotated 90 deg clockwise; moving forward 0.8m")
            twist.linear.x = 0.0
            self.cmd_vel_publisher.publish(twist)
            return
        # Phase: Move forward 0.8m after rotation
        elif self.phase == 'move_forward_0_8m':
            current_pos = self.pose[:2] # Current x,y
            self.distance_moved = np.linalg.norm(current_pos - self.initial_position)
            if self.distance_moved < 0.7:
                if not self.is_obstacle_in_sector(-30, 30, 0.5): # ±30deg sector, 0.5m threshold
                    twist.linear.x = move_speed
                else:
                    twist.linear.x = 0.0
                    self.get_logger().warn("Obstacle detected during initial forward movement - stopping")
                twist.angular.z = 0.0
            else:
                twist.linear.x = 0.0
                self.phase = 'navigate_to_wp'
                self.current_waypoint_index = 0
                self.get_logger().info("Moved 0.8m; start navigating to waypoint 1")
            self.cmd_vel_publisher.publish(twist)
            return
        # Phase: Navigate to current waypoint
        elif self.phase == 'navigate_to_wp':
            wp = self.waypoints[self.current_waypoint_index]
            position = self.pose[:2]
            yaw = self.pose[2]
            dist_to_wp = np.linalg.norm(np.array(wp[:2]) - position)
            desired_yaw = wp[2]
            yaw_error_goal = self.angle_diff(desired_yaw, yaw)
            # If near waypoint within 0.2m
            if dist_to_wp < 0.2:
                # Align orientation if needed
                if abs(yaw_error_goal) > 0.05:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.5 * np.sign(yaw_error_goal)
                    self.get_logger().info(f"Aligning orientation at waypoint {self.current_waypoint_index + 1}")
                else:
                    self.get_logger().info(f"Reached waypoint {self.current_waypoint_index + 1} with desired orientation.")
                    # Handle waypoint-dependent next phase
                    if self.current_waypoint_index == 0:
                        self.phase = 'stop_at_wp1'
                    elif self.current_waypoint_index < len(self.waypoints) - 1:
                        self.current_waypoint_index += 1
                        self.phase = 'navigate_to_wp'
                        self.get_logger().info(f"Proceeding to waypoint {self.current_waypoint_index + 1}")
                    else:
                        self.reached_all_waypoints = True
                        self.get_logger().info("Reached all waypoints; task complete.")
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                self.cmd_vel_publisher.publish(twist)
                return
            # Calculate heading to waypoint and error
            heading_to_wp = math.atan2(wp[1] - position[1], wp[0] - position[0])
            yaw_error = self.angle_diff(heading_to_wp, yaw)
            # Check for obstacle ahead in ±30deg sector within 0.5m
            obstacle_ahead = self.is_obstacle_in_sector(-30, 30, 0.5)
            if obstacle_ahead:
                twist.linear.x = 0.0
                twist.angular.z = 0.8 * np.sign(yaw_error)
                self.get_logger().warn(f"Obstacle detected ahead during navigation to waypoint {self.current_waypoint_index + 1}, turning to avoid")
            else:
                # Angular correction and forward speed control
                if abs(yaw_error) > 0.15:
                    twist.linear.x = move_speed * 0.7
                    twist.angular.z = 0.8 * np.sign(yaw_error)
                else:
                    twist.linear.x = move_speed
                    twist.angular.z = 0.0
            self.cmd_vel_publisher.publish(twist)
            return
        # Phase: Stop for 2 seconds at waypoint 1
        elif self.phase == 'stop_at_wp1':
            if self.wp1_reach_time is None:
                self.wp1_reach_time = time.time()
                # Publish DOCK_STATION message with waypoint 1 coordinates
                wp1 = self.waypoints[0]
                detection_msg = String()
                detection_msg.data = f"DOCK_STATION,{wp1[0]:.2f},{wp1[1]:.2f}"
                self.detection_pub.publish(detection_msg)
                self.get_logger().info(f"Published DOCK_STATION at waypoint 1: {detection_msg.data}")
            current_time = time.time()
            if current_time - self.wp1_reach_time >= 2.0:
                self.phase = 'move_forward_3_4m_after_wp1'
                self.after_wp1_phase = 'not_started'
                self.wp1_reach_time = None
                self.get_logger().info("Stopped for 2 seconds at waypoint 1; proceeding")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_publisher.publish(twist)
            return
        # Phase: Move forward 3.4m after waypoint 1
        elif self.phase == 'move_forward_3_4m_after_wp1':
            if self.after_wp1_phase == 'not_started':
                self.initial_position = np.array(self.pose[:2])
                self.after_wp1_phase = 'moving_forward'
            if self.after_wp1_phase == 'moving_forward':
                current_pos = self.pose[:2]
                distance_moved = np.linalg.norm(current_pos - self.initial_position)
                if distance_moved < 3.3:
                    if not self.is_obstacle_in_sector(-30, 30, 0.5):
                        twist.linear.x = move_speed
                        twist.angular.z = 0.0
                    else:
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.get_logger().warn("Obstacle detected during 3.4m forward after waypoint 1; stopping")
                    self.cmd_vel_publisher.publish(twist)
                else:
                    self.initial_yaw = self.pose[2]
                    # Rotate anticlockwise approx 72 degrees (pi/2.5 rad)
                    self.target_yaw = self.normalize_angle(self.initial_yaw + math.pi / 2.5)
                    self.phase = 'rotate_90_anticlockwise'
                    self.get_logger().info("Moved 3.4m forward; starting ~72 deg anticlockwise rotation")
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_vel_publisher.publish(twist)
            return
        # Phase: Rotate ~72 degrees anticlockwise
        elif self.phase == 'rotate_90_anticlockwise':
            yaw_error = self.angle_diff(self.target_yaw, self.pose[2])
            if abs(yaw_error) > 0.05:
                twist.angular.z = rotate_speed * np.sign(yaw_error)
                twist.linear.x = 0.0
            else:
                twist.angular.z = 0.0
                twist.linear.x = 0.0
                self.phase = 'move_forward_0_5m_after_rotation'
                self.after_rotation_phase = 'not_started'
                self.get_logger().info("Rotation done; starting move forward 0.5m")
            self.cmd_vel_publisher.publish(twist)
            return
        # Phase: Move forward 0.5m after rotation
        elif self.phase == 'move_forward_0_5m_after_rotation':
            if self.after_rotation_phase == 'not_started':
                self.initial_position = np.array(self.pose[:2])
                self.after_rotation_phase = 'moving_forward'
            if self.after_rotation_phase == 'moving_forward':
                current_pos = self.pose[:2]
                distance_moved = np.linalg.norm(current_pos - self.initial_position)
                if distance_moved < 0.7:
                    if not self.is_obstacle_in_sector(-30, 30, 0.5):
                        twist.linear.x = move_speed
                        twist.angular.z = 0.0
                    else:
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.get_logger().warn("Obstacle detected during 0.5m forward after rotation; stopping")
                    self.cmd_vel_publisher.publish(twist)
                else:
                    self.phase = 'navigate_to_wp'
                    self.current_waypoint_index = 1
                    self.get_logger().info("Moved 0.5m forward; start navigating to waypoint 2")
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_vel_publisher.publish(twist)
            return
    def is_obstacle_in_sector(self, start_deg, end_deg, threshold):
        '''
        Purpose:
        ---
        Checks if there is any detected obstacle within a specific angular sector in front of the robot.
        Input Arguments:
        ---
        start_deg : [int]
            Start angle in degrees relative to front of robot to check obstacle.
        end_deg : [int]
            End angle in degrees relative to front of robot to check obstacle.
        threshold : [float]
            Distance threshold below which obstacle detected (in meters).
        Returns:
        ---
        bool
            True if obstacle detected in sector, False otherwise.
        '''
        if len(self.lidar_ranges) == 0:
            return False
        total_readings = len(self.lidar_ranges)
        center_index = total_readings // 2
        degrees_per_index = 360.0 / total_readings
        start_index = center_index + int(start_deg / degrees_per_index)
        end_index = center_index + int(end_deg / degrees_per_index)
        start_index = max(0, start_index)
        end_index = min(total_readings, end_index)
        sector_ranges = self.lidar_ranges[start_index:end_index]
        valid_ranges = sector_ranges[np.isfinite(sector_ranges)]
        if valid_ranges.size == 0:
            return False
        return np.min(valid_ranges) < threshold
    def angle_diff(self, angle1, angle2):
        '''
        Purpose:
        ---
        Computes shortest difference between two angles in radians.
        Input Arguments:
        ---
        angle1 : [float]
            First angle in radians.
        angle2 : [float]
            Second angle in radians.
        Returns:
        ---
        float
            Smallest signed difference angle1 - angle2 in [-pi, pi].
        '''
        delta = angle1 - angle2
        return math.atan2(math.sin(delta), math.cos(delta))
    def normalize_angle(self, angle):
        '''
        Purpose:
        ---
        Normalizes angle to be within [-pi, pi].
        Input Arguments:
        ---
        angle : [float]
            Angle in radians to normalize.
        Returns:
        ---
        float
            Normalized angle within [-pi, pi].
        '''
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
def main(args=None):
    rclpy.init(args=args)
    node = EbotNavigator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
