#!/usr/bin/env python3

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import numpy as np
import math
from sklearn.linear_model import RANSACRegressor


class ShapeDetector(Node):
    def __init__(self):
        super().__init__('shape_detector')
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Publishers
        self.shape_detected_pub = self.create_publisher(
            String, '/shape_detected_signal', 10)
        
        # Current robot position for transforming shape centroid
        self.current_x = 0.0  
        self.current_y = 0.0  
        
        # Detection parameters
        self.min_points_per_line = 5
        self.ransac_threshold = 0.195
        self.angle_tolerance = 0.12
        self.detection_cooldown = 2.0
        
        # Track buffers and last detection times per shape type
        self.detection_buffer = {
            "FERTILIZER_REQUIRED": [],
            "BAD_HEALTH": [],
        }
        self.buffer_length = 3
        
        now = self.get_clock().now()
        self.last_detection_time = {
            "FERTILIZER_REQUIRED": now,
            "BAD_HEALTH": now,
        }
    
    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
    
    def scan_callback(self, msg):
        points = self.laser_to_cartesian(msg)
        if len(points) < 10:
            return
        
        lines = self.detect_lines_ransac(points)
        lines = self.filter_short_lines(lines, min_length=0.1)
        if len(lines) < 3:
            return
        
        shape_type = self.classify_shape(lines)
        if shape_type is not None:
            current_time = self.get_clock().now()
            buffer = self.detection_buffer[shape_type]
            last_time = self.last_detection_time[shape_type]
            time_diff = (current_time - last_time).nanoseconds / 1e9 if last_time is not None else None
            
            buffer.append(shape_type)
            if len(buffer) > self.buffer_length:
                buffer.pop(0)
            
            if (buffer.count(shape_type) > self.buffer_length // 2 and
                (time_diff is None or time_diff > self.detection_cooldown)):
                    
                centroid = self.compute_polygon_centroid(lines)
                
                # Publish signal
                signal_msg = String()
                signal_msg.data = f"{shape_type},{centroid[0]:.2f},{centroid[1]:.2f}"
                self.shape_detected_pub.publish(signal_msg)
                
                self.last_detection_time[shape_type] = current_time
                self.detection_buffer[shape_type] = []

                self.get_logger().info(f"Shape detected signal sent: {signal_msg.data}")
    
    def laser_to_cartesian(self, msg):
        points = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append([x, y])
            angle += msg.angle_increment
        return np.array(points)
    
    def detect_lines_ransac(self, points):
        lines = []
        remaining_points = points.copy()
        while len(remaining_points) > self.min_points_per_line:
            if len(remaining_points) < self.min_points_per_line:
                break
            
            X = remaining_points[:, 0].reshape(-1, 1)
            y = remaining_points[:, 1]
            try:
                ransac = RANSACRegressor(
                    residual_threshold=self.ransac_threshold,
                    min_samples=self.min_points_per_line
                )
                ransac.fit(X, y)
                inlier_mask = ransac.inlier_mask_
                inliers = remaining_points[inlier_mask]
                if len(inliers) >= self.min_points_per_line:
                    lines.append(inliers)
                    remaining_points = remaining_points[~inlier_mask]
                else:
                    break
            except Exception:
                break
        return lines
    
    def filter_short_lines(self, lines, min_length=0.1):
        filtered = []
        for line in lines:
            length = np.linalg.norm(line[-1] - line[0])
            if length >= min_length:
                filtered.append(line)
        return filtered
    
    def classify_shape(self, lines):
        merged_lines = self.merge_collinear_lines(lines)
        num_edges = len(merged_lines)
        if not self.is_closed_polygon(merged_lines):
            return None
        if num_edges == 3:
            return "FERTILIZER_REQUIRED"
        elif num_edges == 4:
            return "BAD_HEALTH"
        
        return None
    
    def merge_collinear_lines(self, lines):
        if len(lines) < 2:
            return lines
        merged = []
        used = set()
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            current_line = line1.copy()
            for j, line2 in enumerate(lines):
                if i != j and j not in used:
                    if self.are_collinear(line1, line2):
                        current_line = np.vstack([current_line, line2])
                        used.add(j)
            merged.append(current_line)
            used.add(i)
        return merged
    
    def are_collinear(self, line1, line2):
        dir1 = self.compute_line_direction(line1)
        dir2 = self.compute_line_direction(line2)
        angle_diff = abs(math.atan2(dir1[1], dir1[0]) - math.atan2(dir2[1], dir2[0]))
        return angle_diff < self.angle_tolerance
    
    def compute_line_direction(self, line):
        start = line[0]
        end = line[-1]
        direction = end - start
        norm = np.linalg.norm(direction)
        return direction / norm if norm > 0 else direction
    
    def is_closed_polygon(self, lines):
        if len(lines) < 3:
            return False
        endpoints = []
        for line in lines:
            endpoints.append(line[0])
            endpoints.append(line[-1])
        closure_threshold = 0.3
        first_point = endpoints[0]
        last_point = endpoints[-1]
        distance = np.linalg.norm(first_point - last_point)
        return distance < closure_threshold
    
    def compute_polygon_centroid(self, lines):
        points = np.vstack(lines)
        centroid_laser = np.mean(points, axis=0)
        centroid_global_x = self.current_x + centroid_laser[0]
        centroid_global_y = self.current_y + centroid_laser[1]
        return (centroid_global_x, centroid_global_y)

def main(args=None):
    rclpy.init(args=args)
    detector = ShapeDetector()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
