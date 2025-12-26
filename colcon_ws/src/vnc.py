#!/usr/bin/python3

'''
*
*  This script detects bad fruits and ArUco markers from camera feed
*  and publishes their transforms relative to base_link
*  
*  Team ID:		1924
*  Author List:		Shah Vaidik Sanjaykumar
*  Theme:               Krishi coBot
*  Filename:		vnc.py
*  Functions:		__init__, depthimagecb, colorimagecb, bad_fruit_detection,
*                   aruco_detection, process_image, main
*  Nodes:		    Publishing: TF transforms for fruits and markers
*                   Subscribing: /camera/image_raw, /camera/depth/image_raw
*  Global Variables: TEAM_ID , DISABLE_MULTITHREADING , SHOW_IMAGE
'''

# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import tf2_ros
import cv2
import numpy as np
import tf_transformations

# SHOW_IMAGE: Flag to enable/disable OpenCV window visualization
SHOW_IMAGE = True

# DISABLE_MULTITHREADING: Flag to disable concurrent callback execution
DISABLE_MULTITHREADING = False

# TEAM_ID: Unique identifier for transform frame naming
TEAM_ID = 1924

class FruitsTF(Node):
    '''
    Purpose:
    ---
    ROS2 node that detects bad fruits and ArUco markers from RGB-D camera
    and broadcasts their transforms to the TF tree
    
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    None
    
    Example call:
    ---
    node = FruitsTF()
    '''
    
    def __init__(self):
        '''
        Purpose:
        ---
        Initialize the FruitsTF node with subscribers, publishers, and parameters
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        Automatically called when creating FruitsTF() instance
        '''
        super().__init__('fruits_tf')
        
        # bridge: CvBridge object for converting ROS Image messages to OpenCV format
        self.bridge = CvBridge()
        
        # cv_image: Stores the latest RGB image from camera
        self.cv_image = None
        
        # depth_image: Stores the latest depth image from camera
        self.depth_image = None
        
        # Configure callback group for multithreading
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()
        
        # Subscribe to camera RGB and depth topics
        self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)
        
        # TF broadcaster and listener for transform management
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Create timer for periodic image processing (5 Hz)
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)
        
        # Create OpenCV window if visualization enabled
        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)
        
        self.get_logger().info("FruitsTF node started.")
        
        # ArUco detection configuration
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Camera intrinsic parameters (from camera calibration)
        self.sizeCamX = 1280  # Camera resolution width in pixels
        self.sizeCamY = 720   # Camera resolution height in pixels
        self.centerCamX = 642.724365234375  # Principal point X coordinate
        self.centerCamY = 361.9780578613281  # Principal point Y coordinate
        self.focalX = 915.3003540039062  # Focal length in X direction (pixels)
        self.focalY = 914.0320434570312  # Focal length in Y direction (pixels)
        
        # detected_fruits: Dictionary storing transform data for each detected fruit
        self.detected_fruits = {}
        
        # detected_markers: Dictionary storing transform data for each detected ArUco marker
        self.detected_markers = {}
        
        # Track what has been printed to console (avoid duplicate prints)
        self.printed_fruits = set()
        self.printed_markers = set()
    
    def depthimagecb(self, data):
        '''
        Purpose:
        ---
        Callback function for depth image subscription
        
        Input Arguments:
        ---
        `data` : [sensor_msgs.msg.Image]
            Depth image message from camera
        
        Returns:
        ---
        None
        
        Example call:
        ---
        Called automatically by ROS2 when depth image is published
        '''
        try:
            # Convert ROS Image message to OpenCV format (passthrough for depth)
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error("Depth image conversion failed: " + str(e))
    
    def colorimagecb(self, data):
        '''
        Purpose:
        ---
        Callback function for RGB image subscription
        
        Input Arguments:
        ---
        `data` : [sensor_msgs.msg.Image]
            RGB image message from camera
        
        Returns:
        ---
        None
        
        Example call:
        ---
        Called automatically by ROS2 when RGB image is published
        '''
        try:
            # Convert ROS Image message to OpenCV BGR8 format
            self.cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            self.get_logger().error("Color image conversion failed: " + str(e))
    
    def bad_fruit_detection(self, rgb_image):
        '''
        Purpose:
        ---
        Detect bad fruits in RGB image using HSV color thresholding
        
        Input Arguments:
        ---
        `rgb_image` : [numpy.ndarray]
            BGR image from camera
        
        Returns:
        ---
        `bad_fruits` : [list of dict]
            List of detected fruits with contour, bounding box, center, and ID
        
        Example call:
        ---
        fruits = self.bad_fruit_detection(image)
        '''
        bad_fruits = []
        
        # Convert BGR image to HSV color space for color-based detection
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # Define HSV range for bad fruit color (tuned for specific fruit appearance)
        lower = np.array([13, 23, 160])  # Lower bound [H, S, V]
        upper = np.array([18, 30, 170])  # Upper bound [H, S, V]
        
        # Create binary mask where fruit color pixels are white
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # fruit_id: Counter for assigning unique IDs to detected fruits
        fruit_id = 1
        
        # Process each detected contour
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter out small contours (noise)
            if area > 200:
                # Get bounding rectangle coordinates
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Calculate center of bounding box
                cX = int(x + w / 2)
                cY = int(y + h / 2)
                
                # Store fruit data
                bad_fruits.append({
                    'contour': cnt,
                    'bbox': (x, y, w, h),
                    'center': (cX, cY),
                    'id': fruit_id
                })
                fruit_id += 1
        
        return bad_fruits
    
    def aruco_detection(self, frame):
        '''
        Purpose:
        ---
        Detect ArUco markers in the image and annotate them
        
        Input Arguments:
        ---
        `frame` : [numpy.ndarray]
            BGR image for marker detection
        
        Returns:
        ---
        `frame` : [numpy.ndarray]
            Annotated image with detected markers drawn
        `aruco_data` : [tuple]
            Tuple containing (corners, ids) of detected markers
        
        Example call:
        ---
        annotated_frame, (corners, ids) = self.aruco_detection(image)
        '''
        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers using predefined dictionary
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        # Annotate detected markers on the frame
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners)
            
            # Draw center point and ID for each marker
            for corner, marker_id in zip(corners, ids.flatten()):
                pts = corner[0]
                center_x = int(np.mean(pts[:, 0]))
                center_y = int(np.mean(pts[:, 1]))
                
                # Draw center circle
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 255), -1)
                
                # Draw marker ID text
                cv2.putText(frame, f"id={marker_id}", (center_x + 5, center_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return frame, (corners, ids)
    
    def process_image(self):
        '''
        Purpose:
        ---
        Main processing function that detects fruits/markers and publishes transforms
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        Called automatically by timer at 5Hz
        '''
        # Wait until both color and depth images are available
        if self.cv_image is None or self.depth_image is None:
            return
        
        # Detect bad fruits in the current image
        bad_fruits = self.bad_fruit_detection(self.cv_image)
        
        # Detect ArUco markers and get annotated frame
        frame, aruco_data = self.aruco_detection(self.cv_image.copy())
        corners, ids = aruco_data
        
        # Get current timestamp for transform messages
        now_msg = self.get_clock().now().to_msg()
        
        # --- Process each detected bad fruit ---
        for fruit in bad_fruits:
            fruit_id = fruit['id']
            cX, cY = fruit["center"]
            
            # Get depth value at fruit center
            depth_val = self.depth_image[cY, cX]
            
            # Skip if depth is invalid
            if depth_val == 0 or np.isnan(depth_val):
                continue
            
            distance = float(depth_val)
            
            # Convert 2D pixel coordinates to 3D camera coordinates using pinhole camera model
            x3d = distance * (cX - self.centerCamX) / self.focalX
            y3d = distance * (cY - self.centerCamY) / self.focalY
            z3d = distance
            
            try:
                # Look up transform from base_link to camera_link
                trans_cam = self.tf_buffer.lookup_transform(
                    "base_link", "camera_link", now_msg,
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                
                cam_trans = trans_cam.transform.translation
                cam_rot = trans_cam.transform.rotation
                
                # Create transformation matrix from quaternion
                cam_matrix = tf_transformations.quaternion_matrix(
                    (cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w)
                )
                cam_matrix[0:3, 3] = [cam_trans.x, cam_trans.y, cam_trans.z]
                
                # Convert camera frame coordinates to ROS frame coordinates
                # Camera: Z-forward, X-right, Y-down → ROS: X-forward, Y-left, Z-up
                x_ros, y_ros, z_ros = z3d, -x3d, -y3d
                
                # Transform fruit position from camera frame to base_link frame
                fruit_cam_pose = np.array([x_ros, y_ros, z_ros, 1.0])
                fruit_base_pose = np.dot(cam_matrix, fruit_cam_pose)
                
                # Store transform data for continuous broadcasting
                self.detected_fruits[fruit_id] = {
                    'position': [fruit_base_pose[0], fruit_base_pose[1], fruit_base_pose[2]],
                    'rotation': cam_rot
                }
                
                # Print fruit position only once (not every frame)
                if fruit_id not in self.printed_fruits:
                    print(f"bad_fruit_{fruit_id}: ({fruit_base_pose[0]:.3f}, {fruit_base_pose[1]:.3f}, {fruit_base_pose[2]:.3f})")
                    self.printed_fruits.add(fruit_id)
            
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for fruit {fruit_id}: {e}")
                continue
        
        # --- Process each detected ArUco marker ---
        if ids is not None:
            for corner, marker_id in zip(corners, ids.flatten()):
                pts = corner[0]
                center_x = int(np.mean(pts[:, 0]))
                center_y = int(np.mean(pts[:, 1]))
                
                # Get depth value at marker center
                depth_val = self.depth_image[center_y, center_x]
                
                # Skip if depth is invalid
                if depth_val == 0 or np.isnan(depth_val):
                    continue
                
                distance = float(depth_val)
                
                # Convert 2D pixel coordinates to 3D camera coordinates
                x3d = distance * (center_x - self.centerCamX) / self.focalX
                y3d = distance * (center_y - self.centerCamY) / self.focalY
                z3d = distance
                
                try:
                    # Look up transform from base_link to camera_link
                    trans_cam = self.tf_buffer.lookup_transform(
                        "base_link", "camera_link", now_msg,
                        timeout=rclpy.duration.Duration(seconds=0.5)
                    )
                    
                    cam_trans = trans_cam.transform.translation
                    cam_rot = trans_cam.transform.rotation
                    
                    # Create transformation matrix
                    cam_matrix = tf_transformations.quaternion_matrix(
                        (cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w)
                    )
                    cam_matrix[0:3, 3] = [cam_trans.x, cam_trans.y, cam_trans.z]
                    
                    # Convert camera frame coordinates to ROS frame coordinates
                    x_ros, y_ros, z_ros = z3d, -x3d, -y3d
                    
                    # Transform marker position from camera frame to base_link frame
                    aruco_cam_pose = np.array([x_ros, y_ros, z_ros, 1.0])
                    aruco_base_pose = np.dot(cam_matrix, aruco_cam_pose)
                    
                    # Store transform data for continuous broadcasting
                    self.detected_markers[marker_id] = {
                        'position': [aruco_base_pose[0], aruco_base_pose[1], aruco_base_pose[2]],
                        'rotation': cam_rot
                    }
                    
                    # Print marker position only once
                    if marker_id not in self.printed_markers:
                        print(f"Aruco_marker_{marker_id}: ({aruco_base_pose[0]:.3f}, {aruco_base_pose[1]:.3f}, {aruco_base_pose[2]:.3f})")
                        self.printed_markers.add(marker_id)
                
                except Exception as e:
                    self.get_logger().warn(f"TF transform failed for ArUco marker {marker_id}: {e}")
                    continue
        
        # Continuously broadcast all detected transforms (fruits and markers)
        now_msg = self.get_clock().now().to_msg()
        
        # Broadcast fruit transforms
        for fruit_id, data in self.detected_fruits.items():
            t_base = TransformStamped()
            t_base.header.stamp = now_msg
            t_base.header.frame_id = "base_link"
            t_base.child_frame_id = f"{TEAM_ID}_bad_fruit_{fruit_id}"
            t_base.transform.translation.x = data['position'][0]
            t_base.transform.translation.y = data['position'][1]
            t_base.transform.translation.z = data['position'][2]
            t_base.transform.rotation = data['rotation']
            self.tf_broadcaster.sendTransform(t_base)
        
        # Broadcast ArUco marker transforms
        for marker_id, data in self.detected_markers.items():
            t_aruco = TransformStamped()
            t_aruco.header.stamp = now_msg
            t_aruco.header.frame_id = "base_link"
            t_aruco.child_frame_id = f"aruco_marker_{marker_id}"
            t_aruco.transform.translation.x = data['position'][0]
            t_aruco.transform.translation.y = data['position'][1]
            t_aruco.transform.translation.z = data['position'][2]
            t_aruco.transform.rotation = data['rotation']
            self.tf_broadcaster.sendTransform(t_aruco)
        
        # Display annotated frame if visualization enabled
        if SHOW_IMAGE:
            cv2.imshow('fruits_tf_view', frame)
            cv2.waitKey(1)

def main(args=None):
    '''
    Purpose:
    ---
    Main function to initialize and run the FruitsTF node
    
    Input Arguments:
    ---
    `args` : [list]
        Command line arguments (optional)
    
    Returns:
    ---
    None
    
    Example call:
    ---
    main()
    '''
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create node instance
    node = FruitsTF()
    
    try:
        # Spin node to execute callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        node.get_logger().info('Shutting down FruitsTF')
        node.destroy_node()
        rclpy.shutdown()
        
        # Close OpenCV windows if visualization was enabled
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

