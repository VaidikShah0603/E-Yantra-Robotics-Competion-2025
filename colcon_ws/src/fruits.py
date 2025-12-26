#!/usr/bin/env python3

'''

*
*  This script autonomously picks bad fruits and throws them in dustbin
*  using delta twist commands and TF transforms
*  
*  Team ID:		1924
*  Author List:		Shah Vaidik Sanjaykumar
*  Filename:		fruits.py
*  Theme:               Krishi coBot
*  Functions:		__init__, store_fruit_coordinates, get_current_pose,
*                   compute_pose_error, compute_distance, find_nearest_unpicked_fruit,
*                   check_if_stuck, publish_velocity, control_loop, main
*  Nodes:		    Publishing: /delta_twist_cmds
*                   Subscribing: TF transforms
*  Global Variable: TEAM_ID

'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math
from scipy.spatial.transform import Rotation as R

# TEAM_ID: Unique identifier for TF frame naming
TEAM_ID = 1924

class FruitPickAndThrowNode(Node):
    '''
    Purpose:
    ---
    ROS2 node that autonomously picks bad fruits and throws them in dustbin
    using anticlockwise scanning, nearest neighbor selection, and stuck recovery
    
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    None
    
    Example call:
    ---
    node = FruitPickAndThrowNode()
    '''
    
    def __init__(self):
        '''
        Purpose:
        ---
        Initialize the FruitPickAndThrowNode with publishers, TF listeners,
        and state machine parameters
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        Automatically called when creating FruitPickAndThrowNode() instance
        '''
        super().__init__('fruit_pick_throw_node')
        
        # Publisher for delta twist commands to control robot end-effector
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        # TF buffer and listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # bad_fruits: List storing all detected bad fruit information
        self.bad_fruits = []
        
        # bad_fruit_frames: TF frame names to look up for bad fruits
        self.bad_fruit_frames = [
            f'{TEAM_ID}_bad_fruit_1',
            f'{TEAM_ID}_bad_fruit_2',
            f'{TEAM_ID}_bad_fruit_3'
        ]
        
        # waypoint_position: Target position to reach during anticlockwise rotation [x, y, z] in meters
        self.waypoint_position = [-0.159, 0.501, 0.415]
        
        # waypoint_approach_distance: Stop rotation when within this distance from waypoint (meters)
        self.waypoint_approach_distance = 0.6
        
        # pick_orientation: Desired end-effector orientation for picking fruits [x, y, z, w] quaternion
        self.pick_orientation = [0.029, 0.997, 0.045, 0.033]
        
        # dustbin_position: Target position for dropping fruits [x, y, z] in meters
        self.dustbin_position = [-0.806, 0.010, 0.182]
        
        # dustbin_orientation: Desired orientation when dropping fruits [x, y, z, w] quaternion
        self.dustbin_orientation = [-0.684, 0.726, 0.05, 0.008]
        
        # Control gains for position and orientation error correction
        self.linear_gain = 1.5  # Proportional gain for linear velocity
        self.angular_gain = 3.0  # Proportional gain for angular velocity
        
        # Position and orientation tolerances for target reaching (meters and radians)
        self.position_tolerance = 0.08
        self.orientation_tolerance = 0.1
        
        # Dustbin-specific tolerances (relaxed for easier completion)
        self.dustbin_position_tolerance = 0.1
        self.dustbin_orientation_tolerance = 0.10
        
        # State machine variables
        self.state = "STORE_COORDINATES"  # Current state of the robot
        self.target_fruit = None  # Currently targeted fruit dictionary
        
        # stored_fruit_positions: Dictionary mapping fruit_id to [x, y, z] position
        self.stored_fruit_positions = {}
        
        # original_fruit_z: Dictionary storing original Z coordinates for stuck recovery
        self.original_fruit_z = {}
        
        # Stuck detection parameters
        self.last_position = None  # Previous robot position for movement tracking
        self.stuck_start_time = None  # Time when robot stopped moving
        self.stuck_threshold = 10.0  # Seconds before considering robot stuck
        self.movement_threshold = 0.02  # Minimum movement (meters) to not be stuck
        self.stuck_maneuver_step = 0  # Current step in unstuck maneuver (0-3)
        
        # lift_target_z: Target Z coordinate for lifting operations (meters)
        self.lift_target_z = None
        
        # unstuck_target_pos: Target position during unstuck maneuver
        self.unstuck_target_pos = None
        
        # unstuck_target_ori: Target orientation during unstuck maneuver
        self.unstuck_target_ori = None
        
        # Anticlockwise rotation parameters
        self.circular_angle = 0.0  # Current angle in circular trajectory (radians)
        self.circular_center = [0.0, 0.0, 0.7]  # Center of circular path [x, y, z]
        self.circular_radius = 0.6  # Radius of circular path (meters)
        self.circular_velocity = 0.6  # Angular velocity of rotation (rad/s)
        self.rotation_start_time = None  # Time when rotation started
        
        # Create timer for control loop execution at 10 Hz
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Fruit Pick and Throw node started")
    
    def store_fruit_coordinates(self):
        '''
        Purpose:
        ---
        Read TF transforms of all bad fruits and store their coordinates
        relative to base_link frame
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        `stored_fruits` : [list of dict]
            List containing fruit data with id, position, picked status, frame_id
        
        Example call:
        ---
        fruits = self.store_fruit_coordinates()
        '''
        stored_fruits = []
        
        # Iterate through all expected fruit frames
        for frame_id in self.bad_fruit_frames:
            try:
                # Look up transform from base_link to fruit frame
                transform = self.tf_buffer.lookup_transform(
                    'base_link', frame_id, rclpy.time.Time(), 
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                
                # Extract position from transform
                trans = transform.transform.translation
                position = [trans.x, trans.y, trans.z]
                
                # Extract fruit ID from frame name
                fruit_id = int(frame_id.split('_')[-1])
                
                # Create fruit data dictionary
                fruit_data = {
                    'id': fruit_id,
                    'position': position,
                    'picked': False,  # Initially not picked
                    'frame_id': frame_id
                }
                
                stored_fruits.append(fruit_data)
                
                # Store position in lookup dictionaries
                self.stored_fruit_positions[fruit_id] = position.copy()
                self.original_fruit_z[fruit_id] = position[2]  # Store original Z for stuck recovery
                
                self.get_logger().info(f"Stored bad_fruit_{fruit_id}: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
                
            except TransformException:
                # Frame not available yet, skip
                pass
        
        return stored_fruits
    
    def get_current_pose(self):
        '''
        Purpose:
        ---
        Get current position and orientation of robot end-effector (wrist_3_link)
        relative to base_link
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        `position` : [list of float]
            Current position [x, y, z] in meters, or None if transform unavailable
        `orientation` : [list of float]
            Current orientation [x, y, z, w] quaternion, or None if transform unavailable
        
        Example call:
        ---
        pos, ori = self.get_current_pose()
        '''
        try:
            # Look up transform from base_link to wrist_3_link (end-effector)
            transform = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time())
            
            # Extract position and orientation
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            
            return [translation.x, translation.y, translation.z], [rotation.x, rotation.y, rotation.z, rotation.w]
        
        except TransformException:
            # Transform not available
            return None, None
    
    def compute_pose_error(self, current_pos, current_ori, target_pos, target_ori):
        '''
        Purpose:
        ---
        Compute position and orientation error between current and target pose
        
        Input Arguments:
        ---
        `current_pos` : [list of float]
            Current position [x, y, z] in meters
        `current_ori` : [list of float]
            Current orientation [x, y, z, w] quaternion
        `target_pos` : [list of float]
            Target position [x, y, z] in meters
        `target_ori` : [list of float]
            Target orientation [x, y, z, w] quaternion
        
        Returns:
        ---
        `pos_err` : [list of float]
            Position error [dx, dy, dz] in meters
        `rot_vec` : [list of float]
            Orientation error as rotation vector [rx, ry, rz] in radians
        
        Example call:
        ---
        pos_error, ori_error = self.compute_pose_error(cur_pos, cur_ori, tgt_pos, tgt_ori)
        '''
        # Compute position error (target - current)
        pos_err = [target_pos[i] - current_pos[i] for i in range(3)]
        
        try:
            # Convert quaternions to rotation objects
            r_current = R.from_quat(current_ori)
            r_target = R.from_quat(target_ori)
            
            # Compute relative rotation
            rel_rot = r_target * r_current.inv()
            
            # Convert to rotation vector (axis-angle representation)
            rot_vec = rel_rot.as_rotvec()
        except:
            # If rotation computation fails, return zero error
            rot_vec = [0, 0, 0]
        
        return pos_err, rot_vec
    
    def compute_distance(self, pos1, pos2):
        '''
        Purpose:
        ---
        Compute Euclidean distance between two 3D points
        
        Input Arguments:
        ---
        `pos1` : [list of float]
            First position [x, y, z] in meters
        `pos2` : [list of float]
            Second position [x, y, z] in meters
        
        Returns:
        ---
        `distance` : [float]
            Euclidean distance in meters
        
        Example call:
        ---
        dist = self.compute_distance([0, 0, 0], [1, 1, 1])
        '''
        return math.sqrt(sum((pos1[i] - pos2[i])**2 for i in range(3)))
    
    def find_nearest_unpicked_fruit(self, current_pos):
        '''
        Purpose:
        ---
        Find the nearest unpicked fruit from current robot position
        
        Input Arguments:
        ---
        `current_pos` : [list of float]
            Current robot position [x, y, z] in meters
        
        Returns:
        ---
        `nearest_fruit` : [dict or None]
            Fruit dictionary with id, position, picked status, or None if no fruits remain
        
        Example call:
        ---
        fruit = self.find_nearest_unpicked_fruit(current_position)
        '''
        # Filter for unpicked fruits only
        unpicked_fruits = [f for f in self.bad_fruits if not f['picked']]
        
        if not unpicked_fruits:
            return None
        
        min_distance = float('inf')
        nearest_fruit = None
        
        # Find fruit with minimum distance
        for fruit in unpicked_fruits:
            fruit_pos = self.stored_fruit_positions[fruit['id']]
            distance = self.compute_distance(current_pos, fruit_pos)
            
            if distance < min_distance:
                min_distance = distance
                nearest_fruit = fruit
        
        return nearest_fruit
    
    def check_if_stuck(self, current_pos):
        '''
        Purpose:
        ---
        Check if robot is stuck (not moving for specified duration)
        
        Input Arguments:
        ---
        `current_pos` : [list of float]
            Current robot position [x, y, z] in meters
        
        Returns:
        ---
        `is_stuck` : [bool]
            True if robot hasn't moved for stuck_threshold seconds, False otherwise
        
        Example call:
        ---
        if self.check_if_stuck(current_position):
            # Execute stuck recovery
        '''
        # Initialize tracking on first call
        if self.last_position is None:
            self.last_position = current_pos
            self.stuck_start_time = self.get_clock().now()
            return False
        
        # Compute movement since last check
        movement = self.compute_distance(current_pos, self.last_position)
        
        if movement < self.movement_threshold:
            # Robot not moving significantly
            elapsed = (self.get_clock().now() - self.stuck_start_time).nanoseconds / 1e9
            
            if elapsed > self.stuck_threshold:
                # Stuck for too long
                return True
        else:
            # Robot is moving, reset timer
            self.stuck_start_time = self.get_clock().now()
            self.last_position = current_pos
        
        return False
    
    def publish_velocity(self, pos_err, ori_err):
        '''
        Purpose:
        ---
        Publish velocity command based on position and orientation errors
        with saturation limits
        
        Input Arguments:
        ---
        `pos_err` : [list of float]
            Position error [dx, dy, dz] in meters
        `ori_err` : [list of float]
            Orientation error [rx, ry, rz] in radians
        
        Returns:
        ---
        None
        
        Example call:
        ---
        self.publish_velocity(position_error, orientation_error)
        '''
        # Create Twist message
        twist = Twist()
        
        # Apply proportional control with gains
        twist.linear.x = self.linear_gain * pos_err[0]
        twist.linear.y = self.linear_gain * pos_err[1]
        twist.linear.z = self.linear_gain * pos_err[2]
        twist.angular.x = self.angular_gain * ori_err[0]
        twist.angular.y = self.angular_gain * ori_err[1]
        twist.angular.z = self.angular_gain * ori_err[2]
        
        # Velocity saturation limits
        max_lin = 0.4  # Maximum linear velocity (m/s)
        max_ang = 0.7  # Maximum angular velocity (rad/s)
        
        # Saturate linear velocity
        lin_norm = math.sqrt(twist.linear.x**2 + twist.linear.y**2 + twist.linear.z**2)
        if lin_norm > max_lin:
            scale = max_lin / lin_norm
            twist.linear.x *= scale
            twist.linear.y *= scale
            twist.linear.z *= scale
        
        # Saturate angular velocity
        ang_norm = math.sqrt(twist.angular.x**2 + twist.angular.y**2 + twist.angular.z**2)
        if ang_norm > max_ang:
            scale = max_ang / ang_norm
            twist.angular.x *= scale
            twist.angular.y *= scale
            twist.angular.z *= scale
        
        # Publish velocity command
        self.pub.publish(twist)
    
    def control_loop(self):
        '''
        Purpose:
        ---
        Main control loop implementing state machine for fruit picking task
        Executes at 10 Hz
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        Called automatically by ROS2 timer
        '''
        # Get current robot pose
        current_pos, current_ori = self.get_current_pose()
        
        if current_pos is None:
            # Transform not available yet
            return
        
        # --- STATE: STORE_COORDINATES ---
        # Initial state: detect and store all fruit positions
        if self.state == "STORE_COORDINATES":
            self.bad_fruits = self.store_fruit_coordinates()
            
            if len(self.bad_fruits) > 0:
                self.get_logger().info(f"Stored {len(self.bad_fruits)} fruit positions!")
                self.state = "ANTICLOCKWISE_ROTATION"
                self.rotation_start_time = self.get_clock().now()
            else:
                self.get_logger().warn("Waiting for TF frames...")
        
        # --- STATE: ANTICLOCKWISE_ROTATION ---
        # Perform circular scanning motion until near waypoint
        elif self.state == "ANTICLOCKWISE_ROTATION":
            # Check distance to waypoint
            distance_to_waypoint = self.compute_distance(current_pos, self.waypoint_position)
            
            if distance_to_waypoint < self.waypoint_approach_distance:
                # Close enough to waypoint, start fruit picking
                nearest_fruit = self.find_nearest_unpicked_fruit(current_pos)
                
                if nearest_fruit:
                    self.get_logger().info(f"Approaching fruit: bad_fruit_{nearest_fruit['id']}")
                    self.pub.publish(Twist())  # Stop motion
                    self.target_fruit = nearest_fruit
                    self.state = "MOVE_TO_FRUIT"
                    self.last_position = None
                    self.stuck_maneuver_step = 0
                else:
                    self.state = "COMPLETED"
                return
            
            # Continue anticlockwise circular motion
            elapsed = (self.get_clock().now() - self.rotation_start_time).nanoseconds / 1e9
            self.circular_angle = self.circular_velocity * elapsed
            
            # Calculate target position on circular path
            target_x = self.circular_center[0] + self.circular_radius * math.cos(self.circular_angle)
            target_y = self.circular_center[1] + self.circular_radius * math.sin(self.circular_angle)
            target_z = self.circular_center[2]
            target_pos = [target_x, target_y, target_z]
            
            # Calculate tangent orientation (perpendicular to radius)
            tangent_angle = self.circular_angle + math.pi/2
            target_quat = R.from_euler('z', tangent_angle).as_quat().tolist()
            
            # Compute errors and publish velocity
            pos_err, ori_err = self.compute_pose_error(current_pos, current_ori, target_pos, target_quat)
            self.publish_velocity(pos_err, ori_err)
        
        # --- STATE: MOVE_TO_FRUIT ---
        # Approach target fruit with stuck detection and recovery
        elif self.state == "MOVE_TO_FRUIT":
            if self.target_fruit is None:
                # Fallback: find nearest fruit if target lost
                nearest_fruit = self.find_nearest_unpicked_fruit(current_pos)
                if nearest_fruit:
                    self.target_fruit = nearest_fruit
                else:
                    self.state = "COMPLETED"
                return
            
            # Check if stuck and initiate unstuck maneuver
            if self.check_if_stuck(current_pos) and self.stuck_maneuver_step == 0:
                self.get_logger().warn("Robot stuck! Starting unstuck maneuver...")
                self.stuck_maneuver_step = 1
                self.last_position = None
            
            # --- Execute unstuck maneuver if triggered ---
            if self.stuck_maneuver_step == 1:
                # Step 1: Move Y coordinate -0.3m (sideways)
                target_pos = [current_pos[0], current_pos[1] - 0.3, current_pos[2]]
                pos_err, _ = self.compute_pose_error(current_pos, current_ori, target_pos, current_ori)
                pos_mag = math.sqrt(sum(e**2 for e in pos_err))
                
                if pos_mag < 0.05:
                    self.get_logger().info("Moved Y-0.3m, now rotating 40° clockwise...")
                    self.stuck_maneuver_step = 2
                    
                    # Calculate target orientation (40° clockwise = -40° rotation around Z)
                    current_r = R.from_quat(current_ori)
                    rotation_delta = R.from_euler('z', -math.radians(40))
                    target_r = rotation_delta * current_r
                    self.unstuck_target_ori = target_r.as_quat().tolist()
                else:
                    self.publish_velocity(pos_err, [0, 0, 0])
            
            elif self.stuck_maneuver_step == 2:
                # Step 2: Rotate 40° clockwise
                _, ori_err = self.compute_pose_error(current_pos, current_ori, current_pos, self.unstuck_target_ori)
                ori_mag = math.sqrt(sum(v**2 for v in ori_err))
                
                if ori_mag < 0.1:
                    self.get_logger().info("Rotated 40°, now moving Y+0.6m...")
                    self.stuck_maneuver_step = 3
                else:
                    self.publish_velocity([0, 0, 0], ori_err)
            
            elif self.stuck_maneuver_step == 3:
                # Step 3: Move Y coordinate +0.6m (return and overshoot)
                target_pos = [current_pos[0], current_pos[1] + 0.6, current_pos[2]]
                pos_err, _ = self.compute_pose_error(current_pos, current_ori, target_pos, current_ori)
                pos_mag = math.sqrt(sum(e**2 for e in pos_err))
                
                if pos_mag < 0.05:
                    self.get_logger().info("Unstuck maneuver complete! Resuming fruit approach...")
                    self.stuck_maneuver_step = 0
                    self.last_position = None
                else:
                    self.publish_velocity(pos_err, [0, 0, 0])
            
            else:
                # Normal fruit approach (no stuck maneuver active)
                target_pos = self.stored_fruit_positions[self.target_fruit['id']]
                
                # Compute errors to target fruit position and pick orientation
                pos_err, ori_err = self.compute_pose_error(current_pos, current_ori, target_pos, self.pick_orientation)
                pos_mag = math.sqrt(sum(e**2 for e in pos_err))
                ori_mag = math.sqrt(sum(v**2 for v in ori_err))
                
                # Check if reached fruit
                if pos_mag < self.position_tolerance and ori_mag < self.orientation_tolerance:
                    self.get_logger().info(f'Reached bad_fruit_{self.target_fruit["id"]}')
                    self.pub.publish(Twist())  # Stop motion
                    
                    # Wait for user to attach fruit
                    input(f'Press Enter AFTER attaching bad_fruit_{self.target_fruit["id"]}...')
                    
                    # Transition to lift state
                    self.lift_target_z = current_pos[2] + 0.25
                    self.state = "LIFT_AFTER_ATTACH"
                else:
                    # Continue approaching fruit
                    self.publish_velocity(pos_err, ori_err)
        
        # --- STATE: LIFT_AFTER_ATTACH ---
        # Lift fruit 0.25m upward after attachment
        elif self.state == "LIFT_AFTER_ATTACH":
            target_pos = [current_pos[0], current_pos[1], self.lift_target_z]
            pos_err, _ = self.compute_pose_error(current_pos, current_ori, target_pos, current_ori)
            pos_mag = math.sqrt(sum(e**2 for e in pos_err))
            
            if pos_mag < 0.05:
                self.get_logger().info('Lifted 0.25m! Moving to dustbin...')
                self.pub.publish(Twist())
                self.state = "MOVE_TO_DUSTBIN"
            else:
                # Continue lifting
                self.publish_velocity(pos_err, [0, 0, 0])
        
        # --- STATE: MOVE_TO_DUSTBIN ---
        # Navigate to dustbin position with correct orientation
        elif self.state == "MOVE_TO_DUSTBIN":
            pos_err, ori_err = self.compute_pose_error(current_pos, current_ori, self.dustbin_position, self.dustbin_orientation)
            pos_mag = math.sqrt(sum(e**2 for e in pos_err))
            ori_mag = math.sqrt(sum(v**2 for v in ori_err))
            
            # Check if reached dustbin (with relaxed tolerance)
            if pos_mag < self.dustbin_position_tolerance and ori_mag < self.dustbin_orientation_tolerance:
                self.get_logger().info('Reached dustbin')
                self.pub.publish(Twist())
                
                # Wait for user to detach fruit
                input(f'Press Enter AFTER detaching bad_fruit_{self.target_fruit["id"]}...')
                
                # Mark fruit as picked
                self.target_fruit['picked'] = True
                
                # Check remaining fruits
                unpicked_count = sum(1 for f in self.bad_fruits if not f['picked'])
                
                if unpicked_count == 0:
                    # All fruits completed
                    self.state = "COMPLETED"
                    self.get_logger().info("All fruits processed!")
                else:
                    # More fruits to pick, lift before next approach
                    self.lift_target_z = current_pos[2] + 0.45
                    self.state = "LIFT_AFTER_DETACH"
            else:
                # Continue approaching dustbin
                self.publish_velocity(pos_err, ori_err)
        
        # --- STATE: LIFT_AFTER_DETACH ---
        # Lift 0.45m after detaching fruit before approaching next fruit
        elif self.state == "LIFT_AFTER_DETACH":
            target_pos = [current_pos[0], current_pos[1], self.lift_target_z]
            pos_err, _ = self.compute_pose_error(current_pos, current_ori, target_pos, current_ori)
            pos_mag = math.sqrt(sum(e**2 for e in pos_err))
            
            if pos_mag < 0.05:
                self.get_logger().info('Lifted after detach!')
                self.pub.publish(Twist())
                
                # Find next nearest fruit
                nearest_fruit = self.find_nearest_unpicked_fruit(current_pos)
                if nearest_fruit:
                    self.target_fruit = nearest_fruit
                    self.state = "MOVE_TO_FRUIT"
                    self.last_position = None
                    self.stuck_maneuver_step = 0
                else:
                    self.state = "COMPLETED"
            else:
                # Continue lifting
                self.publish_velocity(pos_err, [0, 0, 0])
        
        # --- STATE: COMPLETED ---
        # Task completed, stop all motion
        elif self.state == "COMPLETED":
            self.pub.publish(Twist())

def main(args=None):
    '''
    Purpose:
    ---
    Main function to initialize and run the FruitPickAndThrowNode
    
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
    node = FruitPickAndThrowNode()
    
    try:
        # Spin node to execute callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        node.pub.publish(Twist())  # Stop robot
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

