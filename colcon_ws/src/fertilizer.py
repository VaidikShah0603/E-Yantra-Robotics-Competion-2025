#!/usr/bin/env python3

'''

*
*  This script performs pick and place operation for fertilizer placement
*  using delta twist commands and precise pose control
*  
*  Team ID:		1924
*  Author List:		Shah Vaidik Sanjaykumar, Aryan Jain
*  Theme:               Krishi coBot 
*  Filename:		fertilizer.py
*  Functions:		__init__, get_current_pose, compute_pose_error, publish_velocity,
*                   control_loop, main
*  Nodes:		    Publishing: /delta_twist_cmds
*                   Subscribing: TF transforms
*  Global Variables: None

'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import math
from scipy.spatial.transform import Rotation as R

class PickAndPlaceNode(Node):
    '''
    Purpose:
    ---
    ROS2 node that performs pick and place operation for fertilizer
    using precise position and orientation control
    
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    None
    
    Example call:
    ---
    node = PickAndPlaceNode()
    '''
    
    def __init__(self):
        '''
        Purpose:
        ---
        Initialize the PickAndPlaceNode with publishers, TF listeners,
        and target pose parameters
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        Automatically called when creating PickAndPlaceNode() instance
        '''
        super().__init__('pick_and_place_node')
        
        # Publisher for delta twist commands to control robot end-effector
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        # TF buffer and listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # pick_pos: Target position for picking fertilizer [x, y, z] in meters
        self.pick_pos = [-0.214, -0.532, 0.557]
        
        # pick_ori: Wrist orientation for pickup [x, y, z, w] quaternion
        self.pick_ori = [0.707, 0.028, 0.034, 0.707]
        
        # drop_pos: Target position for dropping fertilizer [x, y, z] in meters
        self.drop_pos = [0.75, 0.1, 0.320]
        
        # drop_ori: Wrist orientation for dropping (facing downwards) [x, y, z, w] quaternion
        self.drop_ori = [0.98, -0.04, -0.03, 0.16]
        
        # Tolerances for reaching target poses
        self.position_tolerance = 0.08  # meters
        self.orientation_tolerance = 0.08  # radians
        
        # state: Current state in pick-and-place sequence
        self.state = 'MOVE_TO_PICK'
        
        self.get_logger().info('Node started. Moving to pick position.')
    
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
            trans = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time())
            
            # Extract translation (position)
            t = trans.transform.translation
            position = [t.x, t.y, t.z]
            
            # Extract rotation (orientation)
            r = trans.transform.rotation
            orientation = [r.x, r.y, r.z, r.w]
            
            return position, orientation
        
        except TransformException as e:
            # Transform not available
            self.get_logger().warn(f'Transform lookup failed: {e}')
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
            
            # Compute relative rotation (error)
            rel_rot = r_target * r_current.inv()
            
            # Convert to rotation vector (axis-angle representation)
            rot_vec = rel_rot.as_rotvec()
        except:
            # If rotation computation fails, return zero error
            rot_vec = [0, 0, 0]
        
        return pos_err, rot_vec
    
    def publish_velocity(self, pos_err, ori_err):
        '''
        Purpose:
        ---
        Publish velocity command based on position and orientation errors
        using proportional control
        
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
        # Create Twist message for velocity command
        twist = Twist()
        
        # Proportional control gains
        k_linear = 1.0  # Linear velocity gain
        k_angular = 2.0  # Angular velocity gain
        
        # Compute velocity commands
        twist.linear.x = k_linear * pos_err[0]
        twist.linear.y = k_linear * pos_err[1]
        twist.linear.z = k_linear * pos_err[2]
        twist.angular.x = k_angular * ori_err[0]
        twist.angular.y = k_angular * ori_err[1]
        twist.angular.z = k_angular * ori_err[2]
        
        # Publish velocity command
        self.pub.publish(twist)
    
    def control_loop(self):
        '''
        Purpose:
        ---
        Main control loop implementing state machine for pick-and-place task
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        Called continuously while node is spinning
        '''
        # Get current robot pose
        current_pos, current_ori = self.get_current_pose()
        
        if current_pos is None:
            # Transform not available yet
            return
        
        # --- STATE: MOVE_TO_PICK ---
        # Move to pick position with correct orientation
        if self.state == 'MOVE_TO_PICK':
            # Compute errors to pick pose
            pos_err, ori_err = self.compute_pose_error(current_pos, current_ori, self.pick_pos, self.pick_ori)
            
            # Compute error magnitudes
            pos_mag = math.sqrt(sum(e**2 for e in pos_err))
            ori_mag = math.sqrt(sum(e**2 for e in ori_err))
            
            # Check if reached pick position
            if pos_mag < self.position_tolerance and ori_mag < self.orientation_tolerance:
                self.get_logger().info('Reached pick position.')
                self.pub.publish(Twist())  # Stop motion
                
                # Wait for user to attach object
                input('Press Enter AFTER you attach the object...')
                
                # Transition to drop state
                self.state = 'MOVE_TO_DROP'
                self.get_logger().info('Moving to drop position.')
            else:
                # Continue moving toward pick position
                self.publish_velocity(pos_err, ori_err)
        
        # --- STATE: MOVE_TO_DROP ---
        # Move to drop position with correct orientation
        elif self.state == 'MOVE_TO_DROP':
            # Compute errors to drop pose
            pos_err, ori_err = self.compute_pose_error(current_pos, current_ori, self.drop_pos, self.drop_ori)
            
            # Compute error magnitudes
            pos_mag = math.sqrt(sum(e**2 for e in pos_err))
            ori_mag = math.sqrt(sum(e**2 for e in ori_err))
            
            # Check if reached drop position
            if pos_mag < self.position_tolerance and ori_mag < self.orientation_tolerance:
                self.get_logger().info('Reached drop position.')
                self.pub.publish(Twist())  # Stop motion
                
                # Wait for user to detach object
                input('Press Enter AFTER you detach the object...')
                
                # Transition to done state
                self.state = 'DONE'
                self.get_logger().info('Pick and place complete.')
            else:
                # Continue moving toward drop position
                self.publish_velocity(pos_err, ori_err)
        
        # --- STATE: DONE ---
        # Task completed, stop all motion
        elif self.state == 'DONE':
            self.pub.publish(Twist())  # Ensure robot is stopped

def main(args=None):
    '''
    Purpose:
    ---
    Main function to initialize and run the PickAndPlaceNode
    
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
    node = PickAndPlaceNode()
    
    try:
        # Spin node continuously to execute control loop
        while rclpy.ok():
            rclpy.spin_once(node)
            node.control_loop()
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        node.get_logger().info('Shutting down.')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

