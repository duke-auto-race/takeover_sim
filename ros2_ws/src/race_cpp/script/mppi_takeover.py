#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, Float64
import numpy as np

class LidarFollowerNode(Node):
    def __init__(self):
        super().__init__('lidar_follower_node')

        # Publisher for control commands
        self.rc_pub = self.create_publisher(Float64MultiArray, '/rc/virtual', 10)

        # Subscriber for LiDAR scan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/sim_server/scan',
            self.scan_callback,
            10
        )

        # Subscriber for ego vehicle velocity
        self.vel_x_sub = self.create_subscription(
            Float64,
            '/sim_server/vel_x',
            self.vel_x_callback,
            10
        )

        # Control parameters
        self.declare_parameter('mode', 'follow_opponent')  # Mode: 'follow_opponent' or 'track_only'
        self.declare_parameter('target_distance', 8.0)  # Target following distance (meters)
        self.declare_parameter('desired_velocity', 10.0)  # Desired velocity in m/s
        self.declare_parameter('max_throttle', 0.6)     # Maximum throttle command
        self.declare_parameter('min_throttle', -0.3)    # Maximum braking command
        self.declare_parameter('lookahead_angle', 45.0) # Search angle in front (degrees)
        self.declare_parameter('kp_steering', 1.5)      # Proportional gain for steering
        self.declare_parameter('kp_speed', 0.15)        # Proportional gain for speed control
        self.declare_parameter('kd_speed', 0.05)        # Derivative gain for speed control
        self.declare_parameter('kp_velocity', 0.2)      # Proportional gain for velocity tracking
        self.declare_parameter('min_cluster_size', 5)   # Minimum points to form a valid cluster
        self.declare_parameter('max_cluster_gap', 0.5)  # Maximum range gap within cluster (meters)
        
        self.mode = self.get_parameter('mode').value
        self.target_distance = self.get_parameter('target_distance').value
        self.desired_velocity = self.get_parameter('desired_velocity').value
        self.max_throttle = self.get_parameter('max_throttle').value
        self.min_throttle = self.get_parameter('min_throttle').value
        self.lookahead_angle = self.get_parameter('lookahead_angle').value
        self.kp_steering = self.get_parameter('kp_steering').value
        self.kp_speed = self.get_parameter('kp_speed').value
        self.kd_speed = self.get_parameter('kd_speed').value
        self.kp_velocity = self.get_parameter('kp_velocity').value
        self.min_cluster_size = self.get_parameter('min_cluster_size').value
        self.max_cluster_gap = self.get_parameter('max_cluster_gap').value

        # State variables
        self.current_velocity = 0.0
        self.last_distance_error = 0.0
        self.last_time = None

        self.get_logger().info('LiDAR Follower Node Started')
        self.get_logger().info(f'Mode: {self.mode}')
        self.get_logger().info(f'Target distance: {self.target_distance}m')
        self.get_logger().info(f'Desired velocity: {self.desired_velocity} m/s')
        self.get_logger().info(f'Lookahead angle: ±{self.lookahead_angle}°')
        self.get_logger().info(f'Cluster detection: min_size={self.min_cluster_size}, max_gap={self.max_cluster_gap}m')
    
    def vel_x_callback(self, msg: Float64):
        """Update current velocity from ego vehicle"""
        self.current_velocity = msg.data
    
    def find_clusters(self, ranges, angles, valid_mask):
        """Find clusters of consecutive points at similar ranges"""
        clusters = []
        current_cluster_indices = []
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return clusters
        
        for i, idx in enumerate(valid_indices):
            if len(current_cluster_indices) == 0:
                # Start new cluster
                current_cluster_indices.append(idx)
            else:
                # Check if this point is close enough to the previous point in the cluster
                prev_idx = current_cluster_indices[-1]
                range_gap = abs(ranges[idx] - ranges[prev_idx])
                
                if range_gap <= self.max_cluster_gap:
                    # Add to current cluster
                    current_cluster_indices.append(idx)
                else:
                    # Save current cluster if it's large enough, start new one
                    if len(current_cluster_indices) >= self.min_cluster_size:
                        clusters.append(current_cluster_indices.copy())
                    current_cluster_indices = [idx]
        
        # Don't forget the last cluster
        if len(current_cluster_indices) >= self.min_cluster_size:
            clusters.append(current_cluster_indices)
        
        return clusters
    
    def get_cluster_centroid(self, ranges, angles, cluster_indices):
        """Compute the centroid of a cluster"""
        cluster_ranges = ranges[cluster_indices]
        cluster_angles = angles[cluster_indices]
        
        # Convert to Cartesian coordinates for better centroid calculation
        x = cluster_ranges * np.cos(cluster_angles)
        y = cluster_ranges * np.sin(cluster_angles)
        
        # Compute centroid
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        
        # Convert back to polar
        centroid_range = np.sqrt(centroid_x**2 + centroid_y**2)
        centroid_angle = np.arctan2(centroid_y, centroid_x)
        
        return centroid_range, centroid_angle, len(cluster_indices)
    
    def follow_track(self, ranges, angles, valid_mask):
        """Follow the track centerline when no opponent is detected"""
        # Find left and right boundaries
        filtered_ranges = ranges[valid_mask]
        filtered_angles = angles[valid_mask]
        
        if len(filtered_ranges) == 0:
            self.get_logger().warn('No valid points - stopping')
            self.publish_stop_command()
            return
        
        # Split into left and right sides
        # Only consider points around 90° (left) and -90° (right) - perpendicular to vehicle
        angle_tolerance = np.deg2rad(20.0)  # ±30° around perpendicular
        left_mask = (filtered_angles > (np.pi/2 - angle_tolerance)) & (filtered_angles < (np.pi/2 + angle_tolerance))
        right_mask = (filtered_angles > (-np.pi/2 - angle_tolerance)) & (filtered_angles < (-np.pi/2 + angle_tolerance))
        
        left_ranges = filtered_ranges[left_mask]
        right_ranges = filtered_ranges[right_mask]
        
        # Compute average distances to left and right boundaries
        left_dist = np.mean(left_ranges) if len(left_ranges) > 0 else float('inf')
        right_dist = np.mean(right_ranges) if len(right_ranges) > 0 else float('inf')
        
        self.get_logger().info(f"{left_dist}")
        self.get_logger().info(f"{right_dist}")
        
        # Steer to stay centered between boundaries
        # Positive steering if right side is closer (steer left)
        # Negative steering if left side is closer (steer right)
        lateral_error = -(right_dist - left_dist)
        steering = np.clip(0.3 * lateral_error, -0.5, 0.5)
        
        # Velocity tracking for track following
        velocity_error = self.desired_velocity - self.current_velocity
        
        throttle = self.kp_velocity * velocity_error
        # throttle = np.clip(throttle, self.min_throttle, self.max_throttle)
        # self.get_logger().info(f"{throttle}")

        
        self.publish_command(throttle, steering)

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR scan and compute control commands"""
        
        # Get current time
        current_time = self.get_clock().now()
        
        # Convert lookahead angle to radians
        lookahead_rad = np.deg2rad(self.lookahead_angle)
        
        # Find indices within lookahead angle (centered at 0°, forward direction)
        # LaserScan angle_min is typically negative (left), angle_max is positive (right)
        ranges = np.array(msg.ranges)
        num_ranges = len(ranges)
        
        # Create angles array matching the exact length of ranges
        angles = msg.angle_min + np.arange(num_ranges) * msg.angle_increment
        
        # Filter for valid ranges
        valid_mask = (ranges >= msg.range_min) & (ranges < msg.range_max)
        
        # If in track_only mode, skip opponent detection
        if self.mode == 'track_only':
            self.get_logger().info('TRACK ONLY MODE - Following track centerline')
            self.follow_track(ranges, angles, valid_mask)
            return
        
        # Filter for points within lookahead cone
        lookahead_mask = (angles >= -lookahead_rad) & (angles <= lookahead_rad)
        
        # Combined mask
        target_mask = valid_mask & lookahead_mask
        
        if not np.any(target_mask):
            # No valid points detected, stop
            self.get_logger().warn('No target detected in lookahead cone')
            self.publish_stop_command()
            return
        
        # Find clusters of points (opponent vehicle should appear as a cluster)
        clusters = self.find_clusters(ranges, angles, target_mask)
        
        if len(clusters) == 0:
            self.get_logger().info('No clusters detected - Following track')
            self.follow_track(ranges, angles, valid_mask)
            return
        
        # Find the best cluster (closest centroid, or largest cluster near center)
        # Prefer clusters that are more centered (smaller angle) and closer
        best_cluster = None
        best_score = float('inf')
        
        for cluster_indices in clusters:
            centroid_range, centroid_angle, cluster_size = self.get_cluster_centroid(
                ranges, angles, cluster_indices
            )
            
            # Skip large clusters - likely walls or track boundaries
            if cluster_size > 100:
                continue
            
            # Score based on: distance + angle penalty
            # Prefer centered, close clusters
            angle_penalty = 2.0 * abs(centroid_angle)  # Penalize off-center targets
            score = centroid_range + angle_penalty
            
            if score < best_score:
                best_score = score
                best_cluster = (centroid_range, centroid_angle, cluster_size)
        
        if best_cluster is None:
            # No opponent detected - follow track centerline
            self.get_logger().info('No opponent car detected - Following track')
            self.follow_track(ranges, angles, valid_mask)
            return
        
        detected_distance, target_angle, cluster_size = best_cluster
        
        # Log opponent detection
        self.get_logger().info(f'OPPONENT CAR DETECTED - Tracking target')
        
        # Compute steering command
        # Proportional control: steer towards the target
        # Normalize steering to [-1, 1] range (will be scaled by 20° in sim_server)
        steering = self.kp_steering * target_angle
        steering = np.clip(steering, -1.0, 1.0)
        
        # Compute throttle command
        # PD control based on distance error (no velocity tracking during opponent following)
        distance_error = detected_distance - self.target_distance
        
        if self.last_time is not None:
            dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt > 0:
                error_rate = (distance_error - self.last_distance_error) / dt
            else:
                error_rate = 0.0
        else:
            error_rate = 0.0
        
        # PD control for throttle based on distance to opponent
        throttle = self.kp_speed * distance_error + self.kd_speed * error_rate
        throttle = np.clip(throttle, self.min_throttle, self.max_throttle)
        
        # Update state
        self.last_distance_error = distance_error
        self.last_time = current_time
        
        # Publish control command
        self.publish_command(throttle, steering)
        
        # Log status
        self.get_logger().info(
            f'Clusters: {len(clusters)}, Size: {cluster_size} pts, '
            f'Distance: {detected_distance:.2f}m (target: {self.target_distance:.2f}m), '
            f'Angle: {np.rad2deg(target_angle):.1f}°, '
            f'Throttle: {throttle:.2f}, Steering: {steering:.2f}'
        )
    
    def publish_command(self, throttle, steering):
        """Publish control command to /rc/virtual"""
        msg = Float64MultiArray()
        msg.data = [float(throttle), float(steering)]
        self.rc_pub.publish(msg)
    
    def publish_stop_command(self):
        """Publish stop command"""
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0]
        self.rc_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LidarFollowerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop command before shutting down
        node.publish_stop_command()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
