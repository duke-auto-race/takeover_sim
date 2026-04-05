#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, Float64
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np
import torch
import torch.distributions as dist

class LidarFollowerNode(Node):
    def __init__(self):
        super().__init__('lidar_follower_node')

        # Publisher for control commands
        self.rc_pub = self.create_publisher(Float64MultiArray, '/rc/virtual', 10)
        
        # Publisher for visualization markers
        self.marker_pub = self.create_publisher(MarkerArray, '/mppi_rollouts', 10)

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
        
        # MPPI parameters
        self.declare_parameter('mppi_num_samples', 1000)  # Number of parallel trajectory samples
        self.declare_parameter('mppi_horizon', 20)        # Planning horizon steps
        self.declare_parameter('mppi_dt', 0.1)            # Time step for prediction (seconds)
        self.declare_parameter('mppi_lambda', 1.0)        # Temperature parameter
        self.declare_parameter('mppi_sigma_throttle', 0.3)  # Throttle noise std
        self.declare_parameter('mppi_sigma_steering', 0.5)  # Steering noise std
        self.declare_parameter('mppi_cost_distance', 1.0)   # Cost weight for distance to opponent
        self.declare_parameter('mppi_cost_lateral', 2.0)    # Cost weight for lateral offset
        self.declare_parameter('mppi_cost_velocity', 0.5)   # Cost weight for velocity tracking
        self.declare_parameter('mppi_cost_control', 0.01)   # Cost weight for control effort
        self.declare_parameter('mppi_takeover_distance', 3.0)  # Target distance for takeover
        self.declare_parameter('distance_error_threshold', 2.0)  # Distance error threshold for switching
        self.declare_parameter('distance_maintain_time', 10.0)   # Time to maintain distance before MPPI
        self.declare_parameter('max_takeover_distance', 8.0)     # Maximum distance to allow MPPI takeover
        
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
        
        # MPPI parameters
        self.mppi_num_samples = self.get_parameter('mppi_num_samples').value
        self.mppi_horizon = self.get_parameter('mppi_horizon').value
        self.mppi_dt = self.get_parameter('mppi_dt').value
        self.mppi_lambda = self.get_parameter('mppi_lambda').value
        self.mppi_sigma_throttle = self.get_parameter('mppi_sigma_throttle').value
        self.mppi_sigma_steering = self.get_parameter('mppi_sigma_steering').value
        self.mppi_cost_distance = self.get_parameter('mppi_cost_distance').value
        self.mppi_cost_lateral = self.get_parameter('mppi_cost_lateral').value
        self.mppi_cost_velocity = self.get_parameter('mppi_cost_velocity').value
        self.mppi_cost_control = self.get_parameter('mppi_cost_control').value
        self.mppi_takeover_distance = self.get_parameter('mppi_takeover_distance').value
        self.distance_error_threshold = self.get_parameter('distance_error_threshold').value
        self.distance_maintain_time = self.get_parameter('distance_maintain_time').value
        self.max_takeover_distance = self.get_parameter('max_takeover_distance').value

        # State variables
        self.current_velocity = 0.0
        self.last_distance_error = 0.0
        self.last_time = None
        
        # MPPI state tracking
        self.mppi_mode = False
        self.distance_maintain_start = None
        self.opponent_distance = None
        self.opponent_angle = None
        
        # PyTorch device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        self.get_logger().info('LiDAR Follower Node Started')
        self.get_logger().info(f'Mode: {self.mode}')
        self.get_logger().info(f'Target distance: {self.target_distance}m')
        self.get_logger().info(f'Desired velocity: {self.desired_velocity} m/s')
        self.get_logger().info(f'Lookahead angle: ±{self.lookahead_angle}°')
        self.get_logger().info(f'Cluster detection: min_size={self.min_cluster_size}, max_gap={self.max_cluster_gap}m')
        self.get_logger().info(f'MPPI: samples={self.mppi_num_samples}, horizon={self.mppi_horizon}, dt={self.mppi_dt}s')
    
    def publish_trajectories_visualization(self, states, weights):
        """
        Publish trajectory rollouts as RViz markers
        states: [num_samples, horizon+1, 4] (x, y, theta, v)
        weights: [num_samples] normalized weights for each trajectory
        """
        marker_array = MarkerArray()
        
        # Convert to numpy for easier manipulation
        states_np = states.cpu().numpy()
        weights_np = weights.cpu().numpy()
        
        # Visualize a subset of trajectories (top weighted ones + random samples)
        num_viz = min(50, self.mppi_num_samples)  # Visualize top 50 trajectories
        
        # Get indices of top weighted trajectories
        top_indices = np.argsort(weights_np)[-num_viz:]
        
        for idx, traj_idx in enumerate(top_indices):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "mppi_trajectories"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Scale line width based on weight
            weight = weights_np[traj_idx]
            marker.scale.x = 0.01 + 0.05 * weight / weights_np.max()
            
            # Color based on weight (blue=low weight, green=medium, red=high weight)
            color = ColorRGBA()
            if weight < weights_np.max() * 0.3:
                # Blue for low weight
                color.r = 0.0
                color.g = 0.5
                color.b = 1.0
                color.a = 0.3
            elif weight < weights_np.max() * 0.7:
                # Green for medium weight
                color.r = 0.0
                color.g = 1.0
                color.b = 0.0
                color.a = 0.5
            else:
                # Red for high weight (best trajectories)
                color.r = 1.0
                color.g = 0.0
                color.b = 0.0
                color.a = 0.8
            
            marker.color = color
            
            # Add points along trajectory
            for t in range(states_np.shape[1]):
                point = Point()
                point.x = float(states_np[traj_idx, t, 0])  # x position
                point.y = float(states_np[traj_idx, t, 1])  # y position
                point.z = 0.1  # Slightly above ground
                marker.points.append(point)
            
            marker_array.markers.append(marker)
        
        # Add marker for optimal trajectory (weighted mean)
        optimal_marker = Marker()
        optimal_marker.header.frame_id = "ego_racecar/base_link"
        optimal_marker.header.stamp = self.get_clock().now().to_msg()
        optimal_marker.ns = "mppi_optimal"
        optimal_marker.id = 0
        optimal_marker.type = Marker.LINE_STRIP
        optimal_marker.action = Marker.ADD
        optimal_marker.scale.x = 0.08  # Thicker line
        
        # Yellow color for optimal path
        optimal_color = ColorRGBA()
        optimal_color.r = 1.0
        optimal_color.g = 1.0
        optimal_color.b = 0.0
        optimal_color.a = 1.0
        optimal_marker.color = optimal_color
        
        # Compute weighted mean trajectory
        weights_expanded = weights_np[:, np.newaxis, np.newaxis]
        optimal_traj = np.sum(weights_expanded * states_np, axis=0)
        
        for t in range(optimal_traj.shape[0]):
            point = Point()
            point.x = float(optimal_traj[t, 0])
            point.y = float(optimal_traj[t, 1])
            point.z = 0.15  # Slightly higher than other trajectories
            optimal_marker.points.append(point)
        
        marker_array.markers.append(optimal_marker)
        
        # Publish the marker array
        self.marker_pub.publish(marker_array)
    
    def mppi_dynamics(self, state, control):
        """
        Simple kinematic bicycle model for forward prediction
        state: [x, y, theta, v] (position, heading, velocity)
        control: [throttle, steering]
        Returns: next state
        """
        x, y, theta, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        throttle, steering = control[:, 0], control[:, 1]
        
        # Simple dynamics
        # Acceleration from throttle (simplified)
        a = throttle * 5.0  # max acceleration ~5 m/s^2
        
        # Update velocity
        v_next = torch.clamp(v + a * self.mppi_dt, 0.0, 20.0)
        
        # Update heading (steering affects heading rate)
        # steering is in [-1, 1], mapped to max ±20° in sim
        max_steering_angle = np.deg2rad(20.0)
        delta = steering * max_steering_angle
        
        # Wheelbase (approximate)
        L = 0.3
        theta_next = theta + (v * torch.tan(delta) / L) * self.mppi_dt
        
        # Update position
        x_next = x + v * torch.cos(theta) * self.mppi_dt
        y_next = y + v * torch.sin(theta) * self.mppi_dt
        
        return torch.stack([x_next, y_next, theta_next, v_next], dim=1)
    
    def mppi_cost(self, states, controls, opponent_x, opponent_y):
        num_samples = states.shape[0]
        horizon = controls.shape[1]

        total_cost = torch.zeros(num_samples, device=self.device)

        desired_side = -1.0   # right side
        clearance = 3.6       # lateral clearance (meters)

        for t in range(horizon):
            x = states[:, t, 0]
            y = states[:, t, 1]
            v = states[:, t, 3]

            dx = x - opponent_x
            dy = y - opponent_y
            dist = torch.sqrt(dx**2 + dy**2 + 1e-6)

            # 1. collision
            collision_cost = 1.0 / (dist + 1.0)

            # 2. velocity
            desired_v = self.desired_velocity * 1.3
            velocity_cost = self.mppi_cost_velocity * (v - desired_v)**2

            # 3. control smoothing
            control_cost = 4.0 * (controls[:, t, 0]**2 + controls[:, t, 1]**2)

            # 4. side commitment (stay right)
            side_error = (torch.sign(dy + 1e-3) - desired_side)**2
            side_cost = 20.0 * side_error

            # 5. forward progress
            progress = -8.0 * (x - opponent_x)

            # 6. reverse penalty
            reverse_cost = 20.0 * torch.clamp(-v, min=0.0)**2

            # 7. lateral clearance (key fix)
            lateral_gap = torch.abs(dy)
            gap_violation = torch.clamp(clearance - lateral_gap, min=0.0)
            clearance_cost = 30.0 * gap_violation**2

            total_cost += (
                collision_cost +
                velocity_cost +
                control_cost +
                side_cost +
                progress +
                reverse_cost +
                clearance_cost
            )

        return total_cost
    
    def mppi_control(self, current_state, opponent_x, opponent_y):
        state = torch.tensor(current_state, dtype=torch.float32, device=self.device)
        opp_x = torch.tensor(opponent_x, dtype=torch.float32, device=self.device)
        opp_y = torch.tensor(opponent_y, dtype=torch.float32, device=self.device)

        throttle_samples = torch.randn(
            self.mppi_num_samples, self.mppi_horizon, device=self.device
        ) * self.mppi_sigma_throttle

        steering_samples = torch.randn(
            self.mppi_num_samples, self.mppi_horizon, device=self.device
        ) * self.mppi_sigma_steering 
        
        steering_samples += -0.2

        throttle_samples = torch.clamp(throttle_samples, -0.5, 1.0)
        steering_samples = torch.clamp(steering_samples, -1.0, 1.0)

        controls = torch.stack([throttle_samples, steering_samples], dim=2)

        states = torch.zeros(
            self.mppi_num_samples, self.mppi_horizon + 1, 4, device=self.device
        )
        states[:, 0, :] = state

        for t in range(self.mppi_horizon):
            states[:, t + 1, :] = self.mppi_dynamics(states[:, t, :], controls[:, t, :])

        costs = self.mppi_cost(states, controls, opp_x, opp_y)

        weights = torch.softmax(-costs / self.mppi_lambda, dim=0)

        self.publish_trajectories_visualization(states, weights)

        optimal_control = torch.sum(
            weights[:, None, None] * controls, dim=0
        )

        return (
            optimal_control[0, 0].item(),
            optimal_control[0, 1].item()
        )
    
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
        
        # If only 1 or 2 clusters, likely just walls - no opponent detection
        if len(clusters) <= 2:
            self.get_logger().info(f'Only {len(clusters)} cluster(s) - No opponent detected, following track')
            self.follow_track(ranges, angles, valid_mask)
            return
        
        # Find the best cluster (closest centroid, or largest cluster near center)
        # Prefer clusters that are more centered (smaller angle) and closer
        best_cluster = None
        best_score = float('inf')
        
        # If 3+ clusters, choose the smallest one (likely the opponent car vs walls)
        for cluster_indices in clusters:
            centroid_range, centroid_angle, cluster_size = self.get_cluster_centroid(
                ranges, angles, cluster_indices
            )
            
            # Score based on cluster size (smallest is best for opponent detection)
            if cluster_size < best_score:
                best_score = cluster_size
                best_cluster = (centroid_range, centroid_angle, cluster_size)
        
        if best_cluster is None:
            # No opponent detected - follow track centerline
            self.get_logger().info('No opponent car detected - Following track')
            self.follow_track(ranges, angles, valid_mask)
            return
        
        detected_distance, target_angle, cluster_size = best_cluster
        
        # Update opponent tracking state
        self.opponent_distance = detected_distance
        self.opponent_angle = target_angle
        
        # Check if distance error is within threshold
        distance_error = abs(detected_distance - self.target_distance)
        
        # Track how long we've maintained the target distance
        if distance_error <= self.distance_error_threshold:
            if self.distance_maintain_start is None:
                self.distance_maintain_start = current_time
                self.get_logger().info('Started tracking distance maintenance')
            else:
                maintain_duration = (current_time - self.distance_maintain_start).nanoseconds / 1e9
                
                # Switch to MPPI mode after maintaining distance for specified time
                # AND only if within max_takeover_distance
                if not self.mppi_mode and detected_distance <= self.max_takeover_distance:
                    self.mppi_mode = True
                    self.get_logger().info('=' * 60)
                    self.get_logger().info('SWITCHING TO MPPI TAKEOVER MODE!')
                    self.get_logger().info('=' * 60)
        else:
            # Reset if we lose distance maintenance
            if self.distance_maintain_start is not None and not self.mppi_mode:
                self.get_logger().info('Lost distance maintenance, resetting timer')
            self.distance_maintain_start = None
        
        # Exit MPPI mode if distance exceeds max_takeover_distance
        if self.mppi_mode and detected_distance > self.max_takeover_distance:
            self.mppi_mode = False
            self.distance_maintain_start = None
            self.get_logger().info('=' * 60)
            self.get_logger().info(f'EXITING MPPI MODE - Distance > {self.max_takeover_distance}m')
            self.get_logger().info('=' * 60)
        
        # Use MPPI control if in MPPI mode
        if self.mppi_mode:
            self.get_logger().info('MPPI TAKEOVER MODE ACTIVE - Using MPPI Controller')
            
            # Convert opponent position to ego frame coordinates
            opponent_x = detected_distance * np.cos(target_angle)
            opponent_y = detected_distance * np.sin(target_angle)
            
            # Current state: [x, y, theta, v] (ego frame: x=0, y=0, theta=0)
            current_state = [0.0, 0.0, 0.0, self.current_velocity]
            
            # Compute optimal control using MPPI
            throttle, steering = self.mppi_control(current_state, opponent_x, opponent_y)
            
            # Clip to safety limits
            throttle = np.clip(throttle, self.min_throttle, self.max_throttle)
            steering = np.clip(steering, -1.0, 1.0)
        else:
            # Use PD control for following
            if self.distance_maintain_start is not None:
                maintain_duration = (current_time - self.distance_maintain_start).nanoseconds / 1e9
                self.get_logger().info(
                    f'OPPONENT CAR DETECTED - Maintaining distance '
                    f'({maintain_duration:.1f}/{self.distance_maintain_time:.1f}s)'
                )
            else:
                self.get_logger().info('OPPONENT CAR DETECTED - Tracking target')
            
            # Compute steering command
            steering = self.kp_steering * target_angle
            steering = np.clip(steering, -1.0, 1.0)
            
            # Compute throttle command (PD control)
            distance_error_signed = detected_distance - self.target_distance
            
            if self.last_time is not None:
                dt = (current_time - self.last_time).nanoseconds / 1e9
                if dt > 0:
                    error_rate = (distance_error_signed - self.last_distance_error) / dt
                else:
                    error_rate = 0.0
            else:
                error_rate = 0.0
            
            # PD control for throttle based on distance to opponent
            throttle = self.kp_speed * distance_error_signed + self.kd_speed * error_rate
            throttle = np.clip(throttle, self.min_throttle, self.max_throttle)
            
            # Update state
            self.last_distance_error = distance_error_signed
        
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
