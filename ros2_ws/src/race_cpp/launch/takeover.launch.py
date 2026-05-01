from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    rviz_config_path = os.path.join(
        get_package_share_directory('race_cpp'),
        'launch',      
        'race.rviz'
    )

    # Declare launch arguments for follower parameters
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='follow_opponent',
        description='Control mode: follow_opponent or track_only'
    )
    
    target_distance_arg = DeclareLaunchArgument(
        'target_distance',
        default_value='6.0',
        description='Following distance in meters'
    )
    
    desired_velocity_arg = DeclareLaunchArgument(
        'desired_velocity',
        default_value='15.0',
        description='Desired velocity in m/s'
    )
    
    max_throttle_arg = DeclareLaunchArgument(
        'max_throttle',
        default_value='10.0',
        description='Maximum throttle command'
    )
    
    min_throttle_arg = DeclareLaunchArgument(
        'min_throttle',
        default_value='-0.3',
        description='Maximum braking command'
    )
    
    lookahead_angle_arg = DeclareLaunchArgument(
        'lookahead_angle',
        default_value='60.0',
        description='Search cone angle in degrees'
    )
    
    kp_steering_arg = DeclareLaunchArgument(
        'kp_steering',
        default_value='2.0',
        description='Proportional gain for steering control'
    )
    
    kp_speed_arg = DeclareLaunchArgument(
        'kp_speed',
        default_value='2.0',
        description='Proportional gain for speed control'
    )
    
    kd_speed_arg = DeclareLaunchArgument(
        'kd_speed',
        default_value='0.05',
        description='Derivative gain for speed control'
    )
    
    kp_velocity_arg = DeclareLaunchArgument(
        'kp_velocity',
        default_value='1.0',
        description='Proportional gain for velocity tracking'
    )
    
    min_cluster_size_arg = DeclareLaunchArgument(
        'min_cluster_size',
        default_value='5',
        description='Minimum points to form a valid cluster'
    )
    
    max_cluster_gap_arg = DeclareLaunchArgument(
        'max_cluster_gap',
        default_value='0.5',
        description='Maximum range gap within cluster in meters'
    )
    
    ai_vel_arg = DeclareLaunchArgument(
        'ai_vel',
        default_value='10.0',
        description='AI opponent velocity in m/s'
    )
    
    # MPPI-specific parameters
    mppi_num_samples_arg = DeclareLaunchArgument(
        'mppi_num_samples',
        default_value='1000',
        description='Number of parallel trajectory samples'
    )
    
    mppi_horizon_arg = DeclareLaunchArgument(
        'mppi_horizon',
        default_value='200',
        description='Planning horizon steps'
    )
    
    mppi_dt_arg = DeclareLaunchArgument(
        'mppi_dt',
        default_value='0.04',
        description='Time step for prediction (seconds)'
    )
    
    mppi_lambda_arg = DeclareLaunchArgument(
        'mppi_lambda',
        default_value='40.0',
        description='Temperature parameter'
    )
    
    mppi_sigma_throttle_arg = DeclareLaunchArgument(
        'mppi_sigma_throttle',
        default_value='0.3',
        description='Throttle noise std'
    )
    
    mppi_sigma_steering_arg = DeclareLaunchArgument(
        'mppi_sigma_steering',
        default_value='0.8',
        description='Steering noise std'
    )
    
    mppi_cost_distance_arg = DeclareLaunchArgument(
        'mppi_cost_distance',
        default_value='1.0',
        description='Cost weight for distance to opponent'
    )
    
    mppi_cost_lateral_arg = DeclareLaunchArgument(
        'mppi_cost_lateral',
        default_value='8.0',
        description='Cost weight for lateral offset'
    )
    
    mppi_cost_velocity_arg = DeclareLaunchArgument(
        'mppi_cost_velocity',
        default_value='0.5',
        description='Cost weight for velocity tracking'
    )
    
    mppi_cost_control_arg = DeclareLaunchArgument(
        'mppi_cost_control',
        default_value='1000.0',
        description='Cost weight for control effort'
    )
    
    mppi_takeover_distance_arg = DeclareLaunchArgument(
        'mppi_takeover_distance',
        default_value='10.0',
        description='Target distance for takeover'
    )
    
    distance_error_threshold_arg = DeclareLaunchArgument(
        'distance_error_threshold',
        default_value='2.0',
        description='Distance error threshold for switching to MPPI'
    )
    
    distance_maintain_time_arg = DeclareLaunchArgument(
        'distance_maintain_time',
        default_value='0.01',
        description='Time to maintain distance before MPPI (seconds)'
    )
    
    max_takeover_distance_arg = DeclareLaunchArgument(
        'max_takeover_distance',
        default_value='16.0',
        description='Maximum distance to allow MPPI takeover (meters)'
    )

    return LaunchDescription([
        # Declare launch arguments
        mode_arg,
        target_distance_arg,
        desired_velocity_arg,
        max_throttle_arg,
        min_throttle_arg,
        lookahead_angle_arg,
        kp_steering_arg,
        kp_speed_arg,
        kd_speed_arg,
        kp_velocity_arg,
        min_cluster_size_arg,
        max_cluster_gap_arg,
        ai_vel_arg,
        
        # MPPI arguments
        mppi_num_samples_arg,
        mppi_horizon_arg,
        mppi_dt_arg,
        mppi_lambda_arg,
        mppi_sigma_throttle_arg,
        mppi_sigma_steering_arg,
        mppi_cost_distance_arg,
        mppi_cost_lateral_arg,
        mppi_cost_velocity_arg,
        mppi_cost_control_arg,
        mppi_takeover_distance_arg,
        distance_error_threshold_arg,
        distance_maintain_time_arg,
        max_takeover_distance_arg,
        
        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path]
        ),
        
        # Simulation server
        Node(
            package='race_cpp',
            executable='race_sim_server_node',
            name='race_sim_server_node',
            parameters=[
                {"ai_vel": LaunchConfiguration('ai_vel')},
            ],
            output='screen'
        ),
        
        # MPPI Takeover controller
        Node(
            package='race_cpp',
            executable='mppi_takeover.py',
            name='mppi_takeover_node',
            parameters=[
                {"mode": LaunchConfiguration('mode')},
                {"target_distance": LaunchConfiguration('target_distance')},
                {"desired_velocity": LaunchConfiguration('desired_velocity')},
                {"max_throttle": LaunchConfiguration('max_throttle')},
                {"min_throttle": LaunchConfiguration('min_throttle')},
                {"lookahead_angle": LaunchConfiguration('lookahead_angle')},
                {"kp_steering": LaunchConfiguration('kp_steering')},
                {"kp_speed": LaunchConfiguration('kp_speed')},
                {"kd_speed": LaunchConfiguration('kd_speed')},
                {"kp_velocity": LaunchConfiguration('kp_velocity')},
                {"min_cluster_size": LaunchConfiguration('min_cluster_size')},
                {"max_cluster_gap": LaunchConfiguration('max_cluster_gap')},
                # MPPI parameters
                {"mppi_num_samples": LaunchConfiguration('mppi_num_samples')},
                {"mppi_horizon": LaunchConfiguration('mppi_horizon')},
                {"mppi_dt": LaunchConfiguration('mppi_dt')},
                {"mppi_lambda": LaunchConfiguration('mppi_lambda')},
                {"mppi_sigma_throttle": LaunchConfiguration('mppi_sigma_throttle')},
                {"mppi_sigma_steering": LaunchConfiguration('mppi_sigma_steering')},
                {"mppi_cost_distance": LaunchConfiguration('mppi_cost_distance')},
                {"mppi_cost_lateral": LaunchConfiguration('mppi_cost_lateral')},
                {"mppi_cost_velocity": LaunchConfiguration('mppi_cost_velocity')},
                {"mppi_cost_control": LaunchConfiguration('mppi_cost_control')},
                {"mppi_takeover_distance": LaunchConfiguration('mppi_takeover_distance')},
                {"distance_error_threshold": LaunchConfiguration('distance_error_threshold')},
                {"distance_maintain_time": LaunchConfiguration('distance_maintain_time')},
                {"max_takeover_distance": LaunchConfiguration('max_takeover_distance')},
            ],
            output='screen'
        ),
    ])