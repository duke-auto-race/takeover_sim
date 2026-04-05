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
        # default_value='follow_opponent',
        default_value='follow_opponent',
        description='Control mode: follow_opponent or track_only'
    )
    
    target_distance_arg = DeclareLaunchArgument(
        'target_distance',
        default_value='10.0',
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
        
        # LiDAR follower controller
        Node(
            package='race_cpp',
            executable='pd_track.py',
            name='lidar_follower_node',
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
            ],
            output='screen'
        ),
    ])
