from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='scout_cpp',
            executable='scout_relay_node',
            name='scout_relay'
        ),
        Node(
            package='scout_cpp',
            executable='scout_sim_server_node',
            name='scout_sim_server'
        ),
        Node(
            package='scout_cpp',
            executable='scout_traj_server_node',
            name='scout_traj_server'
        ),
        Node(
            package='scout',
            executable='acoustic_kf',
            name='acoustic_kf'
        )
    ])
