from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """
    two nodes here:
    - leader (w/ sine wave)
    - follower (w/ velocity cmd)
    """
    rviz_config_path = os.path.join(
        get_package_share_directory('race_cpp'),
        'launch',      
        'race.rviz'
    )
    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path]
        ),
    ])