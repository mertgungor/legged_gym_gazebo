import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Set the parameters for the occupancy map server
    params = {
        'point_cloud_topic': '/camera/depth/color/points',
        'octomap_topic': '/octomap',
        'resolution': '0.1',
        'max_range': '10.0',
        'publish_octomap': 'true',
        'map_frame': 'map',
        'sensor_frame': 'camera_depth_optical_frame',
        'octomap_binary': 'true',
        'octomap_frame_id': 'map',
        'octomap_resolution': '0.1',
        'octomap_prob_hit': '0.7',
        'octomap_prob_miss': '0.4',
        'octomap_threshold': '0.5',
        'octomap_publish_binary': 'true',
        'octomap_publish_full': 'true',
        'octomap_publish_markers': 'false',
        'octomap_full_min': '0.1',
        'octomap_full_max': '0.9'
    }

    # Create a Node object for the occupancy map server
    node = Node(
        package='moveit_ros_occupancy_map_monitor',
        executable='moveit_ros_occupancy_map_server',
        parameters=[params],
        output='screen'
    )

    # Create a LaunchDescription object and add the node to it
    ld = LaunchDescription()
    ld.add_action(node)

    return ld
