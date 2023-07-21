from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():

    rviz_path = PathJoinSubstitution(
        [FindPackageShare('quadruped_model'), 'rviz', 'octomap.rviz']
    ),

    return LaunchDescription([

        Node(
            package="octomap_server",
            executable="octomap_server_node",
            parameters=[{
                "frame_id" : "base",
                "sensor_model.max_range" : 5.0,
                "resolution" : 0.05,
            }],

            remappings=[('/cloud_in', '/velodyne_points')],
            # remappings=[('/cloud_in', '/camera/depth/color/points')],

    
            output={
                "stdout": "screen",
                "stderr": "screen",
            },
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_path],
        )
         
    ])


# ros2 run octomap_server octomap_saver_node --ros-args -p octomap_path:=map.bt
