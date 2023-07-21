from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument

from ament_index_python.packages import get_package_share_path

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():

    rviz_path = PathJoinSubstitution(
        [FindPackageShare('quadruped_model'), 'rviz', 'octomap.rviz']
    ),


    return LaunchDescription([

        DeclareLaunchArgument(
            name="rviz",
            default_value="false",
            description="Whether to start RViz"
        ),

        Node(
            package="octomap_server",
            executable="octomap_server_node",
            parameters=[{
                "frame_id" : "odom",
                "resolution" : 0.05,
                "file_path" : "/home/mert/quadruped_gazebo/src/quadruped_model/maps/dummy.bt",
                "octomap_topic" : "/octomap_full",
            }],

            # remappings=[('/cloud_in', '/velodyne_points')],
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