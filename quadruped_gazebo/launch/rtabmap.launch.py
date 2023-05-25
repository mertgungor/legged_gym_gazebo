from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare



def generate_launch_description():
    
    rtabmap_launch = PathJoinSubstitution(
        [FindPackageShare("rtabmap_launch"), "launch", "rtabmap.launch.py"]
    )

    return LaunchDescription([

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rtabmap_launch),
            launch_arguments={
                'rtabmap_args'      : "--delete_db_on_start --Optimizer/GravitySigma 0.3",
                'depth_topic'       : "/camera_1/depth/image_raw",
                "rgb_topic"         : "/camera_1/image_raw",
                "camera_info_topic" : "/camera_1/camera_info",
                "approx_sync"       : "false", 
                "frame_id"          : "camera_link_1",
                "qos"               : "1",
                "rviz"              : "true",
            }.items()
        ),


    ])
