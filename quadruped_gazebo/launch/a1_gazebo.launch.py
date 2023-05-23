from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, Command
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_path

pkg_path = get_package_share_path('quadruped_gazebo')
default_rviz_config_path = pkg_path / 'rviz/urdf.rviz'


world_path = PathJoinSubstitution(
        [FindPackageShare("quadruped_gazebo"), "worlds", "default.world"]
    )

config_path = PathJoinSubstitution(
        [FindPackageShare("quadruped_gazebo"), "config", "a1_ros_control.yaml"]
    )

gazebo_path = PathJoinSubstitution(
        [FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"]
    )

    

robot_description = ParameterValue(Command(['xacro ', PathJoinSubstitution([str(pkg_path), 'urdf/a1/a1.urdf'])]))

def generate_launch_description():

    return LaunchDescription([

        DeclareLaunchArgument(
            name="rviz",
            default_value="false",
            description="Whether to start RViz"
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            remappings=[('/joint_states', '/quadruped_gazebo/joint_states')]
        ),

        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', str(default_rviz_config_path)],
            condition=IfCondition(LaunchConfiguration("rviz"))
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_path),
            launch_arguments={
                'use_sim_time': str("true"),
                'world': world_path,
            }.items()
        ),

        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[{"robot_description":robot_description}, config_path],
            # remappings=[
            #     ("/diffbot_base_controller/cmd_vel_unstamped", "/cmd_vel"),
            # ],
            output={
                "stdout": "screen",
                "stderr": "screen",
            },
        ),
         
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'my_bot', '-z 0.5'],
            output='screen'
        ),

        ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'start',
                'joint_states_controller'],
            output='screen',
        ),

        ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'start',
                'joint_group_effort_controller'],
            output='screen'
        ),


    ])