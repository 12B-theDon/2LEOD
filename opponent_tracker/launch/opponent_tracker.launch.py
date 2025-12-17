from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/opponent_bundle.joblib',
        description='Path to the serialized opponent model bundle.'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    config_file = PathJoinSubstitution(
        [FindPackageShare('opponent_tracker'), 'config', 'opponent_tracker.yaml']
    )

    node = Node(
        package='opponent_tracker',
        executable='opponent_odom_node.py',
        name='opponent_odom_node',
        output='screen',
        parameters=[config_file, {'model_path': LaunchConfiguration('model_path')}],
        remappings=[
            ('/scan', '/scan'),
            ('/odom', '/odom'),
            ('/opponent_odom', '/opponent_odom'),
        ],
        arguments=['--ros-args', '--log-level', 'info'],
    )

    return LaunchDescription([
        model_path_arg,
        use_sim_time_arg,
        node,
    ])


#  ros2 launch opponent_tracker opponent_tracker.launch.py --model_path /path/to/models/opponent_bundle.joblib --all.
