from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _bool_to_str(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value).lower()


def _evaluate_config():
    config_dir = Path(get_package_share_directory('opponent_tracker')) / 'config'
    bag_config_path = config_dir / 'eval_opponent.yaml'
    with open(bag_config_path, 'r') as f:
        return yaml.safe_load(f) or {}


_DEFAULTS = _evaluate_config()


def _node_defaults():
    node_info = _DEFAULTS.get('evaluate_opponent', {}).get('ros__parameters', {})
    return {k: v for k, v in node_info.items()}


def generate_launch_description():
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value=_DEFAULTS.get('bag_path', ''),
        description='Path to the ROS2 bag to play.',
    )
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=_DEFAULTS.get('model_path', 'models/opponent_bundle.joblib'),
        description='Joblib bundle used for inference.',
    )
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value=_DEFAULTS.get('frame_id', 'odom'),
        description='Frame used for the published paths and odometry.',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value=_bool_to_str(_DEFAULTS.get('use_sim_time', 'true')),
        description='Enable ROS2 simulation time for playback.',
    )
    csv_path_arg = DeclareLaunchArgument(
        'csv_path',
        default_value=_DEFAULTS.get('csv_path', ''),
        description='Optional CSV file path to log predictions (empty disables).',
    )
    rmse_plot_arg = DeclareLaunchArgument(
        'rmse_plot_path',
        default_value=_DEFAULTS.get('rmse_plot_path', 'rmse_trajectory.png'),
        description='PNG filename for the rainbow RMSE trajectory plot.',
    )
    rviz_enabled_arg = DeclareLaunchArgument(
        'rviz_enabled',
        default_value=_bool_to_str(_DEFAULTS.get('rviz_enabled', True)),
        description='Whether to launch RViz2 automatically.',
    )
    rviz_rel = _DEFAULTS.get('rviz_config', 'rviz/eval_opponent.rviz')
    rviz_parts = [part for part in rviz_rel.split('/') if part]
    if not rviz_parts:
        rviz_parts = ['rviz', 'eval_opponent.rviz']
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution(
            [FindPackageShare('opponent_tracker'), *rviz_parts]
        ),
        description='RViz configuration for visualizing odometry, paths, and markers.',
    )
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=str(Path(get_package_share_directory('opponent_tracker')) / 'config' /
                          'eval_opponent.yaml'),
        description='Evaluation node parameter file.',
    )

    bag_play = ExecuteProcess(
        cmd=[
            'ros2',
            'bag',
            'play',
            LaunchConfiguration('bag_path'),
            '--clock',
        ],
        output='screen',
    )

    evaluation_node = Node(
        package='opponent_tracker',
        executable='evaluate_opponent',
        name='evaluate_opponent',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            _node_defaults(),
            {
                'model_path': LaunchConfiguration('model_path'),
                'frame_id': LaunchConfiguration('frame_id'),
                'csv_path': LaunchConfiguration('csv_path'),
                'rmse_plot_path': LaunchConfiguration('rmse_plot_path'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            },
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='eval_opponent_rviz',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('rviz_enabled')),
    )

    return LaunchDescription([
        bag_path_arg,
        model_path_arg,
        frame_id_arg,
        use_sim_time_arg,
        csv_path_arg,
        rmse_plot_arg,
        rviz_enabled_arg,
        rviz_config_arg,
        config_file_arg,
        bag_play,
        evaluation_node,
        rviz_node,
    ])
