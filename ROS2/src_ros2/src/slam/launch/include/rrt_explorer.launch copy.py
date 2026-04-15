import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
# 增加了 GroupAction, IncludeLaunchDescription 等导入
from launch.actions import (DeclareLaunchArgument, SetEnvironmentVariable, 
                            OpaqueFunction, GroupAction, IncludeLaunchDescription)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

def launch_setup(context):
    pkg_share = get_package_share_directory('slam') 
    
    compiled = os.environ.get('need_compile', 'False')
    if compiled == 'True':
        navigation_package_path = get_package_share_directory('navigation')
    else:
        # 注意：硬编码路径在不同机器上可能失效
        navigation_package_path = '/home/ubuntu/ros2_ws/src/navigation'

    # 定义配置
    use_sim_time = LaunchConfiguration('use_sim_time')
    scan_topic = LaunchConfiguration('scan_topic')
    map_frame = LaunchConfiguration('map_frame')
    odom_frame = LaunchConfiguration('odom_frame')
    base_frame = LaunchConfiguration('base_frame')
    params_file = LaunchConfiguration('params_file')
    use_teb = LaunchConfiguration('use_teb')

    # 定义参数及其默认值
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')
    scan_topic_arg = DeclareLaunchArgument('scan_topic', default_value='scan')
    map_frame_arg = DeclareLaunchArgument('map_frame', default_value='map')
    odom_frame_arg = DeclareLaunchArgument('odom_frame', default_value='odom')
    base_frame_arg = DeclareLaunchArgument('base_frame', default_value='base_footprint')
    
    default_params_path = os.path.join(navigation_package_path, 'config/nav2_params.yaml')
    params_file_arg = DeclareLaunchArgument('params_file', default_value=default_params_path)
    use_teb_arg = DeclareLaunchArgument('use_teb', default_value='true')

    # 使用 get 防止环境变量缺失导致崩溃
    robot_name = LaunchConfiguration('robot_name', default=os.environ.get('HOST', 'default_robot')).perform(context)
    # master_name 虽然获取了但下面没用到，建议保留或删除
    master_name = LaunchConfiguration('master_name', default=os.environ.get('MASTER', 'default_master')).perform(context)

    remappings = [
        ('/tf', 'tf'),
        ('/tf_static', 'tf_static'),
        ('/map', 'map'),
        ('/map_metadata', 'map_metadata'),
        ('scan', scan_topic),
        ('echoes', scan_topic)
    ]

    bringup_cmd_group = GroupAction([
        Node(
            name='nav2_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            parameters=[params_file, {'autostart': True}],
            arguments=['--ros-args', '--log-level', 'info'],
            remappings=remappings,
            output='screen'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(navigation_package_path, 'launch/include/navigation_base.launch.py')),
            launch_arguments={
                'namespace': robot_name,
                'use_namespace': 'true',
                'use_sim_time': use_sim_time,
                'autostart': 'true',
                'params_file': params_file,
                'use_teb': use_teb
            }.items()),
    ])

    rrt_explorer_node = Node(
        package='frontier_exploration',
        executable='exploration_node',
        name='exploration_node'
    )

    return [
        scan_topic_arg,
        map_frame_arg,
        odom_frame_arg,
        base_frame_arg,
        use_sim_time_arg,
        use_teb_arg,
        params_file_arg,
        # rrt_explorer_node,
        bringup_cmd_group,
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])