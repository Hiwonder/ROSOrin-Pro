import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription, LaunchService
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction,ExecuteProcess

from launch_ros.actions import PushRosNamespace
from launch.actions import TimerAction,GroupAction


def launch_setup(context, *args, **kwargs):
    function_mode = LaunchConfiguration('function').perform(context)
    conf = LaunchConfiguration('conf', default=0.45)
    conf_arg = DeclareLaunchArgument('conf', default_value=conf)

    mode = LaunchConfiguration('mode', default=1)
    mode_arg = DeclareLaunchArgument('mode', default_value=mode)

    debug = LaunchConfiguration('debug', default=False)
    debug_arg = DeclareLaunchArgument('debug', default_value=debug)


    interruption = LaunchConfiguration('interruption', default=False)
    interruption_arg = DeclareLaunchArgument('interruption', default_value=interruption)

    camera_topic = LaunchConfiguration('camera_topic', default='/depth_cam/rgb0/image_raw')
    camera_topic_arg = DeclareLaunchArgument('camera_topic', default_value=camera_topic)


    offline = LaunchConfiguration('offline', default='false')
    offline_arg = DeclareLaunchArgument('offline', default_value=offline)

    interruption = LaunchConfiguration('interruption', default=False)
    interruption_arg = DeclareLaunchArgument('interruption', default_value=interruption)


    controller_package_path = get_package_share_directory('controller')


    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, 'launch/controller.launch.py')),
    )

    peripherals_package_path = get_package_share_directory('peripherals')
    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )

    lidar_node_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(peripherals_package_path, 'launch/lidar.launch.py')),
    )


    
    line_following_node =   Node(
                    package='app',
                    executable='line_following',
                    output='screen',
                    parameters=[{'debug': debug}],
    )

    object_tracking_node =   Node(
            package='app',
            executable='object_tracking',
            output='screen',
            parameters=[{'debug': debug}],
            )


    navigation_package_path = get_package_share_directory('navigation')
    map_name = LaunchConfiguration('map', default='map_01').perform(context)
    robot_name = LaunchConfiguration('robot_name', default=os.environ['HOST'])
    master_name = LaunchConfiguration('master_name', default=os.environ['MASTER'])

    map_name_arg = DeclareLaunchArgument('map', default_value=map_name)
    master_name_arg = DeclareLaunchArgument('master_name', default_value=master_name)
    robot_name_arg = DeclareLaunchArgument('robot_name', default_value=robot_name)
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(navigation_package_path, 'launch/navigation.launch.py')),
        launch_arguments={
            'sim': 'false',
            'map': map_name,
            'robot_name': robot_name,
            'master_name': master_name,
            'use_teb': 'true',
        }.items(),
    )

    navigation_controller_node = Node(
        package='large_models_examples',
        executable='navigation_controller',
        output='screen',
        parameters=[{'map_frame': 'map', 'nav_goal': '/nav_goal'}]
    )

    rviz_node = ExecuteProcess(
            cmd=['rviz2', 'rviz2', '-d', os.path.join(navigation_package_path, 'rviz/navigation_controller.rviz')],
            output='screen'
    )


    llm_agent_progress_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('large_models_examples'), 'large_models_examples/function_calling/llm_agent_progress.launch.py')),
    )


    navigation_start_launch = GroupAction(
     actions=[
         PushRosNamespace(robot_name),
         # rviz_node可用另外一个平台启动(rviz node can be started by another platform)
         rviz_node,
         TimerAction(
             period=3.0,
             actions=[
                navigation_launch,
                navigation_controller_node
                ],
         ),
      ]
    )

    if function_mode == 'navigation':
        return[
            map_name_arg,
            master_name_arg,
            robot_name_arg,

            GroupAction(
                actions=[
                    PushRosNamespace(robot_name),
                    # rviz_node可用另外一个平台启动(rviz node can be started by another platform)
                    rviz_node,
                    TimerAction(
                        period=3.0,
                        actions=[
                            llm_agent_progress_launch,
                            line_following_node,
                            navigation_launch,
                            navigation_controller_node
                            ],
                    ),
                ]
        )

        ]
    else: 
        return[
            depth_camera_launch,
            controller_launch,

            lidar_node_launch,
            line_following_node,
            object_tracking_node,

            llm_agent_progress_launch,

        ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'function',
            default_value='default',
            description='The function to execute'),
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
