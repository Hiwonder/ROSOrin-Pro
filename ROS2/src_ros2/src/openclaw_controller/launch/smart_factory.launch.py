import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, OpaqueFunction, TimerAction, ExecuteProcess

def launch_setup(context):
    compiled = os.environ['need_compile']
    if compiled == 'True':
        slam_package_path = get_package_share_directory('slam')
        kinematics_package_path = get_package_share_directory('kinematics')
        navigation_package_path = get_package_share_directory('navigation')
        large_models_examples_package_path = get_package_share_directory('large_models_examples')
    else:
        slam_package_path = '/home/ubuntu/ros2_ws/src/slam'
        navigation_package_path = '/home/ubuntu/ros2_ws/src/navigation'
        kinematics_package_path = '/home/ubuntu/ros2_ws/src/driver/kinematics'
        large_models_examples_package_path = '/home/ubuntu/ros2_ws/src/large_models_examples'

    map_name = LaunchConfiguration('map_name', default='map')
    map_name_arg = DeclareLaunchArgument('map_name', default_value=map_name)

    camera_topic = LaunchConfiguration('camera_topic', default='/depth_cam/rgb0/image_raw')
    camera_topic_arg = DeclareLaunchArgument('camera_topic', default_value=camera_topic)

    file_name = LaunchConfiguration('file_name', default='road_network_factory')
    file_name_arg = DeclareLaunchArgument('file_name', default_value=file_name)

    track_and_grab_node = Node(
        package='openclaw_controller',
        executable='track_and_grab',
        output='screen',
        parameters=[
            {'enable_disp': False},
        ],
    )

    kinematics_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(kinematics_package_path, 'launch/kinematics_node.launch.py')),
    )

    road_network_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('large_models_examples'),
                'large_models_examples/road_network/road_network.launch.py'
            )
        ),
        launch_arguments={
            'map_name': map_name,
            'camera_topic': camera_topic,
            'file_name': file_name,
            'use_yolo_detect': 'false',
        }.items(),
    )

    return [
        map_name_arg,
        camera_topic_arg,
        file_name_arg,

        track_and_grab_node,
        kinematics_launch,
        road_network_launch,
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    # Create a LaunchDescription object. (创建一个LaunchDescription对象)
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
