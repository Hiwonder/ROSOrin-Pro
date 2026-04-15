import os
from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.actions import IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration 
from launch.actions import DeclareLaunchArgument 
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    compiled = os.environ['need_compile']
    namespace = LaunchConfiguration('namespace', default='')
    namespace_arg = DeclareLaunchArgument('namespace', default_value=namespace)
    frame_prefix = LaunchConfiguration('frame_prefix', default='')
    frame_prefix_arg = DeclareLaunchArgument('frame_prefix', default_value=frame_prefix)

    if compiled == 'True':
        robot_controller_package_path = get_package_share_directory('ros_robot_controller')
        rosorin_description_package_path = get_package_share_directory('rosorin_description')
        peripherals_package_path = get_package_share_directory('peripherals')
    else:
        robot_controller_package_path = '/home/ubuntu/ros2_ws/src/driver/ros_robot_controller'
        rosorin_description_package_path = '/home/ubuntu/ros2_ws/src/simulations/rosorin_description/'
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'

    robot_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(robot_controller_package_path, 'launch/ros_robot_controller.launch.py')]),
        launch_arguments={
            'imu_frame': [namespace, '/imu_frame'],
        }.items()
    )

    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(rosorin_description_package_path, 'launch/robot_description.launch.py')]),
        launch_arguments={
            'frame_prefix': frame_prefix,
            'use_gui': 'false',
            'use_rviz': 'false',
            'use_sim_time': 'false',
            'use_namespace': 'false',
            'namespace': namespace,
        }.items()
    )

    tf_broadcaster_imu_node = Node(
        package='peripherals',
        executable='tf_broadcaster_imu',
        output='screen',
        namespace=namespace,
        parameters=[
            {'imu_topic': 'imu'},
            {'imu_frame': [namespace, '/imu_frame']}, 
            {'imu_link':  [namespace, '/imu_link']}
        ]
    )

    imu_filter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(peripherals_package_path, 'launch/imu_filter.launch.py')]),
        launch_arguments={
            'imu_raw_topic': '/ros_robot_controller/imu_raw',
            'imu_topic': 'imu',
            'namespace': namespace,
        }.items()
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(peripherals_package_path) + '/rviz/imu_view.rviz'],
        output='screen',
    )

    return LaunchDescription([
        namespace_arg,
        frame_prefix_arg,
        robot_controller_launch,
        robot_description_launch,
        tf_broadcaster_imu_node,
        imu_filter_launch,
        rviz_node,
    ])

if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()

