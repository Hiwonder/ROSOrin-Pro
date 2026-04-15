#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    serial_port = LaunchConfiguration('serial_port', default='/dev/lidar')
    serial_baudrate = LaunchConfiguration('serial_baudrate', default='230400') 
    frame_id = LaunchConfiguration('frame_id', default='lidar_frame')
    version = LaunchConfiguration('version', default=4)

    lidar_frame = LaunchConfiguration('lidar_frame', default='lidar_frame')
    scan_raw = LaunchConfiguration('scan_raw', default='scan_raw')

    lidar_frame_arg = DeclareLaunchArgument('lidar_frame', default_value=lidar_frame)
    scan_raw_arg = DeclareLaunchArgument('scan_raw', default_value=scan_raw)
    machine_type = os.environ['MACHINE_TYPE']

    PI = 3.1415926
    # Perform a static transformation uniformly, separate from URDF, need to confirm the LiDAR installation method(统一做一次静态变换，和URDF做一次分离，需要确认好激光雷达的安装方式
    statis_tf_trans_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_laser',
            arguments=[
                '0', '0', '0.0',
                '0', '0', '0',
                'lidar_link',
                frame_id
            ]

    )

    return LaunchDescription([

        DeclareLaunchArgument(
            'serial_port',
            default_value=serial_port,
            description='Specifying usb port to connected lidar'),

        DeclareLaunchArgument(
            'serial_baudrate',
            default_value=serial_baudrate,
            description='Specifying usb port baudrate to connected lidar'),
        
        DeclareLaunchArgument(
            'frame_id',
            default_value=frame_id,
            description='Specifying frame_id of lidar'),

        DeclareLaunchArgument(
            'version',
            default_value=version,
            description='Specifying version of lidar'),

        Node(
            package='sclidar_ros2',
            executable='sclidar',
            name='sclidar_scan_publisher',
            parameters=[{'port': serial_port, 
                         'baudrate': serial_baudrate, 
                         'version': version,
                         'frame_id': frame_id}],
            output='screen',
            remappings=[('/scan', scan_raw)] 
        ),
        
        # statis_tf_trans_node,
        lidar_frame_arg,
        scan_raw_arg,
    ])

