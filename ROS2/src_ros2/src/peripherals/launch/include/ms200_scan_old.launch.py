#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    lidar_frame = LaunchConfiguration('lidar_frame', default='laser_frame')
    scan_raw = LaunchConfiguration('scan_raw', default='scan_raw')

    lidar_frame_arg = DeclareLaunchArgument('lidar_frame', default_value=lidar_frame)
    scan_raw_arg = DeclareLaunchArgument('scan_raw', default_value=scan_raw)

    # LiDAR publisher node
    ordlidar_node = Node(
        package='oradar_lidar',
        executable='oradar_scan',
        name='MS200',
        output='screen',
        parameters=[
            {'device_model': 'MS200'},
            {'frame_id': lidar_frame},
            {'scan_topic': 'MS200/scan'},
            {'port_name': '/dev/ldlidar'},
            {'baudrate': 230400},
            {'angle_min': 0.0},
            {'angle_max': 360.0},
            {'range_min': 0.05},
            {'range_max': 25.0},
            {'clockwise': False},
            {'motor_speed': 15}
        ],
        remappings=[('/MS200/scan', scan_raw)] 
    )

    # base_link to laser_frame tf node
    base_link_to_laser_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_base_laser',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser_frame']
    )

    # Define LaunchDescription variable
    ord = LaunchDescription()

    # Add actions to LaunchDescription
    ord.add_action(lidar_frame_arg)
    ord.add_action(scan_raw_arg)
    ord.add_action(ordlidar_node)
    #ord.add_action(base_link_to_laser_tf_node)

    return ord
