#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 定义与第一个文件一致的参数配置 (Define configurations consistent with the first file)
    serial_port = LaunchConfiguration('serial_port', default='/dev/lidar')
    serial_baudrate = LaunchConfiguration('serial_baudrate', default='230400') 
    frame_id = LaunchConfiguration('frame_id', default='base_laser')
    scan_raw = LaunchConfiguration('scan_raw', default='scan')

    # 2. 声明启动参数 (Declare Launch Arguments)
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value=serial_port, description='Specifying usb port')
    
    serial_baudrate_arg = DeclareLaunchArgument(
        'serial_baudrate', default_value=serial_baudrate, description='Specifying baudrate')

    frame_id_arg = DeclareLaunchArgument(
        'frame_id', default_value=frame_id, description='Specifying frame_id')
    
    scan_raw_arg = DeclareLaunchArgument(
        'scan_raw', default_value=scan_raw, description='Specifying scan topic name')

    # LDROBOT LiDAR publisher node
    ldlidar_node = Node(
        package='ldlidar',
        executable='ldlidar',
        name='STL06N',
        output='screen',
        parameters=[
            {'product_name': 'LDLiDAR_STL06N'},
            {'topic_name': 'scan'}, # 内部话题名，后续通过 remappings 映射
            {'frame_id': frame_id},
            {'enable_serial_or_network_communication': True},
            {'port_name': serial_port},     # 适配第一个文件的参数
            {'port_baudrate': serial_baudrate}, # 适配第一个文件的参数
            {'server_ip': '192.168.1.200'},
            {'server_port': '2000'},
            {'laser_scan_dir': True},
            {'enable_angle_crop_func': False},
            {'angle_crop_min': 135.0},
            {'angle_crop_max': 225.0},
            {'measure_point_freq': 5000}
        ],
        remappings=[('scan', scan_raw)] # 适配第一个文件的重映射逻辑
    )

    # base_link to base_laser tf node
    base_link_to_laser_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_base_laser_stl06p',
        # arguments 使用动态的 frame_id
        arguments=['0', '0', '0.18', '0', '0', '0', 'base_link', frame_id]
    )

    # Define LaunchDescription variable
    ld = LaunchDescription()

    # 添加参数声明 (Add Arguments)
    ld.add_action(serial_port_arg)
    ld.add_action(serial_baudrate_arg)
    ld.add_action(frame_id_arg)
    ld.add_action(scan_raw_arg)

    # 添加节点 (Add Nodes)
    ld.add_action(ldlidar_node)
    ld.add_action(base_link_to_laser_tf_node)

    return ld