#!/usr/bin/env python3

import os
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
    machine_type = os.environ['MACHINE_TYPE']
    if 'Pro' in machine_type:
        lidar_scan_dist_range = [0.1,12.0]
        lidar_exp_angles = [90.0,270.0]
    else:
        lidar_scan_dist_range = [0.05,12.0]
        lidar_exp_angles = [-1.0,-1.0]

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
            {'range_min': lidar_scan_dist_range[0]},
            {'range_max': lidar_scan_dist_range[1]},
            {'clockwise': False},
            {'motor_speed': 15},
            {'expand_angle_occlusion_start': lidar_exp_angles[0]},
            {'expand_angle_occlusion_end': lidar_exp_angles[1]},
        ],
        remappings=[('/MS200/scan', scan_raw)] 
    )


    # Define LaunchDescription variable
    ord = LaunchDescription()

    # Add actions to LaunchDescription
    ord.add_action(lidar_frame_arg)
    ord.add_action(scan_raw_arg)
    ord.add_action(ordlidar_node)
    #ord.add_action(base_link_to_laser_tf_node)

    return ord
