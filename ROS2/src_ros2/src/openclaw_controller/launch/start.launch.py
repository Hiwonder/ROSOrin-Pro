#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription, LaunchService
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context, *args, **kwargs):
    use_yolo = LaunchConfiguration('use_yolo').perform(context)
    stream = LaunchConfiguration('stream').perform(context).lower() == 'true'
    feishu_enable = LaunchConfiguration('feishu_enable').perform(context).lower() == 'true'
    
    openclaw_controller_package_path = get_package_share_directory('openclaw_controller')
    large_models_package_path = get_package_share_directory('large_models')
    
    vocal_detect_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(large_models_package_path, 'launch/vocal_detect.launch.py'
    )),
    )
    
    tts_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
        large_models_package_path, 'launch/tts_node.launch.py'
        )),
    )

    yolo_node = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
        openclaw_controller_package_path,'launch/yolo_node.launch.py'
        )),
    )

    robot_base_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
        openclaw_controller_package_path,'launch/robot_base_control.launch.py'
        )),
    )

    voice_openclaw_node = Node(
        package='openclaw_controller',
        executable='voice_openclaw',
        output='screen',
        parameters=[{
            'stream': stream,
            'feishu_enable': feishu_enable,
        }],
    )
    
    nodes = [
        vocal_detect_node,
        tts_node,
        robot_base_launch,
        voice_openclaw_node,
    ]
    
    if use_yolo == 'true':
        nodes.append(yolo_node)
    
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_yolo',
            default_value='false',
            description='Enable YOLO node'),
        DeclareLaunchArgument(
            'stream',
            default_value='false',
            description='Enable stream mode for voice_openclaw'),
        DeclareLaunchArgument(
            'feishu_enable',
            default_value='false',
            description='Enable feishu message sending'),
        OpaqueFunction(function=launch_setup)
    ])


if __name__ == '__main__':
    ld = generate_launch_description()
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
