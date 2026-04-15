import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context):
    compiled = os.environ['need_compile']
    machine_type = os.environ.get('MACHINE_TYPE')
    start = LaunchConfiguration('start', default='false')
    start_arg = DeclareLaunchArgument('start', default_value=start)
    only_line_follow = LaunchConfiguration('only_line_follow', default='false')
    only_line_follow_arg = DeclareLaunchArgument('only_line_follow', default_value=only_line_follow)


    conf = LaunchConfiguration('conf', default=0.75)
    conf_arg = DeclareLaunchArgument('conf', default_value=conf)
    model_choice = LaunchConfiguration('model', default='yolo26')
    model_arg = DeclareLaunchArgument('model', default_value=model_choice)
    model_choice_str = model_choice.perform(context)
    camera_topic = LaunchConfiguration('camera_topic', default='depth_cam/rgb0/image_raw')
    camera_topic_arg = DeclareLaunchArgument('camera_topic', default_value=camera_topic)


    self_driving_node = Node(
        package='example',
        executable='self_driving',
        output='screen',
        parameters=[{'start': start, 'only_line_follow': only_line_follow, 'use_depth_cam': True}],
    )


    if '11' in model_choice_str:
        model_name = 'best_traffic_11'
    if '26' in model_choice_str:
        model_name = 'best_traffic_26'
    yolo_node = Node(
            package='example',
            executable='yolo_node',
            output='screen',
            parameters=[{
                'image_topic': camera_topic,
                'classes': ['go', 'right', 'park', 'red', 'green', 'crosswalk'],
                'engine': model_name,
                'conf': conf,
                'task': 'detect',
                'display': False}]
    )

    return [start_arg,

            conf_arg,
            model_arg,
            camera_topic_arg,

            only_line_follow_arg,
            # depth_camera_launch,
            # controller_launch,
            yolo_node,
            self_driving_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象(create a LaunchDescription object)
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()

