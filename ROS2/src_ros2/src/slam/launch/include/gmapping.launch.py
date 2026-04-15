import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable,OpaqueFunction
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def launch_setup(context):
    pkg_share = get_package_share_directory('slam') 
    
    compiled = os.environ.get('need_compile', 'False')
    if compiled == 'True':
        slam_package_path = os.path.join(pkg_share)
    else:
        slam_package_path = '/home/ubuntu/ros2_ws/src/slam'

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    scan_topic = LaunchConfiguration('scan_topic')
    scan_topic_arg = DeclareLaunchArgument('scan_topic', default_value='scan')

    map_frame = LaunchConfiguration('map_frame', default='map')
    map_frame_arg = DeclareLaunchArgument('map_frame', default_value=map_frame)

    odom_frame = LaunchConfiguration('odom_frame', default='odom')
    odom_frame_arg = DeclareLaunchArgument('odom_frame', default_value=odom_frame)

    base_frame = LaunchConfiguration('base_frame', default='base_footprint')
    base_frame_arg = DeclareLaunchArgument('base_frame', default_value=base_frame)

    slam_params = RewrittenYaml(
        source_file=os.path.join(slam_package_path, 'config','gmapping.yaml'),
        param_rewrites={
            'use_sim_time': use_sim_time,
            'map_frame': map_frame,
            'odom_frame': odom_frame,
            'base_frame': base_frame,
            },
        convert_types=True
    )

    remappings=[
        ('/tf', 'tf'),
        ('/tf_static', 'tf_static'),
        ('/map', 'map'),
        ('/map_metadata', 'map_metadata'),
        ('scan', scan_topic),
        ('echoes', scan_topic)
    ]

    gmapping_node = Node(
        package='slam_gmapping',
        executable='slam_gmapping',
        name='slam_gmapping',
        output='screen',
        parameters=[
            slam_params,
        ],
        remappings=remappings
    )

    return [
        scan_topic_arg,
        map_frame_arg,
        odom_frame_arg,
        base_frame_arg,
        use_sim_time_arg,

        gmapping_node,
    ]



def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
