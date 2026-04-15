import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable,OpaqueFunction
from launch_ros.actions import Node

def launch_setup(context):
    pkg_share = get_package_share_directory('slam') 
    
    compiled = os.environ.get('need_compile', 'False')
    if compiled == 'True':
        slam_config_package_path = os.path.join(pkg_share, 'config')
    else:
        slam_config_package_path = '/home/ubuntu/ros2_ws/src/slam/config'

    default_slam_config = 'cartographer_2d.lua'

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    resolution = LaunchConfiguration('resolution')
    resolution_arg = DeclareLaunchArgument('resolution', default_value='0.05')

    publish_period_sec = LaunchConfiguration('publish_period_sec')
    publish_period_sec_arg = DeclareLaunchArgument('publish_period_sec', default_value='1.0')

    configuration_directory = LaunchConfiguration('configuration_directory')
    configuration_directory_arg = DeclareLaunchArgument('configuration_directory', default_value=slam_config_package_path)

    configuration_basename = LaunchConfiguration('configuration_basename')
    configuration_basename_arg = DeclareLaunchArgument('configuration_basename', default_value=default_slam_config)

    scan_topic = LaunchConfiguration('scan_topic')
    scan_topic_arg = DeclareLaunchArgument('scan_topic', default_value='scan')


    remappings=[
        ('/tf', 'tf'),
        ('/tf_static', 'tf_static'),
        ('/map', 'map'),
        ('/map_metadata', 'map_metadata'),
        ('scan', scan_topic),
        ('echoes', scan_topic)
    ]


    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-configuration_directory', configuration_directory,
            '-configuration_basename', configuration_basename
        ],
        remappings=remappings
    )

    cartographer_occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-resolution', resolution,
            '-publish_period_sec', publish_period_sec
        ]
    )

    return [
        resolution_arg,
        scan_topic_arg,
        use_sim_time_arg,
        publish_period_sec_arg,
        configuration_basename_arg,
        configuration_directory_arg,

        cartographer_node,
        cartographer_occupancy_grid_node,
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
