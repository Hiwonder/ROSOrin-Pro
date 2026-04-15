import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # init param
    compiled = os.environ['need_compile']
    lidar_type = os.environ['LIDAR_TYPE']
    machine_type = os.environ['MACHINE_TYPE']

    # Declare arguments
    lidar_frame = LaunchConfiguration('lidar_frame', default='lidar_frame')
    scan_raw = LaunchConfiguration('scan_raw', default='scan_raw')
    scan_topic = LaunchConfiguration('scan_topic', default='scan')
    

    lidar_frame_arg = DeclareLaunchArgument('lidar_frame',default_value='lidar_frame',description='TF frame ID for the lidar')
    scan_raw_arg = DeclareLaunchArgument('scan_raw',default_value='scan_raw',description='Topic name for lidar_raw scan data')
    scan_topic_arg = DeclareLaunchArgument('scan_topic',default_value='scan',description='Topic name for lidar scan data')

    # Path to the launch file and package directory
    peripherals_package_path = get_package_share_directory('peripherals')
    if lidar_type == 'MS200':
        # lidar_launch_path = os.path.join(peripherals_package_path, 'launch/include/ms200_scan.launch.py')
        lidar_launch_path = os.path.join(peripherals_package_path, 'launch/include/ldlidar_LD19.launch.py')
    elif lidar_type == 'LD19':
        lidar_launch_path = os.path.join(peripherals_package_path, 'launch/include/ldlidar_LD19.launch.py')
    elif lidar_type == 'SCLIDAR':
        lidar_launch_path = os.path.join(peripherals_package_path, 'launch/include/sclidar.launch.py')
    else: 
        lidar_launch_path = os.path.join(peripherals_package_path, 'launch/include/ms200_scan.launch.py')
    # Include Lidar launch file


    lidar_utils_args = []
    laser_filters_config = ''
    if 'Pro' in machine_type:
        # scan_rotated_topic = 'scan_rotated'
        # scan_rotator_node = Node(
        #     package='lidar_utils', # source code in third party code
        #     executable='scan_rotator',
        #     name='scan_rotator_node',
        #     output='screen',
        #     parameters=[{'rotation_degree': 180.0}], # set transform to rotate lidar by 180 deg
        #     remappings=[
        #         ('/scan_raw', scan_raw),
        #         ('/scan_rotated', scan_rotated_topic)
        #     ]
        # )
        # lidar_utils_args.append(scan_rotator_node)
        if lidar_type == 'A1':
            laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_a1.yaml')
        elif lidar_type == 'G4':
            laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_g4.yaml')
        elif lidar_type == 'LD14P':
            laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_ld14p.yaml')
        elif lidar_type == 'LD19':
            laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_ld19.yaml')
        elif lidar_type == 'MS200':
            laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_ms200.yaml')
        elif lidar_type == 'SCLIDAR':
            laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_sclidar.yaml')

        laser_filter_node = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        output='screen',
        parameters=[laser_filters_config],
        remappings=[('scan', scan_raw),
                    ('scan_filtered', scan_topic)]
        )
        lidar_utils_args.append(laser_filter_node)
    else:
        scan_rotated_topic = scan_raw
        scan_raw = scan_topic
        laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_default.yaml')
        

    # pro use scan_raw ,other use scan
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(lidar_launch_path),
        launch_arguments={
            'scan_raw': scan_raw,
            'frame_id': lidar_frame
        }.items()
    )

    return LaunchDescription([
        scan_topic_arg,
        scan_raw_arg,
        lidar_frame_arg,

        lidar_launch,
        ] + 
        lidar_utils_args
    )


if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()

