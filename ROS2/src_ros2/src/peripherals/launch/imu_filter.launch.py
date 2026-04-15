import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration 
from launch.actions import DeclareLaunchArgument 

def generate_launch_description():
    compiled = os.environ.get('need_compile', 'False')
    namespace = LaunchConfiguration('namespace', default='')
    namespace_arg = DeclareLaunchArgument('namespace', default_value=namespace)
    if compiled == 'True':
        calibration_package_path = get_package_share_directory('calibration')
    else:
        calibration_package_path = '/home/ubuntu/ros2_ws/src/calibration'
    
    calib_file_path = os.path.join(calibration_package_path, 'config/imu_calib.yaml')
    if not os.path.exists(calib_file_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_file_path}")

    imu_calib_node = Node(
        package='imu_calib',
        executable='apply_calib',
        name='imu_calib',
        namespace=namespace,
        output='screen',
        parameters=[{"calib_file": calib_file_path}],
        remappings=[
            ('raw', 'ros_robot_controller/imu_raw'),
            ('corrected', 'imu_corrected')
            ]
        )

    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        namespace=namespace,
        output='screen',
        parameters=[
            {'fixed_frame': "imu_link",
            'use_mag': False,
            'publish_tf': False,
            'world_frame': "enu",
            'orientation_stddev': 0.05}
        ],
        remappings=[
            ('/tf', '/tf'),
            ('imu/data_raw', 'imu_corrected'),
            ('imu/data', 'imu')
        ]
    )

    return LaunchDescription([
        namespace_arg,
        imu_calib_node,
        imu_filter_node
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象(create a LaunchDescription object)
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()

