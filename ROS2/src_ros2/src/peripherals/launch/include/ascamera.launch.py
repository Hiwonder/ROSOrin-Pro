from launch import LaunchDescription
import launch_ros.actions
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(
            namespace="ascamera",
            package='ascamera',
            executable='ascamera_node',
            respawn=True,
            output='both',
            parameters=[
                {"usb_bus_no": -1},
                {"usb_path": "null"},
                {"confiPath": "/home/ubuntu/third_party/camera_third_ws/src/ascamera/configurationfiles"},
                {"color_pcl": True},
                {"pub_tfTree": True},
                {"depth_width": 640},
                {"depth_height": 480},
                {"rgb_width": 640},
                {"rgb_height": 480},
                {"fps": 15},
            ],
            remappings=[
                ("/ascamera/camera_publisher/depth0/camera_info", "/depth_cam/depth0/camera_info"),
                ("/ascamera/camera_publisher/depth0/image_raw", "/depth_cam/depth0/image_raw"),
                ("/ascamera/camera_publisher/depth0/points", "/depth_cam/depth0/points"),
                ("/ascamera/camera_publisher/rgb0/camera_info", "/depth_cam/rgb0/camera_info"),
                ("/ascamera/camera_publisher/rgb0/image", "/depth_cam/rgb0/image_raw"),
            ],

        ),
        Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        #arguments = ['0', '0', '0', '-1.57', '0', '-1.57', 'depth_cam', 'ascamera_camera_link_0']
        arguments = ['0', '0', '0', '-1.57', '0', '-1.57', 'camera_link0', 'ascamera_camera_link_0']
        ),
        
        Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        #arguments = ['0', '0', '0', '-1.57', '0', '-1.57', 'depth_cam', 'ascamera_camera_link_0']
        arguments = ['0', '0', '0', '-1.57', '0', '-1.57', 'camera_link0', 'ascamera_color_0']
        ),
      ])

