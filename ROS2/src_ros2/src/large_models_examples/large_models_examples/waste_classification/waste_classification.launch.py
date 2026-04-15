import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context):
    compiled = os.environ['need_compile']
    start = LaunchConfiguration('start', default='false')
    start_arg = DeclareLaunchArgument('start', default_value=start)
    display = LaunchConfiguration('display', default='true')
    display_arg = DeclareLaunchArgument('display', default_value=display)
    start = LaunchConfiguration('start', default='false')
    start_arg = DeclareLaunchArgument('start', default_value=start)

    conf = LaunchConfiguration('conf', default=0.60)
    conf_arg = DeclareLaunchArgument('conf', default_value=conf)
    model_choice = LaunchConfiguration('model', default='yolo26')
    model_arg = DeclareLaunchArgument('model', default_value=model_choice)
    model_choice_str = model_choice.perform(context)
    camera_topic = LaunchConfiguration('camera_topic', default='depth_cam/rgb0/image_raw')
    camera_topic_arg = DeclareLaunchArgument('camera_topic', default_value=camera_topic)

    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
        controller_package_path = get_package_share_directory('controller')
        kinematics_package_path = get_package_share_directory('kinematics')
        example_package_path = get_package_share_directory('example')
        app_package_path = get_package_share_directory('app')
    else:
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'
        controller_package_path = '/home/ubuntu/ros2_ws/src/driver/controller'
        kinematics_package_path = '/home/ubuntu/ros2_ws/src/driver/kinematics'
        example_package_path = '/home/ubuntu/ros2_ws/src/example'
        app_package_path = '/home/ubuntu/ros2_ws/src/app'
        

    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, 'launch/controller.launch.py')),
    )

    kinematics_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(kinematics_package_path, 'launch/kinematics_node.launch.py')),
    )



    if '11' in model_choice_str:
        model_name = 'best_garbage_11'
    if '26' in model_choice_str:
        model_name = 'best_garbage_26'

    yolo_node = Node(
            package='example',
            executable='yolo_node',
            output='screen',
            parameters=[{
                'image_topic': camera_topic,
                'classes':  ['BananaPeel','BrokenBones','CigaretteEnd','DisposableChopsticks','Ketchup','Marker','OralLiquidBottle','Plate','PlasticBottle','StorageBattery','Toothbrush', 'Umbrella'],
                'engine': model_name,
                'conf': conf,
                'task': 'obb',
                'display': False}]
    )

    waste_classification_node = Node(
        package='large_models_examples',
        executable='waste_classification',
        output='screen',
        parameters=[ {'start': start, 'display': display, 'app': False}],
    )

    return [
            start_arg,
            display_arg,
            
            conf_arg,
            model_arg,
            camera_topic_arg,
            
            depth_camera_launch,
            controller_launch,
            kinematics_launch,
            yolo_node,
            waste_classification_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    # Create a LaunchDescription object. 创建一个LaunchDescription对象
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()


