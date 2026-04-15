import os
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction

def launch_setup(context, *args, **kwargs):
    compiled = os.environ.get('need_compile', 'False')
    namespace = LaunchConfiguration('namespace', default='')
    use_namespace = LaunchConfiguration('use_namespace', default='false')
    action_name = LaunchConfiguration('action_name', default='').perform(context)

    namespace_arg = DeclareLaunchArgument('namespace', default_value=namespace)
    use_namespace_arg = DeclareLaunchArgument('use_namespace', default_value=use_namespace)
    action_name_arg = DeclareLaunchArgument('action_name', default_value=action_name)

    if compiled == 'True':
        controller_package_path = get_package_share_directory('controller')
    else:
        controller_package_path = '/home/ubuntu/ros2_ws/src/driver/controller'
    
    if action_name != '':
        action_param = RewrittenYaml(
            source_file=os.path.join(controller_package_path, 'config/init_pose.yaml'),
            param_rewrites={
                'action_name': action_name,
                },
            convert_types=True
        )
    else:
        action_param = os.path.join(controller_package_path, 'config/init_pose.yaml')
    

    init_pose_node = Node(
        package='controller',
        executable='init_pose',
        name='init_pose',
        namespace=namespace,
        output='screen',
       parameters=[action_param],  
    )

    return [
        namespace_arg,
        use_namespace_arg,
        init_pose_node
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()


