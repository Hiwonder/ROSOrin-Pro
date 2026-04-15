import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
	robot_name = LaunchConfiguration('robot')
	robot_name_arg = DeclareLaunchArgument('robot',default_value='', )
	#package_path = get_package_share_directory('yahboomcar_multi')
	#nav2_bringup_dir = get_package_share_directory('nav2_bringup')
	lifecycle_nodes = ['map_server']
	# param_file = os.path.join(get_package_share_directory('multi'),'param','robot1_amcl_param.yaml')
	param_file_path = PathJoinSubstitution([get_package_share_directory('multi'),'param',PythonExpression(["'", robot_name, "_amcl_param.yaml'"])])
	amcl_node = Node(
    	name=PythonExpression(["'", robot_name, "_amcl'"]),
        package='nav2_amcl',
        executable='amcl',
        parameters=[param_file_path],
        remappings=[
					('/initialpose', ['/', robot_name, '/initialpose']),
					('amcl_pose', ['/', robot_name, '/amcl_pose']),
					('particle_cloud', ['/', robot_name, '/particle_cloud'])
					# ('/map', '/map'),
					# ('/scan', '/robot_1/scan')

					],
        output = "screen"
    )
    
	life_node = Node(
    	name=PythonExpression(["'", robot_name, "_amcl_lifecycle_manager'"]),
    	package='nav2_lifecycle_manager',
    	executable='lifecycle_manager',
    	output='screen',
    	parameters=[
			{'use_sim_time': False},
			{'autostart': True},
			{'node_names': [PythonExpression(["'", robot_name, "_amcl'"])]}
			]
    	)
    	
    
	return LaunchDescription([
    	#lifecycle_nodes,
    	#use_sim_time,\
		robot_name_arg,
    	amcl_node,
    	life_node,
    	#base_link_to_laser_tf_node
    ])
    
    
