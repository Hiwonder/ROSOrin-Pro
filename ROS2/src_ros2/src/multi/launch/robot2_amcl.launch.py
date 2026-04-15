import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
	#package_path = get_package_share_directory('yahboomcar_multi')
	#nav2_bringup_dir = get_package_share_directory('nav2_bringup')
	lifecycle_nodes = ['map_server']
	param_file = os.path.join(get_package_share_directory('multi'),'param','robot2_amcl_param.yaml')
	amcl_node = Node(
    	name="robot2_amcl",
        package='nav2_amcl',
        executable='amcl',
        parameters=[os.path.join(get_package_share_directory('multi'),'param','robot2_amcl_param.yaml')],
        remappings=[
			('/initialpose', '/robot_2/initialpose'),
			('amcl_pose', '/robot_2/amcl_pose'),
			('particle_cloud', '/robot_2/particle_cloud'),
			],
        output = "screen"
    )
    
	life_node = Node(
    	name="robot2_amcl_lifecycle_manager",
    	package='nav2_lifecycle_manager',
    	executable='lifecycle_manager',
    	output='screen',
    	parameters=[{'use_sim_time': False},{'autostart': True},{'node_names': ['robot2_amcl']}]
    	)
    	
    
	return LaunchDescription([
    	#lifecycle_nodes,
    	#use_sim_time,\
    	amcl_node,
    	life_node,
    	#base_link_to_laser_tf_node
    ])
    
    
