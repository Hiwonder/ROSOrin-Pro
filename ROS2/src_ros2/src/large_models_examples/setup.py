import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'large_models_examples'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'large_models_examples'), glob(os.path.join('large_models_examples', '*.*'))),
        (os.path.join('share', package_name, 'large_models_examples', 'navigation_transport'), glob(os.path.join('large_models_examples', 'navigation_transport', '*.*'))),
        (os.path.join('share', package_name, 'large_models_examples', 'transport_dietitianl'), glob(os.path.join('large_models_examples', 'transport_dietitianl', '*.*'))),
        (os.path.join('share', package_name, 'large_models_examples', 'function_calling'), glob(os.path.join('large_models_examples', 'function_calling', '*.*'))),
        (os.path.join('share', package_name, 'large_models_examples', 'road_network'), glob(os.path.join('large_models_examples', 'road_network', '*.*'))),
        (os.path.join('share', package_name, 'large_models_examples', 'function_calling','road_network_llm'), glob(os.path.join('large_models_examples', 'function_calling', 'road_network_llm', '*.*'))),
        (os.path.join('share', package_name, 'large_models_examples', 'color_sorting'), glob(os.path.join('large_models_examples', 'color_sorting', '*.*'))),
        (os.path.join('share', package_name, 'large_models_examples', 'vllm_track_arm'), glob(os.path.join('large_models_examples', 'vllm_track_arm', '*.*'))),
        (os.path.join('share', package_name, 'large_models_examples', 'waste_classification'), glob(os.path.join('large_models_examples', 'waste_classification', '*.*'))),
        (os.path.join('share', package_name, 'large_models_examples', 'object_transport'), glob(os.path.join('large_models_examples', 'object_transport', '*.*'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='2436210442@qq.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigation_controller = large_models_examples.navigation_controller:main',
            'llm_control_move = large_models_examples.llm_control_move:main',
            'llm_control_move_offline = large_models_examples.llm_control_move_offline:main',
            'llm_color_track = large_models_examples.llm_color_track:main',
            'llm_visual_patrol = large_models_examples.llm_visual_patrol:main',
            'vllm_with_camera = large_models_examples.vllm_with_camera:main',
            'vllm_track = large_models_examples.vllm_track:main',
            'vllm_navigation = large_models_examples.vllm_navigation:main',
            # function calling
            'llm_control = large_models_examples.function_calling.llm_control:main',
            # road network llm
            'road_network_navigator = large_models_examples.road_network.road_network_navigator:main',
            'nav2_execution_node = large_models_examples.road_network.nav2_execution_node:main',
            'road_network_tool = large_models_examples.function_calling.road_network_llm.road_network_tool:main',
            
            'vllm_track_arm = large_models_examples.vllm_track_arm.vllm_track_arm:main',
            
            'automatic_pick = large_models_examples.navigation_transport.automatic_pick:main',
            'automatic_transport = large_models_examples.transport_dietitianl.automatic_transport:main',
            'vllm_navigation_transport = large_models_examples.navigation_transport.vllm_navigation_transport:main',
            'vllm_transport_dietitianl = large_models_examples.transport_dietitianl.vllm_transport_dietitianl:main',


            'llm_object_sorting = large_models_examples.color_sorting.llm_object_sorting:main',
            'object_sorting = large_models_examples.color_sorting.object_sorting:main',

            'waste_classification = large_models_examples.waste_classification.waste_classification:main',
            'llm_waste_classification = large_models_examples.waste_classification.llm_waste_classification:main',

            'object_transport = large_models_examples.object_transport.object_transport:main',
            'vllm_object_transport = large_models_examples.object_transport.vllm_object_transport:main',
            
        ],
    },
)
