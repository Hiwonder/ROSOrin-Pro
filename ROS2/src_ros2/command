colcon build --event-handlers  console_direct+  --cmake-args  -DCMAKE_BUILD_TYPE=Release --symlink-install
🌸
#关闭所有ros节点
~/.stop_ros.sh

#编译
cd ~/ros2_ws && ~/.build.sh
#colcon build --event-handlers  console_direct+  --cmake-args  -DCMAKE_BUILD_TYPE=Release --symlink-install
#单独编译某个包
colcon build --event-handlers  console_direct+  --cmake-args  -DCMAKE_BUILD_TYPE=Release --symlink-install --packages-select xxx


🌸
#######################################################
#线速度校准(ROSOrin_Mecanum, ROSOrin_Acker)
ros2 launch calibration linear_calib.launch.py

#角速度校准(ROSOrin_Mecanum)
ros2 launch calibration angular_calib.launch.py

#imu校准
ros2 launch ros_robot_controller ros_robot_controller.launch.py
ros2 run imu_calib do_calib --ros-args -r imu:=/ros_robot_controller/imu_raw --param output_file:=/home/ubuntu/ros2_ws/src/calibration/config/imu_calib.yaml

#查看imu校准效果
ros2 launch peripherals imu_view.launch.py

#深度摄像头RGB图像可视化(点击topic)
ros2 launch peripherals depth_camera.launch.py
rviz2


#雷达数据可视化
ros2 launch peripherals lidar_view.launch.py
#######################################################


🌸
#######################################################
#雷达功能
ros2 launch app lidar_node.launch.py debug:=true
#雷达避障(ROSOrin_Mecanum, ROSOrin_Acker)
ros2 service call /lidar_app/enter std_srvs/srv/Trigger {}
ros2 service call /lidar_app/set_running interfaces/srv/SetInt64 "{data: 1}"

#雷达跟随(ROSOrin_Mecanum, ROSOrin_Acker)
ros2 service call /lidar_app/enter std_srvs/srv/Trigger {}
ros2 service call /lidar_app/set_running interfaces/srv/SetInt64 "{data: 2}"

#雷达警卫(ROSOrin_Mecanum)
ros2 service call /lidar_app/enter std_srvs/srv/Trigger {}
ros2 service call /lidar_app/set_running interfaces/srv/SetInt64 "{data: 3}"

#巡线
ros2 launch app line_following_node.launch.py debug:=true
ros2 service call /line_following/enter std_srvs/srv/Trigger {}
#鼠标左键点击画面取色
ros2 service call /line_following/set_running std_srvs/srv/SetBool "{data: True}"

#目标跟踪
ros2 launch app object_tracking_node.launch.py debug:=true
ros2 service call /object_tracking/enter std_srvs/srv/Trigger {}
#鼠标左键点击画面取色
ros2 service call /object_tracking/set_running std_srvs/srv/SetBool "{data: True}"
#######################################################


🌸
##################### MediaPipe 1 ##################################
ros2 launch peripherals depth_camera.launch.py

#二维码生成
cd ~/ros2_ws/src/example/example/qrcode && python3 qrcode_creater.py
#二维码检测
cd ~/ros2_ws/src/example/example/qrcode && python3 qrcode_detecter.py

#人脸检测
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 face_detect.py

#人脸网格
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 face_mesh.py

#手关键点检测
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 hand.py

#肢体关键点检测
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 pose.py

#背景分割
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 self_segmentation.py

#整体检测
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 holistic.py

#3D物体检测
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 objectron.py

#指尖轨迹识别
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 hand_gesture.py

#颜色识别
cd ~/ros2_ws/src/example/example/color_detect && python3 color_detect_demo.py
##################################

🌸
##################### MediaPipe 2 ####################
# 肢体体感控制
ros2 launch example body_control.launch.py

# 人体融合RGB控制
ros2 launch example body_and_rgb_control.launch.py

# 人体姿态检测（跌倒检测）
ros2 launch example fall_down_detect.launch.py 

# 人体跟踪
ros2 launch example body_track.launch.py

# AR标签识别
ros2 launch example ar_detect.launch.py
ros2 service call /ar_detect/set_model interfaces/srv/SetString "{data: 'bicycle'}"
#  'bicycle' (自行车)
#  'fox' (狐狸) 
#  'chair' (椅子)
#  'cow' (牛)
#  'wolf' (狼)
#  'rectangle' (简单的立方体框)

#无人驾驶
## 摄像头角度调节
ros2 launch example self_driving.launch.py debug_mode:=true
## 功能启动
ros2 launch example self_driving.launch.py
#######################################################

🌸
#######################################################
#2D建图
ros2 launch slam slam.launch.py

#rviz查看建图效果
ros2 launch slam rviz_slam.launch.py

#键盘控制(可选)
ros2 launch peripherals teleop_key_control.launch.py

#保存地图
#/home/ubuntu/ros2_ws/src/slam/maps/map_01.yaml
cd ~/ros2_ws/src/slam/maps && ros2 run nav2_map_server map_saver_cli -f "map_01" --ros-args -p map_subscribe_transient_local:=true

#cd ~/ros2_ws/src/slam/maps && ros2 run nav2_map_server map_saver_cli -f "保存名称" --ros-args -p map_subscribe_transient_local:=true -r __ns:=/robot_1


# cartographer
ros2 launch slam slam.launch.py slam_method:=cartographer
# gmapping
ros2 launch slam slam.launch.py slam_method:=gmapping

#######################################################


🌸
#######################################################
#2D导航
##rviz发布导航目标
ros2 launch navigation rviz_navigation.launch.py

ros2 launch navigation navigation.launch.py map:=地图名称
#######################################################

🌸
#######################################################
#3D建图(depth_cam)
ros2 launch slam rtabmap_slam.launch.py

#rviz查看建图效果
ros2 launch slam rviz_rtabmap.launch.py

#键盘控制(可选)
ros2 launch peripherals teleop_key_control.launch.py
#######################################################

🌸
#######################################################
#3D导航(depth_cam)
ros2 launch navigation rtabmap_navigation.launch.py

#rviz发布导航目标
ros2 launch navigation rviz_rtabmap_navigation.launch.py
#######################################################

🌸
#######simulations#######
#urdf可视化
ros2 launch rosorin_description display.launch.py
#######################################################

🌸
####### 2D Vision ########
# 手势识别跟随识别(debug)
ros2 launch example hand_trajectory_node.launch.py debug:=true
ros2 service call /hand_trajectory/start std_srvs/srv/Trigger {}

#手势控制 （食指上下左右控制机器人前进后退）
ros2 launch example hand_gesture_control_node.launch.py

########################

🌸
####### 3D Vision ########
# 防跌落debug
ros2 launch example prevent_falling.launch.py debug:=true
# 过独木桥debug
ros2 launch example cross_bridge.launch.py debug:=true
########################

🌸
#################语音控制应用##########################
# 初始化init 测试麦克风
ros2 launch xf_mic_asr_offline mic_init.launch.py

# 更换appid之后需要执行
ros2 launch xf_mic_asr_offline mic_init.launch.py enable_setting:=true

# 语音控制小车移动
ros2 launch xf_mic_asr_offline voice_control_move.launch.py

# 语音控制颜色识别
ros2 launch xf_mic_asr_offline voice_control_color_detect.launch.py

# 语音控制多点导航
ros2 launch xf_mic_asr_offline voice_control_navigation.launch.py map:=map_01
################################################


🌸
########使用上位机software在关闭APP自启范围的情况下需要先打开摄像头服务 ########
ros2 launch peripherals depth_camera.launch.py

#lab_tool
python3 ~/software/lab_tool/main.py

#servo_tool（调整舵机偏差）
python3 ~/software/servo_tool/main.py
########################


🌸
#######large_models_examples#######
#机体控制(支持离线 offline),离线提示词和在线提示词不相同
ros2 launch large_models_examples llm_control_move.launch.py
# ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "前后左右，横向移动"}'
# 离线的提示词 offline
# ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "前进一秒，后退一秒"}'

# 追踪红色物体(支持离线 offline)
ros2 launch large_models_examples llm_color_track.launch.py
#ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "追踪红色物体"}'

# 沿着黑线走，遇到障碍就停下(支持离线 offline)
ros2 launch large_models_examples llm_visual_patrol.launch.py
# ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "沿着黑线走"}'


# 描述下你看到了什么
ros2 launch large_models_examples vllm_with_camera.launch.py
# ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "描述下你看到了什么"}'

# 追踪前面的小球
ros2 launch large_models_examples vllm_track.launch.py
# ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "追踪前面的小球"}'


# 去厨房看看大门有没有关，然后回来告诉我
ros2 launch large_models_examples vllm_navigation.launch.py map:=map_01
#ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "去厨房有什么，然后回来告诉我"}'

########################


#### function_calling ###
🌸
ros2 launch large_models_examples llm_control_progress.launch.py
# ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "给我讲个笑话吧"}'
# ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "沿着黑线走，遇到障碍物就停下，接着向左转，最后给我描述一下你看到了什么"}'

ros2 launch large_models_examples llm_control_progress.launch.py function:=navigation
# ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "去厨房有什么，然后回来告诉我"}'
#

🌸
# 路网规划
## 检查和标记路网
cd /home/ubuntu/ros2_ws/src/large_models_examples/large_models_examples/road_network
python3 road_network_builder.py
## 启动路网规划
ros2 launch large_models_examples road_network.launch.py
## 发送路网标记
ros2 topic pub /request_waypoint std_msgs/msg/Int32 "data: 1" --once

## llm 路网规划
# Pi5 和Jetson 用 Vmware 启动RVIZ 
# ros2 run rviz2 rviz2 -d ~/ros2/src/navigation_road_network.rviz
# ros2 run rviz2 rviz2 -d ~/ros2/src/navigation_road_network_pro.rviz
ros2 launch large_models_examples road_network_tool.launch.py
ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "去公园看看有什么"}'
########################



