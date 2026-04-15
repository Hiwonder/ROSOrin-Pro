#!/usr/bin/env python3
# encoding: utf-8

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Twist
from nav_msgs.msg import Path 
from std_srvs.srv import Trigger
from interfaces.msg import ObjectsInfo
from std_msgs.msg import Bool, Int32
from servo_controller_msgs.msg import ServosPosition, ServoPosition
from servo_controller.bus_servo_control import set_servo_position
from visualization_msgs.msg import Marker, MarkerArray
import time
import math
import yaml
import os
import collections
import numpy as np

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class RoadNetworkNavigator(Node):
    def __init__(self):
        super().__init__('road_network_navigator')

        # --- 1. Parameters and Configuration --- (--- 1. 参数与配置 ---)
        self.declare_parameter('file_name', 'road_network')
        
        self.frame_id = 'map'
        self.base_frame_id = 'base_link'
        self.cmd_vel_topic = '/controller/cmd_vel'
        
        # --- 2. State Variables --- (--- 2. 状态变量 ---)
        self.waypoints = {}        
        self.adjacency_list = {}   
        self.current_index = 0     
        self.target_index = None   
        self.active_goal_index = None 
        self.is_moving = False
        
        # Stores the current dense path for breakpoint recovery (存储当前密集路径用于断点恢复)
        self.current_execution_path = [] 
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.debug_last_reach_time = 0.0
        self.min_reach_interval = 0.5 
        
        self.traffic_signs_status = None 
        self.last_sign_time = 0.0
        self.sign_validity_period = 3.0
        self.waiting_for_light = False     
        self.traffic_light_waypoint_id = 16
        self.stop_signal_triggered = False

        # --- 3. Communication Interfaces ---(--- 3. 通信接口 ---)
        self.path_cmd_pub = self.create_publisher(Path, '/final_path_commands', 1)
        self.nav_pub = self.create_publisher(PoseStamped, '/nav_goal', 1) 
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoint_markers', 1)
        self.mecanum_pub = self.create_publisher(Twist, '%s' % self.cmd_vel_topic, 1)
        self.final_reach_pub = self.create_publisher(Bool, '/road_network_navigator/reach_final', 1)
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)

        # Robotic arm initialization (机械臂初始化)
        time.sleep(0.5) 
        set_servo_position(self.joints_pub, 1, ((10, 500), (5, 500), (4, 200), (3, 50), (2, 750), (1, 500)))  # 初始姿态
        time.sleep(1.5)
        self.get_logger().info("arm initialization complete")
        
        self.create_subscription(Bool, '/navigation_controller/reach_goal', self.reach_callback, 1)
        self.create_subscription(Int32, '/request_waypoint', self.command_callback, 1)

        self.declare_parameter('use_yolo_detect', True)
        self.use_yolo_detect = self.get_parameter('use_yolo_detect').get_parameter_value().bool_value
        self.get_logger().info(f'use_yolo_detect: {self.use_yolo_detect}')
        if self.use_yolo_detect:
            self.create_subscription(ObjectsInfo, '/yolo/object_detect', self.get_object_callback, 1)
            # Start visual recognition service（启动视觉识别服务）
            self.yolo_client = self.create_client(Trigger, '/yolo/start')
            if self.yolo_client.wait_for_service(timeout_sec=10.0):
                self.yolo_client.call_async(Trigger.Request())
                self.get_logger().info("\033[1;32m yolo start\033[0m")
            else:
                self.get_logger().warn("\033[1;32m yolo no start\033[0m")

        # --- 4.Initialization ---（--- 4. 初始化 ---）
        self.load_waypoints_from_file()
        self.create_timer(1.0, self.publish_visualization) # Static road network visualization（静态路网可视化）
        self.create_timer(0.2, self.check_traffic_light_loop) # Traffic light logic（红绿灯逻辑）
        self.create_timer(0.1, self.timer_dynamic_viz_callback) # Dynamic path conversion（动态路径剪裁）

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(self.frame_id, self.base_frame_id,rclpy.time.Time())
            return [trans.transform.translation.x, trans.transform.translation.y]
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def get_object_callback(self, msg):
        """Process detected object information from vision （处理视觉检测到的目标信息）"""
        self.objects_info = msg.objects
        if self.objects_info:
            for i in self.objects_info:
                class_name = i.class_name
                if class_name == 'red' or class_name == 'green':
                    self.traffic_signs_status = i
                    self.last_sign_time = self.get_clock().now().nanoseconds / 1e9
                    # Calculate target bounding box distance （计算目标面积判断距离）
                    # box[0]=xmin, box[1]=ymin, box[2]=xmax, box[3]=ymax
                    width = abs(i.box[2] - i.box[0])
                    height = abs(i.box[3] - i.box[1])
                    area = width * height

                    if class_name == 'red' and area > 2300:
                        self.stop_signal_triggered = True 
                    break


    def check_traffic_light_loop(self):
        """Traffic light logic core: stop and resume (红绿灯逻辑核心：停车与恢复)"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Obtain the latest memorized status and area (获取最新的记忆状态与面积)
        time_since_sign = current_time - self.last_sign_time
        
        memory_status = None
        current_area = 0  

        # Only signals recognized within the last 3 seconds are valid (只有在3秒内 识别过的信号才有效）
        if self.traffic_signs_status and time_since_sign < 3.0:
            memory_status = self.traffic_signs_status.class_name           
            # Calculate area （计算面积） 
            # box: [xmin, ymin, xmax, ymax]
            box = self.traffic_signs_status.box
            width = abs(box[2] - box[0])
            height = abs(box[3] - box[1])
            current_area = width * height
        else:
            memory_status = None 

        # Case A: Currently stopped and waiting（情况 A: 正在停车等待中)
        if self.waiting_for_light:
            self.mecanum_pub.publish(Twist()) # Maintain brake（持续刹车）
            
            self.get_logger().info(f"Waiting for green light... Status: {memory_status}, Area: {int(current_area)}", throttle_duration_sec=1.0)
            if memory_status == 'green':
                self.get_logger().info(">>> Green light detected! Resuming movement.")
                self.waiting_for_light = False
                self.stop_signal_triggered = False
                self.resume_path_from_current_pose()
            return

        # Case B: Determine if a stop is needed while moving（情况 B: 移动中判定是否需要停车）
        if self.is_moving:
            robot_pos = self.get_robot_pose()
            if robot_pos is None: return

            if self.traffic_light_waypoint_id not in self.waypoints: return
            
            light_pose = self.waypoints[self.traffic_light_waypoint_id]
            dx = robot_pos[0] - float(light_pose[0])
            dy = robot_pos[1] - float(light_pose[1])
            dist = math.sqrt(dx**2 + dy**2)

            # Set a larger detection range, combined with visual area to determine stop timing_ units/meters （设置较大检测范围，结合视觉面积判定停车时机 单位/米）
            check_dist_range = 1.5 
            
            if dist < check_dist_range and memory_status == 'red' and self.stop_signal_triggered:
                    self.get_logger().warn(f"!!! RED LIGHT STOP: Area {int(current_area)} (Dist: {dist:.2f}m) !!!")
                    
                    self.waiting_for_light = True
                    self.is_moving = False
                    self.stop_signal_triggered = False
                    
                    self.mecanum_pub.publish(Twist()) # Emergency stop（急停）
                    
                    # Send empty path to cancel controller task （发送空路径取消控制器任务）
                    empty_path = Path()
                    empty_path.header.frame_id = self.frame_id
                    self.path_cmd_pub.publish(empty_path)
                
                

    def load_waypoints_from_file(self):
        file_name = self.get_parameter('file_name').get_parameter_value().string_value
        file_path = '/home/ubuntu/ros2_ws/src/large_models_examples/large_models_examples/road_network/config/' + file_name + '.yaml'
        self.get_logger().info(f"Loading waypoint file: {file_path}")
        if not os.path.exists(file_path):
            self.get_logger().error(f"File not found: {file_path}")
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if 'waypoints' in data:
                    raw_list = data['waypoints']
                    self.waypoints = {} 
                    self.adjacency_list = {}
                    for item in raw_list:
                        nid = item['id']
                        self.waypoints[nid] = item['pose'] 
                        neighbors = item.get('to', [])
                        self.adjacency_list[nid] = neighbors
                    self.get_logger().info(f"Network loaded successfully with {len(self.waypoints)} nodes")
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")

    def build_graph_log(self):
        pass

    def find_shortest_path(self, start_idx, goal_idx):
        """Breadth-First Search for shortest path （广度优先搜索最短路径）"""
        if start_idx == goal_idx: return [start_idx]
        if start_idx not in self.adjacency_list or goal_idx not in self.adjacency_list:
            self.get_logger().error(f"Start {start_idx} or Goal {goal_idx} not in network")
            return None
        queue = collections.deque([(start_idx, [start_idx])])
        visited = set([start_idx])
        while queue:
            current, path = queue.popleft()
            if current not in self.adjacency_list: continue
            for neighbor in self.adjacency_list[current]:
                if neighbor == goal_idx: return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None 

    def command_callback(self, msg):
        target = msg.data
        if target not in self.waypoints:
            self.get_logger().warn(f"Target node ID {target} does not exist!")
            return
        
        self.target_index = target
        self.get_logger().info(f"Received command: Planning from Node {self.current_index} to Node {target}")
        
        if target == self.current_index:
            self.get_logger().info("Already at the target node.")
            msg_finish = Bool(); msg_finish.data = True
            self.final_reach_pub.publish(msg_finish)
            return

        # New task: normal planning and generating a completely new path（新任务：正常规划并生成全新路径）
        self.generate_new_path_sequence()

    def generate_new_path_sequence(self):
        """Plan and generate interpolated path （规划并生成插值路径）"""
        if self.target_index is None: return

        # 1. Plan node list（1. 规划节点列表）
        node_path_list = self.find_shortest_path(self.current_index, self.target_index)
        if not node_path_list or len(node_path_list) < 2:
            self.get_logger().error("No valid path found.")
            self.is_moving = False
            return
        
        # Traffic light truncation logic（红绿灯截断逻辑）
        segment_end_index = self.target_index 

        # Visualization (planned logical path) （可视化 (规划的逻辑路径)）
        self.publish_planned_path(node_path_list)

        # 2. Generate complete dense Path message （2. 生成完整密集 Path 消息）
        full_path_msg = Path()
        full_path_msg.header.frame_id = self.frame_id
        
        #  Traverse nodes to generate interpolation（遍历节点生成插值）
        for i in range(len(node_path_list) - 1):
            start_id = node_path_list[i]
            end_id = node_path_list[i+1]
            
            start_pt = self.waypoints[start_id]
            end_pt = self.waypoints[end_id]     
            
            segment_points = self.create_dense_points(start_pt, end_pt)
            full_path_msg.poses.extend(segment_points)

        self.send_path_to_controller(full_path_msg, segment_end_index)

    def send_path_to_controller(self, path_msg, segment_end_index):
        """Send path to controller and record（发送路径至控制器并记录）"""
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        total_points = len(path_msg.poses)
        if total_points > 0:
            # Save currently sent path for recovery use（保存当前发送的路径，供恢复使用）
            self.current_execution_path = path_msg.poses 
            self.path_cmd_pub.publish(path_msg) 
            self.get_logger().info(f"Path command sent with {total_points} points") 
            self.is_moving = True
            self.active_goal_index = segment_end_index


    def timer_dynamic_viz_callback(self):
        """Real-time tracking and publishing of remaining path visualization line （实时裁减并发布剩余路径可视化线）"""
        # If currently not running a path, or path is empty, clear the line（如果当前没有在跑路径，或者路径为空，就清空线）
        if not self.is_moving or not self.current_execution_path:  
            self.publish_dense_path_marker([]) 
            return

        # 1. Get robot's current position （1. 获取机器人当前位置）
        robot_pos = self.get_robot_pose()
        if robot_pos is None: return

        # 2. Find the path point index closest to the robot （2. 找到离机器人最近的路径点索引）
        rx, ry = robot_pos[0], robot_pos[1]
        min_dist = float('inf')
        closest_idx = 0
        
        # Traverse path to find the point index closest to the robot （遍历路径寻找离机器人最近的点索引）
        for i, pose_stamped in enumerate(self.current_execution_path):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            dist = (px - rx)**2 + (py - ry)**2
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 3. Extract remaining path (from the closest point onward) (3. 截取剩余路径 (从最近点开始往后画))
        # Keep one or two points before the closest point (保留最近点前一两个点)
        start_viz_idx = max(0, closest_idx - 1)
        remaining_poses = self.current_execution_path[start_viz_idx:]

        # 4. Publish visualization Marker (4. 发布可视化 Marker)
        self.publish_dense_path_marker(remaining_poses)

    def publish_dense_path_marker(self, pose_list):
        """
        Function specifically for drawing dense paths (专门用于绘制密集路径的函数)
        """
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.ns = "active_path"
        marker.id = 8888
        marker.type = Marker.LINE_STRIP
        
        # If list is empty or too short, delete Marker (如果列表为空或太短，就删除 Marker)
        if len(pose_list) < 2:
            marker.action = Marker.DELETE
        else:
            marker.action = Marker.ADD

        marker.scale.x = 0.08  # Line width (线宽)
        marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 1.0 

        # Convert PoseStamped list to Point list (将 PoseStamped 列表转换为 Point 列表)
        for pose_stamped in pose_list:
            p = Point()
            p.x = pose_stamped.pose.position.x
            p.y = pose_stamped.pose.position.y
            p.z = 0.05 
            marker.points.append(p)
        
        ma = MarkerArray()
        ma.markers.append(marker)
        self.marker_pub.publish(ma)
        
    def resume_path_from_current_pose(self):
        """Find breakpoint from historical path and resume following (从历史路径中寻找断点并恢复跟随)"""
        robot_pos = self.get_robot_pose()
        if robot_pos is None:
            self.get_logger().error("Cannot get robot pose, breakpoint recovery failed")
            return

        if not self.current_execution_path:
            self.get_logger().warn("No historical path in memory, attempting to replan...")
            self.generate_new_path_sequence()
            return

        # 1. Find index of closest point (1. 寻找最近点的索引)
        min_dist = float('inf')
        closest_idx = 0
        
        # For performance, you could avoid traversing all points, but for safety you traverse all first (这里为了性能，可以不遍历所有点，但为了安全先遍历所有)
        rx, ry = robot_pos[0], robot_pos[1]
        
        # Simple traversal to find closest point (if there are many points, numpy or kdTree is recommended, but hundreds of points is fast with loop) (简单的遍历寻找最近点 (如果点非常多，建议用numpy或KDTree，但几百个点循环很快))
        for i, pose_stamped in enumerate(self.current_execution_path):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            dist = (px - rx)**2 + (py - ry)**2
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 2. Extract path (2. 截取路径)
        # [Strategy] To prevent robot from stepping in place, we take points after the closest point ([策略] 为了防止机器人原地踏步，我们取最近点之后的点)
        # If closest point is already the last point, we've arrived (如果最近点已经是最后一个点，说明到了)
        if closest_idx >= len(self.current_execution_path) - 1:
            self.get_logger().warn("Robot is already near the path end; forcing arrival.")
            self.is_moving = False
            return

        # Slightly offset index forward, e.g., move 2-5 points ahead, giving robot forward tendency (稍微向前偏移索引，比如往后数 2-5 个点，让机器人有向前的趋势)
        start_slice_idx = min(closest_idx + 2, len(self.current_execution_path) - 1)
        
        remaining_poses = self.current_execution_path[start_slice_idx:]
        
        self.get_logger().info(f"Breakpoint recovery: resuming from point {closest_idx}/{len(self.current_execution_path)}")


        recovered_path = Path()
        recovered_path.header.frame_id = self.frame_id
        recovered_path.poses = remaining_poses
        self.send_path_to_controller(recovered_path, self.active_goal_index)

    def create_dense_points(self, p1, p2):
        """Generate path points through linear interpolation (线性插值生成路径点)"""
        points = []
        step = 0.05 
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < 0.01: return [] 

        num = int(dist / step)
        if num < 1: num = 1
        
        # Calculate orientation of travel direction (计算路径行进方向的朝向)
        travel_yaw = math.atan2(dy, dx)
        travel_q = self.euler_to_quaternion(travel_yaw)

        # Check if endpoint has custom angle (check if list has third value) (判断终点是否有自定义角度 (检查列表是否有第3个数))
        final_q = travel_q # Default endpoint orientation equals travel orientation (默认终点朝向等于行进朝向)
        if len(p2) >= 3:
            custom_yaw = float(p2[2]) # Read third number from YAML (读取 YAML 里的第三个数)
            final_q = self.euler_to_quaternion(custom_yaw)

        # Generate intermediate interpolation points (all use travel orientation travel_q) (生成中间插值点 (全部使用行进方向 travel_q))
        for j in range(num):
            t = j / float(num)
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(p1[0]) + dx * t
            pose.pose.position.y = float(p1[1]) + dy * t
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = travel_q[0]
            pose.pose.orientation.y = travel_q[1]
            pose.pose.orientation.z = travel_q[2]
            pose.pose.orientation.w = travel_q[3]
            points.append(pose)
            
        end_pose = PoseStamped()
        end_pose.header.frame_id = self.frame_id
        end_pose.pose.position.x = float(p2[0])
        end_pose.pose.position.y = float(p2[1])
        end_pose.pose.position.z = 0.0

        # Apply final orientation (应用最终朝向)
        end_pose.pose.orientation.x = final_q[0]
        end_pose.pose.orientation.y = final_q[1]
        end_pose.pose.orientation.z = final_q[2]
        end_pose.pose.orientation.w = final_q[3]
        points.append(end_pose)
        return points

    def reach_callback(self, msg):
        """Received /navigation_controller/reach_gpal signal (收到 /navigation_controller/reach_goal 信号)"""
        if not msg.data:
            # 1. First check if it's actively stopped by red light (1. 首先检查是不是因为红灯主动叫停的)
            if self.waiting_for_light:
                self.get_logger().info(">> Goal cancelled during red light wait; pausing recovery.")
                self.is_moving = False  # Ensure state is marked as stopped (确保状态标记为停止)
                return  # Hand control to check_traffic_light_loop (把控制权交给 check_traffic_light_loop)

            # Obstacle failure handling: call new breakpoint recovery function (遇到障碍物失败处理：调用新的断点恢复函数)
            self.get_logger().warn(">>> Goal failed/aborted; attempting [Breakpoint Recovery] ...")
            self.is_moving = False 
            self.resume_path_from_current_pose() # <-- Use new logic (<--- 使用新逻辑)
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        time_diff = current_time - self.debug_last_reach_time
        
        if time_diff < self.min_reach_interval:
            return
        
        self.debug_last_reach_time = current_time
        
        if self.is_moving and self.active_goal_index is not None:
            self.current_index = self.active_goal_index
            self.get_logger().info(f">>> Reached segment end: Node {self.current_index}")
            
            # Final point check (终点检查)
            if self.current_index == self.target_index:
                self.get_logger().info("\033[1;32m=== Task Completed: Reached final target ===\033[0m")
                self.is_moving = False
                self.target_index = None
                self.active_goal_index = None
                self.publish_planned_path([]) 
                finish_msg = Bool()
                finish_msg.data = True
                self.final_reach_pub.publish(finish_msg)
            else:
                self.get_logger().info("Segment finished, planning remaining path...")
                # Normally reached intermediate point, continue next segment (正常到达中间点，继续下一段)
                self.generate_new_path_sequence()

    def publish_planned_path(self, path_indices):
        """Publish logical planned path (发布逻辑规划路径)"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.ns = "active_path"
        marker.id = 8888
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD if len(path_indices) > 1 else Marker.DELETE
        marker.scale.x = 0.08 
        marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 1.0 

        for idx in path_indices:
            if idx in self.waypoints:
                pt_data = self.waypoints[idx]
                p = Point(x=float(pt_data[0]), y=float(pt_data[1]), z=0.05) 
                marker.points.append(p)
        
        ma = MarkerArray()
        ma.markers.append(marker)
        self.marker_pub.publish(ma)

    def publish_visualization(self):
        """
        Draw global road network nodes and connections (绘制全局路网节点和连线)
        1. Draw connections (LINE_LIST) (1. 绘制连线 (LINE_LIST))
        2. Draw red solid circles (SPHERE_LIST) (2. 绘制红色实心圆 (SPHERE_LIST))
        3. Draw node numbers (TEXT_VIEW_FACING) (3. 绘制节点编号 (TEXT_VIEW_FACING))
        """
        if not self.waypoints: return
        marker_array = MarkerArray()

        # Draw road network connections (Green Lines) (绘制路网连线 (Green Lines))
        edge_marker = Marker()
        edge_marker.header.frame_id = self.frame_id
        edge_marker.ns = "graph_edges"
        edge_marker.id = 9999
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.05
        edge_marker.color.r = 0.0 
        edge_marker.color.g = 1.0 
        edge_marker.color.b = 0.0 
        edge_marker.color.a = 0.6
        
        for start_id, neighbors in self.adjacency_list.items():
            start_pose = self.waypoints[start_id]
            for end_id in neighbors:
                if end_id in self.waypoints:
                    end_pose = self.waypoints[end_id]
                    p1 = Point(x=float(start_pose[0]), y=float(start_pose[1]), z=0.0)
                    p2 = Point(x=float(end_pose[0]), y=float(end_pose[1]), z=0.0)
                    edge_marker.points.append(p1)
                    edge_marker.points.append(p2)
        marker_array.markers.append(edge_marker)

        # Draw numbers (Text) & fill solid circle points (绘制编号 (Text) & 填充圆球点) 
        for nid, pose_data in self.waypoints.items():

            # === A. Create red flat disk (simulate with cylinder) === (=== A. 创建红色扁平圆盘 (使用圆柱体模拟) ===)
            disk_marker = Marker()
            disk_marker.header.frame_id = self.frame_id
            disk_marker.ns = "node_disks"  # Namespace (命名空间)
            disk_marker.id = nid           
            disk_marker.type = Marker.CYLINDER # Use cylinder (使用圆柱体)
            disk_marker.action = Marker.ADD
            
            # Set position (设置位置)
            disk_marker.pose.position.x = float(pose_data[0])
            disk_marker.pose.position.y = float(pose_data[1])
            disk_marker.pose.position.z = 0.01 # Slightly above ground to prevent overlap flicker (稍微离地一点点，防止重叠闪烁)
            
            # Set orientation (default upright) （设置方向 (默认竖直朝上即可））
            disk_marker.pose.orientation.w = 1.0
            
            # Set dimensions (flatten cylinder) （设置尺寸 (把圆柱体压扁））
            disk_marker.scale.x = 0.10  # 直径
            disk_marker.scale.y = 0.10  # 直径
            disk_marker.scale.z = 0.01  
            
            # Set color (red) （设置颜色 （红色））
            disk_marker.color.r = 1.0
            disk_marker.color.g = 0.0
            disk_marker.color.b = 0.0
            disk_marker.color.a = 1.0

            marker_array.markers.append(disk_marker)
           
           
            #  === B. Create text numbers === (=== B. 创建文字编号 ===)
            text_marker = Marker()
            text_marker.header.frame_id = self.frame_id
            text_marker.ns = "node_ids"
            text_marker.id = nid
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.text = str(nid)
            text_marker.scale.z = 0.10 
            text_marker.color.r = 1.0 
            text_marker.color.g = 0.0 
            text_marker.color.b = 0.0
            text_marker.color.a = 1.0
            text_marker.pose.position.x = float(pose_data[0])
            text_marker.pose.position.y = float(pose_data[1])+0.1
            text_marker.pose.position.z = 0.05
            marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)

    def euler_to_quaternion(self, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return [0.0, 0.0, sy, cy]

def main(args=None):
    rclpy.init(args=args)
    node = RoadNetworkNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
