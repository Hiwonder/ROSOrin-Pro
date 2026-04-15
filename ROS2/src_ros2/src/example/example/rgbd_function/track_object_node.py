#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/12/18
# @author:aiden
# 物体跟踪(Object tracking)
import os
import cv2
import time
import rclpy
import queue
import signal
import threading
import numpy as np
from sdk import pid
import open3d as o3d
import message_filters
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from servo_controller.bus_servo_control import set_servo_position

class TrackObjectNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.pid_x = pid.PID(1.5, 0, 0)
        self.pid_y = pid.PID(1.5, 0, 0)
        self.pid_z = pid.PID(3, 0, 0)
        self.x_speed, self.y_speed, self.z_speed = 0.007, 0.007, 0.04
        self.stop_distance = 0.4
        self.x_stop = -0.1
        self.scale = 4
        self.proc_size = [int(640/self.scale), int(400/self.scale)] 
        self.linear_x, self.linear_y, self.angular = 0, 0, 0
        self.haved_add = False
        self.get_point = False
        self.display = 1
        self.running = True
        self.pc_queue = queue.Queue(maxsize=1)
        self.target_cloud = o3d.geometry.PointCloud() # 要显示的点云(Point cloud to be displayed)
        # 裁剪roi
        # x, y, z
        roi = np.array([
            [-0.7, -0.5, 0],
            [-0.7, 0.3, 0],
            [0.7,  0.3, 0],
            [0.7,  -0.5, 0]], 
            dtype = np.float64)
        # y 近+， x左- (Increase y, decrease x)
        self.vol = o3d.visualization.SelectionPolygonVolume()
        # 裁剪z轴，范围(Clip the z-axis, range)
        self.vol.orthogonal_axis = 'Z'
        self.vol.axis_max = 0.9
        self.vol.axis_min = -0.3
        self.vol.bounding_polygon = o3d.utility.Vector3dVector(roi)

        self.t0 = time.time()
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1) # 舵机控制(Servo control)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)  # 底盘控制(Chassis control)
        
        timer_cb_group = ReentrantCallbackGroup()

        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()

        # self.client = self.create_client(SetBool, '/depth_cam/set_ldp_enable')
        # self.client.wait_for_service()

        camera_name = 'depth_cam'
        rgb_sub = message_filters.Subscriber(self, Image, '/%s/rgb0/image_raw' % camera_name)
        depth_sub = message_filters.Subscriber(self, Image, '/%s/depth0/image_raw' % camera_name)
        info_sub = message_filters.Subscriber(self, CameraInfo, '/%s/depth0/camera_info' % camera_name)
        
        # 同步时间戳, 时间允许有误差在0.02s(Synchronize timestamps, with an allowable time error of 0.02 seconds)
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, info_sub], 3, 0.02)
        sync.registerCallback(self.multi_callback) #执行反馈函数(Execute the feedback function)

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        set_servo_position(self.joints_pub, 1, ((1, 500), (2, 765), (3, 85), (4, 150), (5, 500), (10, 200)))

        # msg = SetBool.Request()
        # msg.data = False
        # self.send_request(self.client, msg)

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def multi_callback(self, ros_rgb_image, ros_depth_image, depth_camera_info):
        try:
            # ros格式转为numpy(Convert ROS format to numpy)
            rgb_image = np.ndarray(shape=(ros_rgb_image.height, ros_rgb_image.width, 3), dtype=np.uint8, buffer=ros_rgb_image.data)
            depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16, buffer=ros_depth_image.data)

            rgb_image = cv2.resize(rgb_image, tuple(self.proc_size), interpolation=cv2.INTER_NEAREST)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            depth_image = cv2.resize(depth_image, tuple(self.proc_size), interpolation=cv2.INTER_NEAREST)
            # self.get_logger().info('\033[1;32m%s\033[0m' % str(depth_camera_info))
            intrinsic = o3d.camera.PinholeCameraIntrinsic(int(depth_camera_info.width / self.scale),
                                                               int(depth_camera_info.height / self.scale),
                                                               int(depth_camera_info.k[0] / self.scale), int(depth_camera_info.k[4] / self.scale),
                                                               int(depth_camera_info.k[2] / self.scale), int(depth_camera_info.k[5] / self.scale))
            o3d_image_rgb = o3d.geometry.Image(rgb_image)
            o3d_image_depth = o3d.geometry.Image(np.ascontiguousarray(depth_image))

            # rgbd_function --> point_cloud
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image_rgb, o3d_image_depth, convert_rgb_to_intensity=False)
            # cpu占用大 (High CPU usage)
            pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)#, extrinsic=extrinsic)ic)
            
            # 裁剪(Crop)
            roi_pc = self.vol.crop_point_cloud(pc)
            
            if len(roi_pc.points) > 0:
                # 去除最大平面，即地面, 距离阈4mm，邻点数，迭代次数(Remove the largest plane, i.e., the ground, with a distance threshold of 4 mm, number of neighbors, and iteration count)
                plane_model, inliers = roi_pc.segment_plane(distance_threshold=0.05,
                         ransac_n=10,
                         num_iterations=50)
                
                # 保留内点(Keep the inliers)
                inlier_cloud = roi_pc.select_by_index(inliers, invert=True)
                self.target_cloud.points = inlier_cloud.points
                self.target_cloud.colors = inlier_cloud.colors
            else:
                self.target_cloud.points = roi_pc.points
                self.target_cloud.colors = roi_pc.colors
            # 转180度方便查看(Rotate 180 degrees for easier viewing)
            self.target_cloud.transform(np.asarray([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
            try:
                self.pc_queue.put_nowait(self.target_cloud)
            except queue.Full:
                pass

            # fps = int(1.0/(time.time() - self.t0))
            # print('\r', 'FPS: ' + str(fps), end='')
        except BaseException as e:
            print('callback error:', e)
        self.t0 = time.time()

    def shutdown(self, signum, frame):
        self.running = False

    def main(self):
        if self.display:
            # 创建可视化窗口(Create a visualization window)
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='point cloud', width=640, height=400, visible=1)
        while self.running:
            if not self.haved_add:
                if self.display:
                    try:
                        point_cloud = self.pc_queue.get(block=True, timeout=2)
                    except queue.Empty:
                        continue
                    vis.add_geometry(point_cloud)
                self.haved_add = True
            if self.haved_add:
                try:
                    point_cloud = self.pc_queue.get(block=True, timeout=2)
                except queue.Empty:
                    continue
                # 刷新(Refresh)
                points = np.asarray(point_cloud.points)
                if len(points) > 0:
                    twist = Twist()
                    min_index = np.argmax(points[:, 2])
                    min_point = points[min_index]
                    if len(point_cloud.colors) < min_index:
                        continue
                    point_cloud.colors[min_index] = [255, 255, 0]
                    
                    # kdtree = o3d.geometry.KDTreeFlann(point_cloud)
                    # 搜索指定坐标点的 10 个最近邻点(Search for the 10 nearest neighbors of the specified coordinate point)
                    # nearest_points = kdtree.search_knn_vector_3d(min_point, 8)
                    # 对 10 个点进行着色(Color the 10 points)
                    # for i in nearest_points[1]:
                        # point_cloud.colors[i] = [255, 255, 0]
                    distance = min_point[-1]
                    self.pid_x.SetPoint = self.stop_distance
                    if abs(distance - self.stop_distance) < 0.1:
                        distance = self.stop_distance
                    self.pid_x.update(-distance)  #更新pid(Update pid)
                    tmp = self.x_speed - self.pid_x.output
                    self.linear_x = tmp
                    if tmp > 0.3:
                        self.linear_x = 0.3
                    if tmp < -0.3:
                        self.linear_x = -0.3
                    if abs(tmp) < 0.008:
                        self.linear_x = 0.0
                    twist.linear.x = float(self.linear_x)
                    if 'Mecanum' in self.machine_type:
                        y_distance = min_point[0]
                        self.pid_y.SetPoint = self.x_stop
                        if abs(y_distance - self.x_stop) < 0.03:
                            y_distance = self.x_stop
                        self.pid_y.update(y_distance)  #更新pid(Update pid)
                        tmp = self.y_speed + self.pid_y.output
                        self.linear_y = tmp
                        if tmp > 0.3:
                            self.linear_y = 0.3
                        if tmp < -0.3:
                            self.linear_y = -0.3
                        if abs(tmp) < 0.008:
                            self.linear_y = 0.0
                        twist.linear.y = float(self.linear_y)
                    else:
                        y_distance = min_point[0]
                        self.pid_y.SetPoint = self.x_stop
                        if abs(y_distance - self.x_stop) < 0.05:
                            y_distance = self.x_stop
                        self.pid_z.update(y_distance)  #更新pid(Update pid)
                        tmp = self.z_speed + self.pid_z.output
                        self.angular = tmp
                        if tmp > 1:
                            self.angular = 1.0
                        if tmp < -1:
                            self.angular = -1.0
                        if abs(tmp) < 0.05:
                            self.angular = 0.0
                        twist.angular.z = float(self.angular)
                    # print(y_distance, self.x_stop, self.angular)
                    # print(min_point)
                    if self.display:
                        vis.update_geometry(point_cloud)
                        vis.poll_events()
                        vis.update_renderer()
                    self.mecanum_pub.publish(twist)
                else:
                    self.mecanum_pub.publish(Twist())

            else:
                time.sleep(0.01)
        self.mecanum_pub.publish(Twist())
        # 销毁所有显示的几何图形(Destroy all displayed geometries" or "Delete all displayed geometric shapes)
        vis.destroy_window()
        self.get_logger().info('\033[1;32m%s\033[0m' % 'shutdown')
        rclpy.shutdown()

def main():
    node = TrackObjectNode('track_object')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()

