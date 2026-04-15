#!/usr/bin/env python3
# coding: utf-8
"""
读取 transform.yaml → 计算相机 roll/pitch/yaw → 提供 Trigger 服务返回
"""

import math, yaml, numpy as np, rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

def plane_to_rpy(plane):
    """根据平面法向量求 roll / pitch / yaw（ZX-Y 形式，yaw 置 0）"""
    nx, ny, nz = plane[:3] / np.linalg.norm(plane[:3])
    # 绕 X 把 ny 消成 0
    roll  = math.degrees(math.atan2(ny, nz))
    nz_   =  math.cos(math.radians(roll)) * nz + math.sin(math.radians(roll)) * ny
    # 再绕 Y 把 nx 消成 0
    pitch = math.degrees(math.atan2(-nx, nz_))
    yaw   = 0.0
    return roll, pitch, yaw

class CameraOffsetNode(Node):
    def __init__(self):
        super().__init__('camera_offset_node')

        # ① 声明 transform_file
        self.declare_parameter(
            'transform_file',
            '/home/ubuntu/ros2_ws/src/example/example/rgbd_function/config/transform.yaml')

        # ★ ② 先声明 3 个偏差参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_roll_offset',  0.0),
                ('camera_pitch_offset', 0.0),
                ('camera_yaw_offset',   0.0),
            ])

        # ③ 然后再更新
        self.update_offsets()

        # 提供 Trigger 服务
        self.create_service(Trigger, 'get_camera_offset', self.handle_trigger)

        # 若想在文件改动后自动刷新，可开个 timer 周期性调用 self.update_offsets()

    # ------------------------------------------------------------
    def update_offsets(self):
        tf_file = self.get_parameter('transform_file').get_parameter_value().string_value
        with open(tf_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        plane = np.array(data.get('plane', [0,0,1]))
        roll, pitch, yaw = plane_to_rpy(plane)

        # 把结果写进参数服务器
        self.set_parameters([
            rclpy.parameter.Parameter('camera_roll_offset',  rclpy.Parameter.Type.DOUBLE, roll),
            rclpy.parameter.Parameter('camera_pitch_offset', rclpy.Parameter.Type.DOUBLE, pitch),
            rclpy.parameter.Parameter('camera_yaw_offset',   rclpy.Parameter.Type.DOUBLE, yaw)
        ])
        self.get_logger().info(
            f'Updated offsets  roll={roll:.2f}°, pitch={pitch:.2f}°, yaw={yaw:.2f}°')

    # ------------------------------------------------------------
    def handle_trigger(self, req, res):
        roll  = self.get_parameter('camera_roll_offset').value
        pitch = self.get_parameter('camera_pitch_offset').value
        yaw   = self.get_parameter('camera_yaw_offset').value
        res.success = True
        res.message = f'{roll:.2f},{pitch:.2f},{yaw:.2f}'
        return res

def main():
    rclpy.init()
    rclpy.spin(CameraOffsetNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
