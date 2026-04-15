#!/usr/bin/env python3
# encoding: utf-8
"""
Plane fitting node for Orbbec / Realsense‑style深度相机。
订阅 depth 图 + camera_info，同步取 1 帧做 SVD 平面拟合，
将得到的平面参数  [a, b, c, d]  写入 transform.yaml 的  plane  字段。

在任何包中放置本脚本后：
    ros2 run <your_pkg> plane_fit_node --ros-args \
        -p output_file:=/home/ubuntu/ros2_ws/src/example/config/transform.yaml

默认一次完成即退出，避免长期占用带宽 / 内存。
"""

import os
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import message_filters

__all__ = ["main"]


class PlaneFitNode(Node):
    """Fit a plane to depth data and save it to YAML."""

    def __init__(self):
        super().__init__("plane_fit_node")

        # ---------- 参数 ----------
        # 输出文件；缺省写到当前工作目录
        self.output_file = self.declare_parameter("output_file", "/home/ubuntu/ros2_ws/src/example/example/rgbd_function/config/transform.yaml").value
        # 最大抽样点数，防止一次抓太多像素
        self.max_points = int(self.declare_parameter("max_samples", 30000).value)

        # ---------- 订阅 ----------
        depth_sub = message_filters.Subscriber(
            self, Image, "/ascamera/camera_publisher/depth0/image_raw"
        )
        info_sub = message_filters.Subscriber(
            self, CameraInfo, "/ascamera/camera_publisher/depth0/camera_info"
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, info_sub], queue_size=10, slop=0.05
        )
        self.sync.registerCallback(self.callback)

        self.done = False
        self.get_logger().info(
            "[PlaneFitNode] waiting for the first synchronised depth frame …"
        )

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def depth_to_numpy(msg: Image) -> np.ndarray:
        """Convert a sensor_msgs/Image depth message into metres (float32)."""
        if msg.encoding == "32FC1":
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        elif msg.encoding in ("16UC1", "mono16"):
            depth = (
                np.frombuffer(msg.data, dtype=np.uint16)
                .reshape(msg.height, msg.width)
                .astype(np.float32)
                / 1000.0
            )  # 毫米 ➔ 米
        else:
            raise ValueError(
                f"Unsupported depth encoding: {msg.encoding}; expected 32FC1, 16UC1 or mono16."
            )
        return depth

    def save_plane(self, plane):
        """Merge / create transform.yaml with new plane field."""
        data = {}
        if os.path.isfile(self.output_file):
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception:
                self.get_logger().warn("Existing YAML unreadable — recreating file …")
        data["plane"] = [float(v) for v in plane]
        with open(self.output_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    # ------------------------------------------------------------------
    # 主回调
    # ------------------------------------------------------------------
    def callback(self, depth_msg: Image, info_msg: CameraInfo):
        if self.done:
            return
        # ---- depth → ndarray ----
        try:
            depth = self.depth_to_numpy(depth_msg)
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        # ---- 有效像素掩码： >20 cm —— 去掉飞点与背景零值 ----
        mask = depth > 0.2
        ys, xs = np.where(mask)
        if xs.size < 1500:
            self.get_logger().warn("Not enough depth points; waiting for another frame …")
            return

        # ---- 随机下采样 ----
        if xs.size > self.max_points:
            choice = np.random.choice(xs.size, self.max_points, replace=False)
            xs, ys = xs[choice], ys[choice]
        zs = depth[ys, xs]

        # 保证一维 & float32
        xs = xs.astype(np.float32).reshape(-1)
        ys = ys.astype(np.float32).reshape(-1)
        zs = zs.astype(np.float32).reshape(-1)

        # ---- 像素 → 相机坐标 ----
        fx, fy, cx, cy = (
            info_msg.k[0],  # fx
            info_msg.k[4],  # fy
            info_msg.k[2],  # cx
            info_msg.k[5],  # cy
        )
        x3 = (xs - cx) * zs / fx
        y3 = (ys - cy) * zs / fy
        pts = np.column_stack((x3, y3, zs))  # 形状 (N,3)

        # ---- SVD 平面拟合 ----
        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
        normal = vh[2, :]  # 第 3 行特征向量 = 法向量
        if normal[2] > 0:
            normal = -normal  # Z 指向上方
        d = -float(normal.dot(centroid))
        plane = normal.tolist() + [d]

        # ---- 写入 YAML ----
        self.save_plane(plane)
        self.get_logger().info(f"[PlaneFitNode] plane = {plane}  ➔  {self.output_file}")
        self.done = True
        self.get_logger().info("[PlaneFitNode] done; shutting down.")
        rclpy.shutdown()


# ----------------------------------------------------------------------
# 入口函数
# ----------------------------------------------------------------------

def main():
    rclpy.init()
    _ = PlaneFitNode()
    rclpy.spin(_)


if __name__ == "__main__":
    main()
