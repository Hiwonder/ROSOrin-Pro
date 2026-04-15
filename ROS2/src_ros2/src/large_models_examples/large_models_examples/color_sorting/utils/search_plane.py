#!/usr/bin/env python3
# encoding: utf-8
# Compute the plane equation from three points using the depth camera intrinsics. 通过深度相机的内参和三点来计算平面方程
import time
import random
import numpy as np

class SearchPlane:
    def __init__(self, depth_image_width, depth_image_height, depth_camera_intrinsics, roi_rect=None):
        """
         Initialize the plane search class. 初始化平面搜索类
        Args:
            depth_image_width: Width of the depth image 深度图像宽度
            depth_image_height: Height of the depth image 深度图像高度
            depth_camera_intrinsics: Camera intrinsic parameters 深度相机内参 [fx, fy, cx, cy]
            roi_rect: Region of interest rectangle 感兴趣区域矩形 [x, y, width, height]
        """
        # Camera intrinsics 相机内参
        self.fx = depth_camera_intrinsics[0]  # Focal length in x-direction 焦距x
        self.fy = depth_camera_intrinsics[1]  # Focal length in y-direction 焦距y
        self.cx = depth_camera_intrinsics[2]  # Principal point in x-direction 光心x
        self.cy = depth_camera_intrinsics[3]  # Principal point in y-direction 光心y

        # Image dimensions 图像尺寸
        self.width = depth_image_width
        self.height = depth_image_height

        # Set the ROI. ROI区域设置
        self.roi_rect = roi_rect
        self.roi_mask = np.zeros(shape=[self.height, self.width], dtype=np.uint8)
        if self.roi_rect is not None:
            self.roi_mask[roi_rect[1]:roi_rect[1] + roi_rect[3],
            roi_rect[0]:roi_rect[0] + roi_rect[2]] = 1
        else:
            self.roi_mask[:, :] = 1

    def calculate_distance(self, point1, point2):
        """Compute the Euclidean distance between two points. 计算两点之间的欧氏距离"""
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def estimate_plane(self, points, normalize=True):
        """
        Estimate plane parameters using three 3D points. 使用三个点估计平面参数
        Args:
            points: A 3x3 array of 3D point coordinates. 3x3的点坐标数组
            normalize: Whether to normalize the normal vector. 是否归一化法向量
        Returns:
            Plane parameters. 平面参数 [a, b, c, d] 其中 ax + by + cz + d = 0
        """
        vector1 = points[1, :] - points[0, :]
        vector2 = points[2, :] - points[0, :]

        # Check if the points are collinear. 检查点是否共线
        if not np.all(vector1):
            return None

        # Check whether two vectors are parallel. 检查两个向量是否平行
        direction_ratio = vector2 / vector1
        if not ((direction_ratio[0] != direction_ratio[1]) or
                (direction_ratio[2] != direction_ratio[1])):
            return None

        # Compute the plane's normal vector. 计算平面法向量
        normal = np.cross(vector1, vector2)
        if normalize:
            normal = normal / np.linalg.norm(normal)
        
        # Ensure the normal vector points toward the camera. 确保法向量朝向摄像头
        if normal[2] > 0:  # Assume the camera view direction is [0, 0, 1]. 假设摄像头视线方向为 [0, 0, 1]
            normal = -normal

        # Compute the 'd' parameter in the plane equation. 计算平面方程的d参数
        d = -np.dot(normal, points[0, :])
        return np.append(normal, d)

    def ransac_plane_fit(self, points,
                         distance_threshold=0.05,
                         confidence=0.99,
                         sample_size=3,
                         max_iterations=1000,
                         min_point_distance=0.1):
        """
         Fit a plane using the RANSAC algorithm. 使用RANSAC算法拟合平面
        Args:
            points: Point cloud data 点云数据
            distance_threshold: Inlier distance threshold 内点判定阈值
            confidence: Desired confidence level 置信度
            sample_size: Number of points sampled 采样点数
            max_iterations: Maximum number of RANSAC iterations 最大迭代次数
            min_point_distance: Minimum distance between sampled points 采样点之间的最小距离
        """
        random.seed(time.time())
        best_inlier_count = -999
        iteration_count = 0
        required_iterations = 10
        total_points = len(points)
        point_indices = range(total_points)

        while iteration_count < required_iterations:
            # Randomly sample three points 随机采样三个点
            sampled_indices = random.sample(point_indices, sample_size)

            # Check that sampled points meet the minimum distance requirement. 检查采样点之间的距离是否满足最小距离要求
            if (self.calculate_distance(points[sampled_indices[0]], points[sampled_indices[1]]) < min_point_distance or
                    self.calculate_distance(points[sampled_indices[0]],
                                            points[sampled_indices[2]]) < min_point_distance or
                    self.calculate_distance(points[sampled_indices[1]],
                                            points[sampled_indices[2]]) < min_point_distance):
                continue

            # Estimate the plane parameters 估计平面参数
            plane_params = self.estimate_plane(points[sampled_indices, :])
            if plane_params is None:
                continue

            # Compute distances from all points to the estimated plane 计算所有点到平面的距离
            distances = np.abs(np.dot(points, plane_params[:3]) + plane_params[3])
            inlier_mask = distances < distance_threshold
            inlier_count = np.sum(inlier_mask)

            # Update the best-fitting model 更新最佳模型
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_plane_params = plane_params
                best_inlier_mask = inlier_mask
                best_sample = sampled_indices

                # Update the required number of iterations 更新所需迭代次数
                inlier_ratio = inlier_count / total_points
                required_iterations = int(np.log(1 - confidence) /
                                          np.log(1 - inlier_ratio ** sample_size))

            iteration_count += 1
            if iteration_count > max_iterations:
                return None, None, None

        return np.where(best_inlier_mask)[0], best_plane_params, best_sample

    def find_plane(self, depth_image):
        """
        Search for a dominant plane in the depth image. 在深度图像中寻找平面
        Args:
            depth_image: Depth image with unit in millimeters. 深度图像（单位：毫米）
        Returns:
            inlier_indices: Indices of inlier points on the plane 平面内点的索引
            plane_params: plane parameters 平面参数
            sampled_indices: Indices of the points used to estimate the plane 用于估计平面的采样点索引
        """
        # Apply ROI mask 应用ROI掩码
        if self.roi_rect is not None:
            depth_image = depth_image * self.roi_mask

        # Extract valid depth points 提取有效深度点
        valid_mask = depth_image > 10
        rows, cols = np.where(valid_mask)
        depths = depth_image[valid_mask] / 1000.0  # 转换为米

        # Compute 3D coordinates from depth 计算三维坐标
        points_3d = np.vstack([
            (cols - self.cx) * depths / self.fx,  # X-coordinate X坐标
            (rows - self.cy) * depths / self.fy,  # Y-coordinate Y坐标
            depths  # Z-coordinate Z坐标
        ]).T

        # Fit a plane using RANSAC 使用RANSAC拟合平面
        return self.ransac_plane_fit(points_3d, distance_threshold=0.01)
