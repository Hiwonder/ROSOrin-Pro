#!/usr/bin/python3
# coding=utf8
# Map pixel coordinates from the undistorted (rectified) image back to the original (distorted) image. 将校正后图像的像素点反推回原始（带畸变）图像中的像素点

import numpy as np

def distort_point(x, y, k1, k2, p1, p2, k3=0):
    """
    Compute distorted normalized image coordinates based on the forward distortion model. 根据正向畸变模型计算带畸变归一化坐标
    输入：
      x, y：Normalized coordinates (before distortion). 归一化坐标（原始/待求变量）
      k1, k2, k3：Radial distortion coefficients. 径向畸变参数
      p1, p2：Tangential distortion coefficients. 切向畸变参数
    Returns 输出：
      x_dist, y_dist：Distorted normalized coordinates after applying the distortion model.经过畸变映射后的归一化坐标
    """
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
    x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return x_dist, y_dist

def invert_undistort_point(xu, yu, k1, k2, p1, p2, k3=0, max_iter=100, tol=1e-10):
    """
    Iteratively solve for the original normalized coordinates (x, y) such that their distorted form equals the given rectified
 normalized coordinates (xu, yu). 通过迭代方法求解原始归一化坐标 (x, y)，使得经过正向畸变映射后等于已知校正后的归一化坐标 (xu, yu)
    Initial guess: (x, y) = (xu, yu), assuming distortion is small and they are close. 初始值取 (x, y) = (xu, yu)（假设畸变较小时差异不大）
    """
    x = xu
    y = yu
    for i in range(max_iter):
        # Compute the distorted normalized coordinates from the current estimate. 计算当前估计下的畸变映射结果
        x_dist, y_dist = distort_point(x, y, k1, k2, p1, p2, k3)
        # calculation error 计算误差
        dx = xu - x_dist
        dy = yu - y_dist
        if dx * dx + dy * dy < tol:
            # Exit the loop when the solution converges. 收敛后退出
            break
        # Use a fixed step size update here. A damping factor can be added to improve convergence if needed. 这里采用简单的固定步长更新，可根据实际情况加入阻尼因子改善收敛性
        x += dx
        y += dy
    return x, y

def undistorted_to_distorted_pixel(u_undist, v_undist, camera_matrix, dist_coeffs):
    """
    Use the camera intrinsic matrix and distortion coefficients to map undistorted image pixels back to the corresponding
pixel coordinates in the original distorted image.根据相机内参与畸变系数，将校正后图像的像素点反推回原始（带畸变）图像中的像素点
    
    Input 输入：
      u_undist, v_undist：Pixel coordinates in the undistorted (rectified) image. 校正后图像的像素坐标
      camera_matrix：3x3 intrinsic matrix in the form. 3x3 相机内参矩阵，形如 [[fx, 0, cx],
                                                    [0, fy, cy],
                                                    [0,  0,  1]]
      dist_coeffs：Distortion coefficients [k1, k2, p1, p2, (k3)]. k3 is optional; if not available, only the first 4 parameters are used.
                  畸变系数数组 [k1, k2, p1, p2, (k3)]，若没有 k3 可只传前 4 个参数
      
    Output 输出：
      u_dist, v_dist：Pixel coordinates in the original (distorted) image. 原始图像（带畸变）的像素坐标
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Convert the undistorted pixel coordinates to normalized image coordinates. 将校正后的像素坐标转换为归一化坐标
    xu = (u_undist - cx) / fx
    yu = (v_undist - cy) / fy

    # Extract distortion parameters. 提取畸变参数
    k1 = dist_coeffs[0]
    k2 = dist_coeffs[1]
    p1 = dist_coeffs[2]
    p2 = dist_coeffs[3]
    k3 = 0.0
    if len(dist_coeffs) >= 5:
        k3 = dist_coeffs[4]
        
    # Use an iterative method to compute the original normalized coordinates. 利用迭代法反求原始归一化坐标
    xd, yd = invert_undistort_point(xu, yu, k1, k2, p1, p2, k3)
    
    # Convert the recovered normalized coordinates back to distorted pixel coordinates. 将原始归一化坐标转换为像素坐标
    u_dist = xd * fx + cx
    v_dist = yd * fy + cy
    return int(u_dist), int(v_dist)


