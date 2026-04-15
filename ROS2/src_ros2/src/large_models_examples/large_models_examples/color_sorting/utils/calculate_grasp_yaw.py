#!/usr/bin/python3
# coding=utf8
# @Author: Aiden
# @Date: 2024/12/31
"""
Gripper rotation angle calculation module for robotic arm during object grasping.

Angle selection strategy:
1. First, create two mutually perpendicular rectangles centered on the object to represent possible grasping directions.
   These rectangles are parallel to the sides of the object.
2. Check whether these two rectangles intersect with other objects — any intersection indicates a collision.
3. If neither direction results in a collision, choose the one with the smaller rotation angle.
4. If one direction leads to a collision, choose the other.
5. If both directions lead to collisions, grasping is not possible.

机械臂抓取物体时的夹爪旋转角度计算模块

角度选择策略：
1. 首先在物体的中心创建两个互相垂直的矩形代表夹爪可能的抓取方向，两个矩形和物体的两边是平行的
2. 检测这两个矩形是否与其他物体有相交，有相交即发生碰撞
3. 如果都没有碰撞，选择旋转角度较小的方向
4. 如果有一个方向发生碰撞，选择另一个方向
5. 如果都发生碰撞，则无法抓取
"""

import cv2
import math
import numpy as np

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points. 计算两点间的欧氏距离

    Args:
        point1: Coordinates of the first point. 第一个点的坐标 [x, y]
        point2: Coordinates of the second point. 第二个点的坐标 [x, y]

    Returns:
        float: Distance between the two points. 两点间的距离
       """
    return round(math.sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2)), 5)


def calculate_intersection_area(rect1, rect2):
    """Calculate the intersection area of two rotated rectangles, used for collision detection. 计算两个旋转矩形的相交面积，用来判断是否有碰撞

    Args:
        rect1: First rotated rectangle. 第一个旋转矩形 ((x,y), (w,h), angle)
        rect2: Second rotated rectangle. 第二个旋转矩形 ((x,y), (w,h), angle)

    Returns:
        float: Intersection area. Returns None if they do not intersect. 相交面积，如果不相交返回None
    """
    intersection = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
    return cv2.contourArea(intersection) if intersection is not None else None


def detect_collision(target_point, points, rect1, rect2, collision_radius):
    """Check whether the target rectangle collides with surrounding objects by computing the intersection area of rectangles.
    检测目标矩形与周围物体是否存在碰撞，通过计算两个矩形的相交面积来判断

    Args:
        target_point: Pixel position of the target object. 目标物体的像素位置信息
        points: Pixel positions of all objects. 所有物体的位置信息
        rect1: First test rectangle. 检测用的第一个矩形
        rect2: Second test rectangle, perpendicular to rect1. 检测用的第二个矩形(与rect1垂直)
        collision_radius: Radius for collision detection. 碰撞检测半径

    Returns:
        list: List of rectangles that are in collision. Returns None if no collision occurs. 发生碰撞的矩形列表，无碰撞返回None
    """

    collision_rects = []  # Gripper rectangles that would result in a collision. 会发生碰撞的矩形夹取矩形
    
    if len(points) <= 1:
        return collision_rects

    for i in points:  # Iterate through all objects. 遍历所有物体
        if i[0] == target_point[0] and i[1] == target_point[1]:
            continue

        other_point = i[2]  # Coordinates of the object to check collision with the target. 和目标物体进行碰撞检测的物体坐标
        if calculate_distance(target_point[2], other_point) >= collision_radius:  # If the distance between them exceeds the collision detection radius. 如果他们间的距离超过了碰撞检测半径
            continue

        check_rect = (i[2], i[3], i[-1])
        # logger.info(f'check_rect: {check_rect} rect1: {rect1} rect2: {rect2}') 
        # Simplify collision detection by using the minimum bounding rectangle of other objects to compute intersection area with the target's gripper rectangle. 简化碰撞检测，用其他物体的最小外接矩形来和目标物体的夹取矩形进行相交面积判断
        if calculate_intersection_area(rect1, check_rect) is not None:
            collision_rects.append(1)
        if calculate_intersection_area(rect2, check_rect) is not None:
            collision_rects.append(2)

    return collision_rects


def get_parallel_line(rect):
    """
    Line that passes through the center of the rectangle and is parallel to its longer side. 经过矩形中心且与长边平行的线。
    
    Parameters 参数:
        rect: Rectangle parameters. 矩形参数 ((cx, cy), (width, height), angle)
    """
    # Extract rectangle parameters. 提取矩形参数
    (cx, cy), (width, height), angle = rect
    
    # Convert angle to radians. 将角度转换为弧度
    theta = np.radians(angle)
    
    # Compute direction vector along the longer side. 计算方向向量（沿长边方向）
    dx = np.cos(theta) * max(width, height)  # x component of the long side. 长边的 x 分量
    dy = np.sin(theta) * max(width, height)  # y component of the long side. 长边的 y 分量

    # Compute the two endpoints of the line. 计算线的两个端点
    point1 = (int(cx - dx / 2), int(cy - dy / 2))
    point2 = (int(cx + dx / 2), int(cy + dy / 2))
    
    return point1, point2 

def calculate_gripper_yaw_angle(target_points, points, gripper_size, yaw):
    """Calculate the optimal rotation angle for the gripper. 计算最优的夹爪旋转角度

    Args:
        targettarget_points: Pixel position of the target object. 目标物体的像素位置信息
        points: Pixel positions of all objects. 所有物体的像素位置信息,[label, index, (center_x, center_y), [width, height], angle]
        gripper_size: Maximum opening size of the gripper when grasping. 夹爪抓取物体时张的最大时的尺寸, [width, height]
        yaw
    Returns:
        tuple: (Optimal rotation angle, gripper contour points. 最优旋转角度, 夹爪轮廓点)
    """
    center_x, center_y = target_points[2][0], target_points[2][1]  # Pixel position of the target object. 目标物体的像素位置
    angle = target_points[-1] # Rotation angle of the target object is returned by cv2.minAreaRect(). 目标物体的旋转角度，是通过cv2.minAreaRect()返回的角度
    
    # Retrieve gripper size. 获取夹爪尺寸
    gripper_width, gripper_height = gripper_size
    
    # Taking the target's center as the origin, objects farther than this radius will not cause collisions. 以物体中心为圆心，和物体超过这个半径距离的物体是不会碰撞的
    half_width = gripper_width // 2
    half_height = gripper_height // 2
    collision_radius = math.sqrt(half_width ** 2 + half_height ** 2)

    angle1 = angle
    angle2 = angle - 90

    # Create two perpendicular rectangles based on the gripper size. 根据夹爪的尺寸创建两个互相垂直的矩形
    rect1 = ((center_x, center_y), (gripper_width, gripper_height), angle1)
    rect2 = ((center_x, center_y), (gripper_width, gripper_height), angle2)
    
    # Detect which gripper rectangles would cause collisions. 检测会发生碰撞的夹取矩形
    collisions = detect_collision(target_points, points, rect1, rect2, collision_radius)

    line1 = get_parallel_line(rect1)
    line2 = get_parallel_line(rect2)

    yaw1 = yaw + angle1
    if yaw1 < 0:
        yaw2 = yaw1 + 90
    else:
        yaw2 = yaw1 - 90 

    if len(collisions) == 0:
        if abs(yaw1) < abs(yaw2):
            return yaw1, line1, rect1
        else:
            return yaw2, line2, rect2
    elif len(collisions) == 1:
        if collisions[0] == 1:
            return yaw2, line2, rect2
        else:
            return yaw1, line1, rect1
    else:
        return None
