#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/01/31
# @author:aiden
# Check whether the position has changed. 判断位置是否发生位置变化
import math
import numpy as np

def calculate_e_distance(point1, point2):
    # Compute the Euclidean distance between two points. 计算两个点间的欧式距离
    e_distance = int(round(math.sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))))
        
    return e_distance

def position_change_or_not(last_point, current_points, distance):
    # Compare the previous point with all current points to find a match within the threshold distance — treat it as unchanged. 将上一次的某点与当前所有点对比，检测是否有距离符合设定的点，即当作位置没有改变的点
    for p in current_points:
        if last_point[0] == p[0]:
            dis = calculate_e_distance(last_point[2], p[2])
            if dis < distance:
                current_points.remove(p)
                p[0] = last_point[0]
                p[1] = last_point[1]
                return False, p, current_points
    
    return True, None, current_points
    
def position_reorder(current_points, last_points, distance):
    # Distance is measured in pixels. distance 单位像素
    # Compare previous and current positions. iI unchanged, retain the same color label, otherwise reassign a new label starting from 1. 对比上一次和当前点的位置，如果位置没有改变，则相应的颜色标签不变，否则从1开始重新安排标签序号
    new_points = []
    haved_change_points = []
    for p in last_points:  # Compare all previous points with current points to detect unchanged positions. 对上一次的所有点和当前点进行位置对比
        res, not_change_point, haved_change_points = position_change_or_not(p, current_points, distance)   
        if not res:  # If a point hasn't changed, record it as part of the newly ordered list of points. 如果没有改变，就将此点记录下来作为重新排序的新点
            new_points.extend([not_change_point])
    if haved_change_points != [] and new_points != []:
        names = np.array(new_points, dtype=object)[:, 0].tolist()
        index = np.array(new_points, dtype=object)[:, 1].tolist()
        combined = [name + str(index) for name, index in zip(names, index)]
        for p in haved_change_points:
            new_index = 0
            while True:
                new_index += 1
                new_name = p[0] + str(new_index)
                if new_name not in combined:
                    p[1] = new_index
                    new_points.extend([p])
                    combined.append(new_name)
                    break

    return new_points

