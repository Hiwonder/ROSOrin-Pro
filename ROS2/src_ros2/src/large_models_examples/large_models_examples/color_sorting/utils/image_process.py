#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2022/10/22
import cv2
import math
import numpy as np

range_rgb = {
    'red': (255, 0, 50),
    'blue': (0, 50, 255),
    'green': (0, 255, 50),
    'black': (0, 0, 0),
    'white': (255, 255, 255)
}

class CropLayer(object):
    def __init__(self, params, blobs):
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY,
                                self.startX:self.endX]]

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from utils.plots import colors'

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def show_faces(detect_img, result_img, boxes, landmarks, bbox_color=(0, 255, 0), ll_color=(0, 0, 255)):
    """Draw bounding boxes and face landmarks on image."""
    detect_size = detect_img.shape[:2]
    show_size = result_img.shape[:2]
    for bb, ll in zip(boxes, landmarks):
        p1 = point_remapped(bb[:2], detect_size, show_size, data_type=int)
        p2 = point_remapped(bb[2:4], detect_size, show_size, data_type=int)
        cv2.rectangle(result_img, p1, p2, bbox_color, 2)
        for i, p in enumerate(ll):
            x, y = point_remapped(p, detect_size, show_size, data_type=int)
            cv2.circle(result_img, (x, y), 2, ll_color, 2)
    return result_img


def mp_face_location(results, img):
    h, w, c, = img.shape
    boxes = []
    keypoints = []
    if results.detections:
        for detection in results.detections:
            x_min = detection.location_data.relative_bounding_box.xmin
            y_min = detection.location_data.relative_bounding_box.ymin
            width = detection.location_data.relative_bounding_box.width
            height = detection.location_data.relative_bounding_box.height
            x_min, y_min = max(x_min * w, 0), max(y_min * h, 0)
            x_max, y_max = min(x_min + width * w, w), min(y_min + height * h, h)
            boxes.append((x_min, y_min, x_max, y_max))
            relative_keypoints = detection.location_data.relative_keypoints
            keypoints.append([(point.x * w, point.y * h) for point in relative_keypoints])
    return boxes, keypoints

def draw_tags(image, tags, corners_color=(0, 125, 255), center_color=(0, 255, 0)):
    for tag in tags:
        corners = tag.corners.astype(int)
        center = tag.center.astype(int)
        cv2.putText(image, "%d"%tag.tag_id, (int(center[0] - (7 * len("%d"%tag.tag_id))), int(center[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if corners_color is not None:
            for p in corners:
                cv2.circle(image, tuple(p.tolist()), 5, corners_color, -1)
        if center_color is not None:
            cv2.circle(image, tuple(center.tolist()), 8, center_color, -1)
    return image

def get_area_max_contour(contours, threshold=50):
    """
    Get the contour whose area is the largest. Filter out those whose area is too small. 获取轮廓中面积最重大的一个, 过滤掉面积过小的情况
    :param contours: contour list 轮廓列表
    :param threshold: Area threshold. Contour whose area is less than this value will be filtered out. 面积阈值, 小于这个面积的轮廓会被过滤
    :return: if the maximum contour area is greater than this threshold, return the largest contour, otherwise return None. 如果最大的轮廓面积大于阈值则返回最大的轮廓, 否则返回None
    """
    contour_area_max = 0
    area_max_contour = None

    for c in contours:  # Iterate through all contours. 历遍所有轮廓
        contour_area_temp = math.fabs(cv2.contourArea(c))  # Compute the area of each contour. 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > threshold:  # Filter out noise or irrelevant contours. 过滤干扰
                area_max_contour = c

    return area_max_contour,  contour_area_max  # Return the largest contour. 返回最大的轮廓

def bgr8_to_jpeg(value, quality=75):
    """
    Convert data in the format of cv bgr8 into jpg. 将cv bgr8格式数据转换为jpg格式
    :param value: original data 原始数据
    :param quality:  jpg quality. Maximum value is 100. jpg质量 最大值100
    :return:
    """
    return bytes(cv2.imencode('.jpg', value)[1])

class GetObjectSurface:
    def __init__(self, proto_path, model_path):
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        cv2.dnn_registerLayer("Crop", CropLayer)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Use CUDA as the computation backend. 使用 CUDA 作为后端
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # Set the target to CUDA. 目标为 CUDA

    def adaptive_threshold(self, gray_image):
        # Segment using adaptive threshold, and filter out the side view. 用自适应阈值先进行分割, 过滤掉侧面
        # cv2.ADAPTIVE_THRESH_MEAN_C： All neighboring pixel values have equal weights. 邻域所有像素点的权重值是一致的
        # cv2.ADAPTIVE_THRESH_GAUSSIAN _C ： The wights if each point are related to the distance between each neighboring pixel and the center pixel, and are calculated using the Gaussian function. 与邻域各个像素点到中心点的距离有关，通过高斯方程得到各个点的权重值
        binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        return binary

    def edge_detect(self, bgr_image):
        (H, W) = bgr_image.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr_image, scalefactor=1.0, size=(W, H),
                                     mean=(104.00698794, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)

        self.net.setInput(blob)
        hed = self.net.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        mask = (255 * hed).astype("uint8")
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask = 255 - cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        return mask

    def get_top_surface(self, bgr_image):
        # to extract only the top surface of the object 为了只提取物体最上层表面
        # kernel = np.array([[0, -1, 0],
                       # [-1, 5, -1],
                       # [0, -1, 0]])

        # sharpened = cv2.filter2D(rgb_image, -1, kernel)
        # image_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        # image_mb = cv2.medianBlur(image_gray, 3)  # median filtering 中值滤波

        # binary = self.adaptive_threshold(image_mb)  # adaptive thresholding 阈值自适应
        # image_gs = cv2.GaussianBlur(rgb_image, (5, 5), 5)  # Gaussian blur for noise reduction 高斯模糊去噪
        mask = self.edge_detect(bgr_image)
        # mask1 = cv2.bitwise_and(binary,
                                # mask)  # merge two extracted images and retain their common areas 合并两个提取出来的图像，保留他们共有的地方
        roi_image_mask = cv2.bitwise_and(bgr_image, bgr_image,
                                         mask=mask)  # mask the original image to retain the area to be recognized 和原图做遮罩，保留需要识别的区域
        return roi_image_mask
