# object_detector/detector.py
"""
检测模块：提供红/蓝物体检测、获取中心点、计算红蓝中点连线等功能。
"""

import cv2
import numpy as np
from .color_filter import create_red_mask, create_blue_mask, apply_morphology
from .shape_utils import filter_contours, draw_detections
from .config import DetectionConfig as cfg


def detect_object(frame, color='red', debug_mask=False):
    """
    检测物体，返回绘制了检测框的图像。
    （兼容旧接口，内部调用 detect_object_with_centers）
    """
    output, _ = detect_object_with_centers(frame, color, debug_mask=False, only_largest=True)
    if debug_mask:
        _, mask, _ = detect_object_with_centers(frame, color, debug_mask=True, only_largest=True)
        return output, mask
    return output


def detect_object_with_centers(frame, color='red', debug_mask=False, only_largest=True, exclude_mask=None):
    """
    检测指定颜色的物体，返回图像和物体中心点。

    :param frame: BGR 图像
    :param color: 'red' 或 'blue'
    :param debug_mask: 若为 True，返回 (输出图像, 掩膜, 中心点列表)
    :param only_largest: 是否只保留面积最大的物体（推荐 True，适用于单个桩桶）
    :param exclude_mask: 可选的排除掩膜（与 frame 同尺寸），排除区域将从颜色掩膜中减去
    :return:
        - debug_mask=False 时: (输出图像, 中心点列表)  中心点为 [(cx,cy), ...]
        - debug_mask=True  时: (输出图像, 掩膜, 中心点列表)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 颜色掩膜与绘制参数
    if color == 'red':
        mask = create_red_mask(hsv)
        box_color = (0, 255, 0)      # 绿色框
        label = "Red Object"
    elif color == 'blue':
        mask = create_blue_mask(hsv)
        box_color = (255, 0, 0)      # 蓝色框
        label = "Blue Object"
    else:
        raise ValueError("color 参数仅支持 'red' 或 'blue'")

    # 减去排除区域（例如红色已检测的区域）
    if exclude_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))

    # 形态学去噪
    mask = apply_morphology(mask, frame.shape[1])

    # 轮廓提取与筛选
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = frame.shape[0] * frame.shape[1]
    valid_rects, valid_contours = filter_contours(contours, image_area, return_contours=True)

    # 只保留面积最大的物体（如桩桶通常只有一个）
    if only_largest and valid_rects:
        largest_idx = max(range(len(valid_rects)), key=lambda i: valid_rects[i][2] * valid_rects[i][3])
        valid_rects = [valid_rects[largest_idx]]
        valid_contours = [valid_contours[largest_idx]]

    # 提取中心点坐标
    centers = [(int(x + w/2), int(y + h/2)) for (x, y, w, h) in valid_rects]

    # 绘制检测结果
    output = frame.copy()
    draw_detections(output, valid_rects, box_color=box_color, label=label)

    if debug_mask:
        # 生成仅包含检测到轮廓的干净掩膜，而非全量形态学掩膜
        clean_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(clean_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
        return output, clean_mask, centers
    return output, centers


def compute_red_blue_midpoint(frame, draw=True):
    """
    检测红、蓝桩桶，计算它们中心连线的中点。

    :param frame: BGR 图像
    :param draw: 是否在返回的图像上绘制连线和中点标记
    :return:
        - midpoint: (mid_x, mid_y) 中点坐标，若任一颜色未检测到则为 None
        - result_img: 绘制后的图像（若 draw=False 则返回原始 frame 的副本）
        当 draw=True 时，连线和中点用黄色绘制。
    """
    # 分别检测红蓝物体，只取最大物体
    # 先检测红色并获取其掩膜，用于排除红色区域（如反光条）在蓝色检测中产生误检
    img_red, red_mask, red_centers = detect_object_with_centers(
        frame, 'red', only_largest=True, debug_mask=True
    )
    img_blue, blue_centers = detect_object_with_centers(
        img_red, 'blue', only_largest=True, exclude_mask=red_mask
    )

    midpoint = None
    result_img = img_red.copy()  # 红框已画

    # 如果红蓝都检测到了
    if red_centers and blue_centers:
        red_c = red_centers[0]
        blue_c = blue_centers[0]
        mid_x = int((red_c[0] + blue_c[0]) / 2)
        mid_y = int((red_c[1] + blue_c[1]) / 2)
        midpoint = (mid_x, mid_y)

        if draw:
            # 画连线 (黄色)
            cv2.line(result_img, red_c, blue_c, (0, 255, 255), 2)
            # 画中点 (黄色实心圆)
            cv2.circle(result_img, midpoint, 8, (0, 255, 255), -1)
            cv2.putText(result_img, "Midpoint", (mid_x + 12, mid_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # 同时画蓝色框（因为 img_red 只画了红框，需要补画蓝框）
            result_img, _ = detect_object_with_centers(
                result_img, 'blue', only_largest=True, exclude_mask=red_mask
            )

    else:
        # 如果只检测到一种颜色，可直接返回红框图像
        result_img = img_red

    return midpoint, result_img


# ---- 保留旧接口兼容性 ----
def detect_red_rectangle(frame, debug_mask=False):
    return detect_object(frame, color='red', debug_mask=debug_mask)

def detect_blue_rectangle(frame, debug_mask=False):
    return detect_object(frame, color='blue', debug_mask=debug_mask)