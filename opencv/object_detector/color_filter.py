# object_detector/color_filter.py
"""
颜色过滤模块：生成红色、蓝色掩膜，并支持将白色区域合并到目标颜色中。
"""
import cv2
import numpy as np
from .config import DetectionConfig as cfg


def create_white_mask(hsv_image):
    """生成白色二值掩膜（高亮度、低饱和度）"""
    return cv2.inRange(hsv_image, cfg.WHITE_LOWER, cfg.WHITE_UPPER)


def create_red_mask(hsv_image):
    """生成红色二值掩膜，可合并白色区域（如反光条）"""
    # 红色本身的掩膜（两段 H 合并）
    mask1 = cv2.inRange(hsv_image, cfg.RED_LOWER1, cfg.RED_UPPER1)
    mask2 = cv2.inRange(hsv_image, cfg.RED_LOWER2, cfg.RED_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 如果需要，合并白色
    if cfg.MERGE_WHITE_TO_RED:
        white_mask = create_white_mask(hsv_image)
        red_mask = cv2.bitwise_or(red_mask, white_mask)

    return red_mask


def create_blue_mask(hsv_image):
    """生成蓝色二值掩膜，可合并白色区域（如反光条）"""
    blue_mask = cv2.inRange(hsv_image, cfg.BLUE_LOWER, cfg.BLUE_UPPER)

    if cfg.MERGE_WHITE_TO_BLUE:
        white_mask = create_white_mask(hsv_image)
        blue_mask = cv2.bitwise_or(blue_mask, white_mask)

    return blue_mask


def apply_morphology(mask, image_width):
    """开闭运算去噪"""
    ksize = max(cfg.MIN_KERNEL_SIZE, image_width // cfg.MORPH_KERNEL_DIVIDER)
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask