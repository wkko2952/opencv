# object_detector/shape_utils.py
import cv2
from .config import DetectionConfig as cfg

def filter_contours(contours, image_area, return_contours=False):
    """
    遍历轮廓，返回所有通过筛选的 (x,y,w,h) 列表
    筛选条件：面积占比、矩形度、宽高比
    """
    valid_rects = []
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / image_area
        if area_ratio < cfg.MIN_AREA_RATIO or area_ratio > cfg.MAX_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        extent = area / (w * h)
        # 宽高比：取 max(w/h, h/w) 兼容 90° 旋转
        aspect_ratio = max(float(w)/h, float(h)/w)

        if abs(aspect_ratio - cfg.TARGET_ASPECT) > cfg.ASPECT_TOL:
            continue
        if extent < cfg.EXTENT_THRESH:
            continue

        valid_rects.append((x, y, w, h))
        valid_contours.append(cnt)
    if return_contours:
        return valid_rects, valid_contours
    return valid_rects
# object_detector/shape_utils.py (只需修改 draw_detections)
def draw_detections(frame, rects, box_color=(0, 255, 0), label="Object"):
    """在画面上绘制检测框和中心点"""
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 3)
        cx, cy = int(x + w/2), int(y + h/2)
        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
    return frame