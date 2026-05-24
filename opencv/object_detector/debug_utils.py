# object_detector/debug_utils.py
"""
调试工具模块
提供掩膜显示、实时调参窗口（非阻塞）、独立调参窗口（阻塞式）等功能。
支持对红/蓝/白色阈值、形态学核大小、面积和形状参数的实时调整。
"""

import cv2
import numpy as np
from .config import DetectionConfig as cfg

# ---------- 公共辅助 ----------
# 形态学核大小用 DIVIDER 表示，越小核越大；实时调参时直接修改该值
_last_kernel_divider = cfg.MORPH_KERNEL_DIVIDER


# ---------- 掩膜显示 ----------
def debug_mask_window(frame, color='red'):
    """
    显示指定颜色的掩膜窗口（红或蓝），用于观察颜色提取效果。
    注意：显示的掩膜已经包含了白色合并（若配置开启）。
    """
    from .detector import detect_object
    _, mask = detect_object(frame, color=color, debug_mask=True)
    cv2.imshow(f"{color.capitalize()} Mask (Debug)", mask)
    return mask


# ========== 1. 实时非阻塞调参（推荐，与主循环共存） ==========
def create_tuning_window():
    """
    创建调参滑动条窗口（不进入循环）。
    调用后需在主循环中反复调用 update_tuning_params() 实时生效。
    """
    cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tuner", 900, 650)

    # ----- 颜色选择 -----
    cv2.createTrackbar("Color 0=R 1=B", "Tuner", 0, 1, lambda x: None)

    # ----- 红色阈值（两组） -----
    cv2.createTrackbar("R_H1_min", "Tuner", cfg.RED_LOWER1[0], 180, lambda x: None)
    cv2.createTrackbar("R_S1_min", "Tuner", cfg.RED_LOWER1[1], 255, lambda x: None)
    cv2.createTrackbar("R_V1_min", "Tuner", cfg.RED_LOWER1[2], 255, lambda x: None)
    cv2.createTrackbar("R_V1_max", "Tuner", cfg.RED_UPPER1[2], 255, lambda x: None)
    cv2.createTrackbar("R_H2_min", "Tuner", cfg.RED_LOWER2[0], 180, lambda x: None)
    cv2.createTrackbar("R_S2_min", "Tuner", cfg.RED_LOWER2[1], 255, lambda x: None)
    cv2.createTrackbar("R_V2_min", "Tuner", cfg.RED_LOWER2[2], 255, lambda x: None)
    cv2.createTrackbar("R_V2_max", "Tuner", cfg.RED_UPPER2[2], 255, lambda x: None)

    # ----- 蓝色阈值（单组） -----
    cv2.createTrackbar("B_H_min", "Tuner", cfg.BLUE_LOWER[0], 180, lambda x: None)
    cv2.createTrackbar("B_H_max", "Tuner", cfg.BLUE_UPPER[0], 180, lambda x: None)
    cv2.createTrackbar("B_S_min", "Tuner", cfg.BLUE_LOWER[1], 255, lambda x: None)
    cv2.createTrackbar("B_V_min", "Tuner", cfg.BLUE_LOWER[2], 255, lambda x: None)
    cv2.createTrackbar("B_V_max", "Tuner", cfg.BLUE_UPPER[2], 255, lambda x: None)

    # ----- 白色阈值（用于合并到红/蓝掩膜） -----
    cv2.createTrackbar("W_V_min", "Tuner", cfg.WHITE_LOWER[2], 255, lambda x: None)
    cv2.createTrackbar("W_S_max", "Tuner", cfg.WHITE_UPPER[1], 255, lambda x: None)
    cv2.createTrackbar("W_V_max", "Tuner", cfg.WHITE_UPPER[2], 255, lambda x: None)
    # 解释：白色 = H:0~180, S:0~W_S_max, V:W_V_min~W_V_max

    # ----- 形态学核大小 -----
    # 滑动条数值 = MORPH_KERNEL_DIVIDER，范围 10~300
    initial_kernel = cfg.MORPH_KERNEL_DIVIDER
    cv2.createTrackbar("Kernel Div", "Tuner", initial_kernel, 300, lambda x: None)

    # ----- 连续面积占比（无预设，直接调节） -----
    cv2.createTrackbar("Min Area %", "Tuner", int(cfg.MIN_AREA_RATIO * 1000), 200, lambda x: None)
    cv2.createTrackbar("Max Area %", "Tuner", int(cfg.MAX_AREA_RATIO * 1000), 1000, lambda x: None)

    # ----- 形状参数 -----
    cv2.createTrackbar("Aspect Tol*100", "Tuner", int(cfg.ASPECT_TOL * 100), 100, lambda x: None)
    cv2.createTrackbar("Extent *100", "Tuner", int(cfg.EXTENT_THRESH * 100), 100, lambda x: None)

    print("实时调参窗口已创建。滑动条即时生效，按 't' 关闭。")


def update_tuning_params():
    """
    从滑动条读取当前值，更新 cfg 配置（包括白色、核大小等）。
    :return: (color_idx, None) 或 (None, None) 若窗口已关闭
    """
    global _last_kernel_divider

    if cv2.getWindowProperty("Tuner", cv2.WND_PROP_VISIBLE) < 0:
        return None, None

    # 颜色选择
    color_idx = cv2.getTrackbarPos("Color 0=R 1=B", "Tuner")

    # 红色
    r_h1_min = cv2.getTrackbarPos("R_H1_min", "Tuner")
    r_s1_min = cv2.getTrackbarPos("R_S1_min", "Tuner")
    r_v1_min = cv2.getTrackbarPos("R_V1_min", "Tuner")
    r_v1_max = cv2.getTrackbarPos("R_V1_max", "Tuner")
    r_h2_min = cv2.getTrackbarPos("R_H2_min", "Tuner")
    r_s2_min = cv2.getTrackbarPos("R_S2_min", "Tuner")
    r_v2_min = cv2.getTrackbarPos("R_V2_min", "Tuner")
    r_v2_max = cv2.getTrackbarPos("R_V2_max", "Tuner")

    # 蓝色
    b_h_min = cv2.getTrackbarPos("B_H_min", "Tuner")
    b_h_max = cv2.getTrackbarPos("B_H_max", "Tuner")
    b_s_min = cv2.getTrackbarPos("B_S_min", "Tuner")
    b_v_min = cv2.getTrackbarPos("B_V_min", "Tuner")
    b_v_max = cv2.getTrackbarPos("B_V_max", "Tuner")

    # 白色
    w_v_min = cv2.getTrackbarPos("W_V_min", "Tuner")
    w_s_max = cv2.getTrackbarPos("W_S_max", "Tuner")
    w_v_max = cv2.getTrackbarPos("W_V_max", "Tuner")

    # 形态学核分频系数
    kernel_div = cv2.getTrackbarPos("Kernel Div", "Tuner")
    if kernel_div < 1:
        kernel_div = 1

    # 面积 (连续)
    min_area_ratio = cv2.getTrackbarPos("Min Area %", "Tuner") / 1000.0
    max_area_ratio = cv2.getTrackbarPos("Max Area %", "Tuner") / 1000.0
    aspect_tol = cv2.getTrackbarPos("Aspect Tol*100", "Tuner") / 100.0
    extent_thresh = cv2.getTrackbarPos("Extent *100", "Tuner") / 100.0

    # ----- 更新配置对象 -----
    # 红
    cfg.RED_LOWER1 = (r_h1_min, r_s1_min, r_v1_min)
    cfg.RED_UPPER1 = (10, 255, r_v1_max)
    cfg.RED_LOWER2 = (r_h2_min, r_s2_min, r_v2_min)
    cfg.RED_UPPER2 = (180, 255, r_v2_max)
    # 蓝
    cfg.BLUE_LOWER = (b_h_min, b_s_min, b_v_min)
    cfg.BLUE_UPPER = (b_h_max, 255, b_v_max)
    # 白 (H 全范围，S 上限与 V 下限/上限可调)
    cfg.WHITE_LOWER = (0, 0, w_v_min)
    cfg.WHITE_UPPER = (180, w_s_max, w_v_max)
    # 形态学
    cfg.MORPH_KERNEL_DIVIDER = kernel_div
    # 面积与形状
    cfg.MIN_AREA_RATIO = min_area_ratio
    cfg.MAX_AREA_RATIO = max_area_ratio
    cfg.ASPECT_TOL = aspect_tol
    cfg.EXTENT_THRESH = extent_thresh

    return color_idx, None   # 返回 None 大小索引，保留接口兼容


def destroy_tuning_window():
    """安全关闭调参窗口"""
    if cv2.getWindowProperty("Tuner", cv2.WND_PROP_VISIBLE) >= 0:
        cv2.destroyWindow("Tuner")


def print_current_tuning_params():
    """打印当前调参窗口的所有参数（可直接复制到 config.py）"""
    if cv2.getWindowProperty("Tuner", cv2.WND_PROP_VISIBLE) < 0:
        print("调参窗口未打开")
        return

    print("\n========== 当前调参参数 (复制到 config.py) ==========")
    # 颜色
    color_idx = cv2.getTrackbarPos("Color 0=R 1=B", "Tuner")
    print(f"# 当前颜色: {'Red' if color_idx==0 else 'Blue'}")
    # 红
    print(f"RED_LOWER1 = ({cv2.getTrackbarPos('R_H1_min','Tuner')}, {cv2.getTrackbarPos('R_S1_min','Tuner')}, {cv2.getTrackbarPos('R_V1_min','Tuner')})")
    print(f"RED_UPPER1 = (10, 255, {cv2.getTrackbarPos('R_V1_max','Tuner')})")
    print(f"RED_LOWER2 = ({cv2.getTrackbarPos('R_H2_min','Tuner')}, {cv2.getTrackbarPos('R_S2_min','Tuner')}, {cv2.getTrackbarPos('R_V2_min','Tuner')})")
    print(f"RED_UPPER2 = (180, 255, {cv2.getTrackbarPos('R_V2_max','Tuner')})")
    # 蓝
    print(f"BLUE_LOWER = ({cv2.getTrackbarPos('B_H_min','Tuner')}, {cv2.getTrackbarPos('B_S_min','Tuner')}, {cv2.getTrackbarPos('B_V_min','Tuner')})")
    print(f"BLUE_UPPER = ({cv2.getTrackbarPos('B_H_max','Tuner')}, 255, {cv2.getTrackbarPos('B_V_max','Tuner')})")
    # 白
    print(f"WHITE_LOWER = (0, 0, {cv2.getTrackbarPos('W_V_min','Tuner')})")
    print(f"WHITE_UPPER = (180, {cv2.getTrackbarPos('W_S_max','Tuner')}, {cv2.getTrackbarPos('W_V_max','Tuner')})")
    # 形态学
    print(f"MORPH_KERNEL_DIVIDER = {cv2.getTrackbarPos('Kernel Div','Tuner')}")
    # 面积/形状
    print(f"MIN_AREA_RATIO = {cv2.getTrackbarPos('Min Area %','Tuner')/1000:.3f}")
    print(f"MAX_AREA_RATIO = {cv2.getTrackbarPos('Max Area %','Tuner')/1000:.3f}")
    print(f"ASPECT_TOL = {cv2.getTrackbarPos('Aspect Tol*100','Tuner')/100:.2f}")
    print(f"EXTENT_THRESH = {cv2.getTrackbarPos('Extent *100','Tuner')/100:.2f}")
    print("========================================================\n")


# ========== 2. 独立阻塞式调参窗口（备选） ==========
def interactive_tuning():
    """
    独立的阻塞调参窗口（会暂停主程序）。
    与实时调参相同的滑动条，按 's' 打印参数，按 'q' 退出。
    适合快速单独调参，无需运行检测循环。
    """
    cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tuner", 900, 650)

    # 创建滑动条（与 create_tuning_window 完全一致）
    cv2.createTrackbar("Color 0=R 1=B", "Tuner", 0, 1, lambda x: None)
    cv2.createTrackbar("R_H1_min", "Tuner", cfg.RED_LOWER1[0], 180, lambda x: None)
    cv2.createTrackbar("R_S1_min", "Tuner", cfg.RED_LOWER1[1], 255, lambda x: None)
    cv2.createTrackbar("R_V1_min", "Tuner", cfg.RED_LOWER1[2], 255, lambda x: None)
    cv2.createTrackbar("R_V1_max", "Tuner", cfg.RED_UPPER1[2], 255, lambda x: None)
    cv2.createTrackbar("R_H2_min", "Tuner", cfg.RED_LOWER2[0], 180, lambda x: None)
    cv2.createTrackbar("R_S2_min", "Tuner", cfg.RED_LOWER2[1], 255, lambda x: None)
    cv2.createTrackbar("R_V2_min", "Tuner", cfg.RED_LOWER2[2], 255, lambda x: None)
    cv2.createTrackbar("R_V2_max", "Tuner", cfg.RED_UPPER2[2], 255, lambda x: None)
    cv2.createTrackbar("B_H_min", "Tuner", cfg.BLUE_LOWER[0], 180, lambda x: None)
    cv2.createTrackbar("B_H_max", "Tuner", cfg.BLUE_UPPER[0], 180, lambda x: None)
    cv2.createTrackbar("B_S_min", "Tuner", cfg.BLUE_LOWER[1], 255, lambda x: None)
    cv2.createTrackbar("B_V_min", "Tuner", cfg.BLUE_LOWER[2], 255, lambda x: None)
    cv2.createTrackbar("B_V_max", "Tuner", cfg.BLUE_UPPER[2], 255, lambda x: None)
    cv2.createTrackbar("W_V_min", "Tuner", cfg.WHITE_LOWER[2], 255, lambda x: None)
    cv2.createTrackbar("W_S_max", "Tuner", cfg.WHITE_UPPER[1], 255, lambda x: None)
    cv2.createTrackbar("W_V_max", "Tuner", cfg.WHITE_UPPER[2], 255, lambda x: None)
    cv2.createTrackbar("Kernel Div", "Tuner", cfg.MORPH_KERNEL_DIVIDER, 300, lambda x: None)
    cv2.createTrackbar("Min Area %", "Tuner", int(cfg.MIN_AREA_RATIO * 1000), 200, lambda x: None)
    cv2.createTrackbar("Max Area %", "Tuner", int(cfg.MAX_AREA_RATIO * 1000), 1000, lambda x: None)
    cv2.createTrackbar("Aspect Tol*100", "Tuner", int(cfg.ASPECT_TOL * 100), 100, lambda x: None)
    cv2.createTrackbar("Extent *100", "Tuner", int(cfg.EXTENT_THRESH * 100), 100, lambda x: None)

    print("独立调参窗口已启动。按 's' 打印参数，按 'q' 退出。")

    while True:
        # 实时更新 cfg（复用 update 逻辑，但不影响主循环）
        update_tuning_params()

        # 显示当前参数面板
        param_img = np.zeros((550, 900, 3), dtype=np.uint8)
        y = 30
        cv2.putText(param_img, "Independent Tuner (blocks main loop)", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        y += 40
        color_idx = cv2.getTrackbarPos("Color 0=R 1=B", "Tuner")
        cv2.putText(param_img, f"Color: {'Red' if color_idx==0 else 'Blue'}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        y += 25
        cv2.putText(param_img, f"Red L1:({cv2.getTrackbarPos('R_H1_min','Tuner')},{cv2.getTrackbarPos('R_S1_min','Tuner')},{cv2.getTrackbarPos('R_V1_min','Tuner')})", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # … 可继续添加更多显示，此处从简
        cv2.putText(param_img, "Press 's' to print, 'q' to quit", (20, y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Tuner", param_img)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print_current_tuning_params()

    cv2.destroyWindow("Tuner")