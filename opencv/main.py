# main.py
"""
主程序：实时检测红/蓝物体，支持按键切换、掩膜调试、实时调参，
以及红蓝桩桶中心连线中点计算与显示。
"""
import cv2
from object_detector import detect_object, compute_red_blue_midpoint
from object_detector.debug_utils import (
    debug_mask_window,
    create_tuning_window,
    update_tuning_params,
    destroy_tuning_window,
    print_current_tuning_params,
)

# ---------- 工具函数 ----------
def safe_destroy_windows(window_names):
    """安全关闭指定名称的窗口（存在才关闭）"""
    for wname in window_names:
        if cv2.getWindowProperty(wname, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(wname)


def main():
    # ---- 摄像头初始化 ----
    cap = cv2.VideoCapture(0)   # 使用 ZED 时替换为 open_zed_camera()
    if not cap.isOpened():
        print("摄像头打开失败")
        return

    # ---- 按键提示 ----
    print("=== 控制按键 ===")
    print("  r - 只检测红色")
    print("  b - 只检测蓝色")
    print("  a - 红蓝同时检测")
    print("  m - 显示 / 隐藏 红蓝中点（需先按 a）")
    print("  d - 显示 / 隐藏 掩膜窗口")
    print("  t - 打开 / 关闭 实时调参窗口")
    print("  s - 打印当前调参参数（需先打开调参窗口）")
    print("  q - 退出程序")
    print("=================")

    # ---- 状态变量 ----
    current_color = 'red'       # 当前检测颜色：'red' 或 'blue'
    show_debug     = False      # 是否显示掩膜窗口
    dual_mode      = False      # 是否同时检测红蓝
    show_midpoint  = False      # 是否显示红蓝中点（仅在 dual_mode=True 时有效）
    tuning_mode    = False      # 是否开启实时调参

    # ---- 主循环 ----
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 实时调参更新（非阻塞）
        if tuning_mode:
            color_idx, size_idx = update_tuning_params()
            if color_idx is None:          # 调参窗口被人为关闭
                tuning_mode = False

        # 2. 物体检测与中点计算
        result = frame.copy()

        if dual_mode:
            if show_midpoint:
                # 使用中点检测函数，自动绘制红蓝框、连线及中点
                midpoint, result = compute_red_blue_midpoint(frame, draw=True)
            else:
                # 仅绘制红蓝框，不显示中点
                result_red = detect_object(frame, color='red')
                result = detect_object(result_red, color='blue')
        else:
            # 单色检测
            result = detect_object(frame, color=current_color)

        # 3. 掩膜调试窗口
        if show_debug:
            if dual_mode:
                debug_mask_window(frame, 'red')
                debug_mask_window(frame, 'blue')
            else:
                debug_mask_window(frame, current_color)

        # 4. 显示主画面
        cv2.imshow("Detection", result)

        # 5. 键盘响应
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):                     # 退出
            break

        elif key == ord('r'):                   # 纯红检测
            current_color = 'red'
            dual_mode = False
            print("→ 红色检测模式")

        elif key == ord('b'):                   # 纯蓝检测
            current_color = 'blue'
            dual_mode = False
            print("→ 蓝色检测模式")

        elif key == ord('a'):                   # 红蓝同时检测
            dual_mode = not dual_mode
            if dual_mode:
                print("→ 红蓝同时检测")
            else:
                print("→ 单色检测（当前颜色：{}）".format(current_color))

        elif key == ord('m'):                   # 切换中点显示
            show_midpoint = not show_midpoint
            if dual_mode:
                print("→ 中点显示：{}".format('开' if show_midpoint else '关'))
            else:
                print("→ 中点显示仅在红蓝模式(按a)下有效")

        elif key == ord('d'):                   # 切换掩膜窗口
            show_debug = not show_debug
            if not show_debug:
                safe_destroy_windows(["Red Mask (Debug)", "Blue Mask (Debug)"])

        elif key == ord('t'):                   # 实时调参窗口
            tuning_mode = not tuning_mode
            if tuning_mode:
                create_tuning_window()
                print("→ 调参窗口已开启，滑动条即时生效")
            else:
                destroy_tuning_window()
                print("→ 调参窗口已关闭")

        elif key == ord('s'):                   # 打印参数
            if tuning_mode:
                print_current_tuning_params()
            else:
                print("→ 请先按 't' 打开调参窗口")

    # ---- 资源释放 ----
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()