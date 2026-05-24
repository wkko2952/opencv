# opencv

---

## 中文版 (Chinese)

使用 Python 库中的 OpenCV 构建的视觉识别  
*This is visual identity*

### 功能
- 实现对红色和蓝色的锥桶识别，并且用框框出
- 若识别到两个不同颜色的锥桶，会计算其中点位置
- 加入实时调试模块，以及实时显示掩膜
- 打印当前调试参数，方便快速调参，优化程序

### 实现逻辑
1. 颜色识别：根据颜色以及光照强度生成灰度图（红蓝分开，方便解耦调试）
2. 框选：框选中一定体积的同色物体
3. 计算中点：框体中心连线的中点

### 该程序优势
1. 环境搭建简单，基本无需搭建
2. 功能代码模块化架构，方便对单个功能模块代码进行单一调整，强可维护性
3. 实时调试模块，方便进行优化调参

### 该程序劣势
1. 识别易受到环境干扰，鲁棒性不足
2. 中心点计算依赖于单个识别体的识别精度
3. 仅仅基于电脑自带摄像头进行识别

### 优化方向
1. 增强其鲁棒性，加入更多识别条件
2. 考虑加入外部摄像头的接入模块，可以使用更好的摄像头进行使用
3. 在优化了前面的识别的代码的基础上，使用更精确的框体，优化中心点的计算

---

## English Version

Visual identity built with OpenCV in Python

### Features
- Identify red and blue cones, and draw bounding boxes around them
- If two cones of different colors are detected, calculate the midpoint of their centers
- Include a real-time debug module and live mask display
- Print current debug parameters for easy tuning and optimization

### Implementation Logic
1. Color recognition: Generate grayscale images based on color and lighting intensity (red and blue separated for decoupled debugging)
2. Bounding box selection: Select objects of the same color with a certain size
3. Midpoint calculation: Midpoint of the line connecting the centers of two bounding boxes

### Advantages
1. Simple environment setup – almost no configuration required
2. Modular code architecture – easy to adjust individual modules, high maintainability
3. Real-time debug module – convenient for optimization and parameter tuning

### Disadvantages
1. Recognition is susceptible to environmental interference, limited robustness
2. Midpoint calculation depends on the recognition accuracy of individual objects
3. Only uses the computer's built-in camera

### Improvement Directions
1. Enhance robustness by adding more recognition criteria
2. Consider adding external camera support for better hardware
3. After optimizing the recognition code, use more precise bounding boxes to improve midpoint calculation
