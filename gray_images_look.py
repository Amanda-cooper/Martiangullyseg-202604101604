import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors


def visualize_grayscale(image_path):
    # 读取灰度图
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray_img is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # 创建图形和子图布局
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(left=0.1, bottom=0.35)

    # 显示原始灰度图
    ax1 = plt.subplot(2, 3, 1)
    orig_img = ax1.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('原始灰度图')
    plt.colorbar(orig_img, ax=ax1)

    # 对比度增强后的图像
    ax2 = plt.subplot(2, 3, 2)
    min_val = np.min(gray_img)
    max_val = np.max(gray_img)
    contrast_img = np.clip((gray_img - min_val) * (255 / (max_val - min_val + 1e-7)), 0, 255).astype(np.uint8)
    contrast_plot = ax2.imshow(contrast_img, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('对比度增强')
    plt.colorbar(contrast_plot, ax=ax2)

    # 直方图均衡化
    ax3 = plt.subplot(2, 3, 3)
    equ_img = cv2.equalizeHist(gray_img)
    equ_plot = ax3.imshow(equ_img, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('直方图均衡化')
    plt.colorbar(equ_plot, ax=ax3)

    # 伪彩色图像（使用jet颜色映射）
    ax4 = plt.subplot(2, 3, 4)
    color_img = cv2.applyColorMap(contrast_img, cv2.COLORMAP_JET)
    color_plot = ax4.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    ax4.set_title('伪彩色映射')

    # 自定义伪彩色（突出低灰度值）
    ax5 = plt.subplot(2, 3, 5)
    # 创建自定义颜色映射：黑色->蓝色->青色->绿色->黄色->红色
    colors = ["black", "blue", "cyan", "green", "yellow", "red"]
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=256)
    custom_plot = ax5.imshow(contrast_img, cmap=cmap_custom, vmin=0, vmax=255)
    ax5.set_title('自定义颜色映射')
    plt.colorbar(custom_plot, ax=ax5)

    # 灰度直方图
    ax6 = plt.subplot(2, 3, 6)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    ax6.plot(hist, color='k')
    ax6.set_title('灰度直方图')
    ax6.set_xlim([0, 256])
    ax6.set_xlabel('灰度值')
    ax6.set_ylabel('像素数量')

    # 添加交互式控件
    ax_slider = plt.axes([0.25, 0.15, 0.5, 0.03])
    slider = Slider(ax_slider, '放大倍数', 1.0, 5.0, valinit=1.0, valstep=0.5)

    # 更新函数
    def update(val):
        zoom = slider.val
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.set_xlim(0, gray_img.shape[1] / zoom)
            ax.set_ylim(gray_img.shape[0] / zoom, 0)
        plt.draw()

    slider.on_changed(update)

    # 添加重置按钮
    reset_ax = plt.axes([0.8, 0.05, 0.1, 0.04])
    button = Button(reset_ax, '重置视图', color='lightgoldenrodyellow', hovercolor='0.975')

    def reset(event):
        slider.reset()
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.set_xlim(0, gray_img.shape[1])
            ax.set_ylim(gray_img.shape[0], 0)
        plt.draw()

    button.on_clicked(reset)

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 替换为你的图像路径
    image_path = "D:/unet-pytorch-main\miou_out/detection-results-TS/0072.png"
    visualize_grayscale(image_path)