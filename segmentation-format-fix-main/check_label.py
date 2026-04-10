import cv2
import os
import numpy as np

# 指定包含标签图像的文件夹路径，使用原始字符串（在路径前加r）避免转义字符问题
folder_path = r'G:\improve\eg'

# 检查文件夹是否存在
if not os.path.exists(folder_path):
    print(f"错误：文件夹不存在 - {folder_path}")
else:
    # 遍历文件夹中的所有图片文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):  # 不区分大小写检查PNG文件
            image_path = os.path.join(folder_path, filename)

            # 检查文件是否存在
            if not os.path.isfile(image_path):
                print(f"错误：文件不存在 - {image_path}")
                continue

            # 读取图像
            label_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # 检查是否成功读取图像
            if label_image is None:
                print(f"错误：无法读取文件 - {image_path}")
                continue

            # 获取标签种类数
            unique_labels = np.unique(label_image)

            # 打印每张图片的文件名和对应的唯一标签值
            print(f"文件: {filename}, 标签中的唯一值: {unique_labels}")