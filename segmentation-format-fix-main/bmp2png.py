from PIL import Image
import os

# 定义源文件夹和目标文件夹
source_folder = 'H:/segment/unet-pytorch-main/datasets/SegmentationClass_Origin'  # 替换为你的源文件夹路径
target_folder = 'H:/segment/unet-pytorch-main/datasets/SegmentationClass_Origin1'  # 替换为你的目标文件夹路径

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith(".bmp"):   ##如果标签格式为jpg，将bmp修改为jpg即可
        # 构造完整的文件路径
        file_path = os.path.join(source_folder, filename)

        # 打开图像文件
        with Image.open(file_path) as img:
            # 构造目标文件名（将.bmp替换为.png）
            target_filename = os.path.splitext(filename)[0] + '.png'
            target_path = os.path.join(target_folder, target_filename)

            # 保存转换后的图像
            img.save(target_path, 'PNG')

print("转换完成！")