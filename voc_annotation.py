import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# 设置数据集划分比例
train_percent = 0.7  # 训练集比例
val_percent = 0.1  # 验证集比例
test_percent = 0.2  # 测试集比例

VOCdevkit_path = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("生成数据集划分文件...")

    # 定位标签文件和保存路径
    seg_class_path = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    save_path = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')

    # 获取所有png标签文件（不带扩展名）
    total_files = [f[:-4] for f in os.listdir(seg_class_path) if f.endswith('.png')]
    num_files = len(total_files)
    random.shuffle(total_files)

    # 计算各集合数量
    train_count = int(num_files * train_percent)
    val_count = int(num_files * val_percent)
    test_count = num_files - train_count - val_count

    print(f"总样本数: {num_files}")
    print(f"训练集大小: {train_count}")
    print(f"验证集大小: {val_count}")
    print(f"测试集大小: {test_count}")

    # 划分数据集
    train_files = total_files[:train_count]
    val_files = total_files[train_count:train_count + val_count]
    test_files = total_files[train_count + val_count:]

    # 确保不重复不遗漏
    assert len(set(train_files + val_files + test_files)) == num_files

    # 写入文件
    os.makedirs(save_path, exist_ok=True)
    for name, data in [('train', train_files), ('val', val_files), ('test', test_files)]:
        with open(os.path.join(save_path, f'{name}.txt'), 'w') as f:
            f.write('\n'.join(data))

    print("数据集划分文件生成完成")

    print("检查数据集格式...")
    class_counts = np.zeros(256, dtype=int)
    for filename in tqdm(total_files):
        png_path = os.path.join(seg_class_path, f"{filename}.png")

        # 检查文件是否存在
        if not os.path.exists(png_path):
            raise FileNotFoundError(f"标签文件不存在: {png_path}")

        # 检查是否为单通道图像
        img = Image.open(png_path)
        if img.mode not in ['L', 'P']:
            raise ValueError(f"标签 {filename}.png 应为灰度图/调色板模式, 实际模式: {img.mode}")

        # 统计像素值
        img_array = np.array(img)
        flat = img_array.flatten()
        class_counts += np.bincount(flat, minlength=256)

    # 输出统计结果
    print("\n像素值统计:")
    print("-" * 37)
    print(f"| {'像素值':^15} | {'出现次数':^15} |")
    print("-" * 37)
    for i, count in enumerate(class_counts):
        if count > 0:
            print(f"| {i:^15} | {count:^15} |")
            print("-" * 37)

    # 检查常见问题
    if class_counts[255] > 0 and class_counts[0] > 0 and np.sum(class_counts[1:255]) == 0:
        print("警告: 标签仅包含0和255，二分类问题应使用0和1！")
    elif class_counts[0] == num_files * img.size[0] * img.size[1]:
        print("错误: 所有标签均为全背景图！")

    print("\n数据格式验证完成")
    print("提示: JPEGImages中应存放.jpg图像，SegmentationClass中存放对应.png标签")