import os
import shutil
import random

# 设置文件夹路径
dataset_path = 'E:/BME/YOLO/nii_2_yolo'  # 原始数据集文件夹路径
output_path = 'E:/BME/YOLO/yolodata'  # YOLOv8数据集文件夹路径

# 设置划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 确保划分比例之和为1
assert train_ratio + val_ratio + test_ratio == 1.0, "Train, val and test ratios must sum to 1"

# 创建YOLOv8数据集文件夹结构
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

# 获取所有图像文件名
image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]

# 打乱数据集
random.shuffle(image_files)

# 计算每个数据集的样本数量
total_images = len(image_files)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)
test_count = total_images - train_count - val_count

# 划分数据集
train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

def copy_files(file_list, split):
    for file_name in file_list:
        # 复制图像文件
        src_image_path = os.path.join(dataset_path, file_name)
        dst_image_path = os.path.join(output_path, split, 'images', file_name)
        shutil.copyfile(src_image_path, dst_image_path)

        # 复制对应的标签文件
        label_name = os.path.splitext(file_name)[0] + '.txt'
        src_label_path = os.path.join(dataset_path, label_name)
        dst_label_path = os.path.join(output_path, split, 'labels', label_name)
        if os.path.exists(src_label_path):
            shutil.copyfile(src_label_path, dst_label_path)

# 复制文件到各个数据集文件夹
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("数据集划分完成！")
