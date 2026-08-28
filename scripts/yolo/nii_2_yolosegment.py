import os
import numpy as np
import nibabel as nib

# 设置文件夹路径
dataset_path = 'E:/BME/YOLO/label_Tr'  # nii.gz文件夹路径
output_path = 'E:/BME/YOLO/test'  # 输出文件夹路径

# 获取所有nii.gz文件名
nii_files = [f for f in os.listdir(dataset_path) if f.endswith('.nii.gz')]


def find_polygon_vertices(binary_image):
    vertices = []
    height, width = binary_image.shape

    # 遍历每一行，找出顶点
    for y in range(height):
        x_start = None

        # 找到每一行的起始点和终止点
        for x in range(width):
            if binary_image[y, x] == 1 and x_start is None:
                x_start = x
            elif binary_image[y, x] == 0 and x_start is not None:
                # 发现了一个顶点
                vertices.append((y, (x_start + x - 1) // 2))
                x_start = None

        # 处理行末尾的顶点
        if x_start is not None:
            vertices.append((y, width - 1))

    return vertices


def convert_nii_to_txt(nii_file):
    # 读取nii.gz文件
    nii_path = os.path.join(dataset_path, nii_file)
    img = nib.load(nii_path)
    data = img.get_fdata()

    # 假设nii.gz文件中的标注已经转换为二值化图像，并存储在data中
    # 这里假设data是二值化图像数据的示例，可以直接处理

    polygons = []
    for slice_idx in range(data.shape[2]):
        binary_image = data[:, :, slice_idx]  # 获取二值化图像的每个切片
        vertices = find_polygon_vertices(binary_image)
        polygons.append(vertices)

    # 构建.txt文件名（可以根据需求调整文件名格式）
    txt_file = os.path.splitext(nii_file)[0] + '.txt'
    txt_path = os.path.join(output_path, txt_file)

    with open(txt_path, 'w') as f:
        for polygon in polygons:
            # 将多边形顶点坐标写入.txt文件，每行一个多边形的顶点坐标
            for point in polygon:
                f.write(f"{point[0]} {point[1]}\n")  # 假设每个顶点坐标是以空格分隔的格式


# 调用函数转换每个nii.gz文件
for nii_file in nii_files:
    convert_nii_to_txt(nii_file)

print("数据集转换完成！")
