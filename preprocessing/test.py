import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('darkgrid')

img1_path = 'imagesTr/001_0000.nii.gz'

img = nib.load(img1_path)

# 获取图像数据
data = img.get_fdata()

# 可视化每一层切片
num_slices = data.shape[-1]

# 设置子图的行数和列数
num_rows = num_slices // 10 + 1  # 每行显示10个切片
num_cols = min(num_slices, 10)

# 设置子图的大小
fig, axes = plt.subplots(num_rows, num_cols, figsize=(50,50))

# 遍历每一层切片并可视化
for i in range(num_slices):
    row_idx = i // 10
    col_idx = i % 10

    # 在子图中显示每一层切片
    axes[row_idx, col_idx].imshow(data[:, :, i], cmap='gray')
    axes[row_idx, col_idx].axis('off')  # 关闭坐标轴

# 如果切片数量不是10的倍数，隐藏多余的子图
for i in range(num_slices, num_rows * num_cols):
    row_idx = i // 10
    col_idx = i % 10
    fig.delaxes(axes[row_idx, col_idx])

plt.show()

