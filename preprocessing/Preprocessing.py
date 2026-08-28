## using simpleITK to load and save data.
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt

infer_path = r'imagesTr'
infer_list = sorted(os.listdir(infer_path))

# 文件读取
for filename in infer_list:
    # Construct the full path to the image file
    filepath = os.path.join(infer_path, filename)
    itk_img = sitk.ReadImage(filepath)
    # 设置灰度范围为 0 - 255
    itk_img = sitk.RescaleIntensity(itk_img, outputMinimum=0, outputMaximum=255)
    # 窗宽设置50-100
    # itk_img = sitk.IntensityWindowing(itk_img,windowMinimum=50,windowMaximum=100)
    # 转为矩阵
    img = sitk.GetArrayFromImage(itk_img)
    x, y, z = itk_img.GetSize()
    # 展平图像数组
    image_array_flat = img.flatten()

    # 绘制直方图
    plt.hist(image_array_flat, bins=100, color='blue', alpha=0.7)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Image Histogram')

    # 图像显示
    plt.figure(figsize=(5.12,5.12))
    plt.imshow(img[:, :, 50], cmap='gray')  # 替换50为你想展示的切片编号
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    # for img_idx in img:
    break



# ## save
# out = sitk.GetImageFromArray(img)
# out.SetSpacing(itk_img.GetSpacing())
# out.SetOrigin(itk_img.GetOrigin())
# out.SetDirection(itk_img.GetDirection())
#
# sitk.WriteImage(out, 'simpleitk_save.nii.gz')
