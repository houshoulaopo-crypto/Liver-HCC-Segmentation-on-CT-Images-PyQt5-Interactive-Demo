import os
import numpy as np
import nibabel as nib
import cv2
from ultralytics import YOLO

output_folder = 'E:/BME/作品ID7200分割结果/YOLO/test_predict/inferTs'
input_folder = 'E:/BME/作品ID7200分割结果/liver_tumor_segmentation_testset50/inferTs'  # 输入文件夹

# 加载模型
model = YOLO('E:/BME/作品ID7200分割结果/YOLO/liver_tumor_detect/weights/best.pt')


def apply_window(image, center, width):
    min_value = center - (width / 2)
    max_value = center + (width / 2)
    windowed_image = np.clip(image, min_value, max_value)
    return windowed_image


def check_prediction(nifti_file):
    """
    检查nii.gz文件中是否有预测结果。
    假设预测结果存在于特定的标签范围内（例如：标签值 > 0）。
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()

    # 假设预测结果的标签值大于0
    if np.any(data > 0):
        return True
    return False


def predict_and_save(image_file):
    # 加载 nii.gz 图像
    img_nii = nib.load(image_file)
    img_data = img_nii.get_fdata()

    # 创建一个空的数组用于存储推理结果
    result_data = np.zeros(img_data.shape)

    # 对每个切片进行推理
    for i in range(img_data.shape[2]):
        slice_img = img_data[:, :, i]

        # 将切片转换为 3 通道图像，并确保数据类型为 uint8
        slice_img = apply_window(slice_img, 150, 400)
        slice_img = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        slice_img = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)

        # 推理
        results = model(slice_img, device='cpu')

        for result in results:
            # 提取检测框、置信度和类别标签
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                print(f'Detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf}, class={cls}')

                # 将检测结果存储在 result_data 中
                result_data[y1:y2, x1:x2, i] = cls + 1  # 使用 cls + 1，以便 0 保持为空白背景

    # 保存结果为 nii.gz 文件
    result_nii = nib.Nifti1Image(result_data, img_nii.affine, img_nii.header)
    output_nii_path = os.path.join(output_folder, os.path.basename(image_file).replace('.nii.gz', '_output.nii.gz'))
    nib.save(result_nii, output_nii_path)

    print(f"Saved result to {output_nii_path}")


def main():
    # 读取输入文件夹中的所有 nii.gz 文件并进行处理
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(input_folder, file_name)
            print(f'Processing {file_path}...')

            if check_prediction(file_path):
                print(f"Prediction already exists in {file_path}")
            else:
                print(f"No prediction found in {file_path}. Running prediction...")
                predict_and_save(file_path)


if __name__ == "__main__":
    main()
