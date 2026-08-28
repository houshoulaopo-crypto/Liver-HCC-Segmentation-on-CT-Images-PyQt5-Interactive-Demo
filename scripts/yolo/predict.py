import torch
import numpy as np
import nibabel as nib
import cv2
from ultralytics import YOLO

output_folder = 'E:/BME/ID7200 segmentation result/YOLO/test_predict/inferTs/151'
# 加载模型
model = YOLO('E:/BME/ID7200 segmentation result/YOLO/train4/weights/best.pt')

def apply_window(image, center, width):
    min_value = center - (width / 2)
    max_value = center + (width / 2)
    windowed_image = np.clip(image, min_value, max_value)
    return windowed_image

def predict_and_save(image_file):
    # 加载 nii.gz 图像
    img_nii = nib.load(image_file)
    img_data = img_nii.get_fdata()

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

                # 在图像上绘制检测框
                cv2.rectangle(slice_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(slice_img, 'tumor:'+ f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

        # 保存处理后的切片
        cv2.imwrite(f'{output_folder}/output_slice_{i}.png', slice_img)

# 使用函数进行预测和保存
predict_and_save('E:/BME/Qt/final/imagesTr/151_0000.nii.gz')
