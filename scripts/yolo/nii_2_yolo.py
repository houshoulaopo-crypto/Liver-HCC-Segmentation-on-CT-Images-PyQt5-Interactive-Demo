import os
import nibabel as nib
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split


def read_nii_gz(file_path):
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data


def apply_window(image, center, width):
    min_value = center - (width / 2)
    max_value = center + (width / 2)
    windowed_image = np.clip(image, min_value, max_value)
    return windowed_image


def normalize_image(image):
    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm_image


def extract_rois(img_slice, threshold=0.5):
    rois = []
    contours, _ = cv2.findContours((img_slice > threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rois.append((x, y, w, h))
    return rois


def convert_to_yolo_format(rois, img_shape):
    yolo_rois = []
    height, width = img_shape
    for x, y, w, h in rois:
        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        w /= width
        h /= height
        yolo_rois.append((cx, cy, w, h))
    return yolo_rois


def save_yolo_format(img_slice, yolo_rois, output_img_path, output_label_path):
    # Save the image
    cv2.imwrite(output_img_path, img_slice)

    # Save the labels
    with open(output_label_path, 'w') as f:
        for roi in yolo_rois:
            f.write(f"0 {roi[0]} {roi[1]} {roi[2]} {roi[3]}\n")


def process_nii_files(label_nii_path, train_nii_path, output_dir, window_center, window_width):
    # Read label and train nii.gz files
    label_data = read_nii_gz(label_nii_path)
    train_data = read_nii_gz(train_nii_path)

    file_pairs = []

    # Iterate over all slices
    for slice_idx in range(label_data.shape[2]):
        label_slice = label_data[:, :, slice_idx]
        train_slice = train_data[:, :, slice_idx]

        # Apply windowing and normalization to the training slice
        windowed_image = apply_window(train_slice, window_center, window_width)
        normalized_image = normalize_image(windowed_image)

        # Extract ROIs from label slice
        rois = extract_rois(label_slice)

        if rois:  # If there are any ROIs in this slice
            # Convert to YOLO format
            yolo_rois = convert_to_yolo_format(rois, normalized_image.shape)

            # Output file paths
            base_name = os.path.basename(label_nii_path).replace('.nii.gz', '')
            output_img_path = os.path.join(output_dir, f"{base_name}_slice{slice_idx}.jpg")
            output_label_path = os.path.join(output_dir, f"{base_name}_slice{slice_idx}.txt")

            # Save in YOLO format
            save_yolo_format(normalized_image, yolo_rois, output_img_path, output_label_path)

            # Store file paths for train/test split
            file_pairs.append((output_img_path, output_label_path))

    return file_pairs


def split_dataset(file_pairs, train_ratio=0.8):
    train_files, val_files = train_test_split(file_pairs, train_size=train_ratio, random_state=42)
    return train_files, val_files


def write_split_files(file_pairs, output_txt_path):
    with open(output_txt_path, 'w') as f:
        for img_path, _ in file_pairs:
            f.write(f"{img_path}\n")


def main(label_dir, train_dir, output_dir, window_center, window_width, train_ratio=0.8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_file_pairs = []

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.nii.gz'):
            train_file = label_file[0:3] + '_0000' + '.nii.gz'  # Assuming label and train files have the same names
            label_file_path = os.path.join(label_dir, label_file)
            train_file_path = os.path.join(train_dir, train_file)

            if os.path.exists(train_file_path):
                file_pairs = process_nii_files(label_file_path, train_file_path, output_dir, window_center,
                                               window_width)
                all_file_pairs.extend(file_pairs)
            else:
                print(f"Train file {train_file_path} not found for label file {label_file_path}")

    # Split dataset into training and validation sets
    train_files, val_files = split_dataset(all_file_pairs, train_ratio)

    # Write train.txt and val.txt
    write_split_files(train_files, os.path.join(output_dir, "train.txt"))
    write_split_files(val_files, os.path.join(output_dir, "val.txt"))


if __name__ == "__main__":
    label_dir = "E:/BME/YOLO/label_Tr"  # 标签文件夹路径
    train_dir = "E:/BME/YOLO/image_Tr"  # 训练图像文件夹路径
    output_dir = "E:/BME/YOLO/nii_2_yolo"  # 输出文件夹路径
    window_center = 150  # 窗位
    window_width = 400  # 窗宽
    train_ratio = 0.8  # 训练集比例
    main(label_dir, train_dir, output_dir, window_center, window_width, train_ratio)
