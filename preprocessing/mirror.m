% 读取图像
input_image_path = '151.png';
output_image_path = 'E:\BME\liver_tumor_segmentation\mirror_151.png';

% 读取图像
img = imread(input_image_path);

% 上下镜像
mirrored_img = flipud(img);

% 逆时针旋转90度
rotated_img = imrotate(mirrored_img, 90);

% 保存镜像和旋转后的图像
imwrite(rotated_img, output_image_path);
