clc;clear;close all;
path = 'labelsTr';
output_path = 'processed_labels_data';
Files = dir(fullfile(path,'*.nii.gz'));
Filesname = {Files.name}';	
len = length(Files);
for index = 280:len
    nii = load_nii([path ,'\',Filesname{index}]);  % 装载.nii数据
    img = nii.img;  %文件包含img和head，img是图像数据
    [~,~,z] = size(img);
    for level = 1:z
        Image = img(:,:,level);
        [h,w] = size(Image);
        Image_pro = zeros(h,w);
        for i =1:h
           Image_pro(i,:) = Image(h-i+1,:);
        end
        processed_nii = nii;
        processed_nii.img(:,:,z) = Image_pro;
        [~, filename, ~] = fileparts(Filesname{index});
        output_filename = fullfile(output_path, [filename, '_processed.nii']);
        save_nii(processed_nii, output_filename);
    end
    disp(['第', num2str(index), '个nii处理完成']);
end