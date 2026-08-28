clc;clear;close all;
path = 'imagesTr';
output_path = 'processed_train_data';
Files = dir(fullfile(path,'*.nii.gz'));
Filesname = {Files.name}';	
len = length(Files);
for index = 1:len
    nii = load_nii([path ,'\',Filesname{index}]);  % 装载.nii数据
    img = nii.img;  %文件包含img和head，img是图像数据
    [~,~,z] = size(img);
    for level = 1:z
        Image = img(:,:,level);
        [h,w] = size(Image);
        window_low =-100;
        window_high = 400;
        for i =1:h
            for j =1:w
                if Image(i,j) >= window_low && Image(i,j) <= window_high
                    continue
                elseif Image(i,j) < window_low
                        Image(i,j) = window_low;
                else 
                    Image(i,j) = window_high;
                end
            end
        end
        % Image = imadjust(Image,[0,1]);
        % Image = histeq(Image);
        Image = mat2gray(Image);%归一化
        % 保存处理后的图像到输出文件夹
        processed_nii = nii;
        processed_nii.img(:,:,z) = Image;
        [~, filename, ~] = fileparts(Filesname{index});
        output_filename = fullfile(output_path, [filename, '_processed.nii']);
        save_nii(processed_nii, output_filename);
    end
    disp(['第', num2str(index), '个nii处理完成']);
end