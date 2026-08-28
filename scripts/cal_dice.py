import os
import SimpleITK as sitk
import numpy as np
from skimage.measure import regionprops, label
from medpy.metric import binary

# dice测试代码
# 1、路径设置  2、类别设置 3、元组设置 (4、是否启用最大病灶选取后与处理策略)


# 1、路径设置

# 标签路径
root_path = r'E:/BME/liver_tumor_segmentation/labelsTr'
# 预测标签路径
infer_path = r'best_predictions'
infer_list = sorted(os.listdir(infer_path))  # v3 8281 v2 8246

# 2、类别设置
# classes 表示需要分割的类别数(含背景类0)   如二分类任务classes则为2
def cal_dice(seg, gt, classes=2, background_id=0):
    channel_dice = []
    for i in range(classes):
        if i == background_id:
            continue
        cond = i ** 2
        # 计算相交部分
        inter = len(np.where(seg * gt == cond)[0])
        total_pix = len(np.where(seg == i)[0]) + len(np.where(gt == i)[0])
        if total_pix == 0:
            dice = 0
        else:
            dice = (2 * inter) / total_pix
        channel_dice.append(dice)

    return np.array(channel_dice)

# 3、 元组设置 涵盖背景的classes
# classes=2 -> np.zeros((1,1))    classes=3 -> np.zeros((1,2))
sum_result = np.zeros((1,1)) # np.zeros((1,2))

count = 0
sum_result_HD = 0
for data in infer_list:
    pr_path = os.path.join(infer_path, data)
    gt_path = os.path.join(root_path, data)
    image = sitk.ReadImage(pr_path)

    predict = sitk.GetArrayFromImage(sitk.ReadImage(pr_path))  # predict是预测结果
    target = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))  # target是ground true
    #target[target==1]=0
    #target[target==2]=1

    #####################################################################################
    # 后处理策略
    connect_regions = label(predict, connectivity=1, background=0)  # 四连通区域 conn=1
    props = regionprops(connect_regions)  # 对每一个连通区域进行操作
    max_area = 0
    for prop in props:
        if prop.area > max_area:
            mask_new = np.zeros_like(predict)
            for idx in prop.coords:
                mask_new[idx[0]][idx[1]][idx[2]] = 1
            max_area = prop.area

    # 4、最大病灶后处理策略
    # 自行决定是否启用 选取最大病灶区域
    if max_area != 0:
        predict = mask_new * predict
    #####################################################################################

    sum_result += cal_dice(target, predict)  # Cal dice
    #######################################################################################
    print('Processing Data:', data)
    print('Singe case Dice:', cal_dice(target, predict))

    #######################################################################################
    count += 1
    #######################################################################################
    #######################################################################################
sum_result_HD = sum_result_HD / len(infer_list)
sum_result_dice = sum_result / len(infer_list)

print('*' * 50)
print("Average dice score:", sum_result_dice)
print("Total case num:", count)
