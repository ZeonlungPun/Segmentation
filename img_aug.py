import albumentations as A

import numpy as np
import os,cv2

trans = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
            A.GaussNoise(),    # 将高斯噪声应用于输入图像。
        ], p=0.2),   # 应用选定变换的概率
        A.OneOf([
            A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
            A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
            A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
        A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=255, mask_value=0, shift_limit_x=0.2, shift_limit_y=0.2, always_apply=False, p=0.1),
        A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.8),
        A.RandomGridShuffle(grid=(3, 3), always_apply=False, p=0.2),
        #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, always_apply=False, p=0.1),
        A.PiecewiseAffine(scale=(0.05, 0.05), nb_rows=4, nb_cols=4, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, always_apply=False, keypoints_threshold=0.01, p=0.1),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.1),
        A.ElasticTransform(alpha=2, sigma=80, alpha_affine=60, interpolation=1, border_mode=4, value=0, mask_value=0, always_apply=False, approximate=False, same_dxdy=False, p=0.1),
        A.GridDistortion(num_steps=5, distort_limit=0.5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.1),
        A.CoarseDropout(max_holes=8, max_height=128, max_width=128, min_holes=None, min_height=64, min_width=64, fill_value=0, mask_fill_value=0, always_apply=False, p=0.1)])


img_list=os.listdir('/home/kingargroo/seg/zhecheng')
save_path='/home/kingargroo/seg/zhecheng/trans_img'
masks_path='/home/kingargroo/seg/mask'
for index,i in enumerate(img_list):
    img_path=os.path.join('/home/kingargroo/seg/zhecheng',i)
    mask_path=os.path.join(masks_path,i)

    img_array=cv2.imread(img_path)
    mask_array=cv2.imread(mask_path)
    print(mask_array.shape)
    print(img_array.shape)
    img_trans = trans(image=img_array,mask=mask_array)['image']
    mask_trans=trans(image=img_array,mask=mask_array)['mask']
    base_path='/home/kingargroo/seg/'
    new_img_path=os.path.join(base_path,'trans_img')
    new_img_path2=new_img_path+'/'+'trans'+str(i)
    new_mask_path=os.path.join(base_path,'trans_mask')
    new_mask_path2 = new_mask_path + '/' + 'trans' + str(i)
    cv2.imwrite(new_img_path2,img_trans)
    cv2.imwrite(new_mask_path2, mask_trans)


