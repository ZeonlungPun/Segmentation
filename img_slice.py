import albumentations as A
import os
import cv2
import numpy as np

# 定義裁剪轉換
trans = A.Compose(
    [
        A.RandomCrop(height=640, width=640)
    ]
)

# 路徑配置
img_list = os.listdir('/home/kingargroo/fungs_rect_imgs/new2/image')
save_path = '/home/kingargroo/fungs_rect_imgs/img_aug2'
save_masks_path = '/home/kingargroo/fungs_rect_imgs/mask_aug2'
masks_path = '/home/kingargroo/fungs_rect_imgs/new2/mask2'

# 建立保存目錄（如果不存在）
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_masks_path, exist_ok=True)

for index, i in enumerate(img_list):
    img_path = os.path.join('/home/kingargroo/fungs_rect_imgs/new2/image', i)
    mask_path = os.path.join(masks_path,i.replace('.jpg', '.png') )#i.replace('.jpg', '.png')
    img_array = cv2.imread(img_path)

    mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 確保遮罩是單通道
    print(mask_array.shape)
    print(img_array.shape)

    for j in range(6):  # 每張圖生成6個裁剪片段
        # 應用裁剪
        transformed = trans(image=img_array, mask=mask_array)
        img_trans = transformed['image']
        mask_trans = transformed['mask']

        # 檢查裁剪後的遮罩是否包含目標
        if np.any(mask_trans > 0):  # 假設非零像素為目標
            # 保存裁剪後的圖片和遮罩
            new_img_path = os.path.join(save_path, f'trans_{i}_{j}.jpg')
            new_mask_path = os.path.join(save_masks_path, f'trans_{i}_{j}.png')
            cv2.imwrite(new_img_path, img_trans)
            cv2.imwrite(new_mask_path, mask_trans)
        else:
            print(f"跳過裁剪片段 {i}_{j}，因為不包含目標")
