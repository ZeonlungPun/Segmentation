import json
import numpy as np
import cv2
import os


def labelme_json_to_binary_mask(json_path, image_shape):
    # 打開並解析 LabelMe 的 JSON 文件
    with open(json_path, 'r') as f:
        label_data = json.load(f)

    # 創建一個與影像大小相同的空白遮罩，背景為0
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

    # 遍歷 JSON 文件中的所有標註區域
    for shape in label_data['shapes']:
        points = shape['points']

        # 如果 points 只有兩個點，則假設是圓形
        if len(points) == 2:
            # 圓心
            center = tuple(map(int, points[0]))
            # 根據圓心和圓周點計算半徑
            radius = int(np.linalg.norm(np.array(points[0]) - np.array(points[1])))
            # 使用 OpenCV 繪製圓形遮罩，將前景區域設置為255
            cv2.circle(mask, center, radius, color=255, thickness=-1)
        else:
            # 如果是多邊形，則使用多邊形填充
            polygon = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], color=255)

    return mask


def save_binary_mask(mask, save_path):
    # 將二進制遮罩保存為圖像文件（例如 PNG 格式）
    cv2.imwrite(save_path, mask)


all_img_list = "/home/kingargroo/fungs_rect_imgs/img3"
labelme_path = "/home/kingargroo/fungs_rect_imgs/labelme2"
save_path_main = "/home/kingargroo/fungs_rect_imgs/mask3"

# 遍歷圖像目錄
for image_path in os.listdir(all_img_list):
    main_name = image_path.split('.')[0]
    image_path = os.path.join(all_img_list, image_path)
    label_path = os.path.join(labelme_path, main_name + '.json')
    print(label_path)

    image = cv2.imread(image_path)
    image_shape = image.shape[:2]  # 獲取影像的大小 (高, 寬)

    # 生成二進制遮罩
    mask = labelme_json_to_binary_mask(label_path, image_shape)
    save_path = os.path.join(save_path_main, main_name + '.png')
    save_binary_mask(mask, save_path)
    print(f"二進制遮罩已保存至: {save_path}")






