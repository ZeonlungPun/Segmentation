import json
import os
import cv2
import numpy as np


def labelme_json_to_yolo_seg(json_path, image_shape, class_id, save_path):
    # 打開並解析 LabelMe 的 JSON 文件
    with open(json_path, 'r') as f:
        label_data = json.load(f)

    height, width = image_shape
    yolo_annotations = []

    # 遍歷 JSON 文件中的所有標註區域
    for shape in label_data['shapes']:
        points = shape['points']
        points = [(float(x), float(y)) for x, y in points]  # 獲取標註點坐標
        points = np.array(points)

        # 處理只有兩個點的情況 (圓形標註)
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            radius = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 將圓形標註轉換為近似多邊形 (例如，用 36 個點模擬圓形)
            num_points = 36
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            circle_points = [
                (x1 + radius * np.cos(angle), y1 + radius * np.sin(angle))
                for angle in angles
            ]
            points = np.array(circle_points)

        # 歸一化多邊形點坐標
        normalized_points = []
        for (x, y) in points:
            normalized_x = x / width
            normalized_y = y / height
            normalized_points.extend([normalized_x, normalized_y])

        # YOLO Seg 格式：<class_id>  <x1> <y1> ... <xn> <yn>
        yolo_annotation = f"{class_id} " + " ".join(
            map(str, normalized_points))
        yolo_annotations.append(yolo_annotation)

    # 將 YOLO Seg 標註保存到指定文件
    with open(save_path, 'w') as f:
        for annotation in yolo_annotations:
            f.write(annotation + '\n')
    print(f"YOLO Seg annotations saved to {save_path}")


# 路徑設置
all_img_list = "/home/kingargroo/fungs_rect_imgs/img3"
labelme_path = "/home/kingargroo/fungs_rect_imgs/labelme2"
save_path_main = "/home/kingargroo/fungs_rect_imgs/yololabel"
class_id = 0  # 假設只有一個類別，將其設為 0

# 創建 YOLO Seg 標註保存路徑
os.makedirs(save_path_main, exist_ok=True)

for image_file in os.listdir(all_img_list):
    main_name = image_file.split('.')[0]
    image_path = os.path.join(all_img_list, image_file)
    label_path = os.path.join(labelme_path, main_name + '.json')
    save_path = os.path.join(save_path_main, main_name + '.txt')

    # 讀取圖像以獲取圖像尺寸
    image = cv2.imread(image_path)
    image_shape = image.shape[:2]  # (height, width)

    # 將 LabelMe 標註轉換為 YOLO Seg 格式
    labelme_json_to_yolo_seg(label_path, image_shape, class_id, save_path)

