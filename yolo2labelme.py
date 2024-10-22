import json
import os,base64


# 讀取影像的寬高，這在轉換 YOLO 相對坐標時需要
def get_image_size(image_path):
    import cv2
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    return w, h

# 如果想嵌入影像數據，可以使用這個函數將影像轉換為 base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
# 將 YOLO 分割標籤轉換為 LabelMe 格式

def yolo_to_labelme(yolo_label_path, image_path,labelme_path):
    # 讀取影像尺寸
    img_w, img_h = get_image_size(image_path)

    # LabelMe 格式的初始結構
    labelme_annotation = {
        "version": "4.5.6",  # LabelMe版本
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": image_to_base64(image_path),  # 若不想嵌入 imageData，保持為 None
        "imageHeight": img_h,
        "imageWidth": img_w
    }

    # 讀取 YOLO 標籤
    with open(yolo_label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = parts[0]  # 類別 ID
        points = []

        # 每個點的相對座標轉換為絕對座標
        for i in range(1, len(parts), 2):
            x_rel = float(parts[i])
            y_rel = float(parts[i + 1])
            x_abs = x_rel * img_w  # 轉換為絕對 x 座標
            y_abs = y_rel * img_h  # 轉換為絕對 y 座標
            points.append([x_abs, y_abs])

        # 構建 LabelMe 的 shape 格式
        shape = {
            "label": class_id,  # 使用類別 ID 作為標籤
            "points": points,  # 多邊形頂點
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        labelme_annotation["shapes"].append(shape)

    # 保存為 JSON 文件
    json_path = yolo_label_path.replace('.txt', '.json').split('/')[-1]
    json_path=os.path.join(labelme_path,json_path)

    with open(json_path, 'w') as json_file:
        json.dump(labelme_annotation, json_file, indent=4)

    print(f"轉換完成，保存為 {json_path}")

all_img_list = "/home/kingargroo/fungs_rect_imgs/img"
all_yololabel_save_path = "/home/kingargroo/fungs_rect_imgs/label"
labelme_path="/home/kingargroo/fungs_rect_imgs/labelme"
for image_path in os.listdir(all_img_list):
    main_name=image_path.split('.')[0]
    image_path=os.path.join(all_img_list,image_path)
    yolo_label_path=os.path.join(all_yololabel_save_path,main_name+'.txt')
    print(main_name)
    print(yolo_label_path,image_path)


    yolo_to_labelme(yolo_label_path, image_path,labelme_path)
