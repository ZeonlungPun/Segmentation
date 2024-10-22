from ultralytics import SAM
import cv2, os
import numpy as np

# Load a model
model = SAM("./sam2_l.pt")

all_img_list = "/home/kingargroo/fungs_rect_imgs/img"
all_label_save_path = "/home/kingargroo/fungs_rect_imgs/label"
visualization_save_path = "/home/kingargroo/fungs_rect_imgs/visualization"

# 確保保存可視化結果的文件夾存在
if not os.path.exists(visualization_save_path):
    os.makedirs(visualization_save_path)

for path in os.listdir(all_img_list):

    img_path = os.path.join(all_img_list, path)
    img = cv2.imread(img_path)

    imgh, imgw = img.shape[0:2]
    img_title = path.split('.')[0]
    txt_name = os.path.join(all_label_save_path, img_title + '.txt')

    with open(txt_name, 'a+') as txt_file:
        # 檢測一張圖像
        results = model(img)

        # 遍歷一張圖像中的所有物體
        for result in results:
            masks = result.masks

            for mask in masks:
                for seg in mask.xy:
                    segment = np.array(seg, dtype=np.int32)

                    # 過濾過大或過小的區域
                    x, y, w, h = cv2.boundingRect(segment)
                    area = cv2.contourArea(segment)
                    perimeter = cv2.arcLength(segment, True)
                    circularity = 4 * np.pi * (area / (perimeter ** 2 + 1e-16))  # 圓度計算，防止除以零

                    # 過濾掉非細菌部分，只保留圓度接近的區域
                    if w * h > 4000 and w * h < 50000 and 0.085 < circularity < 1.9:

                        # 可視化分割結果：在圖像上繪製分割區域的多邊形
                        cv2.polylines(img, [segment], isClosed=True, color=(0, 255, 0), thickness=2)

                        # 構建 YOLO 分割格式：class_id + 多邊形頂點座標 (相對於圖像尺寸標準化)
                        output_str = '0'  # 假設類別為 0
                        for point in seg:
                            x_rel = point[0] / imgw  # 相對於圖像寬度標準化
                            y_rel = point[1] / imgh  # 相對於圖像高度標準化
                            output_str += f' {x_rel:.6f} {y_rel:.6f}'

                        # 將結果寫入 .txt 文件
                        txt_file.write(output_str + '\n')

    # 保存可視化結果
    visualization_img_path = os.path.join(visualization_save_path, img_title + '_visualized.jpg')
    cv2.imwrite(visualization_img_path, img)




