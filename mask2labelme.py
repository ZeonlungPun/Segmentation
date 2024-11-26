import cv2
import numpy as np
import json
import os
from PIL import Image
import base64



def mask_to_labelme(mask_path,img_dir,output_json_path):

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(mask_path)
    img_path=os.path.join(img_dir,mask_path.split('/')[-1])
    if mask is None:
        print(f"unable to load {mask_path}")
        return


    with open(img_path, "rb") as image_file:
        image_data = image_file.read()
        image_data_base64 = base64.b64encode(image_data).decode('utf-8')


    annotation = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(mask_path),
        "imageData": image_data_base64,
        "imageHeight": mask.shape[0],
        "imageWidth": mask.shape[1]
    }


    labels = np.unique(mask)
    for label in labels:
        if label == 0:
            # 0 is background
            continue


        binary_mask = np.uint8(mask == label)


        contours, hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)


            points = approx.reshape(-1, 2).tolist()


            shape = {
                "label": str(label),
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }

            annotation["shapes"].append(shape)

    # 将标注保存为 JSON 文件
    with open(output_json_path, 'w') as json_file:
        json.dump(annotation, json_file, indent=2)

    print(f"已保存到 {output_json_path}")

mask_dir='/home/kingargroo/fungs_rect_imgs/new/GT'
out_dir='/home/kingargroo/fungs_rect_imgs/new/labelme'
for i in os.listdir(mask_dir):
    mask_image_path = os.path.join(mask_dir,i)
    name=mask_image_path.split('/')[-1].split('.')[0]

    output_json=os.path.join(out_dir,name+'.json')

    mask_to_labelme(mask_image_path,'/home/kingargroo/fungs_rect_imgs/new/Image' ,output_json)
