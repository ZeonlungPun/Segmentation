import os
import json
import numpy as np
import cv2
import shutil

from tqdm import tqdm


# 0-背景，从 1 开始
class_info = [
    {'label':'P', 'type':'polygon', 'color':1},                    # polygon 多段线
]


def labelme2mask_single_img(img_path, labelme_json_path):
    '''
    输入原始图像路径和labelme标注路径，输出 mask
    '''
    img_bgr = cv2.imread(img_path)
    img_mask = np.zeros(img_bgr.shape[:2])  # 创建空白图像 0-背景

    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    for one_class in class_info:  # 按顺序遍历每一个类别
        for each in labelme['shapes']:  # 遍历所有标注，找到属于当前类别的标注
            if each['label'] == one_class['label']:
                if one_class['type'] == 'polygon':  # polygon 多段线标注

                    # 获取点的坐标
                    points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]

                    # 在空白图上画 mask（闭合区域）
                    img_mask = cv2.fillPoly(img_mask, points, color=one_class['color'])

                elif one_class['type'] == 'line' or one_class['type'] == 'linestrip':  # line 或者 linestrip 线段标注

                    # 获取点的坐标
                    points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]

                    # 在空白图上画 mask（非闭合区域）
                    img_mask = cv2.polylines(img_mask, points, isClosed=False, color=one_class['color'],thickness=one_class['thickness'])

                elif one_class['type'] == 'circle':  # circle 圆形标注

                    points = np.array(each['points'], dtype=np.int32)

                    center_x, center_y = points[0][0], points[0][1]  # 圆心点坐标

                    edge_x, edge_y = points[1][0], points[1][1]  # 圆周点坐标

                    radius = np.linalg.norm(np.array([center_x, center_y] - np.array([edge_x, edge_y]))).astype('uint32')  # 半径

                    img_mask = cv2.circle(img_mask, (center_x, center_y), radius, one_class['color'],one_class['thickness'])

                else:
                    print('未知标注类型', one_class['type'])

    return img_mask
img_list=os.listdir('/home/kingargroo/seg/new')
json_base='/home/kingargroo/seg/newlabel'

for i in tqdm(img_list):
    try:
        img_path = os.path.join('/home/kingargroo/seg/new', i)


        labelme_json_path =os.path.join(json_base,i.split('.')[0])
        labelme_json_path=labelme_json_path+'.json'

        img_mask = labelme2mask_single_img(img_path, labelme_json_path)

        mask_path = i.split('.')[0] + '.png'
        final_path=os.path.join('/home/kingargroo/seg/', 'newmask', mask_path)
        #img_mask = cv2.imread(final_path)
        #img_mask=cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
        print(final_path,img_mask.shape)
        cv2.imwrite(final_path, img_mask)

    except Exception as E:
        print(img_path, '转换失败', E)