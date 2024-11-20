# import base64
# import json
# import os
# import os.path as osp
#
# import numpy as np
# import PIL.Image
# from labelme import utils
#
# '''
# 制作自己的语义分割数据集需要注意以下几点：
# 1、我使用的labelme版本是3.16.7，建议使用该版本的labelme，有些版本的labelme会发生错误，
#    具体错误为：Too many dimensions: 3 > 2
#    安装方式为命令行pip install labelme==3.16.7
# 2、此处生成的标签图是8位彩色图，与视频中看起来的数据集格式不太一样。
#    虽然看起来是彩图，但事实上只有8位，此时每个像素点的值就是这个像素点所属的种类。
#    所以其实和视频中VOC数据集的格式一样。因此这样制作出来的数据集是可以正常使用的。也是正常的。
# '''
# if __name__ == '__main__':
#     jpgs_path   = "datasets/JPEGImages"
#     pngs_path   = "datasets/SegmentationClass"
#     # classes     = ["_background_","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#     classes     = ["_background_", "f"]
#
#     count = os.listdir("./datasets/before/")
#     for i in range(0, len(count)):
#         path = os.path.join("./datasets/before", count[i])
#
#         if os.path.isfile(path) and path.endswith('json'):
#             data = json.load(open(path))
#
#             if data['imageData']:
#                 imageData = data['imageData']
#             else:
#                 imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
#                 with open(imagePath, 'rb') as f:
#                     imageData = f.read()
#                     imageData = base64.b64encode(imageData).decode('utf-8')
#
#             img = utils.img_b64_to_arr(imageData)
#             label_name_to_value = {'_background_': 0}
#             for shape in data['shapes']:
#                 label_name = shape['label']
#                 if label_name in label_name_to_value:
#                     label_value = label_name_to_value[label_name]
#                 else:
#                     label_value = len(label_name_to_value)
#                     label_name_to_value[label_name] = label_value
#
#             # label_values must be dense
#             label_values, label_names = [], []
#             for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
#                 label_values.append(lv)
#                 label_names.append(ln)
#             assert label_values == list(range(len(label_values)))
#
#             lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
#
#
#             PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))
#
#             new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
#             for name in label_names:
#                 index_json = label_names.index(name)
#                 index_all = classes.index(name)
#                 new = new + index_all*(np.array(lbl) == index_json)
#
#             utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
#             print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')


import base64
import json
import os
import os.path as osp

import imgviz

from labelme.logger import logger
from labelme import utils


def main():
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )

    # json_file是标注完之后生成的json文件的目录。out_dir是输出目录，即数据处理完之后文件保存的路径
    json_file = r"/home/kingargroo/fungs_rect_imgs/labelme3"
    out_dir1 = r"/home/kingargroo/fungs_rect_imgs/mask4"

    # 如果输出的路径不存在，则自动创建这个路径
    if not osp.exists(out_dir1):
        os.mkdir(out_dir1)
        # 将类别名称转换成数值，以便于计算
    label_name_to_value = {"_background_": 0}
    for file_name in os.listdir(json_file):
        # 遍历json_file里面所有的文件，并判断这个文件是不是以.json结尾
        if file_name.endswith(".json"):
            path = os.path.join(json_file, file_name)
            if os.path.isfile(path):
                data = json.load(open(path))

                # 获取json里面的图片数据，也就是二进制数据
                imageData = data.get("imageData")
                # 如果通过data.get获取到的数据为空，就重新读取图片数据
                if not imageData:
                    imagePath = os.path.join(json_file, data["imagePath"])
                    with open(imagePath, "rb") as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode("utf-8")
                #  将二进制数据转变成numpy格式的数据
                img = utils.img_b64_to_arr(imageData)

                for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                    label_name = shape["label"]
                    if label_name in label_name_to_value:
                        label_value = label_name_to_value[label_name]
                    else:
                        label_value = len(label_name_to_value)
                        label_name_to_value[label_name] = label_value
                lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

                label_names = [None] * (max(label_name_to_value.values()) + 1)
                for name, value in label_name_to_value.items():
                    label_names[value] = name

                lbl_viz = imgviz.label2rgb(
                    label=lbl, image=imgviz.asgray(img), label_names=label_names, loc="rb"
                )

                # out_dir = osp.basename(file_name).replace('.', '_')
                # out_dir = osp.join(out_dir1, out_dir)
                # if not osp.exists(out_dir):
                #     os.mkdir(out_dir)
                #     print(out_dir)

                # 将输出结果保存，
                # PIL.Image.fromarray(img).save(osp.join(out_dir1, "%s_img.jpg" % file_name.split(".")[0]))
                utils.lblsave(osp.join(out_dir1, "%s.png" % file_name.split(".")[0]), lbl)
                # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir1, "%s_label_viz.png"%file_name.split(".")[0]))

                # with open(osp.join(out_dir, "label_names.txt"), "w") as f:
                #     for lbl_name in label_names:
                #         f.write(lbl_name + "\n")

                logger.info("Saved to: {}".format(out_dir1))
    print("label:", label_name_to_value)


if __name__ == "__main__":
    main()