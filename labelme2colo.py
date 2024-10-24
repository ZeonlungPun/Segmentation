import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import imgviz
import numpy as np
import labelme
from sklearn.model_selection import train_test_split

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def to_coco(args, label_files, train):
    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[],
        type="instances",
        annotations=[],
        categories=[],
    )

    # 修改：類別 ID 從 1 開始
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i  # 直接從 0 開始（背景也可以視為一類）
        class_name = line.strip()
        if class_id == 0:
            assert class_name == "__ignore__"  # background:0
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name, )
        )

    if train:
        out_ann_file = osp.join(args.output_dir, "annotations", "instances_train2017.json")
    else:
        out_ann_file = osp.join(args.output_dir, "annotations", "instances_val2017.json")

    for image_id, filename in enumerate(label_files):
        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]
        if train:
            out_img_file = osp.join(args.output_dir, "train2017", base + ".jpg")
        else:
            out_img_file = osp.join(args.output_dir, "val2017", base + ".jpg")

        print("| ", out_img_file)

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}
        segmentations = collections.defaultdict(list)
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )
        if not args.noviz:
            labels, captions, masks = zip(
                *[
                    (class_name_to_id[cnm], cnm, msk)
                    for (cnm, gid), msk in masks.items()
                    if cnm in class_name_to_id
                ]
            )
            viz = imgviz.instances2rgb(
                image=img,
                labels=labels,
                masks=masks,
                captions=captions,
                font_size=15,
                line_width=2,
            )
            out_viz_file = osp.join(
                args.output_dir, "visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default='/home/kingargroo/fungs_rect_imgs/dataano', help="input annotated directory")
    parser.add_argument("--output_dir", default='/home/kingargroo/fungs_rect_imgs/coco2', help="output dataset directory")
    parser.add_argument("--labels", default='/home/kingargroo/fungs_rect_imgs/label.txt', help="labels file")
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        # 清空文件夾，而不是終止程序
        print("Output directory already exists, clearing it:", args.output_dir)
        for root, dirs, files in os.walk(args.output_dir):
            for file in files:
                os.remove(osp.join(root, file))

    os.makedirs(args.output_dir, exist_ok=True)
    print("| Creating dataset dir:", args.output_dir)
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "visualization"), exist_ok=True)


    if not os.path.exists(osp.join(args.output_dir, "annotations")):
        os.makedirs(osp.join(args.output_dir, "annotations"))
    if not os.path.exists(osp.join(args.output_dir, "train2017")):
        os.makedirs(osp.join(args.output_dir, "train2017"))
    if not os.path.exists(osp.join(args.output_dir, "val2017")):
        os.makedirs(osp.join(args.output_dir, "val2017"))

    feature_files = glob.glob(osp.join(args.input_dir, "*.jpg"))
    print('| Image number: ', len(feature_files))

    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    print('| Json number: ', len(label_files))

    x_train, x_test, y_train, y_test = train_test_split(feature_files, label_files, test_size=0.3)
    print("| Train number:", len(y_train), '\t Value number:', len(y_test))

    print("—" * 50)
    print("| Train images:")
    to_coco(args, y_train, train=True)

    print("—" * 50)
    print("| Test images:")
    to_coco(args, y_test, train=False)


if __name__ == "__main__":
    print("—" * 50)
    main()
    print("—" * 50)
