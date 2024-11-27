import os
import json
import numpy as np
import glob
from PIL import Image, ImageDraw
from labelme import utils


class Labelme2COCO:
    def __init__(self, labelme_json=[], save_json_path="./coco.json"):
        """
        :param labelme_json: 所有labelme的json
        :param save_json_path: 保存路徑，json註冊表
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label_set = set()
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, "r") as fp:
                data = json.load(fp)
                self.images.append(self.image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"]
                    if label not in self.label_set:
                        self.label_set.add(label)
                    points = shapes["points"]
                    shape_type = shapes.get("shape_type", "polygon")
                    self.annotations.append(
                        self.annotation(points, label, num, shape_type)
                    )
                    self.annID += 1

        # 編號
        self.label_list = sorted(list(self.label_set))
        for label in self.label_list:
            self.categories.append(self.category(label))

        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(annotation["category_id"])

    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data["imageData"])
        height, width = img.shape[:2]
        image["height"] = height
        image["width"] = width
        image["id"] = num
        image["file_name"] = "../"+self.labelme_json[0].split('/')[-2]+'/'+data["imagePath"].split("\\")[-1]
        self.height = height
        self.width = width
        return image

    def category(self, label):
        category = {}
        category["supercategory"] = label
        category["id"] = len(self.categories) + 1  # ID从1开始
        category["name"] = label
        return category

    def annotation(self, points, label, num, shape_type):
        annotation = {}
        if shape_type == "circle":
            # 圓形標註
            center = np.array(points[0])
            circumference_point = np.array(points[1])
            radius = np.linalg.norm(center - circumference_point)
            num_points = 32  # 用于近似圆形的多边形顶点数
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            circle_points = [
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                ]
                for angle in angles
            ]
            contour = np.array(circle_points)
            x = contour[:, 0]
            y = contour[:, 1]
            area = np.pi * radius * radius
            segmentation = [list(contour.flatten())]
            x_min = center[0] - radius
            y_min = center[1] - radius
            bbox = [x_min, y_min, 2 * radius, 2 * radius]
        else:
            # 多邊形標註
            contour = np.array(points)
            x = contour[:, 0]
            y = contour[:, 1]
            area = 0.5 * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )
            segmentation = [list(np.asarray(points).flatten())]
            bbox = self.getbbox(points)
        annotation["segmentation"] = segmentation
        annotation["iscrowd"] = 0
        annotation["area"] = float(area)
        annotation["image_id"] = num
        annotation["bbox"] = list(map(float, bbox))
        annotation["category_id"] = label  # 后续会更新
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print(f"Label '{label}' not found in categories.")
        exit()
        return -1

    def getbbox(self, points):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min = min(x_coords)
        y_min = min(y_coords)
        width = max(x_coords) - x_min
        height = max(y_coords) - y_min
        return [x_min, y_min, width, height]

    def data2coco(self):
        data_coco = {
            "images": self.images,
            "categories": self.categories,
            "annotations": self.annotations,
        }
        return data_coco

    def save_json(self):
        print("Saving COCO JSON...")
        self.data_transfer()
        data_coco = self.data2coco()
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        with open(self.save_json_path, "w") as f:
            json.dump(data_coco, f, indent=4)
        print(f"COCO JSON saved at: {self.save_json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Labelme annotations to COCO format."
    )
    parser.add_argument(
        "--labelme_images",
        help="Directory containing Labelme images and JSON files.",
        default="/home/kingargroo/fungs_rect_imgs/coco3/val2017",
    )
    parser.add_argument(
        "--output", help="Output JSON file path.", default="instances_val2017.json"
    )
    args = parser.parse_args()
    labelme_json = glob.glob(os.path.join(args.labelme_images, "*.json"))
    Labelme2COCO(labelme_json, args.output)
