import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
import torch, torchvision,cv2,os
from glob import glob
import torchvision.transforms as transforms
from pathlib import Path

def create_data_pairs(input_path, dir_type = 'train'):

    img_paths = Path(input_path +'/' +dir_type + '/images/').glob('*.png')

    pairs = []
    for img_path in img_paths:

        file_name_tmp = str(img_path).split('/')[-1].split('.')
        file_name_tmp.pop(-1)
        file_name = '.'.join((file_name_tmp))

        label_path = Path(input_path +'/'+ dir_type + '/labels/' + file_name + '.txt')



        line_img = input_path+'/'+ dir_type+'/images/'+ file_name + '.png'
        line_annot = input_path+'/'+dir_type+'/labels/' + file_name + '.txt'
        pairs.append([line_img, line_annot])

    return pairs


def create_coco_format(data_pairs):
    data_list = []

    for i, path in enumerate(data_pairs):

        filename = path[0]

        img_h, img_w = cv2.imread(filename).shape[:2]

        img_item = {}
        img_item['file_name'] = filename
        img_item['image_id'] = i
        img_item['height'] = img_h
        img_item['width'] = img_w

        print(str(i), filename)

        annotations = []
        with open(path[1]) as annot_file:
            lines = annot_file.readlines()
            for line in lines:
                if line[-1] == "\n":
                    box = line[:-1].split(' ')
                else:
                    box = line.split(' ')

                class_id = box[0]
                x_c = float(box[1])
                y_c = float(box[2])
                width = float(box[3])
                height = float(box[4])

                x1 = (x_c - (width / 2)) * img_w
                y1 = (y_c - (height / 2)) * img_h
                x2 = (x_c + (width / 2)) * img_w
                y2 = (y_c + (height / 2)) * img_h

                annotation = {
                    "bbox": list(map(float, [x1, y1, x2, y2])),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int(class_id),
                    "iscrowd": 0
                }
                annotations.append(annotation)
            img_item["annotations"] = annotations
        data_list.append(img_item)
    return data_list

input_path = '/home/kingargroo/Documents/corndataset'
train = create_data_pairs(input_path, 'train')
val = create_data_pairs(input_path,  'val')
test = create_data_pairs(input_path,  'test')


train_list = create_coco_format(train)
val_list = create_coco_format(val)
test_list=create_coco_format(test)
for catalog_name, file_annots in [("train", train_list), ("val", val_list),("test",test_list)]:
    DatasetCatalog.register(catalog_name, lambda file_annots = file_annots: file_annots)
    MetadataCatalog.get(catalog_name).set(thing_classes=['corn'])

metadata = MetadataCatalog.get("train")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.DEVICE = 'cuda' # cpu
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 8000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 640 # 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("train").thing_classes)
cfg.DATASETS.TEST = ("val",)
cfg.TEST.EVAL_PERIOD = 500
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

import time as t
s1 = t.time()
try:
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
except:
  None
s2 = t.time()
print(s2-s1)

# visualize test
# from detectron2.utils.visualizer import Visualizer
# import matplotlib.pyplot as plt
# import random
# dataset_dicts = train_list  # 或者 val_list
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     plt.figure(figsize=(12, 8))
#     plt.imshow(vis.get_image())
#     plt.show()