from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset

evaluator = COCOEvaluator("test", cfg, False, output_dir=cfg.OUTPUT_DIR)
test_loader = build_detection_test_loader(cfg, "test")
metrics = inference_on_dataset(trainer.model, test_loader, evaluator)
print(metrics)





cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = 'cuda'
predictor = DefaultPredictor(cfg)
im = cv2.imread("/home/kingargroo/Documents/corndataset/test/images/576_7488_57.png")
outputs = predictor(im)
v = Visualizer(im, metadata=metadata, scale=1., instance_mode =  ColorMode.IMAGE    )
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
img = v.get_image()[:,:,[2,1,0]]
cv2.imwrite('demo.png',img)






