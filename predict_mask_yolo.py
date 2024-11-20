from ultralytics import YOLO
import numpy as np
import cv2, os
from PIL import Image

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

def predict_one_image(model, img):
    results = model.predict(img)
      # 單通道掩膜

    if hasattr(results[0], 'masks') and results[0].masks is not None:
        mask_data = results[0].masks.data.cpu().numpy()  # 提取二值化掩膜數據 (形狀: [num_masks, H, W])
        input_h,input_w=mask_data.shape[1],mask_data.shape[2]
        mask = np.zeros((input_h, input_w), dtype=np.uint8)
        _, nw, nh = resize_image(Image.fromarray(img), (input_w, input_h))
        for i in range(mask_data.shape[0]):  # 遍歷每個掩膜
            binary_mask = mask_data[i]  # 單個掩膜
            mask[binary_mask > 0.5] = 255  # 設定閾值以將掩膜區域設為 255 (白色)
        mask = mask[int((input_h - nh) // 2): int((input_h - nh) // 2 + nh), \
             int((input_w- nw) // 2): int((input_w - nw) // 2 + nw)]

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return mask

def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('L', size, (0,))  # 'L' mode for single channel (grayscale)
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):
    print('Num classes', num_classes)
    # 創建confusion matrix
    hist = np.zeros((num_classes, num_classes))

    # gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]
    # pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]
    gt_imgs = [os.path.join(gt_dir, x) for x in png_name_list]
    pred_imgs = [os.path.join(pred_dir, x) for x in png_name_list]


    #  image-label pair
    for ind in range(len(gt_imgs)):

        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        # 將標籤結果轉化成0開頭的值
        pred = np.where(pred > 0, 1, 0)
        label = np.where(label > 0, 1, 0)


        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue


        #   hist_matrix : num_class x num_class
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            )
            )
    # iou for all images
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    # iou for each class
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                  + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
                round(Precision[ind_class] * 100, 2)))


    # 所有類別的miou值
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
    return np.array(hist, int), IoUs, PA_Recall, Precision

def preict_all(model, img_dir, save_dir):
    img_list = os.listdir(img_dir)
    for name in img_list:
        img_path = os.path.join(img_dir, name)
        img = cv2.imread(img_path)
        if img is not None:
            pred_mask = predict_one_image(model, img)
            pred_path = os.path.join(save_dir, name.replace('.jpg', '.png'))
            cv2.imwrite(pred_path, pred_mask)

def get_all_predict_and_true(true_dir, pred_dir):
    name_list = os.listdir(true_dir)
    gt_imgs, pred_imgs = [], []
    for name in name_list:
        true_path = os.path.join(true_dir, name)
        pred_path = os.path.join(pred_dir, name.replace('.jpg', '.png'))
        true_mask = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
        true_mask = np.where(true_mask > 0, 1, 0)
        gt_imgs.append(true_mask)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = np.where(pred_mask > 0, 1, 0)
        pred_imgs.append(pred_mask)
    return gt_imgs, pred_imgs


model = YOLO('runs/segment/train8/weights/best.pt')
true_dir = '/home/kingargroo/fungs_rect_imgs/voc2/train/images'
save_dir = '/home/kingargroo/fungs_rect_imgs/voc2/train/masks2'
preict_all(model, true_dir, save_dir)
# gt_dir='/home/kingargroo/fungs_rect_imgs/voc2/test/masks'
# pred_dir='/home/kingargroo/fungs_rect_imgs/voc2/test/masks2'
# image_ids=os.listdir(gt_dir)
# hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes=2, name_classes=['background','bacteria'])  # 执行计算mIoU的函数
