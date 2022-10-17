# -*- coding: UTF-8 -*-
import argparse
from re import L
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random, array, zeros, logical_and, logical_or, mean
import copy
from glob import glob
from os.path import join
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    # coords[:, 8].clamp_(0, img0_shape[1])  # x5
    # coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def calc_iou(pred_landmarks, target_landmarks, image_shape):
    h, w = image_shape[:2]
    target_landmarks = (target_landmarks.reshape(-1, 2) * (w, h)).astype('int32')
    pred_landmarks = (array(pred_landmarks).reshape(-1, 2) * (w, h)).astype('int32')
    pred_mask = cv2.fillPoly(zeros(image_shape).astype('uint8'), [pred_landmarks], [255, 255, 255])
    target_mask = cv2.fillPoly(zeros(image_shape).astype('uint8'), [target_landmarks], [255, 255, 255])
    intersection = logical_and(pred_mask > 0, target_mask > 0).sum()
    union = logical_or(pred_mask > 0, target_mask > 0).sum()
    return intersection / union


def mask_iou(pred_landmarks, target_landmarks, image_shape):
    pred_landmarks = pred_landmarks.cpu().numpy()
    target_landmarks = target_landmarks.cpu().numpy()
    h, w = image_shape[:2]
    target_landmarks = (target_landmarks.reshape(4, 2) * (w, h)).astype('int32')
    pred_landmarks = (array(pred_landmarks).reshape(-1, 4, 2)).astype('int32')
    ious = []
    target_mask = cv2.fillPoly(zeros(image_shape).astype('uint8'), [target_landmarks], [255, 255, 255])
    for pred_landmark in pred_landmarks:
        pred_mask = cv2.fillPoly(zeros(image_shape).astype('uint8'), [pred_landmark], [255, 255, 255])
        intersection = logical_and(pred_mask > 0, target_mask > 0).sum()
        union = logical_or(pred_mask > 0, target_mask > 0).sum()
        ious.append(intersection / union)
    return torch.tensor(ious).reshape(-1, 1).cuda()


def seg_eval(model, opt, device):
    # Load model
    image_dir = opt.image
    img_size = opt.img_size
    conf_thres = opt.conf_thres
    iou_thres = opt.iou_thres

    image_pths = glob(join(image_dir, '*.jpg'))
    ious = []

    total_time = 0
    # image_pths = image_pths[:500]
    for image_path in tqdm(image_pths):
        txt_path = image_path.replace('.jpg', '.txt')
        with open(txt_path, 'r') as f:
            labels = f.readlines()[0][:-1].split(' ')
            target_landmarks = array([float(el) for el in labels[5:]])

        orgimg = cv2.imread(image_path)  # BGR
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found ' + image_path
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        # Run inference
        t0 = time.time()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        start = time.time()
        pred = model(img)[0]
        total_time += time.time() - start

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        # print('img.shape: ', img.shape)
        # print('orgimg.shape: ', orgimg.shape)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
            gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

                for j in range(det.size()[0]):
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = (det[j, 5:13].view(1, 8) / gn_lks).view(-1).tolist()
                    class_num = det[j, 13].cpu().numpy()
                    iou = calc_iou(landmarks, target_landmarks, orgimg.shape)
                    ious.append(iou)
    print('mean iou: {} %'.format(round(mean(ious) * 100, 2)))
    print(total_time / len(image_pths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='data/card_images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = load_model(opt.weights, device)
    seg_eval(model, opt, device)
