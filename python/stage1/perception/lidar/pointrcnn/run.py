import os
import logging

import cv2
import torch
import numpy as np

from perception.pointrcnn.lib.config import cfg
from perception.pointrcnn.lib.utils import kitti_utils
from perception.pointrcnn.lib.utils.iou3d import iou3d_utils

from perception.pointrcnn.lib.utils.bbox_transform import decode_bbox_target

def generate_detections(calib, bbox3d, scores, img_shape):
    """
    """
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    detections = []
    for k in range(bbox3d.shape[0]):
        if box_valid_mask[k] == 0:
            continue
        x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        detection = (cfg.CLASSES, -1, -1, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
               bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
               bbox3d[k, 6], scores[k])
        detections.append(detection)

    return detections


def calculate_anchor_size():
    """
    """
    # hard coded value taken from the yaml config file
    CLS_MEAN_SIZE = np.array([[1.52, 1.63, 3.88]], dtype=np.float32)
    MEAN_SIZE = torch.from_numpy(CLS_MEAN_SIZE[0]).cuda()
    return MEAN_SIZE


def inference(model, dataset_item):
    """
    """
    # using a batch of one at inference time
    batch_size = 1
    # set the same seed
    np.random.seed(666)

    inputs = torch.from_numpy(dataset_item['pts_input']).cuda(non_blocking=True).float()
    inputs = inputs.unsqueeze(0)
    ret_dict = model({'pts_input': inputs})

    roi_boxes3d = ret_dict['rois']  # (B, M, 7)
    rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
    rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

    return roi_boxes3d, rcnn_cls, rcnn_reg


def filter_detections(pred_boxes3d, raw_scores, norm_scores, calib, img_shape):
    """
    """
    # using a batch of one at inference time
    k = 0
    final_total = 0

    # scores thresh
    inds = norm_scores > cfg.RCNN.SCORE_THRESH
    cur_inds = inds[k].view(-1)

    pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
    raw_scores_selected = raw_scores[k, cur_inds]
    norm_scores_selected = norm_scores[k, cur_inds]

    # NMS thresh
    # rotated nms
    boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
    keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
    pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
    scores_selected = raw_scores_selected[keep_idx]

    pred_boxes3d_selected = pred_boxes3d_selected.cpu().detach().numpy()
    scores_selected = scores_selected.cpu().detach().numpy()

    final_total += pred_boxes3d_selected.shape[0]

    return generate_detections(calib, pred_boxes3d_selected, scores_selected, img_shape)


def run(model, dataset_item, calib):
    """
    """
    # using a batch of one at inference time
    batch_size = 1
    
    # model inference
    roi_boxes3d, rcnn_cls, rcnn_reg = inference(model, dataset_item)

    # get boxes in 3d
    pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                        anchor_size=calculate_anchor_size(),
                        loc_scope=cfg.RCNN.LOC_SCOPE,
                        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                        get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                        get_ry_fine=True).view(batch_size, -1, 7)

    # scoring
    if rcnn_cls.shape[2] == 1:
        raw_scores = rcnn_cls  # (B, M, 1)
        norm_scores = torch.sigmoid(raw_scores)
        pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
    else:
        pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
        cls_norm_scores = F.softmax(rcnn_cls, dim=1)
        raw_scores = rcnn_cls[:, pred_classes]
        norm_scores = cls_norm_scores[:, pred_classes]

    return filter_detections(pred_boxes3d, raw_scores, norm_scores, calib, dataset_item['img'].shape)
