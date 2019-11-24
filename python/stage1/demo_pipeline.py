import os
import logging

import cv2
import torch
import numpy as np

from perception.pointrcnn.lib.config import cfg
from perception.pointrcnn.lib.utils import kitti_utils
from perception.pointrcnn.lib.utils.iou3d import iou3d_utils
from perception.pointrcnn.lib.net.point_rcnn import PointRCNN
from perception.pointrcnn.lib.utils.bbox_transform import decode_bbox_target

from sampledata import KittiRawData

np.random.seed(1024)  # set the same seed

colors = {
    'yellow': (0,255,255),
    'green': (0, 255, 0),
    'pink': (255, 0, 255),
    'red': (0, 0, 255)
}


def plot_3d_bbox(img, label_info, cam_to_img, is_gt=True):
    # print('current label info: ', label_info)
    alpha = label_info['alpha']
    # theta_ray = label_info['theta_ray']
    box_3d = []
    center = label_info['location']
    dims = label_info['dimension'] 
    
    color = colors['green']
    

    if center[0] > 20 or center[0] < -20:
        return img

    if center[2] < 20 or (2.5 >= center[0] >= -2.5):# in meters
        color = colors['yellow']
    
    if center[2] < 20 and (5 >= center[0] >= -5):# in meters
        color = colors['red']
    
    cam_to_img = cam_to_img#label_info['calib']

    rot_y = label_info['rot_y']

    height, width = img.shape[0], img.shape[1]

    for i in [1, -1]:
        for j in [1, -1]:
            for k in [0, 1]:
                point = np.copy(center)
                point[0] = center[0] + i * dims[1] / 2 * np.cos(-rot_y + np.pi / 2) + (j * i) * dims[2] / 2 * np.cos(
                    -rot_y)
                point[2] = center[2] + i * dims[1] / 2 * np.sin(-rot_y + np.pi / 2) + (j * i) * dims[2] / 2 * np.sin(
                    -rot_y)
                point[1] = center[1] - k * dims[0]
                
                point = np.append(point, 1)
                point = np.dot(cam_to_img, point)
                point = point[:2] / point[2]
                point = point.astype(np.int16)
                box_3d.append(point)
    
    for i in range(4):
        point_1_ = box_3d[2 * i]
        point_2_ = box_3d[2 * i + 1]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color, 1)

    for i in range(8):
        point_1_ = box_3d[i]
        point_2_ = box_3d[(i + 2) % 8]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color, 1)

    return img

    
def draw_3d_bbox(detections, image, camp_to_img):
    for l in detections:
        label_info = {}
        # this angle need a preprocess
        label_info['alpha'] = float(l[3])
        label_info['location'] = np.asarray(l[11: 14], dtype=float)
        label_info['dimension'] = np.asarray(l[8: 11], dtype=float)
        label_info['rot_y'] = float(l[14])
        label_info['box'] = np.asarray(l[4: 7], dtype=float)
        
        image = plot_3d_bbox(image, label_info, camp_to_img)
    return image


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def load_sampledata(scene_base_dir, scene_name):
    
    # data_path = os.path.join(curr_dir, '../data_samples/kitti')
    # base_dir = '/home/sameh/Autonomous-Vehicles/Datasets/Kitti-Raw/kitti_data'
    # data_path = os.path.join(base_dir, scene_date)
    # scene_name = "2011_09_26_drive_0009_sync"
    
    return KittiRawData(scene_base_dir, scene_name)


def init_pointrcnn(dataset, pointrcnn_filename, logger):
    model = PointRCNN(num_classes=dataset.num_classes, use_xyz=True, mode='TEST')
    model.cuda()
    
    if not os.path.isfile(pointrcnn_filename):
        raise FileNotFoundError(pointrcnn_filename)

    logger.info("==> Loading from checkpoint '{}'".format(pointrcnn_filename))
    checkpoint = torch.load(pointrcnn_filename)
    model.load_state_dict(checkpoint['model_state'])
    total_keys = len(model.state_dict().keys())


    model.eval()
    return model


def generate_detections(calib, bbox3d, scores, img_shape):
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
        
        # print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
        #       (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
        #        bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
        #        bbox3d[k, 6], scores[k]), file=f)
    return detections


def run(model, dataset_item, calib):
    """
    """
    np.random.seed(666)
    CLS_MEAN_SIZE = np.array([[1.52, 1.63, 3.88]], dtype=np.float32)
    MEAN_SIZE = torch.from_numpy(CLS_MEAN_SIZE[0]).cuda()
    mode = 'TEST'

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    total_roi_recalled_bbox_list = [0] * 5
    cnt = final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0

    # sample_id, pts_rect, pts_features, pts_input = \
    #         data['sample_id'], data['pts_rect'], data['pts_features'], data['pts_input']

    batch_size = 1

    inputs = torch.from_numpy(dataset_item['pts_input']).cuda(non_blocking=True).float()
    inputs = inputs.unsqueeze(0)
    input_data = {'pts_input': inputs}

    # model inference
    ret_dict = model(input_data)
    
    roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
    roi_boxes3d = ret_dict['rois']  # (B, M, 7)
    seg_result = ret_dict['seg_result'].long()  # (B, N)

    rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
    rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

    anchor_size = MEAN_SIZE
    pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                          anchor_size=anchor_size,
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

    # evaluation
    recalled_num = gt_num = rpn_iou = 0

    disp_dict = {'mode': mode, 'recall': '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox)}

    # save roi and refine results
    roi_boxes3d_np = roi_boxes3d.cpu().detach().numpy()
    pred_boxes3d_np = pred_boxes3d.cpu().detach().numpy()
    roi_scores_raw_np = roi_scores_raw.cpu().detach().numpy()
    raw_scores_np = raw_scores.cpu().detach().numpy()

    # scores thresh
    inds = norm_scores > cfg.RCNN.SCORE_THRESH

    k = 0 # using a batch of one at inference time
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
    pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu().detach().numpy(), scores_selected.cpu().detach().numpy()

    final_total += pred_boxes3d_selected.shape[0]
    image_shape = dataset_item['img'].shape

    return generate_detections(calib, pred_boxes3d_selected, scores_selected, image_shape)


def main(scene_base_dir, scene_name):
    """
    """
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    # output_dir = os.path.join(curr_dir, 'output/{}'.format(scene_name))
    output_dir = os.path.join(curr_dir, 'output')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pointrcnn_model_file = os.path.join(curr_dir, 'PointRCNN.pth')
    log_file = os.path.join(curr_dir, 'log_perception_pipeline.txt')
    
    logger = create_logger(log_file)
    sampledata = load_sampledata(scene_base_dir, scene_name)
    
    pointrcnn_model = init_pointrcnn(sampledata, pointrcnn_model_file, logger)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    shape = sampledata[0]['img'].shape
    out = cv2.VideoWriter('{}/{}.avi'.format(output_dir,scene_name), fourcc, 10, (shape[1], shape[0]), True)

    print('starting...')
    for i in range(sampledata.num_samples):
        dataset_item = sampledata[i]
        detections = run(pointrcnn_model, dataset_item, sampledata.calib)
        
        # draw 3d bounding boxes on input image
        image = draw_3d_bbox(detections, dataset_item['img'], sampledata.calib.P2)
        out.write(image)
        
        if i % 100 == 0:
            print('finished', i, 'samples')

    print('done')
    out.release()



if __name__ == '__main__':
    base_dir = '/home/sameh/Autonomous-Vehicles/Datasets/Kitti-Raw/kitti_data'
    scene_dates = [
        '2011_09_26',
        # '2011_09_28' ,
        '2011_09_29', 
        # '2011_09_30',
        # '2011_10_03'
    ]

    scene_names = [
        # '2011_09_26_drive_0001_sync',
        # '2011_09_26_drive_0011_sync',
        # '2011_09_26_drive_0028_sync',
        # '2011_09_26_drive_0032_sync',
        '2011_09_26_drive_0056_sync',
        # '2011_09_29_drive_0004_sync',
    ]

    for scene_date in scene_dates:
        scene_dir = os.path.join(base_dir, scene_date)
        # scene_names = os.listdir(scene_dir)
        for scene_name in scene_names:
            print('processing images from', scene_name)
            scene_path = os.path.join(scene_dir, scene_name)
            if os.path.isdir(scene_path):
                main(scene_dir, scene_name)
