import os

import cv2
import numpy as np

from perception import yolov3
from perception import psmnet
from perception import pointnets

from perception.yolov3 import run as YOLO
from perception.psmnet import run as PsmNet
from perception.pointnets import run as PseudoLidarPointNet

# from perception.lidar import pointrcnn
# from perception.lidar.pointrcnn import run as LidarPointNet

# from perception import lanes
# from .lanes import run as Lanes

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


def project_disp_to_points(calib, disp, max_high=1):
    disp[disp < 0] = 0
    baseline = 0.54
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1. - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


def not_far_away(detection):
    xmin,ymin,xmax,ymax = detection[1:]
    if xmax - xmin < 20 or ymax - ymin < 20:
        return False
    return True

def run(
        sampledata,
        pointnet_model_file,
        lanes_model_file,
        video_writer,
        logger,
        disp_model_file=None,
        yolov3_weights_file=None,
        yolov3_config_file=None
        ):
    """
    """
    if not disp_model_file:
        raise RuntimeError("Please provide a pre-trained disparity model when using the No-Lidar option")
        
    yolov3_model = yolov3.init_model(yolov3_weights_file, yolov3_config_file, logger)
    disp_model = psmnet.init_model(disp_model_file, logger)
    tf_sess, tf_ops = pointnets.init_model(pointnet_model_file, batch_size=1)
        
    # lanes_model = lanes.init_model(lanes_model_file, logger)
    
    for i in range(sampledata.num_samples):
        # if i > 75: break
        dataset_item = sampledata[i]

        logger.info("[Perception] Running one frame through the pipeline with pseudo-lidar")
        detections_2d = YOLO.run(yolov3_model,  dataset_item)
        logger.debug("YOLOv3 returned {}".format(detections_2d))
        
        logger.debug("Filter out anything other than Car and Pedestrians")
        detections_2d = list(filter(lambda detection: detection[0] in (0, 2,), detections_2d))
        logger.debug("Remove any detection that is too small (far away)")
        detections_2d = list(filter(not_far_away, detections_2d))

        logger.debug("Running a disparity estimation network")
        disp_map = PsmNet.run(disp_model, dataset_item['left_img'], dataset_item['right_img'])
        logger.debug("PSMNet returned a disparity map with shape {}".format(disp_map.shape))
        
        logger.debug("Generating pseudo-lidar from disparity map")
        pseudo_velo = project_disp_to_points(sampledata.calib, disp_map)
        pseudo_velo = np.concatenate([pseudo_velo, np.ones((pseudo_velo.shape[0], 1))], 1)
        pseudo_velo = pseudo_velo.astype(np.float32)

        logger.debug("Running RBG-D 3D object detection with pseudo-lidar")
        detections_3d = []
        for detection in detections_2d:
            result = PseudoLidarPointNet.run(
                    tf_sess, tf_ops,
                    detection,
                    pseudo_velo,
                    dataset_item['left_img_cv2'],
                    sampledata.calib
            )
            detections_3d.append(result)
    
    # draw 3d bounding boxes on input image
    img = draw_3d_bbox(detections_3d, dataset_item['left_img_cv2'], sampledata.calib.P2)
    cv2.imshow('Window', img)
    cv2.waitKey(0)
    # # detect and draw drivable space
    # # img = Lanes.run(lanes_model, img)

    video_writer.write(img)
  