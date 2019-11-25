import os

import cv2
import numpy as np

from perception import pointrcnn
from .pointrcnn import run as Pointnet

from perception import lanes
from .lanes import run as Lanes

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


def run(sampledata, pointrcnn_model_file, lanes_model_file, video_writer, logger):
    """
    """
    pointrcnn_model = pointrcnn.init_model(sampledata, pointrcnn_model_file, logger)
    lanes_model = lanes.init_model(lanes_model_file, logger)
    
    for i in range(sampledata.num_samples):
        dataset_item = sampledata[i]

        detections_3d = Pointnet.run(pointrcnn_model, dataset_item, sampledata.calib)
        
        # draw 3d bounding boxes on input image
        img = draw_3d_bbox(detections_3d, dataset_item['img'], sampledata.calib.P2)

        # detect and draw drivable space
        img = Lanes.run(lanes_model, img)

        video_writer.write(img)
        
        if i % 100 == 0:
            logger.debug('finished {} frames'.format(i))
