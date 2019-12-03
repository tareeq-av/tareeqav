import os

import cv2
import numpy as np

from perception.no_lidar import yolov3
from perception.no_lidar import psmnet

from perception.no_lidar.yolov3 import run as YOLO
# from perception.no_lidar import pointnets
from perception.no_lidar.psmnet import run as PsmNet



# from perception.no_lidar.pointnets import run as PseudoLidarPointNet

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


def run(
        sampledata,
        pointnet_model_file,
        lanes_model_file,
        video_writer,
        logger,
        disp_model_file=None,
        yolov3_weights_file=None,
        yolov3_config_file=None,
        with_lidar=False
        ):
    """
    """
    if with_lidar:
        points_model = pointrcnn.init_model(sampledata, pointnet_model_file, logger)
    else:
        if not disp_model_file:
            raise RuntimeError("Please provide a pre-trained disparity model when using the No-Lidar option")
        
        yolov3_model = yolov3.init_model(yolov3_weights_file, yolov3_config_file, logger)
        disp_model = psmnet.init_model(disp_model_file, logger)
        # points_model = pointnets.init_model(sampledata, pointnet_model_file, logger)
        
        
    # lanes_model = lanes.init_model(lanes_model_file, logger)
    
    for i in range(sampledata.num_samples):
        dataset_item = sampledata[i]

        if with_lidar:
            detections_3d = Pointnet.run(points_model, dataset_item, sampledata.calib)
        else:
            logger.info("Running a 2D object detector")
            detections_2d = YOLO.run(yolov3_model,  dataset_item)
            logger.debug("YOLOv3 returned {}".format(detections_2d))

            logger.info("Running a disparity estimation network")
            disp_map = PsmNet.run(disp_model, dataset_item['left_img'], dataset_item['right_img'])
            logger.debug("PSMNet returned a disparity map with shape {}".format(disp_map.shape))
            
        #     detections_3d = PseudoLidarPointNet.run(detections_2d, pseudo_velo, sampledata.calib)
        
        # # draw 3d bounding boxes on input image
        # img = draw_3d_bbox(detections_3d, dataset_item['img'], sampledata.calib.P2)

        # # detect and draw drivable space
        # # img = Lanes.run(lanes_model, img)

        # cv2.imwrite('./output/images/'+str(i)+'.png', img)
        # video_writer.write(img)
        
        # if i % 100 == 0:
        #     logger.debug('finished {} frames'.format(i))
