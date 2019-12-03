import os
import sys

import cv2
from PIL import Image

import importlib
import numpy as np
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR  = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
MODEL_DIR = os.path.join(CURRENT_DIR, 'models')

sys.path.append(MODEL_DIR)
sys.path.append(os.path.join(CURRENT_DIR, 'train'))

import provider

from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER

import mayavi.mlab as mlab

# Set training configurations
BATCH_SIZE = 1
MODEL_PATH = os.path.join(MODEL_DIR, 'model.ckpt')
GPU_INDEX = 0
NUM_POINT = 1024
MODEL = importlib.import_module('frustum_pointnets_v1')
NUM_CLASSES = 2
NUM_CHANNEL = 4

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def inference(sess, ops, pc, one_hot_vec, batch_size):
    ''' Run inference for frustum pointnets in batch mode '''
    assert pc.shape[0]%batch_size == 0
    num_batches = pc.shape[0]/batch_size
    print('>>>>>>>> NUM BATCHES', num_batches)
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[0],)) # 3D box score 
   
    ep = ops['end_points'] 
    for i in range(num_batches):
        feed_dict = {\
            ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
            ops['one_hot_vec_pl']: one_hot_vec[i*batch_size:(i+1)*batch_size,:],
            ops['is_training_pl']: False}

        batch_logits, batch_centers, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = \
            sess.run([ops['logits'], ops['center'],
                ep['heading_scores'], ep['heading_residuals'],
                ep['size_scores'], ep['size_residuals']],
                feed_dict=feed_dict)

        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
        centers[i*batch_size:(i+1)*batch_size,...] = batch_centers
        heading_logits[i*batch_size:(i+1)*batch_size,...] = batch_heading_scores
        heading_residuals[i*batch_size:(i+1)*batch_size,...] = batch_heading_residuals
        size_logits[i*batch_size:(i+1)*batch_size,...] = batch_size_scores
        size_residuals[i*batch_size:(i+1)*batch_size,...] = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:,:,1] # BxN
        batch_seg_mask = np.argmax(batch_logits, 2) # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask,1) # B,
        heading_prob = np.max(softmax(batch_heading_scores),1) # B
        size_prob = np.max(softmax(batch_size_scores),1) # B,
        batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        scores[i*batch_size:(i+1)*batch_size] = batch_scores 
        # Finished computing scores

    heading_cls = np.argmax(heading_logits, 1) # B
    size_cls = np.argmax(size_logits, 1) # B
    heading_res = np.array([heading_residuals[i,heading_cls[i]] \
        for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i,size_cls[i],:] \
        for i in range(pc.shape[0])])

    return np.argmax(logits, 2), centers, heading_cls, heading_res, \
        size_cls, size_res, scores


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)

    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    
    imgfov_pc_velo = pc_velo[fov_inds,:]
    
    return imgfov_pc_velo, pts_2d, fov_inds


def get_center_view_rot_angle(frustum_angle):
    ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
    can be directly used to adjust GT heading angle '''
    return np.pi/2.0 + frustum_angle

def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc


def get_center_view_point_set(pseudo_velo, frustum_angle):
    ''' Frustum rotation of point clouds.
    NxC points with first 3 channels as XYZ
    z is facing forward, x is left ward, y is downward
    '''
    # Use np.copy to avoid corrupting original data
    point_set = np.copy(pseudo_velo)
    return rotate_pc_along_y(point_set,
            get_center_view_rot_angle(frustum_angle))


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i,2]
        color = cmap[int(640.0/depth),:]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i,0])),
            int(np.round(imgfov_pts_2d[i,1]))),
            2, color=tuple(color), thickness=-1)
    Image.fromarray(img).show() 
    return img

def draw_lidar(pc, color=None, fig=None, bgcolor=(0,0,0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=pts_color, mode=pts_mode, colormap = 'gnuplot', scale_factor=pts_scale, figure=fig)
    
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    
    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)

    # draw fov (todo: update to real sensor spec.)
    fov=np.array([  # 45 degree
        [20., 20., 0.,0.],
        [20.,-20., 0.,0.],
    ],dtype=np.float64)
    
    mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
   
    # draw square region
    TOP_Y_MIN=-20
    TOP_Y_MAX=20
    TOP_X_MIN=0
    TOP_X_MAX=40
    TOP_Z_MIN=-2.0
    TOP_Z_MAX=0.4
    
    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    
    #mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def calculate_frustum_angle(box2d_center_rect):
    return -1 * np.arctan2(box2d_center_rect[0,2],
        box2d_center_rect[0,0])


def get_center_view_rot_angle(frustum_angle):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + frustum_angle


def extract_frustum_data_rgb_detection(detection, pseudo_velo, img, calib, augmentX=1):

    pc_rect = np.zeros_like(pseudo_velo)
    pc_rect[:,0:3] = calib.project_velo_to_rect(pseudo_velo[:,0:3])
    pc_rect[:,3] = pseudo_velo[:,3]
    
    img_height, img_width, img_channel = img.shape

    _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(
                pseudo_velo[:,0:3],
                calib,
                0,
                0,
                img_width,
                img_height)

    # we only care about car, pedestrian and cyclist(later)
    # that's 0 , 3 and 5 from kitti
    # whic are 0,2,  from coco names 
    if detection[0] not in (0, 2,): # car and person from coco names
        return None, None, None

    xmin,ymin,xmax,ymax = detection[1:]

    box_fov_inds = (pc_image_coord[:,0]<xmax) & \
                (pc_image_coord[:,0]>=xmin) & \
                (pc_image_coord[:,1]<ymax) & \
                (pc_image_coord[:,1]>=ymin)

    box_fov_inds = box_fov_inds & img_fov_inds
    
    pc_in_box_fov = pc_rect[box_fov_inds,:]

    # # Get frustum angle (according to center pixel in 2D BOX)
    box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
    
    uvdepth = np.zeros((1,3))
    
    uvdepth[0,0:2] = box2d_center

    uvdepth[0,2] = 20 # some random depth
    
    box2d_center_rect = calib.project_image_to_rect(uvdepth)

    frustum_angle = calculate_frustum_angle(box2d_center_rect)
    
    rot_angle = get_center_view_rot_angle(frustum_angle)

    # # Get point cloud
    point_set = get_center_view_point_set(pc_in_box_fov, frustum_angle)
    
    # # Resample
    choice = np.random.choice(point_set.shape[0], NUM_POINT, replace=True)
    point_set = point_set[choice, :]

    return point_set[:,0:NUM_CHANNEL], rot_angle, frustum_angle


def run(sess, ops, detections_2d, pseudo_velo, img, calib):
    """Test frustum pointents with 2D boxes from a RGB detector.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    """
    num_batches = 1
    batch_size = len(detections_2d)

    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    
    batch_one_hot_to_feed = np.zeros((batch_size, 3))
    batch_data_to_feed = np.zeros((batch_size, NUM_POINT, NUM_CHANNEL))

    # sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)

    for index, detection in enumerate(detections_2d):
        ps, rot_angle, _ = extract_frustum_data_rgb_detection(detection, pseudo_velo, img, calib)
        if isinstance(ps, type(None)):
            print('???????????????????????????????????????', detection)
            continue

        # we have to swtich indices from coco class indices
        # to kitti class indices
        coco_id = detection[0]
        if coco_id == 0: # that's a person and is 1 in kitti classes
            kitti_id = 1
        elif coco_id == 2: # that's a car is 0 in kitti classes
            kitti_id = 0
        else:
            raise Exception("unsupported detection class " + str(coco_id))

        batch_one_hot_to_feed[index][kitti_id] = 1
        batch_data_to_feed[index,...] = ps
        rot_angle_list.append(rot_angle)

    # Run one batch inference
    batch_output, batch_center_pred, \
        batch_hclass_pred, batch_hres_pred, \
        batch_sclass_pred, batch_sres_pred, batch_scores = \
            inference(sess, ops, batch_data_to_feed,
                batch_one_hot_to_feed, batch_size=batch_size)

    for i in range(batch_size):
        center_list.append(batch_center_pred[i,:])
        heading_cls_list.append(batch_hclass_pred[i])
        heading_res_list.append(batch_hres_pred[i])
        size_cls_list.append(batch_sclass_pred[i])
        size_res_list.append(batch_sres_pred[i,:])
        score_list.append(batch_scores[i])

    results = []
    for i in range(len(center_list)):
        result = []
        # idx = id_list[i]
        # output_str = type_list[i] + " -1 -1 -10 "
        # box2d = box2d_list[i]
        # output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(
                    center_list[i],
                    heading_cls_list[i],
                    heading_res_list[i],
                    size_cls_list[i],
                    size_res_list[i],
                    rot_angle_list[i]
                    )

        xmin,ymin,xmax,ymax = detections_2d[i][1:]

        result.extend([detections_2d[i][0], -1, -1, -10,xmin,ymin,xmax,ymax,h,w,l,tx,ty,tz,ry,score_list[i]])
        results.append(result)

        # output_str += "%f %f %f %f %f %f %f %f" % ()
        # if idx not in results: results[idx] = []
        # results[idx].append(output_str)
    
    return results

# if __name__ == '__main__':
#     run([], [], [], None)