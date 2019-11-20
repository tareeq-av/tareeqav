import os
import sys

import cv2
import numpy as np

class Calibration(object):
    def __init__(self, calib):
        
        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return:
        """
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu)**2 + (v - self.cv)**2 + self.fu**2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d**2 - x**2 - y**2)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return pts_rect


class KittiRawData:

    npoints            = 16384
    left_camera        = 'image_02/data'
    lidar              = 'velodyne_points/data'
    classes            = ('Background', 'Car')

    def __init__(self, data_path, scene_name):
        
        self.data_path  = data_path
        self.scene_name = scene_name
        try:
            self.image_ids = sorted(
                os.listdir(
                    '{}/{}/{}'.format(data_path, scene_name, self.left_camera)
                ))

            self.lidar_ids = sorted(
                os.listdir(
                    '{}/{}/{}'.format(data_path, scene_name, self.lidar)
                ))
        except OSError as msg:
            print('Please provide the correct paths:', msg)
            sys.exit(1)
        
        finally:
            self.num_samples = len(self.image_ids)
            self.calib = Calibration(self.__get_calibrations())
            self.idx_list = sorted([idx[:idx.find('.')] for idx in self.image_ids])
            self.num_classes = len(self.classes)

    
    def __get_calibrations(self):
        """
            Opens calibration files and returns the translation and rotation
            matrices for camera 2 and 3, as well as the velodyne to camera
            matrices.

            The Kitti raw data uses different a different naming convention
            for the matrices than the train/test splits.  

            This function opens the calibration files and returns a dictionary
            of the matrices using the train/test split convention as expected
            by the preptrained models.
        """
        
        calib_cam_to_cam = {
            'P_rect_02': 'P2' ,
            'P_rect_03': 'P3',
            'R_rect_00': 'R0',
        }

        calib = {}

        with open('{}/calib_cam_to_cam.txt'.format(self.data_path)) as f:
            for line in f:
                for line in f.readlines()[1:]:
                    key_str, data = line.split(':')
                    if key_str in calib_cam_to_cam:
                        calib[calib_cam_to_cam[key_str]] = np.array(data.strip().split(' '), dtype=np.float32)
        
        with open('{}/calib_velo_to_cam.txt'.format(self.data_path)) as f:
            r,t = [],[]
            for line in f.readlines()[1:]:
                if line and 'T:' in line:
                    t = line.split(':')[1].strip().split(' ')
                elif line and 'R:' in line:
                    r = line.split(':')[1].strip().split(' ')

            r1,r2,r3 = r[0:3],r[3:6],r[6:]
            
            calib['Tr_velo_to_cam'] = np.array(r1+[t[0]]+r2+[t[1]]+r3+[t[2]], dtype=np.float32)
        
        return {
            'P2'          : calib['P2'].reshape(3, 4),
            'P3'          : calib['P3'].reshape(3, 4),
            'R0'          : calib['R0'].reshape(3, 3),
            'Tr_velo2cam' : calib['Tr_velo_to_cam'].reshape(3, 4)
        }

       
    
    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)


        ## HARD CODED FOR NOW ##
        # x_range, y_range, z_range = cfg.PC_AREA_SCOPE
        PC_AREA_SCOPE =  np.array([[-40, 40],
                              [-1,   3],
                              [0, 70.4]])  # x, y, z scope in rect camera coords
        
        x_range, y_range, z_range = PC_AREA_SCOPE
        pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
        range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                        & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                        & (pts_z >= z_range[0]) & (pts_z <= z_range[1])

        pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    def _get_image(self, image_idx):
        # assert False, 'DO NOT USE cv2 NOW, AVOID DEADLOCK'
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(
            '{}/{}/{}'.format(self.data_path, self.scene_name, self.left_camera),
            image_idx
        )
        assert os.path.exists(img_file)
        img = cv2.imread(img_file)  # (H, W, 3) BGR mode
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def get_lidar(self, image_idx):
        lidar_file = os.path.join(
            '{}/{}/{}'.format(self.data_path, self.scene_name, self.lidar),
            image_idx+'.bin'
        )
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    
    def __getitem__(self, index):
        image_idx = self.image_ids[index]

        img = self._get_image(image_idx)
        img_shape = img.shape
        pts_lidar = self.get_lidar(image_idx[:image_idx.find('.')])

        # get valid point (projected points should be in image)
        pts_rect = self.calib.lidar_to_rect(pts_lidar[:, 0:3])
        pts_intensity = pts_lidar[:, 3]

        pts_img, pts_rect_depth = self.calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)

        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        pts_intensity = pts_intensity[pts_valid_flag]

        if self.npoints < len(pts_rect):
            pts_depth = pts_rect[:, 2]
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(pts_rect), dtype=np.int32)
            if self.npoints > len(pts_rect):
                extra_choice = np.random.choice(choice, self.npoints - len(pts_rect), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)

        ret_pts_rect = pts_rect[choice, :]
        ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]

        # ret_pts_rect = pts_rect
        # ret_pts_intensity = pts_intensity - 0.5

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        pts_input = ret_pts_rect

        # prepare input
        sample_info = {}
        sample_info['img'] = img
        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = ret_pts_rect
        sample_info['pts_features'] = ret_pts_features
        # sample_info['rpn_cls_label'] = rpn_cls_label
        # sample_info['rpn_reg_label'] = rpn_reg_label
        # sample_info['gt_boxes3d'] = aug_gt_boxes3d
        return sample_info

if __name__  == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(cur_dir, '../data_samples/kitti')
    scene_name = "2011_09_26_drive_0009_sync"
    
    data = KittiRawData(data_path, scene_name)
    print(data[0])
