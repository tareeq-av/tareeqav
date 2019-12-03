import os
import sys
import logging
import argparse

import cv2

from sampledata import KittiRawData
from perception import pipeline as Perception

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR  = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

def parse_args():
    parser = argparse.ArgumentParser(description='Running the Perception, Planning and Control stacks of the TareeqAV Platform')
    # sample data dir
    parser.add_argument(
            '--sample-data-dir',
            dest='sampledata_dir',
            default=os.path.join(PARENT_DIR, 'sample-data/kitti/2011_09_26'),
            help='Base directory of KITTI RAW Dataset example for demonstration of the platform.'
            )

    # name of the drive to use in the demo
    parser.add_argument(
            '--drive-name',
            dest='drive_name',
            default='2011_09_26_drive_0086_sync',
            help='Name, date and number of the KITTI RAW Dataset example, eg: 2011_09_26_drive_0056_sync.',
            )

    # path to pretrained ponitrcnn model
    parser.add_argument(
            '--pointrcnn-model-file',
            dest='pointrcnn_model_file',
            default=os.path.join(CURRENT_DIR, 'perception/lidar/pointrcnn/PointRCNN.pth'),
            help='Path to the checkpoint of the PointRCNN Model to be used for intefrence.',
            )

    # path to pretrained pointnet++ model
    parser.add_argument(
            '--rgbd-pointnet-model-file',
            dest='rgbd_pointnet_model_file',
            default=os.path.join(CURRENT_DIR, 'perception/no_lidar/pointnets/no_lidar/pointnets/models/model.ckpt'),
            help='Path to the checkpoint of the RGB-D PointNet Model to be used for intefrence.',
            )
    
    # path to pretrained disparity/depth estimation model
    parser.add_argument(
            '--disp-model-file',
            dest='disp_model_file',
            default=os.path.join(CURRENT_DIR, 'perception/no_lidar/psmnet/finetune_300.tar'),
            help='Path to the checkpoint of the Dispary/Depth Estimation Model to be used for intefrence.',
            )

    # path to pretrained YOLO pre-trained weights
    parser.add_argument(
            '--yolov3-weights-file',
            dest='yolov3_weights_file',
            default=os.path.join(CURRENT_DIR, 'perception/no_lidar/yolov3/yolov3.weights'),
            help='Path to the YOLOv3 weights to be used for intefrence.',
            )

    # path to pretrained YOLO pre-trained weights
    parser.add_argument(
            '--yolov3-config-file',
            dest='yolov3_config_file',
            default=os.path.join(CURRENT_DIR, 'perception/no_lidar/yolov3/cfg/yolov3.cfg'),
            help='Path to the YOLOv3 config file.',
            )

    # path to pretrained lane detection model
    parser.add_argument(
            '--lanes-model-file',
            dest='lanes_model_file',
            default=os.path.join(CURRENT_DIR, 'perception/common/lanes/weights_erfnet_road.pth'),
            help='Path to the checkpoint of the Lane Detection Model to be used for intefrence.',
            )

    # use or do not use lidar signal/sensor?
    parser.add_argument(
            '--with-lidar',
            dest='with_lidar',
            action='store_true',
            default=False,
            help='Does this vehicle have a lidar sensor?'

    )

    # enable debug logging level ?
    parser.add_argument(
            '--debug',
            dest='debug',
            action='store_true',
            default=False,
            help='Enables DEBUG Loggin Level'

    )

    return parser.parse_args()


def create_logger(level):
    """
    """
    log_format = '[TareeqAV Stage 1] %(asctime)s  %(levelname)5s  %(message)s'
    log_file = os.path.join(CURRENT_DIR, 'tareeqav_stage1_pipeline.log')

    logging.basicConfig(level=level, format=log_format, filename=log_file)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

    # logFormatter = logging.Formatter("[TareeqAV Stage 1] %(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    # rootLogger = logging.getLogger()

    # fileHandler = logging.FileHandler(log_file)
    # fileHandler.setLevel(level)
    # fileHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(fileHandler)

    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setLevel(level)
    # consoleHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(consoleHandler)

    # return rootLogger


def load_sampledata(scene_base_dir, drive_name, logger):
    return KittiRawData(scene_base_dir, drive_name, logger)


def prepare_output(out_shape, drive_name, fps=10, output_dir='output', with_lidar=False):
    output_dir = os.path.join(CURRENT_DIR, output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    video_name = '{}/{}'.format(output_dir, drive_name)
    if with_lidar:
        video_name += '-with-lidar.avi'
    else:
        video_name += '.avi'
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    
    writer = cv2.VideoWriter(video_name, fourcc, fps, (out_shape[1], out_shape[0]), True)
    return writer
    

def run(
        scene_base_dir,
        drive_name,
        pointnet_model_file,
        lanes_model_file,
        disp_model_file=None,
        yolov3_weights_file=None,
        yolov3_config_file=None,
        with_lidar=False,
        debug=False
    ):
    """
    """
    logger = create_logger(logging.DEBUG if debug else logging.INFO)
    sampledata = load_sampledata(scene_base_dir, drive_name, logger)

    logger.debug("preparing a video writer for visualization of the pipeline")
    video_writer = prepare_output(
            sampledata[0]['left_img_cv2'].shape,
            drive_name, 
            fps=10, 
            output_dir='output'
        )
    
    logger.info("running the perception pipeline...")
    # run the perception stack (3d detection, lane detection, traffic signs)
    Perception.run(
        sampledata,
        pointnet_model_file,
        lanes_model_file,
        video_writer,
        logger,
        disp_model_file=disp_model_file,
        yolov3_weights_file=yolov3_weights_file,
        yolov3_config_file=yolov3_config_file,
        with_lidar=with_lidar
    )

if __name__ == '__main__':
    
    args = parse_args()

    if args.with_lidar:
        pointnet_model_file = args.pointrcnn_model_file
    else:
        pointnet_model_file = args.rgbd_pointnet_model_file
        
    run(
        args.sampledata_dir,
        args.drive_name,
        pointnet_model_file,
        args.lanes_model_file,
        disp_model_file=args.disp_model_file,
        yolov3_weights_file=args.yolov3_weights_file,
        yolov3_config_file=args.yolov3_config_file,
        with_lidar=args.with_lidar,
        debug=args.debug
    )

    # base_dir = '/home/sameh/Autonomous-Vehicles/Datasets/Kitti-Raw/kitti_data'
    # scene_dates = [
    #     '2011_09_26',
    #     # '2011_09_28' ,
    #     '2011_09_29', 
    #     # '2011_09_30',
    #     # '2011_10_03'
    # ]

    # scene_names = [
    #     # '2011_09_26_drive_0001_sync',
    #     # '2011_09_26_drive_0011_sync',
    #     # '2011_09_26_drive_0028_sync',
    #     # '2011_09_26_drive_0032_sync',
    #     '2011_09_26_drive_0056_sync',
    #     # '2011_09_29_drive_0004_sync',
    # ]

    # for scene_date in scene_dates:
    #     scene_dir = os.path.join(base_dir, scene_date)
    #     # scene_names = os.listdir(scene_dir)
    #     for scene_name in scene_names:
    #         print('processing images from', scene_name)
    #         scene_path = os.path.join(scene_dir, scene_name)
    #         if os.path.isdir(scene_path):
    #             main(scene_dir, scene_name)
