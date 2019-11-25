import os
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
            default='2011_09_26_drive_0056_sync',
            help='Name, date and number of the KITTI RAW Dataset example, eg: 2011_09_26_drive_0056_sync.',
            )

    # path to pretrained ponitrcnn model
    parser.add_argument(
            '--pointrcnn-model-file',
            dest='pointrcnn_model_file',
            default=os.path.join(CURRENT_DIR, 'perception/pointrcnn/PointRCNN.pth'),
            help='Path to the checkpoint of the PointRCNN Model to be used for intefrence.',
            )

    # path to pretrained lane detection model
    parser.add_argument(
            '--lanes-model-file',
            dest='lanes_model_file',
            default=os.path.join(CURRENT_DIR, 'perception/lanes/weights_erfnet_road.pth'),
            help='Path to the checkpoint of the Lane Detection Model to be used for intefrence.',
            )

    return parser.parse_args()


def create_logger():
    """
    """
    log_format = '[TareeqAV Stage 1] %(asctime)s  %(levelname)5s  %(message)s'
    log_file = os.path.join(CURRENT_DIR, 'tareeqav_stage1_pipeline.log')
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def load_sampledata(scene_base_dir, drive_name):
    return KittiRawData(scene_base_dir, drive_name)


def prepare_output(out_shape, drive_name, fps=10, output_dir='output'):
    output_dir = os.path.join(CURRENT_DIR, output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    video = '{}/{}.avi'.format(output_dir, drive_name)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    
    writer = cv2.VideoWriter(video, fourcc, fps, (out_shape[1], out_shape[0]), True)
    return writer
    

def run(scene_base_dir, drive_name, pointrcnn_model_file, lanes_model_file):
    """
    """
    logger = create_logger()
    sampledata = load_sampledata(scene_base_dir, drive_name)

    logger.info("preparing a video writer for visualization of the pipeline")
    video_writer = prepare_output(
            sampledata[0]['img'].shape,
            drive_name, 
            fps=10, 
            output_dir='output')
    
    logger.info("running the perception pipeline...")
    # run the perception stack (3d detection, lane detection, traffic signs)
    Perception.run(sampledata, pointrcnn_model_file, lanes_model_file, video_writer, logger)

if __name__ == '__main__':
    args = parse_args()
    run(args.sampledata_dir, args.drive_name, args.pointrcnn_model_file, args.lanes_model_file)

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
