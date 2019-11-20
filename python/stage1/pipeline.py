import os
import logging

import torch
import numpy as np

from sampledata import KittiRawData
from perception.pointrcnn.lib.net.point_rcnn import PointRCNN

np.random.seed(1024)  # set the same seed


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def load_sampledata(curr_dir):
    data_path = os.path.join(curr_dir, '../data_samples/kitti')
    scene_name = "2011_09_26_drive_0001"
    
    return KittiRawData(data_path, scene_name)


def init_pointrcnn(dataset, pointrcnn_filename, logger):
    model = PointRCNN(num_classes=dataset.num_classes, use_xyz=True, mode='TEST')
    model.cuda()
    
    if not os.path.isfile(pointrcnn_filename):
        raise FileNotFoundError(pointrcnn_filename)

    logger.info("==> Loading from checkpoint '{}'".format(pointrcnn_filename))
    checkpoint = torch.load(pointrcnn_filename)
    model.load_state_dict(checkpoint['model_state'])
    total_keys = len(model.state_dict().keys())

    return model


def run():
    """
    """
    pass


def main():
    """
    """
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    pointrcnn_model_file = os.path.join(curr_dir, 'PointRCNN.pth')
    log_file = os.path.join(curr_dir, 'log_perception_pipeline.txt')
    
    logger = create_logger(log_file)
    sampledata = load_sampledata(curr_dir)
    
    pointrcnn_model = init_pointrcnn(sampledata, pointrcnn_model_file, logger)
    


if __name__ == '__main__':
    main()
