import os

import torch
import torch.nn as nn

from perception.yolov3.darknet import Darknet

def init_model(weights_file, config_file, logger, height=416):
    """
    """    
    if not os.path.isfile(weights_file):
        raise FileNotFoundError(weights_file)

    if not os.path.isfile(config_file):
        raise FileNotFoundError(config_file)

    logger.info("==> Loading Yolov3 from '{}'".format(weights_file))
    model = Darknet(config_file)
    model.load_weights(weights_file)
    model.net_info["height"] = height

    model.cuda()
    model.eval()

    return model
