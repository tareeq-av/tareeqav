import os

import torch

from .lib.net.point_rcnn import PointRCNN

def init_pointrcnn(dataset, pointrcnn_filename, logger):
    """
    """
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

