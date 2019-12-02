import os

import torch
import torch.nn as nn

from perception.no_lidar.psmnet.models import stackhourglass

def init_model(psmnet_model_filename, logger, max_disp=192):
    """
    """    
    torch.cuda.manual_seed(1)
    model = stackhourglass(max_disp)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    if not os.path.isfile(psmnet_model_filename):
        raise FileNotFoundError(psmnet_model_filename)

    logger.info("==> Loading PSMNet from '{}'".format(psmnet_model_filename))

    state_dict = torch.load(psmnet_model_filename)
    model.load_state_dict(state_dict['state_dict'])

    model.eval()

    return model
