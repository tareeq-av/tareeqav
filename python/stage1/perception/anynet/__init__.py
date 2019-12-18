import os

import torch
import torch.nn as nn

from perception.anynet.models.anynet import AnyNet

def init_model(anynet_model_filename, logger, max_disp=192):
    """
    """
    args = {
       'init_channels' : 1,
       'maxdisplist' : [12, 3, 3],
       'nblocks' : 2,
       'layers_3d' : 4,
       'channels_3d' : 4,
       'growth_rate' : [4,1,1],
       'with_spn' : True,
       'spn_init_channels': 8,
    }

    torch.cuda.manual_seed(1)
    model = AnyNet(args)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    if not os.path.isfile(anynet_model_filename):
        raise FileNotFoundError(anynet_model_filename)

    logger.info("==> Loading AnyNet from '{}'".format(anynet_model_filename))

    state_dict = torch.load(anynet_model_filename)
    model.load_state_dict(state_dict['state_dict'])

    model.eval()

    return model
