import torch

from .erfnet_road import Net

def init_model(lanes_model_file, logger):
    """
    """
    
    logger.info("==> Loading Lane Detection from '{}'".format(lanes_model_file))
   
    # GPU only mode, setting up
    model = Net()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(lanes_model_file))
    model.eval()

    return model
