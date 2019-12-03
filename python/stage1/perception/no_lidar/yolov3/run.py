from __future__ import division

import torch 
import torch.nn as nn
from torch.autograd import Variable

from darknet import Darknet
from util import load_classes, write_results
from preprocess import prep_image, inp_to_image

def run(model, dataset_item):
    """
    """
    confidence = 0.5
    nms_thesh = 0.4
    num_classes = 80

    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    
    img_, orig_im, dim = prep_image(dataset_item['left_img_cv2'], inp_dim)

    # img_ = torch.unsqueeze(img_, 0)
    img_ = img_.cuda()
    
    dim = torch.FloatTensor(dim).repeat(1, 2)
    # dim  = torch.unsqueeze(dim, 0)
    dim = dim.cuda()
    

    objs = {}

    with torch.no_grad():
        prediction = model(Variable(img_))
        
    output = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)

    dim = torch.index_select(dim, 0, output[:,0].long())
    
    scaling_factor = torch.min(inp_dim/dim,1)[0].view(-1,1)
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*dim[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*dim[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, dim[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, dim[i,1])
    
    results = []
    for x in output.cpu().data:
        result = list()
        result.append(int(x[-1].item()))
        result.extend([int(x[i].item()) for i in range(1, 5)])
        results.append(result)

    torch.cuda.empty_cache()
    
    return results
