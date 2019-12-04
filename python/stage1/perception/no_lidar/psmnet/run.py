from __future__ import print_function

import os

import numpy as np
import torch
from torch.autograd import Variable

from utils import preprocess

def inference(model, imgL, imgR):

    imgL = torch.FloatTensor(imgL).cuda()
    imgR = torch.FloatTensor(imgR).cuda()     

    imgL, imgR = Variable(imgL), Variable(imgR)

    with torch.no_grad():
        output = model(imgL,imgR)

    output = torch.squeeze(output)
    pred_disp = output.data.cpu().numpy()

    return pred_disp

def run(model, imgL_o, imgR_o):

    processed = preprocess.get_transform(augment=False)

    imgL = processed(imgL_o).numpy()
    imgR = processed(imgR_o).numpy()
    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    # pad to (384, 1248)
    top_pad = 384-imgL.shape[2]
    left_pad = 1248-imgL.shape[3]
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    pred_disp = inference(model, imgL, imgR)

    top_pad   = 384-imgL_o.shape[0]
    left_pad  = 1248-imgL_o.shape[1]
    img = pred_disp[top_pad:,:-left_pad]

    return img
