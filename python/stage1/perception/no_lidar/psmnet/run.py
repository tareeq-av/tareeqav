from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import *

torch.cuda.manual_seed(1)
model = stackhourglass(192)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR  = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

state_dict = torch.load(os.path.join(CURRENT_DIR, 'kitti_3d/finetune_300.tar'))
model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    imgL = torch.FloatTensor(imgL).cuda()
    imgR = torch.FloatTensor(imgR).cuda()     

    imgL, imgR = Variable(imgL), Variable(imgR)

    with torch.no_grad():
        output = model(imgL,imgR)
    output = torch.squeeze(output)
    pred_disp = output.data.cpu().numpy()

    return pred_disp


def run(imgL_o, imgR_o):

    # save_path = './output/'
    processed = preprocess.get_transform(augment=False)
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)

    imgL = processed(imgL_o).numpy()
    imgR = processed(imgR_o).numpy()
    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    # pad to (384, 1248)
    top_pad = 384-imgL.shape[2]
    left_pad = 1248-imgL.shape[3]
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    start_time = time.time()
    pred_disp = test(imgL,imgR)
    print('time = %.2f' %(time.time() - start_time))

    top_pad   = 384-imgL_o.shape[0]
    left_pad  = 1248-imgL_o.shape[1]
    img = pred_disp[top_pad:,:-left_pad]

    print('done...')

    # skimage.io.imsave(save_path+idx, (img*256).astype('uint16'))
    # np.save(save_path+idx, img)

    return img
