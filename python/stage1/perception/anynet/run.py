import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from dataloader import preprocess
import utils.logger as logger

from PIL import Image

import cv2
import numpy as np
import skimage

import models.anynet

def test(model, imgL, imgR):

    stages = 3 # zero based index
    
    with torch.no_grad():
        outputs = model(imgL, imgR)

        output = torch.squeeze(outputs[3], 1)
        output = output[:, 4:, :]
        output = output.squeeze().cpu().numpy()
        # print('>>>>>>>>>>>>>', output.shape)
        # cv2.imshow('Window', (output*255).astype('uint16'))
        # cv2.waitKey(0)

        return output    


def process_image_pair(left_img, right_img):
    w, h = left_img.size

    left_img = left_img.crop((w-1232, h-368, w, h))
    right_img = right_img.crop((w-1232, h-368, w, h))

    processed = preprocess.get_transform(augment=False)  
    left_img       = processed(left_img)
    right_img      = processed(right_img)

    return left_img, right_img

def run(model, imgL_o, imgR_o):
    
    imgL_o, imgR_o = process_image_pair(imgL_o, imgR_o)

    imgL_o = np.expand_dims(imgL_o, 0)
    imgR_o = np.expand_dims(imgR_o, 0)

    imgL = torch.FloatTensor(imgL_o).cuda()
    imgR = torch.FloatTensor(imgR_o).cuda()

    return test(model, imgL, imgR)
