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

def test(imgL, imgR, model):

    stages = 3 # zero based index
    model.eval()

    with torch.no_grad():
        outputs = model(imgL, imgR)

        for i in range(stages+1):
            output = torch.squeeze(outputs[i], 1)
            output = output[:, 4:, :]
            output = output.squeeze().cpu().numpy()
            print('>>>>>>>>>>>>>', output.shape)
            cv2.imshow('Window'+str(i), (output*255).astype('uint16'))
            cv2.waitKey(0)


def process_image_pair(left_img, right_img):
    w, h = left_img.size

    left_img = left_img.crop((w-1232, h-368, w, h))
    right_img = right_img.crop((w-1232, h-368, w, h))

    processed = preprocess.get_transform(augment=False)  
    left_img       = processed(left_img)
    right_img      = processed(right_img)

    return left_img, right_img

def main():
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

    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    checkpoint = torch.load('./kitti2015_ck/checkpoint.tar')
    # checkpoint = torch.load('results/finetune_anynet/checkpoint.tar')
    model.load_state_dict(checkpoint['state_dict'])

    imgl_path = '/home/sameh/Autonomous-Vehicles/tareeqav.monodepth/python/sample-data/kitti/2011_09_26/2011_09_26_drive_0086_sync/image_02/data/0000000136.png'
    imgr_path = '/home/sameh/Autonomous-Vehicles/tareeqav.monodepth/python/sample-data/kitti/2011_09_26/2011_09_26_drive_0086_sync/image_03/data/0000000136.png'

    imgL_o = Image.open(imgl_path).convert('RGB')
    imgR_o = Image.open(imgr_path).convert('RGB')

    # imgL_o = np.rollaxis(imgL_o, 2, 0)
    # imgR_o = np.rollaxis(imgR_o, 2, 0)

    # print('>>>>>>>>>>>>>>>',imgL_o.shape)

    imgL_o, imgR_o = process_image_pair(imgL_o, imgR_o)

    imgL_o = np.expand_dims(imgL_o, 0)
    imgR_o = np.expand_dims(imgR_o, 0)

    imgL = torch.FloatTensor(imgL_o).cuda()
    imgR = torch.FloatTensor(imgR_o).cuda()

    test(imgL, imgR, model)
    print('inference time = {:.2f} '.format((time.time() - start_full_time)))


if __name__ == '__main__':
    main()
