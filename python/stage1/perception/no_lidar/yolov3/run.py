from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR  = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()

def run(img):
    args = arg_parse()

    batch_size = 1
    confidence = 0.5
    nms_thesh = 0.4
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes(os.path.join(CURRENT_DIR, 'data/coco.names'))

    #Set up the neural network
    print("Loading network.....")
    model = Darknet(os.path.join(CURRENT_DIR, 'cfg/yolov3.cfg'))
    model.load_weights(os.path.join(CURRENT_DIR, 'yolov3.weights'))
    print("Network successfully loaded")
    
    model.net_info["height"] = 416
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    
    #Set the model in evaluation mode
    model.eval()
    
    read_dir = time.time()

    #Detection phase
    # try:
    #     imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    # except NotADirectoryError:
    #     imlist = []
    #     imlist.append(osp.join(osp.realpath('.'), images))
    # except FileNotFoundError:
    #     print ("No file or directory with the name {}".format(images))
    #     exit()
        
    # if not os.path.exists(args.det):
    #     os.makedirs(args.det)
        
    load_batch = time.time()
    
    # batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    # im_batches = [x[0] for x in batches]
    # orig_ims = [x[1] for x in batches]
    # im_dim_list = [x[2] for x in batches]
    # im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

    prepped = prep_image(img, inp_dim)
    im_batches = [prepped[0]]
    orig_ims = [prepped[1]]
    im_dim_list = [prepped[2]]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    
    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    start_det_loop = time.time()
    
    objs = {}

    for batch in im_batches:
        #load the image 
        start = time.time()
        if CUDA:
            batch = batch.cuda()

        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)
    
        prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)
        output = prediction
        if CUDA:
            torch.cuda.synchronize()
    
    try:
        output
    except NameError:
        print("No detections were made")
        exit()
        
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    results = []
    for x in output.cpu().data:
        result = list()
        result.append(int(x[-1].item()))
        result.extend([int(x[i].item()) for i in range(1, 5)])
        results.append(result)

    torch.cuda.empty_cache()
    
    return results
