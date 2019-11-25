import time

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

# default model resize
RESZIE = (640,360)

### Colors for visualization
# Ego: red, other: blue
COLORS_DEBUG = [(255,0,0), (0,0,255)]

# Road name map
ROAD_MAP = ['Residential', 'Highway', 'City Street', 'Other']

def run(model, img):

    resize_factor = 5
    resized = cv2.resize(img, RESZIE, interpolation = cv2.INTER_AREA)
    
    ### Pytorch conversion
    start_t = time.time()
    input_tensor = torch.from_numpy(resized)
    input_tensor = torch.div(input_tensor.float(), 255)
    input_tensor = input_tensor.permute(2,0,1).unsqueeze(0)
    
    with torch.no_grad():
        input_tensor = Variable(input_tensor).cuda()
        output = model(input_tensor)

    output, output_road = output
    road_type = output_road.max(dim=1)[1][0]
            
    ### Classification
    output = output.max(dim=1)[1]
    output = output.float().unsqueeze(0)

    ### Resize to desired scale for easier clustering
    output = F.interpolate(output, size=(output.size(2) // resize_factor, output.size(3) // resize_factor) , mode='nearest')

    ### Obtaining actual output
    ego_lane_points = torch.nonzero(output.squeeze() == 1)
    other_lanes_points = torch.nonzero(output.squeeze() == 2)

    ego_lane_points = ego_lane_points.view(-1).cpu().numpy()
    other_lanes_points = other_lanes_points.view(-1).cpu().numpy()

    # Convert the image and substitute the colors for egolane and other lane
    output = output.squeeze().unsqueeze(2).data.cpu().numpy()
    output = output.astype(np.uint8)

    output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    output[np.where((output == [1, 1, 1]).all(axis=2))] = COLORS_DEBUG[0]
    output[np.where((output == [2, 2, 2]).all(axis=2))] = COLORS_DEBUG[1]

    # Blend the original image and the output of the CNN
    output = cv2.resize(output, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_NEAREST)
    tmp = cv2.addWeighted(resized, 1, output, 0.4, 0)
    
    # resize back to original size
    img = cv2.resize(tmp, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
    return img
