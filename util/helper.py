import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def convert_map_to_lane_map(ego_map, binary_lane):
    mask = (ego_map[0,:,:] == ego_map[1,:,:]) * (ego_map[1,:,:] == ego_map[2,:,:]) + (ego_map[0,:,:] == 250 / 255)

    if binary_lane:
        return (~ mask)
    return ego_map * (~ mask.view(1, ego_map.shape[1], ego_map.shape[2]))

def convert_map_to_road_map(ego_map):
    mask = (ego_map[0,:,:] == 1) * (ego_map[1,:,:] == 1) * (ego_map[2,:,:] == 1)

    return (~mask)

def collate_fn(batch):
    concated_input=[]
    for i in range(len(batch)):
        concated_input.append(torch.reshape(batch[i][0],(1,18,256,320)))
    concated_input=torch.cat(concated_input,0)
    bbox_list=[]
    category_list=[]
    for i in range(len(batch)):
        bbox_list.append(batch[i][1]['bounding_box'])
        category_list.append(batch[i][1]['category'])
    concated_roadmap=[]
    for i in range(len(batch)):
        concated_roadmap.append(batch[i][2].unsqueeze(0))
    concated_roadmap=torch.cat(concated_roadmap,0).long()
    return [concated_input,bbox_list,category_list,concated_roadmap]

def draw_box(ax, corners, color):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
    
    # the corners are in meter and time 10 will convert them in pixels
    # Add 400, since the center of the image is at pixel (400, 400)
    # The negative sign is because the y axis is reversed for matplotlib
    ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color=color)





