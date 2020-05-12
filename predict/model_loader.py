"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision

# import your model class
from model.bothModelFPN import testModel
from utils.yolo_utils import non_max_suppression


# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1():
    return torchvision.transforms.Compose([
        torchvision.transforms.Pad((7, 0)),
        torchvision.transforms.Resize((128, 160), 0),
        torchvision.transforms.ToTensor(),
    ])


# For road map task
def get_transform_task2():
    return torchvision.transforms.Compose([
        torchvision.transforms.Pad((7, 0)),
        torchvision.transforms.Resize((128, 160), 0),
        torchvision.transforms.ToTensor(),
    ])


class ModelLoader():
    # Fill the information for your team
    team_name = 'team_name'
    round_number = 1
    team_member = ["Jiuhong Xiao", "Xinmeng Li", "Junrong Zha"]
    contact_email = 'jx1190@nyu.edu'
    anchor_file = 'yolo_anchors.txt'

    def __init__(self, model_file='bothModelFPN.pkl'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        self.model = testModel(self.get_anchors())
        self.model.load_state_dict(torch.load(model_file, map_location="cuda:0"))
        self.model = self.model.cuda()
        self.model.eval()
        self.test_flag = None
        self.NUM_SAMPLE_PER_SCENE = 126
        self.count = 0

    def get_anchors(self):
        '''loads the anchors from a file'''
        # with open(self.anchor_file) as f:
        #    anchors = f.readline()
        anchors = "10,9, 29,48, 45,18, 49,21, 86,29"
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        self.count += 1
        if self.test_flag != 'detection' or self.count == self.NUM_SAMPLE_PER_SCENE:
            self.count = 0
            # For LSTM
            # self.model.removeBuffer()
            self.test_flag = 'detection'
        outputs = self.model(samples)
        output_list = [non_max_suppression(outputs[1]), non_max_suppression(outputs[2])]
        temp_list = []
        for i in range(samples.size(0)):
            temp_list.append(torch.cat([output_list[0][i], output_list[1][i]], dim=0))
        output_list = temp_list
        real_output = []
        for i in range(samples.size(0)):
            if output_list[i] is not None:
                x1 = (output_list[i][:, 0] - 400) / 10
                y1 = (output_list[i][:, 1] - 400) / 10
                x2 = (output_list[i][:, 2] - 400) / 10
                y2 = (output_list[i][:, 3] - 400) / 10
                real_output.append(
                    torch.stack([torch.stack([x1, x2, x2, x1], dim=1), torch.stack([y1, y1, y2, y2], dim=1)], dim=1))
            else:
                real_output.append(torch.zeros([1, 2, 4]))
        return real_output

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        self.count += 1
        if self.test_flag != 'roadimage' or self.count == self.NUM_SAMPLE_PER_SCENE:
            self.count = 0
            # For LSTM
            # self.model.removeBuffer()
            self.test_flag = 'roadimage'
        outputs = self.model(samples)
        roadmap = nn.Upsample(scale_factor=2)(outputs[0])
        _, predicted = torch.max(roadmap.data, 1)
        return predicted
