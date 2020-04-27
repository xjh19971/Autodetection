import os
import random

import numpy as np
import pandas as pd
import time
import torchcontrib

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler

from dataset.dataHelper import LabeledDataset
from utils.helper import collate_fn, draw_box
from model.roadModel import trainModel
from tensorboardX import SummaryWriter

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file

image_folder = 'dataset/data'
annotation_csv = 'dataset/data/annotation.csv'

# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(120, 134)

def compute_ts_road_map(road_map1, road_map2):
    tp = (road_map1 * road_map2).sum()

    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)

def test(model, device, test_loader):
    # Set model to evaluation mode
    model.eval()
    # Variable for the total loss
    test_loss = 0
    test_accuracy = 0;
    predict_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    with torch.no_grad():
    # Loop through data points
        batch_num=0
        for batch_idx, data in enumerate(test_loader):
            # Send data to device
            sample, bbox_list, category_list, road_image = data
            sample, road_image = sample.to(device), road_image.to(device)
            # Pass data through model
            output=model(sample)
            output=output.cpu().numpy()
            outputlist=[]
            for i in range(len(output)):
                output1=cv2.resize(output[i][0],(800,800))
                output2 = cv2.resize(output[i][1], (800, 800))
                outputlist.append([output1,output2])
            output=np.array(outputlist)
            output=torch.from_numpy(output)
            _, predicted = torch.max(output.data, 1)
            test_loss+=nn.NLLLoss()(output,road_image.cpu())
            test_accuracy += compute_ts_road_map(predicted,road_image.cpu())
            batch_num+=1;
            # Add number of correct predictions to total num_correct
        # Compute the average test_loss
        avg_test_loss = test_loss/batch_num
        avg_test_accuracy = test_accuracy/batch_num
        # Print loss (uncomment lines below once implemented)
        print('\nTest set: Average loss: {:.4f}\n'.format(avg_test_loss)+
              '\nTest set: Average accuracy: {:.4f}\n'.format(avg_test_accuracy))

if __name__ == '__main__':
    data_transforms = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.Pad((7,0)),
        transforms.Resize((128,160), 0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   #for ImageNet
    ])
    roadmap_transforms = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.Resize((200,200),0),
        transforms.ToTensor()
    ])
    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index,
                                      transform=data_transforms,
                                      roadmap_transform=roadmap_transforms,
                                      extra_info=True
                                      )
    testloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=8, shuffle=True, num_workers=8,
                                              collate_fn=collate_fn)
    # Train model
    start_time = time.time()
    model=trainModel()
    model.load_state_dict(torch.load('best.pkl'))
    model.to(device)
    test(model,device,testloader)