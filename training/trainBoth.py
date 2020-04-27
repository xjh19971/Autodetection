import os
import random

import numpy as np
import pandas as pd
import time
import torchcontrib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler

from dataset.dataHelper import LabeledDataset
from utils.helper import collate_fn, draw_box
from model.bothModel import trainModel
from tensorboardX import SummaryWriter
writer = SummaryWriter('log') #建立一个保存数据用的东西

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file

image_folder = 'dataset/data'
annotation_csv = 'dataset/data/annotation.csv'
anchor_file='yolo_anchors.txt'

# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def train(model, device, train_loader, optimizer, epoch, log_interval = 50):
    # Set model to training mode
    model.train()
    # Loop through data points
    for batch_idx, data in enumerate(train_loader):
        # Send data and target to device
        sample,bbox_list,category_list,road_image=data
        sample, road_image=sample.to(device),road_image.to(device)
        # Zero out the optimizer
        optimizer.zero_grad()
        # Pass data through model
        outputs=model(sample,[bbox_list,category_list])
        # Compute the negative log likelihood loss
        road_loss=nn.NLLLoss()(outputs[0],road_image)
        detection_loss=outputs[2]
        loss=road_loss+detection_loss
        # Backpropagate loss
        loss.backward()
        # Make a step with the optimizer
        optimizer.step()
        # Print loss (uncomment lines below once implemented)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tRoad Loss: {:.6f}\tDetection Loss: {:.6f}'.format(
            epoch, batch_idx * len(sample), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), road_loss.item(),detection_loss.item()))
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tRoad Loss: {:.6f}\tDetection Loss: {:.6f}'.format(
        epoch, len(train_loader.dataset), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), road_loss.item(), detection_loss.item()))

def test(model, device, test_loader):
    # Set model to evaluation mode
    model.eval()
    # Variable for the total loss
    test_road_loss = 0
    test_detection_loss = 0
    with torch.no_grad():
    # Loop through data points
        batch_num=0
        for batch_idx, data in enumerate(test_loader):
            # Send data to device
            sample, bbox_list, category_list, road_image = data
            sample, road_image = sample.to(device), road_image.to(device)
            # Pass data through model
            outputs=model(sample,[bbox_list, category_list])
            road_loss = nn.NLLLoss()(outputs[0], road_image)
            detection_loss = outputs[2]
            test_road_loss += road_loss
            test_detection_loss+= detection_loss
            batch_num+=1
            # Add number of correct predictions to total num_correct
        # Compute the average test_loss
        avg_road_loss = test_road_loss   /batch_num
        avg_detection_loss = test_detection_loss / batch_num
        # Print loss (uncomment lines below once implemented)
        print('\nTest set: Average loss: {:.4f} {:.4f}\n'.format(avg_road_loss,avg_detection_loss))
    return avg_road_loss+avg_detection_loss

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
        transforms.Resize((200,200),0),
        transforms.ToTensor()
    ])
    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index,
                                      transform=data_transforms,
                                      roadmap_transform=roadmap_transforms,
                                      extra_info=True
                                      )
    trainset, testset = torch.utils.data.random_split(labeled_trainset, [int(0.85 * len(labeled_trainset)),
                                                                         len(labeled_trainset)-int(0.85 * len(labeled_trainset))])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8,
                                              collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=8,
                                              collate_fn=collate_fn)
    anchors=get_anchors(anchor_file)
    #sample, target, road_image, extra = iter(trainloader).next()
    #print(torch.stack(sample).shape)
    model=trainModel(anchors)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2,weight_decay=0.01)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.5)
    optimizer = torchcontrib.optim.SWA(optimizer,swa_freq=5,swa_start=200,swa_lr=0.0025)
    print("Model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    last_test_loss=1
    for epoch in range(1, 300 + 1):
        # Train model
        start_time=time.time()
        train(model, device, trainloader, optimizer, epoch)
        test_loss=test(model,device,testloader)
        if(last_test_loss>test_loss):
            torch.save(model.state_dict(), 'parameter.pkl')
            last_test_loss=test_loss
        if epoch <= 300:
            scheduler.step(epoch)
            print("lr_scheduler="+str(scheduler.get_lr())+'\n')
        end_time=time.time()
        print("total_time="+str(end_time-start_time)+'\n')
    optimizer.swap_swa_sgd()
    optimizer.bn_update(trainloader, model)
    test_loss =test(model,device,testloader)
    if (last_test_loss > test_loss):
        torch.save(model.state_dict(), 'parameter.pkl')
        last_test_loss = test_loss

