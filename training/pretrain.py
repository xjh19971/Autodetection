import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import sys
sys.path.insert(0, '/scratch/xl1575/Autodetection')
from dataset.dataHelper import LabeledDataset
from util.helper import collate_fn, draw_box
from model.pretrainModel import trainModel
from tensorboardX import SummaryWriter
writer = SummaryWriter('log') #建立一个保存数据用的东西

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
labeled_scene_index = np.arange(106, 134)

def train(model, device, train_loader, optimizer, epoch, log_interval = 100):
    # Set model to training mode
    model.train()
    # Loop through data points
    for batch_idx, (sample,bbox_list,category_list,road_image) in enumerate(train_loader):
        # Send data and target to device
        sample=sample.to(device)
        # Zero out the optimizer
        optimizer.zero_grad()
        # Pass data through model
        recon=model(sample)
        # Compute the negative log likelihood loss
        loss=nn.MSELoss()(recon,sample)
        # Backpropagate loss
        loss.backward()
        # Make a step with the optimizer
        optimizer.step()
        # Print loss (uncomment lines below once implemented)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(sample), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

if __name__ == '__main__':
    data_transforms = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.Pad((7,0)),
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
    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=4, shuffle=True, num_workers=4,collate_fn=collate_fn)

    #sample, target, road_image, extra = iter(trainloader).next()
    #print(torch.stack(sample).shape)
    model=trainModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    #dummy_input = torch.rand(1, 18, 256, 320).to(device) # 假设输入20张1*28*28的图片
    print("resnet50 have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    for epoch in range(1,11):
        # Train model
        train(model, device, trainloader, optimizer, epoch)
        torch.save(model.state_dict(), 'parameter_pretrain_resnet101_'+str(epoch)+'.pkl')
