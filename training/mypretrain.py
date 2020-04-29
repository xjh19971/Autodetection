import os
import random

import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset.dataHelper import UnlabeledDataset
from utils.helper import collate_fn_unlabeled, draw_box
import torchcontrib
from model.mypretrainModel import trainModel
from tensorboardX import SummaryWriter
import math

writer = SummaryWriter('log')  # 建立一个保存数据用的东西

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file

image_folder = 'dataset/data'
annotation_csv = 'dataset/data/annotation.csv'

# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(134)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)
start_epoch = 150
long_cycle = 30
short_cycle = 5
start_lr = 0.004

def lambdaScheduler(epoch):
    if epoch == 0:
        return 1
    elif epoch < start_epoch:
        return 0.5 ** (math.floor(epoch / long_cycle))
    else:
        if epoch % short_cycle == 0:
            return 0.5 ** math.floor(start_epoch / long_cycle / 2 + 1)
        else:
            return 0.5 ** math.floor(start_epoch / long_cycle / 2 + 1) - \
                   (0.5 ** math.floor(start_epoch / long_cycle / 2 + 1) - 0.5 ** math.floor(start_epoch / long_cycle)) \
                   * (epoch % short_cycle) / short_cycle


def cal_loss(output, target, mu, logvar):
    return nn.MSELoss(reduction='mean')(output, target) + 0.5 * torch.mean(logvar.exp() - logvar - 1 + mu.pow(2))


def train(model, device, train_loader, optimizer, epoch, log_interval=50):
    # Set model to training mode
    model.train()
    # Loop through data points
    for batch_idx, data in enumerate(train_loader):
        # Send data and target to device
        sample, target = data
        sample, target = sample.to(device), target.to(device)
        # Zero out the optimizer
        optimizer.zero_grad()
        # Pass data through model
        output, mu, logvar = model(sample)
        # Compute the negative log likelihood loss
        loss = cal_loss(output, target, mu, logvar)
        # Backpropagate loss
        loss.backward()
        # Make a step with the optimizer
        optimizer.step()
        # Print loss (uncomment lines below once implemented)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, len(train_loader.dataset), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    # Set model to evaluation mode
    model.eval()
    # Variable for the total loss
    test_loss = 0
    with torch.no_grad():
        # Loop through data points
        batch_num = 0
        for batch_idx, data in enumerate(test_loader):
            # Send data to device
            sample, target = data
            sample, target = sample.to(device), target.to(device)
            # Pass data through model
            output, mu, logvar = model(sample)
            test_loss += cal_loss(output, target, mu, logvar)
            batch_num += 1
            # Add number of correct predictions to total num_correct
        # Compute the average test_loss
        avg_test_loss = test_loss / batch_num
        # Print loss (uncomment lines below once implemented)
        print('\nTest set: Average loss: {:.4f}\n'.format(avg_test_loss))
    return avg_test_loss


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Pad((7, 0)),
        transforms.Resize((128, 160), 0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # for ImageNet
    ])
    unlabeled_trainset = UnlabeledDataset(image_folder=image_folder,
                                          scene_index=unlabeled_scene_index,
                                          transform=data_transforms,
                                          first_dim='sample'
                                          )
    trainset, testset = torch.utils.data.random_split(unlabeled_trainset, [int(0.90 * len(unlabeled_trainset)),
                                                                           len(unlabeled_trainset) - int(
                                                                               0.90 * len(unlabeled_trainset))])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8,
                                              collate_fn=collate_fn_unlabeled)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=8,
                                             collate_fn=collate_fn_unlabeled)

    # sample, target, road_image, extra = iter(trainloader).next()
    # print(torch.stack(sample).shape)
    model = trainModel()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=5e-4)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdaScheduler)
    optimizer = torchcontrib.optim.SWA(optimizer)
    print("Model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    last_test_loss = 1
    for epoch in range(1, 200 + 1):
        # Train model
        start_time = time.time()
        train(model, device, trainloader, optimizer, epoch)
        test_loss = test(model, device, testloader)
        scheduler.step(epoch)
        if last_test_loss > test_loss:
            torch.save(model.state_dict(), 'pretrain.pkl')
            last_test_loss = test_loss
        if epoch >= 150 and (epoch + 1) % short_cycle == 0:
            optimizer.update_swa()
        print('lr=' + str(optimizer.param_groups[0]['lr'])+ '\n')
        end_time = time.time()
        print("total_time=" + str(end_time - start_time) + '\n')
    optimizer.swap_swa_sgd()
    model = model.cpu()
    optimizer.bn_update(trainloader, model)
    model.to(device)
    test_loss = test(model, device, testloader)
    if (last_test_loss > test_loss):
        torch.save(model.state_dict(), 'pretrain.pkl')
        last_test_loss = test_loss
