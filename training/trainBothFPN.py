import math
import re
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.nn import DataParallel

import model.bothModelFPN as bothModel
from dataset.dataHelper import LabeledDatasetScene
from utils.helper import collate_fn_lstm, compute_ts_road_map
torch.cuda.set_device(0)
# All the images are saved in image_folder
# All the labels are saved in the annotation_csv file

image_folder = 'dataset/data'
annotation_csv = 'dataset/data/annotation.csv'
anchor_file = 'yolo_anchors.txt'
# You shouldn't change the unlabeled_scene_index
# The first 106 scenes are unlabeled
unlabeled_scene_index = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
labeled_scene_index = np.arange(106, 134)
start_epoch = 200
long_cycle = 40
short_cycle = 5
start_lr = 0.01
gamma = 0.25
pretrain_file = "pretrainfinal128noaug.pkl"


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def lambdaScheduler(epoch):
    if epoch == 0:
        return 1
    elif epoch < start_epoch:
        return gamma ** (math.floor(epoch / long_cycle))
    else:
        if epoch % short_cycle == 0:
            return gamma ** math.floor(start_epoch / long_cycle / 2 + 1)
        else:
            return gamma ** math.floor(start_epoch / long_cycle / 2 + 1) - \
                   (gamma ** math.floor(start_epoch / long_cycle / 2 + 1) - gamma ** math.floor(
                       start_epoch / long_cycle)) \
                   * (epoch % short_cycle) / short_cycle


def train(model, train_loader, optimizer, epoch, log_interval=50):
    # Set model to training mode
    model.train()
    # Loop through data points
    for batch_idx, data in enumerate(train_loader):
        # Send data and target to device
        sample, bbox_list, category_list, road_image = data
        sample, road_image = sample.cuda(), road_image.cuda()
        # Zero out the optimizer
        optimizer.zero_grad()
        # Pass data through model
        outputs = model(sample, [bbox_list, category_list])
        # Compute the negative log likelihood loss
        road_loss = nn.NLLLoss()(outputs[0], road_image)
        detection_loss = outputs[4]
        loss = road_loss + detection_loss
        # Compute the negative log likelihood loss
        output0 = outputs[0].view(-1, 2, 400, 400)
        road_image = road_image.view(-1, 400, 400)
        _, predicted = torch.max(output0.data, 1)
        AUC = compute_ts_road_map(predicted, road_image)
        # Backpropagate loss
        loss.backward()
        # Make a step with the optimizer
        optimizer.step()
        # Print loss (uncomment lines below once implemented)
        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRoad Loss: {:.6f}\tDetection Loss: {:.6f}\tAccuracy: {:.6f}\tPrecision: {:.6f}\tRecall50: {:.6f}'.format(
                    epoch, batch_idx * len(sample), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), road_loss.item(), detection_loss.item(),
                    AUC, (model.yolo0.metrics['precision'] + model.yolo1.metrics['precision'] + model.yolo2.metrics[
                        'precision']) / 3,
                           (model.yolo0.metrics['recall50'] + model.yolo1.metrics['recall50'] + model.yolo2.metrics[
                               'recall50']) / 3))
    print(
        'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRoad Loss: {:.6f}\tDetection Loss: {:.6f}\tAccuracy: {:.6f}\tPrecision: {:.6f}\tRecall50: {:.6f}'.format(
            epoch,len(train_loader.dataset), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item(), road_loss.item(), detection_loss.item(), AUC, (
                           model.yolo0.metrics['precision'] + model.yolo1.metrics['precision'] +
                           model.yolo2.metrics['precision']) / 3,
                   (model.yolo0.metrics['recall50'] + model.yolo1.metrics['recall50'] + model.yolo2.metrics[
                       'recall50']) / 3))


def test(model, test_loader):
    # Set model to evaluation mode
    model.eval()
    # Variable for the total loss
    test_loss = 0
    AUC = 0
    P = 0
    R = 0
    with torch.no_grad():
        # Loop through data points
        batch_num = 0

        for batch_idx, data in enumerate(test_loader):
            # Send data to device
            sample, bbox_list, category_list, road_image = data
            sample, road_image = sample.cuda(), road_image.cuda()
            # Pass data through model
            outputs = model(sample, [bbox_list, category_list])
            road_loss = nn.NLLLoss()(outputs[0], road_image)
            detection_loss = outputs[4]
            test_loss += road_loss + detection_loss
            # Compute the negative log likelihood loss
            output0 = outputs[0].view(-1, 2, 400, 400)
            road_image = road_image.view(-1, 400, 400)
            _, predicted = torch.max(output0.data, 1)
            AUC += compute_ts_road_map(predicted, road_image)
            P += (model.yolo0.metrics['precision'] + model.yolo1.metrics['precision'] + model.yolo2.metrics[
                'precision']) / 3
            R += (model.yolo0.metrics['recall50'] + model.yolo1.metrics['recall50'] + model.yolo2.metrics[
                'recall50']) / 3
            batch_num += 1
            # Add number of correct predictions to total num_correct
        # Compute the average test_loss
        avg_test_loss = test_loss / batch_num
        avg_AUC = AUC / batch_num
        avg_P = P / batch_num
        avg_R = R / batch_num
        # Print loss (uncomment lines below once implemented)
        print('\nTest set: Average loss: {:.4f}\t Accuracy: {:.4f}\tPrecision: {:.4f}\tRecall50: {:.4f}\n'.format(
            avg_test_loss, avg_AUC, avg_P, avg_R))
    return avg_test_loss


if __name__ == '__main__':


    data_transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Pad((7, 0)),
        transforms.Resize((128, 160)),
        transforms.ToTensor(),
    ])
    roadmap_transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((400, 400)),
        transforms.ToTensor()
    ])
    labeled_trainset = LabeledDatasetScene(image_folder=image_folder,
                                           annotation_file=annotation_csv,
                                           scene_index=labeled_scene_index,
                                           transform=data_transforms,
                                           roadmap_transform=roadmap_transforms,
                                           extra_info=False,
                                           scene_batch_size=1
                                           )
    trainset, testset = torch.utils.data.random_split(labeled_trainset, [int(0.90 * len(labeled_trainset)),
                                                                         len(labeled_trainset) - int(
                                                                             0.90 * len(labeled_trainset))])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8,
                                              collate_fn=collate_fn_lstm)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=8,
                                             collate_fn=collate_fn_lstm)
    anchors = get_anchors(anchor_file)
    # sample, target, road_image, extra = iter(trainloader).next()
    # print(torch.stack(sample).shape)

    if pretrain_file is not None:
        model = bothModel.trainModel(anchors, True)
        pretrain_dict = torch.load(pretrain_file)
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                         (k in model_dict and re.search('^efficientNet.*', k) and (not  re.search('^efficientNet._fc.*', k)))}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        #for para in model.efficientNet.parameters():
        #    para.requires_grad = False
    else:
        model = bothModel.trainModel(anchors, False)

    print("Model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model=model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=1e-4)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdaScheduler)

    last_test_loss = 2
    for epoch in range(1, 250 + 1):
        # Train model
        start_time = time.time()
        train(model, trainloader, optimizer, epoch)
        test_loss = test(model, testloader)
        print('lr=' + str(optimizer.param_groups[0]['lr']) + '\n')
        scheduler.step(epoch)
        if last_test_loss > test_loss:
            torch.save(model.state_dict(), 'bothModelFPNpreFT.pkl')
            last_test_loss = test_loss
        end_time = time.time()
        print("total_time=" + str(end_time - start_time) + '\n')
    model = model.cpu()
    model = model.cuda()
    test_loss = test(model, testloader)
    if (last_test_loss > test_loss):
        torch.save(model.state_dict(), 'bothModelFPNpreFT.pkl')
        last_test_loss = test_loss
