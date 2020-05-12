import math
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

import model.bothModel as bothModel
from dataset.dataHelper import LabeledDatasetScene
from utils.helper import collate_fn_lstm, compute_ts_road_map

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

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
labeled_scene_index = np.arange(132, 134)
start_epoch = 200
final_epoch = 250
long_cycle = 40
short_cycle = 5
start_lr = 0.01
gamma = 0.25
pretrain_file = "pretrainfinal.pkl"


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


def train(model, device, train_loader, optimizer, epoch, log_interval=50):
    # Set model to training mode
    model.train()
    # Loop through data points
    for batch_idx, data in enumerate(train_loader):
        # Send data and target to device
        sample, bbox_list, category_list, road_image = data
        sample, road_image = sample.to(device), road_image.to(device)
        # Zero out the optimizer
        optimizer.zero_grad()
        # Pass data through model
        outputs = model(sample, [bbox_list, category_list])
        # Compute the negative log likelihood loss
        road_loss = nn.NLLLoss()(outputs[0], road_image)
        detection_loss = outputs[3]
        loss = road_loss + detection_loss
        # Compute the negative log likelihood loss
        output0 = outputs[0].view(-1, 2, 400, 400)
        road_image = road_image.view(-1, 400, 400)
        _, predicted = torch.max(output0.data, 1)
        AC = compute_ts_road_map(predicted, road_image)
        # Backpropagate loss
        loss.backward()
        # Make a step with the optimizer
        optimizer.step()
        # Print loss (uncomment lines below once implemented)
        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRoad Loss: {:.6f}\tDetection Loss: {:.6f}\tAccuracy: {:.6f}\tPrecision: {:.6f}\tRecall50: {:.6f}'.format(
                    epoch, batch_idx * len(sample), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader.dataset), loss.item(), road_loss.item(), detection_loss.item(),
                    AC, (model.yolo1.metrics['precision'] + model.yolo0.metrics['precision']) / 2,
                           (model.yolo1.metrics['recall50'] + model.yolo0.metrics['recall50']) / 2))
    print(
        'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRoad Loss: {:.6f}\tDetection Loss: {:.6f}\tAccuracy: {:.6f}\tPrecision: {:.6f}\tRecall50: {:.6f}'.format(
            epoch, len(train_loader.dataset), len(train_loader.dataset),
                   100, loss.item(), road_loss.item(), detection_loss.item(), AC,
                   (model.yolo1.metrics['precision'] + model.yolo0.metrics['precision']) / 2,
                   (model.yolo1.metrics['recall50'] + model.yolo0.metrics['recall50']) / 2))


def test(model, device, test_loader):
    # Set model to evaluation mode
    model.eval()
    # Variable for the total loss
    test_loss = 0
    AC = 0
    P = 0
    R = 0
    with torch.no_grad():
        # Loop through data points
        batch_num = 0

        for batch_idx, data in enumerate(test_loader):
            # Send data to device
            sample, bbox_list, category_list, road_image = data
            sample, road_image = sample.to(device), road_image.to(device)
            # Pass data through model
            outputs = model(sample, [bbox_list, category_list])
            road_loss = nn.NLLLoss()(outputs[0], road_image)
            detection_loss = outputs[3]
            test_loss += road_loss + detection_loss
            # Compute the negative log likelihood loss
            output0 = outputs[0].view(-1, 2, 400, 400)
            road_image = road_image.view(-1, 400, 400)
            _, predicted = torch.max(output0.data, 1)
            AC += compute_ts_road_map(predicted, road_image)
            P += (model.yolo1.metrics['precision'] + model.yolo0.metrics['precision']) / 2
            R += (model.yolo1.metrics['recall50'] + model.yolo0.metrics['recall50']) / 2
            batch_num += 1
            # Add number of correct predictions to total num_correct
        # Compute the average test_loss
        avg_test_loss = test_loss / batch_num
        avg_AC = AC / batch_num
        avg_P = P / batch_num
        avg_R = R / batch_num
        # Print loss (uncomment lines below once implemented)
        print('\nTest set: Average loss: {:.4f}\t Accuracy: {:.4f}\tPrecision: {:.4f}\tRecall50: {:.4f}\n'.format(
            avg_test_loss, avg_AC, avg_P, avg_R))
    return avg_test_loss


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Pad((7, 0)),
        transforms.Resize((128, 160)),
        transforms.ToTensor(),
    ])
    roadmap_transforms = transforms.Compose([
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=8,
                                              collate_fn=collate_fn_lstm)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=8,
                                             collate_fn=collate_fn_lstm)
    anchors = get_anchors(anchor_file)

    model = bothModel.trainModel(anchors,device=device)
    if pretrain_file is not None:
        pretrain_dict = torch.load(pretrain_file, map_location=device)
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and re.search('^efficientNet.*', k)}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=1e-8)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdaScheduler)
    print("Model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    last_test_loss = 2
    for epoch in range(1, final_epoch + 1):
        # Train model
        start_time = time.time()
        train(model, device, trainloader, optimizer, epoch)
        test_loss = test(model, device, testloader)
        print('lr=' + str(optimizer.param_groups[0]['lr']) + '\n')
        scheduler.step(epoch)
        torch.save(model.state_dict(), 'bothModel.pkl')
        if last_test_loss > test_loss:
            torch.save(model.state_dict(), 'bothModel.pkl')
            last_test_loss = test_loss
        end_time = time.time()
        print("total_time=" + str(end_time - start_time) + '\n')
    # optimizer.swap_swa_sgd()
    model = model.cpu()
    # optimizer.bn_update(trainloader, model)
    model.to(device)
    test_loss = test(model, device, testloader)
    if (last_test_loss > test_loss):
        torch.save(model.state_dict(), 'bothModel.pkl')
        last_test_loss = test_loss
