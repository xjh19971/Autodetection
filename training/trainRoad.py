import math
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

import model.roadModel as roadModel
from dataset.dataHelper import LabeledDatasetScene
from utils.helper import collate_fn_lstm, compute_ts_road_map

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
start_epoch = 150
final_epoch = 200
long_cycle = 30
short_cycle = 5
start_lr = 0.01
batch_size = 8
pretrain_file = "pretrainfinal.pkl"


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
        output = model(sample)
        output = output.view(-1, 2, 400, 400)
        road_image = road_image.view(-1, 400, 400)
        # Compute the negative log likelihood loss
        loss = nn.NLLLoss()(output, road_image)
        _, predicted = torch.max(output.data, 1)
        AC = compute_ts_road_map(predicted, road_image)
        # Backpropagate loss
        loss.backward()
        # Make a step with the optimizer
        optimizer.step()
        # Print loss (uncomment lines below once implemented)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), AC))
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
        epoch, len(train_loader.dataset), len(train_loader.dataset),
        100, loss.item(), AC))


def test(model, device, test_loader):
    # Set model to evaluation mode
    model.eval()
    # Variable for the total loss
    test_loss = 0
    AC = 0
    with torch.no_grad():
        # Loop through data points
        batch_num = 0
        for batch_idx, data in enumerate(test_loader):
            # Send data to device
            sample, bbox_list, category_list, road_image = data
            sample, road_image = sample.to(device), road_image.to(device)
            # Pass data through model
            output = model(sample)
            output = output.view(-1, 2, 400, 400)
            road_image = road_image.view(-1, 400, 400)
            test_loss += nn.NLLLoss()(output, road_image)
            _, predicted = torch.max(output.data, 1)
            AC += compute_ts_road_map(predicted, road_image)
            batch_num += 1
            # Add number of correct predictions to total num_correct
        # Compute the average test_loss
        avg_test_loss = test_loss / batch_num
        avg_AC = AC / batch_num
        # Print loss (uncomment lines below once implemented)
        print('\nTest set: Average loss: {:.4f}\t Accuracy: {:.4f}\n'.format(avg_test_loss, avg_AC))
    return avg_test_loss


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Pad((7, 0)),
        transforms.Resize((128, 160), 0),
        transforms.ToTensor(),
    ])
    roadmap_transforms = transforms.Compose([
        transforms.Resize((400, 400), 0),
        transforms.ToTensor()
    ])
    labeled_trainset = LabeledDatasetScene(image_folder=image_folder,
                                           annotation_file=annotation_csv,
                                           scene_index=labeled_scene_index,
                                           transform=data_transforms,
                                           roadmap_transform=roadmap_transforms,
                                           extra_info=True,
                                           scene_batch_size=1
                                           )
    trainset, testset = torch.utils.data.random_split(labeled_trainset, [int(0.90 * len(labeled_trainset)),
                                                                         len(labeled_trainset) - int(
                                                                             0.90 * len(labeled_trainset))])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8,
                                              collate_fn=collate_fn_lstm)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8,
                                             collate_fn=collate_fn_lstm)

    if pretrain_file is not None:
        model = roadModel.trainModel(freeze=True, device=device)
        pretrain_dict = torch.load(pretrain_file, map_location=device)
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                         (k in model_dict and re.search('^efficientNet.*', k) and (
                             not re.search('^efficientNet._fc.*', k)))}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
    else:
        model = roadModel.trainModel(freeze=False, device=device)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=1e-4)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdaScheduler)
    print("Model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    last_test_loss = 1
    for epoch in range(1, final_epoch + 1):
        # Train model
        start_time = time.time()
        train(model, device, trainloader, optimizer, epoch)
        test_loss = test(model, device, testloader)
        print('lr=' + str(optimizer.param_groups[0]['lr']) + '\n')
        if last_test_loss > test_loss:
            torch.save(model.state_dict(), 'roadModel.pkl')
            last_test_loss = test_loss
        end_time = time.time()
        print("total_time=" + str(end_time - start_time) + '\n')
    test_loss = test(model, device, testloader)
    if (last_test_loss > test_loss):
        torch.save(model.state_dict(), 'roadModel.pkl')
        last_test_loss = test_loss
