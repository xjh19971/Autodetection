import re

import numpy as np
import pytorch_lightning as pl
import torch
from torchvision import transforms
from argparse import ArgumentParser
import model.bothModel as bothModel
from dataset.dataHelper import LabeledDatasetScene
from utils.helper import collate_fn_lstm
from utils.yolo_utils import get_anchors

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
pretrain_file = None

if __name__ == '__main__':
    parser = ArgumentParser()
    parser= pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--learning_rate', default=0.01)
    parser.set_defaults(max_epochs=3000)
    args=parser.parse_args()
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                              collate_fn=collate_fn_lstm)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                             collate_fn=collate_fn_lstm)
    anchors = get_anchors(anchor_file)

    model = bothModel.trainModel(anchors, freeze=False)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=testloader)
