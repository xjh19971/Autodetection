from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from torchvision import transforms

import model.bothModel as bothModel
from dataset.dataHelper import LabeledDatasetScene
from utils.helper import collate_fn_lstm
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
pretrain_file = None

if __name__ == '__main__':
    parser1 = ArgumentParser()
    trainparser = pl.Trainer.add_argparse_args(parser1, )
    trainparser.add_argument('--batch_size', default=4)
    trainparser.set_defaults(gpus=1)
    trainparser.set_defaults(max_epochs=3000)
    modelparser = bothModel.AutoNet.add_model_specific_args(trainparser)
    args1 = trainparser.parse_args()
    args2 = modelparser.parse_args()
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args1.batch_size, shuffle=True, num_workers=8,
                                              collate_fn=collate_fn_lstm)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args1.batch_size, shuffle=False, num_workers=8,
                                             collate_fn=collate_fn_lstm)

    model = bothModel.trainModel(args2)
    trainer = pl.Trainer.from_argparse_args(args1)
    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=testloader)
