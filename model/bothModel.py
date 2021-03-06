from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from model.backboneModel import EfficientNet, YOLOLayer, BasicBlock
from utils.helper import compute_ts_road_map
from utils.yolo_utils import get_anchors


class AutoNet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.latent = 600
        self.fc_num = 300
        self.hparams = hparams
        self.learning_rate = hparams.learning_rate
        self.anchors = get_anchors(hparams.anchors_file)
        self.anchors1 = np.reshape(self.anchors[0], [1, 2])
        self.anchors0 = self.anchors[1:]
        self.detection_classes = hparams.detection_classes
        super(AutoNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b1', freeze=hparams.freeze)
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=self.latent),
            nn.BatchNorm1d(self.latent),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent, self.fc_num, bias=False),
            nn.BatchNorm1d(self.fc_num),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_num * 6, 25 * 25 * 32, bias=False),
            nn.BatchNorm1d(25 * 25 * 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc1_1 = nn.Sequential(
            nn.Linear(self.latent, self.fc_num, bias=False),
            nn.BatchNorm1d(self.fc_num),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc2_1 = nn.Sequential(
            nn.Linear(self.fc_num * 6, 25 * 25 * 64, bias=False),
            nn.BatchNorm1d(25 * 25 * 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.inplanes = 32
        self.conv0 = self._make_layer(BasicBlock, 32, 2)
        self.deconv0 = self._make_deconv_layer(32, 16)
        self.inplanes = 16
        self.conv1 = self._make_layer(BasicBlock, 16, 2)
        self.deconv1 = self._make_deconv_layer(16, 8)
        self.inplanes = 8
        self.conv2 = self._make_layer(BasicBlock, 8, 2)
        self.deconv2 = self._make_deconv_layer(8, 4)
        self.inplanes = 4
        self.conv3 = self._make_layer(BasicBlock, 4, 2)
        self.deconv3 = self._make_deconv_layer(4, 2)
        self.convfinal = nn.Conv2d(2, 2, 1)

        self.inplanes = 64
        self.conv0_1_detect = self._make_layer(BasicBlock, 64, 2)
        self.convfinal_0 = nn.Conv2d(64, len(self.anchors0) * (self.detection_classes + 5), 1)
        self.yolo0 = YOLOLayer(self.anchors0, self.detection_classes, 800)
        self.conv0_1 = self._make_layer(BasicBlock, 64, 2)
        self.deconv0_1 = self._make_deconv_layer(64, 16)
        self.conv0_1 = self._make_layer(BasicBlock, 64, 2)

        self.inplanes = 16
        self.conv1_1_detect = self._make_layer(BasicBlock, 16, 2)
        self.convfinal_1 = nn.Conv2d(16, len(self.anchors1) * (self.detection_classes + 5), 1)
        self.yolo1 = YOLOLayer(self.anchors1, self.detection_classes, 800)
        self.conv1_1 = self._make_layer(BasicBlock, 16, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, inplanes, outplanes):
        layers = []
        layers.append(
            nn.ConvTranspose2d(inplanes, outplanes, 3, stride=2,
                               padding=1, output_padding=1, bias=False))
        layers.append(nn.BatchNorm2d(outplanes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, detection_target=None):
        x = x.view(-1, 3, 128, 160)
        output_list = self.efficientNet(x)
        x = output_list[3].view(output_list[3].size(0), -1)

        x1 = self.fc1(x)
        x1 = x1.view(-1, self.fc_num * 6)
        x1 = self.fc2(x1)
        x1 = x1.view(x1.size(0), -1, 25, 25)
        x1 = self.conv0(x1)
        x1 = self.deconv0(x1)
        x1 = self.conv1(x1)
        x1 = self.deconv1(x1)
        x1 = self.conv2(x1)
        x1 = self.deconv2(x1)
        x1 = self.conv3(x1)
        x1 = self.deconv3(x1)
        x1 = self.convfinal(x1)

        x2 = self.fc1_1(x)
        x2 = x2.view(-1, self.fc_num * 6)
        x2 = self.fc2_1(x2)
        x2 = x2.view(x2.size(0), -1, 25, 25)
        x2 = self.conv0_1(x2)
        detect_output0 = self.conv0_1_detect(x2)
        detect_output0 = self.convfinal_0(detect_output0)
        detect_output0, detect_loss0 = self.yolo0(detect_output0, detection_target, 800)
        x2 = self.deconv0_1(x2)

        x2 = self.conv1_1(x2)
        detect_output1 = self.conv1_1_detect(x2)
        detect_output1 = self.convfinal_1(detect_output1)
        detect_output1, detect_loss1 = self.yolo1(detect_output1, detection_target, 800)
        total_loss = 0.6 * detect_loss0 + 0.4 * detect_loss1
        return nn.LogSoftmax(dim=1)(x1), detect_output0, detect_output1, total_loss

    def loss_function(self, outputs, road_image):
        road_loss = nn.NLLLoss()(outputs[0], road_image)
        detection_loss = outputs[3]
        loss = road_loss + detection_loss
        return loss

    def training_step(self, batch, batch_idx):
        sample, bbox_list, category_list, road_image = batch
        sample, road_image = sample, road_image
        outputs = self(sample, [bbox_list, category_list])
        loss = self.loss_function(outputs, road_image)
        output0 = outputs[0].view(-1, 2, 400, 400)
        road_image = road_image.view(-1, 400, 400)
        _, predicted = torch.max(output0.data, 1)
        AC = compute_ts_road_map(predicted, road_image)
        P = torch.tensor((self.yolo0.metrics['precision'] + self.yolo1.metrics['precision']) / 2)
        R = torch.tensor((self.yolo0.metrics['recall50'] + self.yolo1.metrics['recall50']) / 2)
        lr = torch.tensor(self.trainer.optimizers[0].param_groups[0]['lr'])
        log_bar = {'roadmap_score': AC, 'precision': P, 'recall': R, 'lr': lr}
        log={'loss': loss,'roadmap_score': AC, 'precision': P, 'recall': R, 'lr': lr}
        return {'loss': loss, 'log': log, 'progress_bar': log_bar}

    def validation_step(self, batch, batch_idx):
        sample, bbox_list, category_list, road_image = batch
        sample, road_image = sample, road_image
        outputs = self(sample, [bbox_list, category_list])
        loss = self.loss_function(outputs, road_image)
        # Pass data through model
        outputs = self(sample, [bbox_list, category_list])
        output0 = outputs[0].view(-1, 2, 400, 400)
        road_image = road_image.view(-1, 400, 400)
        _, predicted = torch.max(output0.data, 1)
        AC = compute_ts_road_map(predicted, road_image)
        P = torch.tensor((self.yolo0.metrics['precision'] + self.yolo1.metrics['precision']) / 2)
        R = torch.tensor((self.yolo0.metrics['recall50'] + self.yolo1.metrics['recall50']) / 2)
        return {'val_loss': loss, 'roadmap_score': AC, 'precision': P, 'recall': R}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_AC = torch.stack([x['roadmap_score'] for x in outputs]).mean()
        avg_P = torch.stack([x['precision'] for x in outputs]).mean()
        avg_R = torch.stack([x['recall'] for x in outputs]).mean()
        log = {'avg_val_loss': avg_val_loss, 'avg_roadmap_score': avg_AC, 'avg_precision': avg_P, 'avg_recall': avg_R}
        return {'log': log, 'val_loss': avg_val_loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--anchors_file', type=str, default='yolo_anchors.txt')
        parser.add_argument('--detection_classes', type=int, default=9)
        parser.add_argument('--freeze', type=bool, default=False)
        parser.add_argument('--learning_rate', type=float, default=0.01)
        return parser


def trainModel(args):
    return AutoNet(args)


def testModel(args):
    return AutoNet(args)
