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
        self.fc_num1 = 400
        self.fc_num2 = 600
        self.hparams = hparams
        self.learning_rate = hparams.learning_rate
        self.anchors = get_anchors(hparams.anchors_file)
        self.anchors1 = np.reshape(self.anchors[0], [1, 2])
        self.anchors0 = self.anchors[1:]
        self.detection_classes = hparams.detection_classes
        super(AutoNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b2', freeze=hparams.freeze)
        self.compressed = nn.Sequential(
            nn.Conv2d(352, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 8 * 10, self.fc_num1, bias=False),
            nn.BatchNorm1d(self.fc_num1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.fc2 = nn.ModuleList([])
        for i in range(6):
            if i != 1 and i != 4:
                self.fc2.append(nn.Sequential(
                    nn.Linear(self.fc_num1, 14 * 13 * 32, bias=False),
                    nn.BatchNorm1d(14 * 13 * 32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                ))
            else:
                self.fc2.append(nn.Sequential(
                    nn.Linear(self.fc_num1, 13 * 18 * 32, bias=False),
                    nn.BatchNorm1d(13 * 18 * 32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                ))
        self.compressed_1 = nn.Sequential(
            nn.Conv2d(352, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.fc1_1 = nn.Sequential(
            nn.Linear(32 * 8 * 10, self.fc_num2, bias=False),
            nn.BatchNorm1d(self.fc_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.fc2_1 = nn.ModuleList([])
        for i in range(6):
            if i != 1 and i != 4:
                self.fc2_1.append(nn.Sequential(
                    nn.Linear(self.fc_num2, 14 * 13 * 64, bias=False),
                    nn.BatchNorm1d(14 * 13 * 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                ))
            else:
                self.fc2_1.append(nn.Sequential(
                    nn.Linear(self.fc_num2, 13 * 18 * 64, bias=False),
                    nn.BatchNorm1d(13 * 18 * 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                ))

        self.reversedfc2 = nn.ModuleList([])
        for i in range(6):
            if i != 1 and i != 4:
                self.reversedfc2.append(nn.Sequential(
                    nn.Linear(14 * 13 * 96, (self.fc_num1 + self.fc_num2)//2, bias=False),
                    nn.BatchNorm1d((self.fc_num1 + self.fc_num2)//2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                ))
            else:
                self.reversedfc2.append(nn.Sequential(
                    nn.Linear(13 * 18 * 96, (self.fc_num1 + self.fc_num2)//2, bias=False),
                    nn.BatchNorm1d((self.fc_num1 + self.fc_num2)//2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                ))
        self.reversedfc1 = nn.Sequential(
            nn.Linear((self.fc_num1 + self.fc_num2)//2, 64 * 8 * 10, bias=False),
            nn.BatchNorm1d(64 * 8 * 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        '''
        self.recompressed = nn.Sequential(
            nn.Conv2d(64, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        '''
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

        self.inplanes = 16
        self.conv1_1_detect = self._make_layer(BasicBlock, 16, 2)
        self.convfinal_1 = nn.Conv2d(16, len(self.anchors1) * (self.detection_classes + 5), 1)
        self.yolo1 = YOLOLayer(self.anchors1, self.detection_classes, 800)
        self.conv1_1 = self._make_layer(BasicBlock, 16, 2)

        self.deconv0_for_self = self._make_deconv_layer(64, 32)
        self.inplanes = 32
        self.conv0_for_self = self._make_layer(BasicBlock, 32, 2)
        self.deconv1_for_self = self._make_deconv_layer(32, 16)
        self.inplanes = 16
        self.conv1_for_self = self._make_layer(BasicBlock, 16, 2)
        self.deconv2_for_self = self._make_deconv_layer(16, 8)
        self.inplanes = 8
        self.conv2_for_self = self._make_layer(BasicBlock, 8, 2)
        self.deconv3_for_self = self._make_deconv_layer(8, 4)
        self.conv3_for_self = nn.Conv2d(4, 3, 1)
        self.upSample = nn.Upsample(scale_factor=2)
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

    def limitedFC1(self, x, fc, filter):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        output = FloatTensor(x.size(0) // 6, filter, 25, 25).fill_(0)
        x = x.view(x.size(0) // 6, 6, -1)
        for i, block in enumerate(fc):
            if i == 0:
                output[:, :, :13, 11:] += block(x[:, i, :]).view(x.size(0), -1, 13, 14)
            elif i == 1:
                output[:, :, 4:22, 12:] += block(x[:, i, :]).view(x.size(0), -1, 18, 13)
            elif i == 2:
                output[:, :, 12:, 11:] += block(x[:, i, :]).view(x.size(0), -1, 13, 14)
            elif i == 3:
                output[:, :, :13, :14] += block(x[:, i, :]).view(x.size(0), -1, 13, 14)
            elif i == 4:
                output[:, :, 4:22, :13] += block(x[:, i, :]).view(x.size(0), -1, 18, 13)
            elif i == 5:
                output[:, :, 12:, :14] += block(x[:, i, :]).view(x.size(0), -1, 13, 14)
        return output

    def ReversedlimitedFC1(self, x, fc):
        output = []
        for i, block in enumerate(fc):
            if i == 0:
                output.append(block(x[:, :, :13, 11:].reshape(x.size(0), -1)))
            elif i == 1:
                output.append(block(x[:, :, 4:22, 12:].reshape(x.size(0), -1)))
            elif i == 2:
                output.append(block(x[:, :, 12:, 11:].reshape(x.size(0), -1)))
            elif i == 3:
                output.append(block(x[:, :, :13, :14].reshape(x.size(0), -1)))
            elif i == 4:
                output.append(block(x[:, :, 4:22, :13].reshape(x.size(0), -1)))
            elif i == 5:
                output.append(block(x[:, :, 12:, :14].reshape(x.size(0), -1)))
        output = torch.stack(output, dim=1)
        output = output.view(output.size(0) * 6, -1)
        return output

    def forward(self, x, detection_target=None, loss_mask=None):
        scene = x.size(0)
        batch = x.size(2)
        # TODO time-step self learning
        real_detection_target=[[],[]]
        for i in range(2):
            for j in range(len(detection_target[i])):
                for k in range(len(detection_target[i][j])):
                    real_detection_target[i].append(detection_target[i][j][k])
        x = x.view(-1, 3, 256, 320)
        output_list = self.efficientNet(x)
        x = output_list[2]
        x1 = self.compressed(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)
        x1_for_x3 = self.limitedFC1(x1, self.fc2, 32)
        x1 = self.conv0(x1_for_x3)
        x1 = self.deconv0(x1)
        x1 = self.conv1(x1)
        x1 = self.deconv1(x1)
        x1 = self.conv2(x1)
        x1 = self.deconv2(x1)
        x1 = self.conv3(x1)
        x1 = self.deconv3(x1)
        x1 = self.convfinal(x1)

        x2 = self.compressed_1(x)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc1_1(x2)
        x2_for_x3 = self.limitedFC1(x2, self.fc2_1, 64)
        x2 = self.conv0_1(x2_for_x3)
        detect_output0 = self.conv0_1_detect(x2)
        detect_output0 = self.convfinal_0(detect_output0)
        detect_output0, detect_loss0 = self.yolo0(detect_output0, real_detection_target, 800, loss_mask)
        x2 = self.deconv0_1(x2)

        x2 = self.conv1_1(x2)
        detect_output1 = self.conv1_1_detect(x2)
        detect_output1 = self.convfinal_1(detect_output1)
        detect_output1, detect_loss1 = self.yolo1(detect_output1, real_detection_target, 800, loss_mask)
        total_loss = 0.6 * detect_loss0 + 0.4 * detect_loss1

        x3 = torch.cat([x1_for_x3, x2_for_x3], dim=1)
        x3 = self.ReversedlimitedFC1(x3, self.reversedfc2)
        x3 = self.reversedfc1(x3)
        x3 = x3.view(x.size(0), -1, 8, 10)
        #x3 = self.recompressed(x3)
        x3 = self.deconv0_for_self(x3)
        x3 = self.conv0_for_self(x3)
        x3 = self.deconv1_for_self(x3)
        x3 = self.conv1_for_self(x3)
        x3 = self.deconv2_for_self(x3)
        x3 = self.conv2_for_self(x3)
        x3 = self.deconv3_for_self(x3)
        x3 = self.conv3_for_self(x3)
        x3 = self.upSample(x3)
        x3 = x3.view(scene, 2, batch, 18, 256, 320)
        return nn.LogSoftmax(dim=1)(x1), detect_output0, detect_output1, total_loss, nn.Sigmoid()(x3)

    def loss_function(self, outputs, road_image, image, loss_mask):
        road_loss = nn.NLLLoss(reduction='none')(outputs[0], road_image.view(-1,400,400))
        road_loss = torch.mean(road_loss * loss_mask.view(-1,1,1))*2
        detection_loss = outputs[3]
        self_loss = nn.BCELoss()(outputs[4], image)
        loss = road_loss + detection_loss + self_loss
        return loss, self_loss

    def training_step(self, batch, batch_idx):
        sample, bbox_list, category_list, road_image, loss_mask = batch
        outputs = self(sample, [bbox_list, category_list], loss_mask)
        loss, self_loss = self.loss_function(outputs, road_image, sample, loss_mask)
        output0 = outputs[0].view(-1, 2, 400, 400)
        road_image = road_image.view(-1, 400, 400)
        _, predicted = torch.max(output0.data, 1)
        AC = compute_ts_road_map(predicted, road_image)
        P = torch.tensor((self.yolo0.metrics['precision'] + self.yolo1.metrics['precision']) / 2)
        R = torch.tensor((self.yolo0.metrics['recall50'] + self.yolo1.metrics['recall50']) / 2)
        lr = torch.tensor(self.trainer.optimizers[0].param_groups[0]['lr'])
        log_bar = {'roadmap_score': AC, 'precision': P, 'recall': R, 'self_loss': self_loss, 'lr': lr}
        log = {'loss': loss, 'roadmap_score': AC, 'precision': P, 'recall': R, 'self_loss': self_loss, 'lr': lr}
        return {'loss': loss, 'log': log, 'progress_bar': log_bar}

    def validation_step(self, batch, batch_idx):
        sample, bbox_list, category_list, road_image, loss_mask = batch
        sample, road_image = sample, road_image
        outputs = self(sample, [bbox_list, category_list], loss_mask)
        loss, self_loss = self.loss_function(outputs, road_image, sample, loss_mask)
        # Pass data through model
        outputs = self(sample, [bbox_list, category_list])
        output0 = outputs[0].view(-1, 2, 400, 400)
        road_image = road_image.view(-1, 400, 400)
        _, predicted = torch.max(output0.data, 1)
        AC = compute_ts_road_map(predicted, road_image)
        P = torch.tensor((self.yolo0.metrics['precision'] + self.yolo1.metrics['precision']) / 2)
        R = torch.tensor((self.yolo0.metrics['recall50'] + self.yolo1.metrics['recall50']) / 2)
        return {'val_loss': loss, 'roadmap_score': AC, 'precision': P, 'recall': R, 'self_loss': self_loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_AC = torch.stack([x['roadmap_score'] for x in outputs]).mean()
        avg_P = torch.stack([x['precision'] for x in outputs]).mean()
        avg_R = torch.stack([x['recall'] for x in outputs]).mean()
        avg_self_loss = torch.stack([x['self_loss'] for x in outputs]).mean()
        log = {'avg_val_loss': avg_val_loss, 'avg_roadmap_score': avg_AC, 'avg_precision': avg_P, 'avg_recall': avg_R,
               'avg_self_loss': avg_self_loss}
        return {'log': log, 'val_loss': avg_val_loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=10)
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
