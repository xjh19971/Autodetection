import numpy as np
import torch
import torch.nn as nn

from model.EfficientNetBackbone import EfficientNet
from model.detectionModel import YOLOLayer


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AutoNet(nn.Module):
    def __init__(self, scene_batch_size, batch_size, step_size, device, anchors, detection_classes, num_classes=2):
        self.fc_num1 = 200
        self.fc_num2 = 200
        self.batch_size = batch_size
        self.step_size = step_size
        self.scene_batch_size = scene_batch_size
        self.num_classes = num_classes
        self.device = device
        self.anchors = anchors
        self.anchors2 = np.reshape(anchors[0], [1, 2])
        self.anchors1 = anchors[1:5, :]
        self.anchors0 = anchors[5:, :]
        self.detection_classes = detection_classes
        super(AutoNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b3')
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
        )
        self.fc1_1_1 = nn.Sequential(
            nn.Linear(384 * 4 * 5, self.fc_num1 * 3, bias=False),
            nn.BatchNorm1d(self.fc_num1 * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc1_1_2 = nn.Sequential(
            nn.Linear(136 * 8 * 10, self.fc_num1 * 3, bias=False),
            nn.BatchNorm1d(self.fc_num1 * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc1_1_3 = nn.Sequential(
            nn.Linear(48 * 16 * 20, self.fc_num1 * 3, bias=False),
            nn.BatchNorm1d(self.fc_num1 * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc2_1_1 = nn.Sequential(
            nn.Linear(self.fc_num1 * 6*3, 25 * 25 * 32, bias=False),
            nn.BatchNorm1d(25 * 25 * 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc2_1_2 = nn.Sequential(
            nn.Linear(self.fc_num1 * 6 *3, 50 * 50 * 8, bias=False),
            nn.BatchNorm1d(50 * 50 * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc2_1_3 = nn.Sequential(
            nn.Linear(self.fc_num1 * 6 *3, 100 * 100 * 2, bias=False),
            nn.BatchNorm1d(100 * 100 * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc1_2_1 = nn.Sequential(
            nn.Linear(384 * 4 * 5, self.fc_num2 * 3, bias=False),
            nn.BatchNorm1d(self.fc_num2 * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc1_2_2 = nn.Sequential(
            nn.Linear(136 * 8 * 10, self.fc_num2 * 3, bias=False),
            nn.BatchNorm1d(self.fc_num2 * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc1_2_3 = nn.Sequential(
            nn.Linear(48 * 16 * 20, self.fc_num2 * 3, bias=False),
            nn.BatchNorm1d(self.fc_num2 * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc2_2_1 = nn.Sequential(
            nn.Linear(self.fc_num2 * 6*3, 25 * 25 * 128, bias=False),
            nn.BatchNorm1d(25 * 25 * 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc2_2_2 = nn.Sequential(
            nn.Linear(self.fc_num2 * 6*3, 50 * 50 * 32, bias=False),
            nn.BatchNorm1d(50 * 50 * 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc2_2_3 = nn.Sequential(
            nn.Linear(self.fc_num2 * 6*3, 100 * 100 * 8, bias=False),
            nn.BatchNorm1d(100 * 100 * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.inplanes = 32
        self.conv0 = self._make_layer(BasicBlock, 32, 2)
        self.deconv0 = self._make_deconv_layer(32, 8)
        self.inplanes = 16
        self.conv1 = self._make_layer(BasicBlock, 16, 2)
        self.deconv1 = self._make_deconv_layer(16, 2)
        self.inplanes = 4
        self.conv2 = self._make_layer(BasicBlock, 4, 2)
        self.deconv2 = self._make_deconv_layer(4, 2)
        self.inplanes = 2
        self.convfinal = nn.Conv2d(2, 2, 1)

        self.inplanes = 128
        self.conv0_1_detect = self._make_layer(BasicBlock, 128, 2)
        self.convfinal_0 = nn.Conv2d(128, len(self.anchors0) * (self.detection_classes + 5), 1)
        self.yolo0 = YOLOLayer(self.anchors0, self.detection_classes, self.device, 800)
        self.conv0_1 = self._make_layer(BasicBlock, 128, 2)
        self.deconv0_1 = self._make_deconv_layer(128, 32)
        self.conv0_1 = self._make_layer(BasicBlock, 128, 2)

        self.inplanes = 64
        self.conv1_1_detect = self._make_layer(BasicBlock, 64, 2)
        self.convfinal_1 = nn.Conv2d(64, len(self.anchors1) * (self.detection_classes + 5), 1)
        self.yolo1 = YOLOLayer(self.anchors1, self.detection_classes, self.device, 800)
        self.conv1_1 = self._make_layer(BasicBlock, 64, 2)
        self.deconv1_1 = self._make_deconv_layer(64, 8)

        self.inplanes = 16
        self.conv2_1_detect = self._make_layer(BasicBlock, 16, 2)
        self.convfinal_2 = nn.Conv2d(16, len(self.anchors2) * (self.detection_classes + 5), 1)
        self.yolo2 = YOLOLayer(self.anchors2, self.detection_classes, self.device, 800)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LSTM):
                nn.init.xavier_normal_(m.all_weights[0][0])
                nn.init.xavier_normal_(m.all_weights[0][1])
                nn.init.xavier_normal_(m.all_weights[1][0])
                nn.init.xavier_normal_(m.all_weights[1][1])

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

    def forward(self, x, detection_target):
        # (S,B,18,H,W)
        scene = x.size(0)
        step = x.size(1)
        x = x.view(-1, 3, 128, 160)
        output_list = self.efficientNet(x)
        feature1 = output_list[2].view(output_list[2].size(0), -1)
        feature2 = output_list[1].view(output_list[1].size(0), -1)
        feature3 = output_list[0].view(output_list[0].size(0), -1)
        featurefc1_1 = self.fc1_1_1(feature1)
        featurefc1_2 = self.fc1_2_1(feature1)
        featurefc2_1 = self.fc1_1_2(feature2)
        featurefc2_2 = self.fc1_2_2(feature2)
        featurefc3_1 = self.fc1_1_3(feature3)
        featurefc3_2 = self.fc1_2_3(feature3)

        x1 = torch.cat([featurefc1_1[:,:self.fc_num1], featurefc2_1[:,:self.fc_num1], featurefc3_1[:,:self.fc_num1]], dim=1)
        x1 = x1.view(-1, self.fc_num1 * 6 * 3)
        x1 = self.fc2_1_1(x1)
        x1 = x1.view(x1.size(0), -1, 25, 25)
        x1 = self.conv0(x1)
        x1 = self.deconv0(x1)  # detection

        x1_1 = torch.cat([featurefc1_1[:,self.fc_num1:self.fc_num1 * 2], featurefc2_1[:,self.fc_num1:self.fc_num1 * 2],
                          featurefc3_1[:,self.fc_num1:self.fc_num1 * 2]], dim=1)
        x1_1 = x1_1.view(-1, self.fc_num1 * 6 * 3)
        x1_1 = self.fc2_1_2(x1_1)
        x1_1 = x1_1.view(x1_1.size(0), -1, 50, 50)
        x1 = torch.cat([x1, x1_1], dim=1)
        x1 = self.conv1(x1)
        x1 = self.deconv1(x1)

        x1_2 = torch.cat(
            [featurefc1_1[:,self.fc_num1 * 2:self.fc_num1 * 3], featurefc2_1[:,self.fc_num1 * 2:self.fc_num1 * 3],
             featurefc3_1[:,self.fc_num1 * 2:self.fc_num1 * 3]], dim=1)
        x1_2 = x1_2.view(-1, self.fc_num1 * 6 * 3)
        x1_2 = self.fc2_1_3(x1_2)
        x1_2 = x1_2.view(x1_2.size(0), -1, 100, 100)
        x1 = torch.cat([x1, x1_2], dim=1)
        x1 = self.conv2(x1)
        x1 = self.deconv2(x1)  # resize conv conv resize conv conv)
        x1 = self.convfinal(x1)

        x2 = torch.cat([featurefc1_2[:,:self.fc_num2], featurefc2_2[:,:self.fc_num2], featurefc3_2[:,:self.fc_num2]], dim=1)
        x2 = x2.view(-1, self.fc_num2 * 6*3)
        x2 = self.fc2_2_1(x2)
        x2 = x2.view(x2.size(0), -1, 25, 25)  # x = x.view(x.size(0)*6,-1,128,160)
        detect_output0 = self.conv0_1_detect(x2)
        detect_output0 = self.convfinal_0(detect_output0)
        detect_output0, detect_loss0 = self.yolo0(detect_output0, detection_target, 800)
        x2 = self.conv0_1(x2)
        x2 = self.deconv0_1(x2)  # detection

        x2_1 = torch.cat([featurefc1_2[:,self.fc_num2:self.fc_num2 * 2], featurefc2_2[:,self.fc_num2:self.fc_num2 * 2],
                          featurefc3_2[:,self.fc_num2:self.fc_num2 * 2]], dim=1)
        x2_1 = x2_1.view(-1, self.fc_num2 * 6*3)
        x2_1 = self.fc2_2_2(x2_1)
        x2_1 = x2_1.view(x2_1.size(0), -1, 50, 50)
        x2 = torch.cat([x2, x2_1], dim=1)
        detect_output1 = self.conv1_1_detect(x2)
        detect_output1 = self.convfinal_1(detect_output1)
        detect_output1, detect_loss1 = self.yolo1(detect_output1, detection_target, 800)
        x2 = self.conv1_1(x2)
        x2 = self.deconv1_1(x2)

        x2_2 = torch.cat(
            [featurefc1_2[:,self.fc_num2 * 2:self.fc_num2 * 3], featurefc2_2[:,self.fc_num2 * 2:self.fc_num2 * 3],
             featurefc3_2[:,self.fc_num2 * 2:self.fc_num2 * 3]], dim=1)
        x2_2 = x2_2.view(-1, self.fc_num2 * 6*3)
        x2_2 = self.fc2_2_3(x2_2)
        x2_2 = x2_2.view(x2_2.size(0), -1, 100, 100)
        x2 = torch.cat([x2, x2_2], dim=1)
        detect_output2 = self.conv2_1_detect(x2)
        detect_output2 = self.convfinal_2(detect_output2)
        detect_output2, detect_loss2 = self.yolo2(detect_output2, detection_target, 800)

        total_loss = 0.4 * detect_loss0 + 0.3 * detect_loss1 + 0.3 * detect_loss2
        
        return nn.LogSoftmax(dim=1)(x1), detect_output0, detect_output1, detect_output2, total_loss


def trainModel(device, anchors, detection_classes=9, scene_batch_size=4, batch_size=8, step_size=4):
    return AutoNet(scene_batch_size, batch_size, step_size, device, anchors, detection_classes)
