import numpy as np
import torch
import torch.nn as nn

from model.backboneModel import EfficientNet, YOLOLayer, BasicBlock


class AutoNet(nn.Module):
    def __init__(self, anchors, detection_classes, freeze=True, device=None):
        self.latent = 1000
        self.fc_num1 = 100
        self.fc_num2 = 100
        self.anchors = anchors
        self.anchors2 = np.reshape(anchors[0], [1, 2])
        self.anchors1 = anchors[1:5, :]
        self.anchors0 = anchors[5:, :]
        self.device = device
        self.detection_classes = detection_classes
        super(AutoNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b3', freeze=freeze)
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=2 * self.latent),
            # nn.Dropout(p=0.25)
        )
        # self.rnn1 = nn.LSTM(self.latent, self.fc_num, 2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent, self.fc_num1, bias=False),
            nn.BatchNorm1d(self.fc_num1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc2 = nn.ModuleList([])
        for i in range(6):
            if i != 1 and i != 4:
                self.fc2.append(nn.Sequential(
                    nn.Linear(self.fc_num1, 14 * 13 * 32, bias=False),
                    nn.BatchNorm1d(14 * 13 * 32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                ))
            else:
                self.fc2.append(nn.Sequential(
                    nn.Linear(self.fc_num1, 13 * 18 * 32, bias=False),
                    nn.BatchNorm1d(13 * 18 * 32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                ))
        # self.rnn1_1 = nn.LSTM(self.latent, self.fc_num, 2, batch_first=True, dropout=0.2)
        self.fc1_1 = nn.Sequential(
            nn.Linear(384 * 4 * 5, self.fc_num2 * 3, bias=False),
            nn.BatchNorm1d(self.fc_num2 * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc1_2 = nn.Sequential(
            nn.Linear(136 * 8 * 10, self.fc_num2 * 3, bias=False),
            nn.BatchNorm1d(self.fc_num2 * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fc2_1 = nn.ModuleList([])
        for i in range(6):
            if i != 1 and i != 4:
                self.fc2_1.append(nn.Sequential(
                    nn.Linear(self.fc_num2 * 2, 14 * 13 * 128, bias=False),
                    nn.BatchNorm1d(14 * 13 * 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                ))
            else:
                self.fc2_1.append(nn.Sequential(
                    nn.Linear(self.fc_num2 * 2, 13 * 18 * 128, bias=False),
                    nn.BatchNorm1d(13 * 18 * 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                ))
        self.fc2_2 = nn.ModuleList([])
        for i in range(6):
            if i != 1 and i != 4:
                self.fc2_2.append(nn.Sequential(
                    nn.Linear(self.fc_num2 * 2, 28 * 26 * 32, bias=False),
                    nn.BatchNorm1d(28 * 26 * 32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                ))
            else:
                self.fc2_2.append(nn.Sequential(
                    nn.Linear(self.fc_num2 * 2, 26 * 36 * 32, bias=False),
                    nn.BatchNorm1d(26 * 36 * 32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                ))
        self.fc2_3 = nn.ModuleList([])
        for i in range(6):
            if i != 1 and i != 4:
                self.fc2_3.append(nn.Sequential(
                    nn.Linear(self.fc_num2 * 2, 56 * 52 * 8, bias=False),
                    nn.BatchNorm1d(56 * 52 * 8),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                ))
            else:
                self.fc2_3.append(nn.Sequential(
                    nn.Linear(self.fc_num2 * 2, 52 * 72 * 8, bias=False),
                    nn.BatchNorm1d(52 * 72 * 8),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                ))
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

        self.inplanes = 128
        self.conv0_1_detect = self._make_layer(BasicBlock, 128, 2)
        self.convfinal_0 = nn.Conv2d(128, len(self.anchors0) * (self.detection_classes + 5), 1)
        self.yolo0 = YOLOLayer(self.anchors0, self.detection_classes, 800, device=device)
        self.conv0_1 = self._make_layer(BasicBlock, 128, 2)
        self.deconv0_1 = self._make_deconv_layer(128, 32)

        self.inplanes = 64
        self.conv1_1_detect = self._make_layer(BasicBlock, 64, 2)
        self.convfinal_1 = nn.Conv2d(64, len(self.anchors1) * (self.detection_classes + 5), 1)
        self.yolo1 = YOLOLayer(self.anchors1, self.detection_classes, 800, device=device)
        self.conv1_1 = self._make_layer(BasicBlock, 64, 2)
        self.deconv1_1 = self._make_deconv_layer(64, 8)

        self.inplanes = 16
        self.conv2_1_detect = self._make_layer(BasicBlock, 16, 2)
        self.convfinal_2 = nn.Conv2d(16, len(self.anchors2) * (self.detection_classes + 5), 1)
        self.yolo2 = YOLOLayer(self.anchors2, self.detection_classes, 800, device=device)
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

    def reparameterise(self, mu, logvar):
        return mu

    def limitedFC1(self, x, fc, filter):
        output = torch.zeros((x.size(0) // 6, filter, 25, 25)).cuda()
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

    def limitedFC2(self, x, fc, filter):
        output = torch.zeros((x.size(0) // 6, filter, 50, 50)).cuda()
        x = x.view(x.size(0) // 6, 6, -1)
        for i, block in enumerate(fc):
            if i == 0:
                output[:, :, :26, 22:] += block(x[:, i, :]).view(x.size(0), -1, 26, 28)
            elif i == 1:
                output[:, :, 8:44, 24:] += block(x[:, i, :]).view(x.size(0), -1, 36, 26)
            elif i == 2:
                output[:, :, 24:, 22:] += block(x[:, i, :]).view(x.size(0), -1, 26, 28)
            elif i == 3:
                output[:, :, :26, :28] += block(x[:, i, :]).view(x.size(0), -1, 26, 28)
            elif i == 4:
                output[:, :, 8:44, :26] += block(x[:, i, :]).view(x.size(0), -1, 36, 26)
            elif i == 5:
                output[:, :, 24:, :28] += block(x[:, i, :]).view(x.size(0), -1, 26, 28)
        return output

    def limitedFC3(self, x, fc, filter):
        output = torch.zeros((x.size(0) // 6, filter, 100, 100)).cuda()
        x = x.view(x.size(0) // 6, 6, -1)
        for i, block in enumerate(fc):
            if i == 0:
                output[:, :, :52, 44:] += block(x[:, i, :]).view(x.size(0), -1, 52, 56)
            elif i == 1:
                output[:, :, 16:88, 48:] += block(x[:, i, :]).view(x.size(0), -1, 72, 52)
            elif i == 2:
                output[:, :, 48:, 44:] += block(x[:, i, :]).view(x.size(0), -1, 52, 56)
            elif i == 3:
                output[:, :, :52, :56] += block(x[:, i, :]).view(x.size(0), -1, 52, 56)
            elif i == 4:
                output[:, :, 16:88, :52] += block(x[:, i, :]).view(x.size(0), -1, 72, 52)
            elif i == 5:
                output[:, :, 48:, :56] += block(x[:, i, :]).view(x.size(0), -1, 52, 56)
        return output

    def forward(self, x, detection_target):
        # (S,B,18,H,W)
        scene = x.size(0)
        step = x.size(1)
        x = x.view(-1, 3, 128, 160)
        output_list = self.efficientNet(x)
        x1 = output_list[3]
        x1 = x1.view(x1.size(0), 2, -1)
        mu = x1[:, 0, :]
        logvar = x1[:, 1, :]
        x1 = self.reparameterise(mu, logvar)

        x1 = self.fc1(x1)
        x1 = self.limitedFC1(x1, self.fc2, 32)
        x1 = self.conv0(x1)
        x1 = self.deconv0(x1)
        x1 = self.conv1(x1)
        x1 = self.deconv1(x1)
        x1 = self.conv2(x1)
        x1 = self.deconv2(x1)
        x1 = self.conv3(x1)
        x1 = self.deconv3(x1)
        x1 = self.convfinal(x1)

        # x2 = self.batch_lstm(x, scene, step, 2)
        feature0 = self.fc1_1(output_list[2].view(output_list[2].size(0), -1))
        feature1 = self.fc1_2(output_list[1].view(output_list[1].size(0), -1))
        x2 = torch.cat([feature0[:, :self.fc_num2], feature1[:, :self.fc_num2]], dim=1)
        x2 = self.limitedFC1(x2, self.fc2_1, 128)
        x2 = self.conv0_1(x2)
        detect_output0 = self.conv0_1_detect(x2)
        detect_output0 = self.convfinal_0(detect_output0)
        detect_output0, detect_loss0 = self.yolo0(detect_output0, detection_target, 800)
        x2 = self.deconv0_1(x2)  # detection

        x2_1 = torch.cat([feature0[:, self.fc_num2:self.fc_num2 * 2], feature1[:, self.fc_num2:self.fc_num2 * 2]],
                         dim=1)
        x2_1 = self.limitedFC2(x2_1, self.fc2_2, 32, )
        x2 = torch.cat([x2, x2_1], dim=1)
        x2 = self.conv1_1(x2)
        detect_output1 = self.conv1_1_detect(x2)
        detect_output1 = self.convfinal_1(detect_output1)
        detect_output1, detect_loss1 = self.yolo1(detect_output1, detection_target, 800)
        x2 = self.deconv1_1(x2)

        x2_2 = torch.cat(
            [feature0[:, self.fc_num2 * 2:self.fc_num2 * 3], feature1[:, self.fc_num2 * 2:self.fc_num2 * 3]], dim=1)
        x2_2 = self.limitedFC3(x2_2, self.fc2_3, 8)
        x2 = torch.cat([x2, x2_2], dim=1)
        detect_output2 = self.conv2_1_detect(x2)
        detect_output2 = self.convfinal_2(detect_output2)
        detect_output2, detect_loss2 = self.yolo2(detect_output2, detection_target, 800)
        total_loss = 0.4 * detect_loss0 + 0.3 * detect_loss1 + 0.3 * detect_loss2
        return nn.LogSoftmax(dim=1)(x1), detect_output0, detect_output1, detect_output2, total_loss


def trainModel(anchors, freeze=False, device=None):
    return AutoNet(anchors, freeze=freeze, device=device)
