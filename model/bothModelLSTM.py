import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.MobileNet import MobileNetV3
import torch
import numpy as np
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
    def __init__(self, scene_batch_size, batch_size, step_size, device,anchors,detection_classes, num_classes=2):
        self.latent = 1000
        self.fc_num = 500
        self.batch_size = batch_size
        self.step_size = step_size
        self.scene_batch_size = scene_batch_size
        self.num_classes = num_classes
        self.device = device
        self.anchors=anchors
        self.detection_classes=detection_classes
        super(AutoNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b4')
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=2 * self.latent),
            # nn.Dropout(p=0.4)
        )
        self.rnn1 = nn.LSTM(self.latent, self.fc_num, 2, batch_first=True, dropout=0.2)
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_num*6, 25 * 25 * 16, bias=False),
            nn.BatchNorm1d(25 * 25 * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.rnn1_1 = nn.LSTM(self.latent, self.fc_num, 2, batch_first=True, dropout=0.2)
        self.fc2_1 = nn.Sequential(
            nn.Linear(self.fc_num * 6, 25 * 25 * 64, bias=False),
            nn.BatchNorm1d(25 * 25 * 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.inplanes = 8
        self.conv0 = self._make_layer(BasicBlock, 8, 1)
        self.deconv0 = self._make_deconv_layer(16, 8)
        self.inplanes = 4
        self.conv1 = self._make_layer(BasicBlock, 4, 1)
        self.deconv1 = self._make_deconv_layer(8, 4)
        self.inplanes = 2
        self.conv2 = self._make_layer(BasicBlock, 2, 1)
        self.deconv2 = self._make_deconv_layer(4, 2)
        self.convfinal = nn.Conv2d(2, 2, 1)

        self.inplanes = 64
        self.conv0_1 = self._make_layer(BasicBlock, 64, 1)
        self.deconv0_1 = self._make_deconv_layer(64, 64)
        self.inplanes = 64
        self.conv1_1 = self._make_layer(BasicBlock, 64, 1)
        self.deconv1_1 = self._make_deconv_layer(64, 64)
        self.inplanes = 64
        self.conv2_1 = self._make_layer(BasicBlock, 64, 1)
        self.deconv2_1 = self._make_deconv_layer(64, 64)
        self.convfinal_1 = nn.Conv2d(64, len(anchors)*(self.detection_classes+5), 1)
        self.yolo1 = YOLOLayer(anchors, self.detection_classes,self.device, 800)

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

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        for i in range(blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, inplanes, outplanes, last=False, num_anchors=None):
        layers = []
        layers.append(
                nn.ConvTranspose2d(inplanes, outplanes, 3, stride=2,
                                   padding=1, output_padding=1, bias=False))
        layers.append(nn.BatchNorm2d(outplanes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def reparameterise(self, mu, logvar):
        return mu

    def batch_lstm(self,x,scene,step,branch):
        x_lstm = []
        h0 = torch.zeros((2, 6 * scene * step, self.fc_num)).to(self.device)
        c0 = torch.zeros((2, 6 * scene * step, self.fc_num)).to(self.device)
        for k in range(step):
            if k<self.step_size:
                x_pad = torch.zeros((6 * scene, self.step_size - k - 1, self.latent)).to(self.device)
                x_lstm_unit = torch.cat([x_pad, x[:, :k + 1, :]], dim=1)
            else:
                x_lstm_unit =x[:, k-self.step_size+1:k + 1, :]
            x_lstm.append(x_lstm_unit)
        x_lstm = torch.cat(x_lstm, dim=0)
        if branch==1:
            x_lstm_out, (ht, ct) = self.rnn1(x_lstm, (h0, c0))
        else:
            x_lstm_out, (ht, ct) = self.rnn1_1(x_lstm, (h0, c0))
        x_lstm_final = []
        for k in range(step):
            x_lstm_unit = x_lstm_out[k * scene * 6:(k + 1) * scene * 6, self.step_size - 1, :]
            x_lstm_final.append(x_lstm_unit)
        x = torch.cat(x_lstm_final, dim=0)
        x = x.view(scene, 6, step, self.fc_num)
        x = x.transpose(1, 2).contiguous()
        x = x.view(scene * step, self.fc_num * 6)
        return x

    def forward(self, x, detection_target):
        # (S,B,18,H,W)
        scene = x.size(0)
        step = x.size(1)
        x = x.view(-1, 3, 128, 160)
        x = self.efficientNet(x)
        x = x.view(x.size(0), 2, -1)
        mu = x[:, 0, :]
        logvar = x[:, 1, :]
        x = self.reparameterise(mu, logvar)
        x = x.view(scene, step, 6, self.latent)
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, step, self.latent)
        x1 = self.batch_lstm(x,scene,step,1)
        x1 = self.fc2(x1)
        x1 = x1.view(x1.size(0), -1, 25, 25)  # x = x.view(x.size(0)*6,-1,128,160)
        x1 = self.deconv0(x1)  # detection
        x1 = self.conv0(x1)
        x1 = self.deconv1(x1)
        x1 = self.conv1(x1)
        x1 = self.deconv2(x1)  # resize conv conv resize conv conv
        x1 = self.conv2(x1)
        x1 = self.convfinal(x1)

        x2 = self.batch_lstm(x,scene,step,1)
        x2 = self.fc2_1(x2)
        x2 = x2.view(x2.size(0), -1, 25, 25)  # x = x.view(x.size(0)*6,-1,128,160)
        x2 = self.conv0_1(x2)
        x2 = self.deconv0_1(x2)  # detection
        x2 = self.conv1_1(x2)
        x2 = self.deconv1_1(x2)
        x2 = self.conv2_1(x2)
        x2 = self.deconv2_1(x2)
        x2 = self.convfinal_1(x2)

        output2, total_loss = self.yolo1(x2, detection_target, 800)
        output1 = x1.view(-1, 2, 200, 200)
        output2 = x2.view(-1, len(self.anchors)*(self.detection_classes+5), 200, 200)
        return nn.LogSoftmax(dim=1)(output1),output2,total_loss


def trainModel(device,anchors,detection_classes=9, scene_batch_size=4, batch_size=8, step_size=4):
    return AutoNet(scene_batch_size, batch_size, step_size, device,anchors,detection_classes)
