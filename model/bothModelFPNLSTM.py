import numpy as np
import torch
import torch.nn as nn

from model.EfficientNetBackbone import EfficientNet
from model.detectionModel import YOLOLayer

torch.cuda.set_device(0)
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
    def __init__(self, scene_batch_size, batch_size, step_size, anchors, detection_classes, num_classes=2,freeze=True,device=None):
        self.latent = 1000
        self.fc_num1 = 400
        self.fc_num2 = 200
        self.batch_size = batch_size
        self.step_size = step_size
        self.scene_batch_size = scene_batch_size
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors1 = np.reshape(anchors[0], [1, 2])
        self.anchors0 = anchors[1:5, :]
        self.detection_classes = detection_classes
        if device is not None:
            torch.cuda.set_device(device)
        super(AutoNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b3',freeze=freeze)
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=self.latent),
            nn.BatchNorm1d(self.latent),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.rnn1 = nn.LSTM(self.latent,self.fc_num1,2,dropout=0.3,batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_num1 * 6, 25 * 25 * 32, bias=False),
            nn.BatchNorm1d(25 * 25 * 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.rnn1_1 = nn.LSTM(384 * 4 * 5,self.fc_num2 * 2,2,dropout=0.3,batch_first=True)
        self.rnn1_2 = nn.LSTM(136 * 8 * 10,self.fc_num2 * 2,2,dropout=0.3,batch_first=True)
        self.fc2_1 = nn.Sequential(
            nn.Linear(self.fc_num2 * 6 * 2, 25 * 25 * 64, bias=False),
            nn.BatchNorm1d(25 * 25 * 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.fc2_2 = nn.Sequential(
            nn.Linear(self.fc_num2 * 6 * 2, 50 * 50 * 8, bias=False),
            nn.BatchNorm1d(50 * 50 * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.inplanes =32
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
        self.deconv0_1 = self._make_deconv_layer(64, 8)

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
            elif isinstance(m, nn.LSTM):
                nn.init.xavier_normal_(m.all_weights[0][0])
                nn.init.xavier_normal_(m.all_weights[0][1])

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


    def batch_lstm(self, x, scene, step,output, rnn):
        x=x.view(scene,step,6,-1)
        x=x.transpose(1,2).contiguous()
        x=x.view(-1,step,x.size(3))
        x_lstm = []
        h0 = torch.zeros((2, scene*step*6, output)).cuda()
        c0 = torch.zeros((2, scene*step*6, output)).cuda()
        for k in range(step):
            if k < self.step_size:
                x_pad = torch.zeros((6 * scene, self.step_size - k - 1,x.size(2))).cuda()
                x_lstm_unit = torch.cat([x_pad, x[:, :k + 1, :]], dim=1)
            else:
                x_lstm_unit = x[:, k - self.step_size + 1:k + 1, :]
            x_lstm.append(x_lstm_unit)
        x_lstm = torch.cat(x_lstm, dim=0)
        x_lstm_out, (ht, ct) = rnn(x_lstm, (h0, c0))
        x_lstm_final = []
        for k in range(step):
            x_lstm_unit = x_lstm_out[k * scene * 6:(k + 1) * scene * 6, self.step_size - 1, :]
            x_lstm_final.append(x_lstm_unit)
        x = torch.cat(x_lstm_final, dim=0)
        x = x.view(scene, 6, step, output)
        x = x.transpose(1, 2).contiguous()
        x = x.view(scene * step*6, output )
        return x

    def forward(self, x, detection_target):
        # (S,B,18,H,W)
        scene = x.size(0)
        step = x.size(1)
        x = x.view(-1, 3, 128, 160)
        output_list = self.efficientNet(x)
        x1 = output_list[3]

        x1 = self.batch_lstm(x1,scene,step,self.fc_num1,self.rnn1)
        x1 = x1.view(-1, self.fc_num1 * 6)
        x1 = self.fc2(x1)
        x1 = x1.view(x1.size(0), -1, 25, 25)  # x = x.view(x.size(0)*6,-1,128,160)
        x1 = self.conv0(x1)
        x1 = self.deconv0(x1)
        x1 = self.conv1(x1)
        x1 = self.deconv1(x1)
        x1 = self.conv2(x1)
        x1 = self.deconv2(x1)
        x1 = self.conv3(x1)
        x1 = self.deconv3(x1)
        x1 = self.convfinal(x1)

        feature0 = self.batch_lstm(output_list[2],scene,step,self.fc_num2*2,self.rnn1_1)
        feature1 = self.batch_lstm(output_list[1],scene,step,self.fc_num2*2,self.rnn1_2)
        x2 = torch.cat([feature0[:, :self.fc_num2], feature1[:, :self.fc_num2]], dim=1)
        x2 = x2.view(-1, self.fc_num2 * 6 * 2)
        x2 = self.fc2_1(x2)
        x2 = x2.view(x2.size(0), -1, 25, 25)  # x = x.view(x.size(0)*6,-1,128,160)
        x2 = self.conv0_1(x2)
        detect_output0 = self.conv0_1_detect(x2)
        detect_output0 = self.convfinal_0(detect_output0)
        detect_output0, detect_loss0 = self.yolo0(detect_output0, detection_target, 800)
        x2 = self.deconv0_1(x2)  # detection

        x2_1 = torch.cat([feature0[:, self.fc_num2:self.fc_num2 * 2], feature1[:, self.fc_num2:self.fc_num2 * 2]],
                         dim=1)
        x2_1 = x2_1.view(-1, self.fc_num2 * 6 * 2)
        x2_1 = self.fc2_2(x2_1)
        x2_1 = x2_1.view(x2_1.size(0), -1, 50, 50)  #
        x2 = torch.cat([x2, x2_1], dim=1)
        x2 = self.conv1_1(x2)
        detect_output1 = self.conv1_1_detect(x2)
        detect_output1 = self.convfinal_1(detect_output1)
        detect_output1, detect_loss1 = self.yolo1(detect_output1, detection_target, 800)

        total_loss = 0.8 * detect_loss0 + 0.2 * detect_loss1

        return nn.LogSoftmax(dim=1)(x1), detect_output0, detect_output1, total_loss


def trainModel(anchors, detection_classes=9, scene_batch_size=4, batch_size=2, step_size=2,device=None,freeze=False):
    return AutoNet(scene_batch_size, batch_size, step_size, anchors, detection_classes,device=device,freeze=freeze)
