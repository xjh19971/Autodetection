import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.MobileNet import MobileNetV3
from model.EfficientNetBackbone import EfficientNet
from model.detectionModel import YOLOLayer

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class AutoNet(nn.Module):
    def __init__(self, anchors,detection_classes,num_classes=2):
        self.inplanes = 64
        self.num_classes = num_classes
        self.detection_classes=detection_classes
        super(AutoNet, self).__init__()
        self.efficientNet=EfficientNet.from_name('efficientnet-b3')
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1800, 25 * 25 * 16, bias=False),
            nn.BatchNorm1d(25 * 25 * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )
        self.fc2_1 = nn.Sequential(
            nn.Linear(1800, 25 * 25 * 48, bias=False),
            nn.BatchNorm1d(25 * 25 * 48),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )
        self.deconv0 = self._make_deconv_layer(16,8)
        self.deconv1 = self._make_deconv_layer(8,4)
        self.deconv2 = self._make_deconv_layer(4,2, last=True)
        self.conv0_0 = nn.Sequential(
            nn.Conv2d(48, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.inplanes=256
        self.conv0_1 = self._make_layer(Bottleneck,64,2)
        self.deconv0_1 = self._make_deconv_layer(256,128)
        self.inplanes=128
        self.conv1_1 = self._make_layer(Bottleneck,32,2)
        self.deconv1_1 = self._make_deconv_layer(128,self.detection_classes+5, last=True,num_anchors=len(anchors))
        self.yolo1=YOLOLayer(anchors, self.detection_classes, 800)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, inplanes, outplanes, last=False, num_anchors=None):
        layers = []
        if last is not True:
            layers.append(
                nn.ConvTranspose2d(inplanes, outplanes, 3, stride=2,
                                   padding=1, output_padding=1, bias=False))
            layers.append(nn.BatchNorm2d(outplanes))
            layers.append(nn.ReLU(inplace=True))
        else:
            if num_anchors is None:
                layers.append(
                    nn.ConvTranspose2d(inplanes, outplanes, 3, stride=2, padding=1,
                                       output_padding=1))
            else:
                layers.append(
                    nn.ConvTranspose2d(inplanes, outplanes * num_anchors, 3, stride=2, padding=1,
                                       output_padding=1))
        return nn.Sequential(*layers)

    def forward(self, x, detection_target):
        batch_size=x.size(0)
        x = x.view(x.size(0)*6,-1,128,160)
        x = self.efficientNet(x)
        x = x.view(batch_size, -1)
        branch = x
        x = self.fc2(x)
        x = x.view(x.size(0),-1,25,25) #x = x.view(x.size(0)*6,-1,128,160)
        x = self.deconv0(x)#detection
        x = self.deconv1(x)
        x = self.deconv2(x)#resize conv conv resize conv conv
        x1 = self.fc2_1(branch)
        x1 = x1.view(x1.size(0),-1,25,25)
        x1 = self.conv0_0(x1)
        x1 = self.conv0_1(x1)
        x1 = self.deconv0_1(x1)#detection
        x1 = self.conv1_1(x1)
        x1 = self.deconv1_1(x1)
        output, total_loss = self.yolo1(x1, detection_target,800)
        return [nn.LogSoftmax(dim=1)(x),output,total_loss]


def trainModel(anchors,detection_classes=9):
    return AutoNet(anchors,detection_classes)
