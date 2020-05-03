import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.MobileNet import MobileNetV3
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
    def __init__(self, anchors,detection_classes,num_classes=2):
        self.latent = 1000
        self.fc_num = 400
        self.num_classes = num_classes
        self.detection_classes=detection_classes
        super(AutoNet, self).__init__()
        self.efficientNet=EfficientNet.from_name('efficientnet-b3')
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=400, bias=False),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )
        self.fc2_1 = nn.Sequential(
            nn.Linear(self.fc_num * 6, 25 * 25 * 16, bias=False),
            nn.BatchNorm1d(25 * 25 * 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.fc2_1 = nn.Sequential(
            nn.Linear(self.fc_num * 6, 25 * 25 * 64, bias=False),
            nn.BatchNorm1d(25 * 25 * 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
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
