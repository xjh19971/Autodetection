import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.MobileNet import MobileNetV3
import numpy as np
from model.EfficientNetBackbone import EfficientNet


class AutoNet(nn.Module):
    def __init__(self, num_classes=2):
        self.latent = 1000
        self.fc_num = 300
        self.num_classes = num_classes
        super(AutoNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b4')
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=2 * self.latent),
            # nn.Dropout(p=0.4)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent, self.fc_num, bias=False),
            nn.BatchNorm1d(self.fc_num),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_num * 6, 25 * 25 * 16, bias=False),
            nn.BatchNorm1d(25 * 25 * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.deconv0 = self._make_deconv_layer(16, 8)
        self.deconv1 = self._make_deconv_layer(8, 4)
        self.deconv2 = self._make_deconv_layer(4, 2, last=True)

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
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

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

    def reparameterise(self, mu, logvar):
        return mu

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(x.size(0) * 6, -1, 128, 160)
        x = self.efficientNet(x)
        x = x.view(x.size(0), 2, -1)
        mu = x[:, 0, :]
        logvar = x[:, 1, :]
        x = self.reparameterise(mu, logvar)
        x = self.fc1(x)
        x = x.view(batch_size, -1)
        x = self.fc2(x)
        x = x.view(x.size(0), -1, 25, 25)  # x = x.view(x.size(0)*6,-1,128,160)
        x = self.deconv0(x)  # detection
        x = self.deconv1(x)
        x = self.deconv2(x)  # resize conv conv resize conv conv
        return nn.LogSoftmax(dim=1)(x)


def trainModel():
    return AutoNet()
