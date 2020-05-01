import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.MobileNet import MobileNetV3

from model.EfficientNetBackbone import EfficientNet


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class AutoPretrainNet(nn.Module):
    def __init__(self, num_classes=3):
        self.latent = 1000
        self.num_classes = num_classes
        super(AutoPretrainNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b4')
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=2*self.latent),
            # nn.Dropout(p=0.4)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 4 * 5 * 256),
            #nn.BatchNorm1d(4 * 5 * 128),
            #nn.ReLU(inplace=True),
            # nn.Dropout(p=0.4)
        )
        self.deconv0 = self._make_deconv_layer(256, 128)
        self.inplanes=128
        self.conv0=self._make_layer(Bottleneck,128,2)
        self.deconv1 = self._make_deconv_layer(128, 64)
        self.inplanes=64
        self.conv1 = self._make_layer(Bottleneck, 64, 2)
        self.deconv2 = self._make_deconv_layer(64, 32)
        self.inplanes=32
        self.conv2 = self._make_layer(Bottleneck, 32, 2)
        self.deconv3 = self._make_deconv_layer(32, 3, last=True)
        self.upSample = nn.Upsample(scale_factor=2)
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
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(x.size(0) * 6, -1, 128, 160)
        x = self.efficientNet(x)
        x = x.view(x.size(0), 2, -1)
        mu = x[:, 0, :]
        logvar = x[:, 1, :]
        x = self.reparameterise(mu, logvar)
        #x = x.view(batch_size, -1)
        x = self.fc2(x)
        x = x.reshape(x.size(0), -1, 4, 5)  # x = x.view(x.size(0)*6,-1,128,160)
        x = self.deconv0(x)  # detection
        x = self.conv0(x)
        x = self.deconv1(x)
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv2(x)
        x = self.deconv3(x)
        x = self.upSample(x)
        x = x.view(batch_size, -1, 128, 160)
        return x, mu, logvar


def trainModel():
    return AutoPretrainNet()
