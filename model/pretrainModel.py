import torch.nn as nn

from model.backboneModel import EfficientNet, BasicBlock
import pytorch_lightning as pl

class AutoPretrainNet(pl.LightningModule):
    def __init__(self, num_classes=3):
        self.latent = 1000
        self.num_classes = num_classes
        super(AutoPretrainNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b3')
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=2 * self.latent),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 4 * 5 * 256),
            nn.BatchNorm1d(4 * 5 * 256),
            nn.ReLU(inplace=True)
        )
        self.deconv0 = self._make_deconv_layer(256, 128)
        self.inplanes = 128
        self.conv0 = self._make_layer(BasicBlock, 128, 2)
        self.deconv1 = self._make_deconv_layer(128, 64)
        self.inplanes = 64
        self.conv1 = self._make_layer(BasicBlock, 64, 2)
        self.deconv2 = self._make_deconv_layer(64, 32)
        self.inplanes = 32
        self.conv2 = self._make_layer(BasicBlock, 32, 2)
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

    def _make_layer(self, block, planes, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, inplanes, outplanes, last=False):
        layers = []
        if last is not True:
            layers.append(
                nn.ConvTranspose2d(inplanes, outplanes, 3, stride=2,
                                   padding=1, output_padding=1, bias=False))
            layers.append(nn.BatchNorm2d(outplanes))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(
                nn.ConvTranspose2d(inplanes, outplanes, 3, stride=2, padding=1,
                                   output_padding=1))
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
        x = self.efficientNet(x, pretrain=True)
        x = x.view(x.size(0), 2, -1)
        mu = x[:, 0, :]
        logvar = x[:, 1, :]
        x = self.reparameterise(mu, logvar)
        x = self.fc2(x)
        x = x.reshape(x.size(0), -1, 4, 5)
        x = self.deconv0(x)
        x = self.conv0(x)
        x = self.deconv1(x)
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv2(x)
        x = self.deconv3(x)
        x = self.upSample(x)
        x = x.view(batch_size, -1, 128, 160)
        return nn.Sigmoid()(x), mu, logvar


def trainModel():
    return AutoPretrainNet()
