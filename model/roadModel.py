import torch.nn as nn

from model.backboneModel import EfficientNet,BasicBlock

class AutoNet(nn.Module):
    def __init__(self, freeze=False,device=None):
        self.latent = 1000
        self.fc_num = 400
        self.device=device
        super(AutoNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b3',freeze=freeze)
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=self.latent * 2),
            nn.BatchNorm1d(self.latent * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent, self.fc_num, bias=False),
            nn.BatchNorm1d(self.fc_num),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_num * 6, 25 * 25 * 32, bias=False),
            nn.BatchNorm1d(25 * 25 * 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        self.inplanes = 32
        self.conv0 = self._make_layer(BasicBlock, 32, 2)
        self.deconv0 = self._make_deconv_layer(32, 16)
        self.inplanes = 16
        self.conv1 = self._make_layer(BasicBlock, 16, 2)
        self.deconv1 = self._make_deconv_layer(16, 8)
        self.inplanes = 8
        self.conv2 = self._make_layer(BasicBlock, 8, 2)
        self.deconv2 = self._make_deconv_layer(8, 4, last=True)
        self.inplanes = 4
        self.conv3 = self._make_layer(BasicBlock, 4, 2)
        self.deconv3 = self._make_deconv_layer(4, 2, last=True)
        self.convfinal = nn.Conv2d(2, 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        for i in range(blocks):
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
        x = x.view(-1, 3, 128, 160)
        x = self.efficientNet(x)
        x = x[3].view(x[3].size(0), 2, -1)
        mu = x[:, 0, :]
        logvar = x[:, 1, :]
        x = self.reparameterise(mu, logvar)
        x = self.fc1(x)
        x = x.view(-1, 6 * self.fc_num)
        x = self.fc2(x)
        x = x.view(x.size(0), -1, 25, 25)  # x = x.view(x.size(0)*6,-1,128,160)
        x = self.conv0(x)
        x = self.deconv0(x)  # detection
        x = self.conv1(x)
        x = self.deconv1(x)
        x = self.conv2(x)
        x = self.deconv2(x)  # resize conv conv resize conv conv
        x = self.conv3(x)
        x = self.deconv3(x)
        x = self.convfinal(x)
        return nn.LogSoftmax(dim=1)(x)

def trainModel(freeze=False,device=None):
    return AutoNet(freeze=freeze,device=device)
