import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.MobileNet import MobileNetV3
import torch

from model.EfficientNetBackbone import EfficientNet


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class AutoNet(nn.Module):
    def __init__(self, scene_batch_size, batch_size, step_size, num_classes=2):
        self.latent = 1000
        self.batch_size = batch_size
        self.step_size = step_size
        self.scene_batch_size = scene_batch_size
        self.num_classes = num_classes
        super(AutoNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b4')
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=2 * self.latent),
            # nn.Dropout(p=0.4)
        )
        self.rnn1 = nn.LSTM(1000, 300, 2, batch_first=True,dropout=0.3)
        self.fc2 = nn.Sequential(
            nn.Linear(1800, 25 * 25 * 16, bias=False),
            nn.BatchNorm1d(25 * 25 * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.deconv0 = self._make_deconv_layer(16, 8)
        self.deconv1 = self._make_deconv_layer(8, 4)
        self.deconv2 = self._make_deconv_layer(4, 1, last=True)

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
                layers.append(nn.Sigmoid())
            else:
                layers.append(
                    nn.ConvTranspose2d(inplanes, outplanes * num_anchors, 3, stride=2, padding=1,
                                       output_padding=1))
        return nn.Sequential(*layers)

    def reparameterise(self, mu, logvar):
        return mu

    def forward(self, x):
        # (S,B,18,H,W)
        output = []
        scene = x.size(0)
        for i in range(scene):
            output_scene = []
            x_scene = x[i, :, :, :, :]
            h0 = torch.zeros((2, 6, 300)).cuda()
            c0 = torch.zeros((2, 6, 300)).cuda()
            x_lstm = torch.zeros((6, self.step_size, 1000)).cuda()
            for j in range(0, self.scene_batch_size, self.batch_size):
                batch_x = x_scene[j:j + self.batch_size, :, :, :]
                batch_x = batch_x.view([batch_x.size(0) * 6, -1, 128, 160])
                batch_x = self.efficientNet(batch_x)
                batch_x = batch_x.view(batch_x.size(0), 2, -1)
                mu = batch_x[:, 0, :]
                logvar = batch_x[:, 1, :]
                batch_x = self.reparameterise(mu, logvar)
                batch_x = batch_x.view([self.batch_size, -1, 1000])
                batch_x = batch_x.transpose(0, 1)
                x_lstm_out_list = []
                for k in range(batch_x.size(1)):
                    x_lstm = x_lstm[:, 1:, :]
                    x_lstm = torch.cat([x_lstm, batch_x[:, k, :].unsqueeze(1)], dim=1)
                    x_lstm_out, (ht, ct) = self.rnn1(x_lstm, (h0, c0))
                    x_lstm_out_list.append(x_lstm_out[:, self.step_size - 1, :])
                ho, c0 = ht, ct
                batch_x = torch.stack(x_lstm_out_list, dim=1).transpose(0, 1)
                batch_x = batch_x.reshape(self.batch_size, -1)
                batch_x = self.fc2(batch_x)
                batch_x = batch_x.view(batch_x.size(0), -1, 25, 25)  # x = x.view(x.size(0)*6,-1,128,160)
                batch_x = self.deconv0(batch_x)  # detection
                batch_x = self.deconv1(batch_x)
                batch_x = self.deconv2(batch_x)  # resize conv conv resize conv conv
                output_scene.append(batch_x)
            output_scene = torch.cat(output_scene)
            output.append(output_scene)
        output = torch.stack(output)
        return output


def trainModel(scene_batch_size=4, batch_size=4, step_size=4):
    return AutoNet(scene_batch_size, batch_size, step_size)
