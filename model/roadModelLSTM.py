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
    def __init__(self, scene_batch_size, batch_size, step_size, device, num_classes=2):
        self.latent = 1000
        self.fc_num=300
        self.batch_size = batch_size
        self.step_size = step_size
        self.scene_batch_size = scene_batch_size
        self.num_classes = num_classes
        self.device = device
        super(AutoNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b4')
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=2 * self.latent),
            # nn.Dropout(p=0.4)
        )
        self.rnn1 = nn.LSTM(1000, 300, 1, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(1800, 25 * 25 * 16, bias=False),
            nn.BatchNorm1d(25 * 25 * 16),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
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
        x_lstm = []
        h0 = torch.zeros((2, 6 * scene * step, self.fc_num)).to(self.device)
        c0 = torch.zeros((2, 6 * scene * step, self.fc_num)).to(self.device)
        for k in range(step):
            x_pad = torch.zeros((6*scene, x.size(1) - k - 1, self.latent)).to(self.device)
            x_lstm_unit = torch.cat([x_pad, x[:, :k + 1, :]], dim=1)
            x_lstm.append(x_lstm_unit)
        x_lstm = torch.cat(x_lstm, dim=0)
        x_lstm_out, (ht, ct) = self.rnn1(x_lstm, (h0, c0))
        x_lstm_final=[]
        for k in range(step):
            x_lstm_unit=x_lstm_out[k*scene*6:(k+1)*scene*6,step-1,:]
            x_lstm_final.append(x_lstm_unit)
        x=torch.cat(x_lstm_final,dim=0)
        x = x.view(scene,6,step,self.fc_num)
        x = x.transpose(1,2).contiguous()
        x = x.view(scene*step,self.fc_num*6)
        x = self.fc2(x)
        x = x.view(x.size(0), -1, 25, 25)  # x = x.view(x.size(0)*6,-1,128,160)
        x = self.deconv0(x)  # detection
        x = self.deconv1(x)
        x = self.deconv2(x)  # resize conv conv resize conv conv
        output = x.view(scene,step,1,200,200)
        return output

def trainModel(device,scene_batch_size=4, batch_size=4, step_size=4):
    return AutoNet(scene_batch_size, batch_size, step_size,device)
