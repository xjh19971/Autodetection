import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        self.num_classes=num_classes
        super(ResNet, self).__init__()
        ### encoder
        self.conv1 = nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion *2 *4, 25*25*num_classes*8)
        self.fc = nn.Linear(512 * block.expansion*2 *4,32*40*num_classes*4)
        #self.upsample = nn.Upsample(scale_factor=2)
        self.deconv0 = self._make_deconv_layer(8,12)
        self.deconv1 = self._make_deconv_layer(12,16)
        self.deconv2 = self._make_deconv_layer(16,18,last=True)
        ### decoder
     #   self.defc = self._make_de_fc(200*200*num_classes,64*80*num_classes*6)
      #  self.deconv3 = self._make_deconv_layer(12,16)
       # self.deconv4 = self._make_deconv_layer(16,18,last=True)

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

    def _make_de_fc(self,a,b):
        layers=[]
        layers.append( nn.Linear(a,b))
        layers.append( nn.BatchNorm2d(b))
        layers.append( nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    def _make_deconv_layer(self, in_chan,out_chan, last=False):
        layers=[]
        if last is not True:
            layers.append( nn.ConvTranspose2d(in_chan,out_chan, 3, stride=2, padding=1, output_padding=1,bias=False))
            layers.append( nn.BatchNorm2d(out_chan))
            layers.append( nn.ReLU(inplace=True))
        else:
            layers.append(
                nn.ConvTranspose2d(in_chan,out_chan, 3, stride=2, padding=1,
                                   output_padding=1))
        return nn.Sequential(*layers)
    def forward(self, x):
    #    print("input",x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
 #       print("after resnet encoder",x.shape)
        x = x.view(x.size(0), -1)
  #      print("change shape",x.shape)
        x = self.fc(x)
   #     print("after fc",x.shape)
        x = x.view(x.size(0),-1,32,40)
    #    print("after change shape",x.shape)
  #      x = self.upsample(x)
     #   print("after upsampling",x.shape)
        x = self.deconv0(x)
       # print("after first deconv",x.shape)
        x = self.deconv1(x)
      #  print("after second deconv",x.shape)
        x = self.deconv2(x)
       # print("after third decov",x.shape)
        #x = x.view(x.size(0), -1)
        #x = defc(x)
        #x = x.view(x.size(0),-1,,80)
       # x = self.deconv3(x)
      #  x = self.deconv4(x)

        return nn.Sigmoid()(x)

def trainModel():
    return ResNet(Bottleneck,[3,4,23,3]);
