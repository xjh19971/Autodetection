import numpy as np
import torch
import torch.nn as nn

from model.backboneModel import EfficientNet,YOLOLayer,BasicBlock

class AutoNet(nn.Module):
    def __init__(self, batch_size, step_size, anchors, detection_classes,freeze=False, device=None):
        self.latent = 1000
        self.fc_num = 400
        self.batch_size = batch_size
        self.step_size = step_size
        self.device = device
        self.anchors = anchors
        self.anchors1 = np.reshape(anchors[0], [1, 2])
        self.anchors2 = anchors[1:]
        self.detection_classes = detection_classes
        super(AutoNet, self).__init__()
        self.efficientNet = EfficientNet.from_name('efficientnet-b3', freeze=freeze)
        feature = self.efficientNet._fc.in_features
        self.efficientNet._fc = nn.Sequential(
            nn.Linear(in_features=feature, out_features=2 * self.latent),
        )
        self.rnn1 = nn.LSTM(self.latent, self.fc_num, 2, batch_first=True, dropout=0.25)
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_num * 6, 25 * 25 * 32, bias=False),
            nn.BatchNorm1d(25 * 25 * 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.rnn1_1 = nn.LSTM(self.latent, self.fc_num, 2, batch_first=True, dropout=0.25)
        self.fc2_1 = nn.Sequential(
            nn.Linear(self.fc_num * 6, 25 * 25 * 64, bias=False),
            nn.BatchNorm1d(25 * 25 * 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.inplanes = 32
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
        self.conv0_1 = self._make_layer(BasicBlock, 64, 2)
        self.deconv0_1 = self._make_deconv_layer(64, 16)
        self.conv0_1_detect = self._make_layer(BasicBlock, 64, 2)
        self.convfinal_0 = nn.Conv2d(64, len(self.anchors2) * (self.detection_classes + 5), 1)
        self.yolo0 = YOLOLayer(self.anchors2, self.detection_classes, 800,device=device)
        self.inplanes = 16
        self.conv1_1_detect = self._make_layer(BasicBlock, 16, 2)
        self.convfinal_1 = nn.Conv2d(16, len(self.anchors1) * (self.detection_classes + 5), 1)
        self.yolo1 = YOLOLayer(self.anchors1, self.detection_classes, 800,device=device)
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
                nn.init.xavier_normal_(m.all_weights[1][0])
                nn.init.xavier_normal_(m.all_weights[1][1])

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

    def reparameterise(self, mu, logvar):
        return mu

    def batch_lstm(self, x, scene, step, branch):
        #x_lstm = []
 #       print("-------- in batch_lstm -------",flush=True)
        h0 = torch.zeros((2, 6 * scene, self.fc_num)).to(self.device)
        c0 = torch.zeros((2, 6 * scene, self.fc_num)).to(self.device)
#        print("hidden layer shape",h0.size(),flush=True)
        #for k in range(step):
         #   if k < self.step_size-1:
          #      x_pad = torch.zeros((6 * scene, self.step_size - k - 1, self.latent)).to(self.device)
        #        print("padding size",x_pad.size(),flush=True)
           #     x_lstm_unit = torch.cat([x_pad, x[:, :k + 1, :]], dim=1)
         #       print("lstm unit after padding",x_lstm_unit.size(),flush=True)
            #else:
             #   x_lstm_unit = x[:, k - self.step_size + 1:k + 1, :]
          #      print("lstm unit no need pad",x_lstm_unit.size(),flush=True)
           # x_lstm.append(x_lstm_unit)
        #x_lstm = torch.cat(x_lstm, dim=0)
        #print("x_lstm",x.size(),flush=True)
        if branch == 1:
            x_lstm_out, (ht, ct) = self.rnn1(x, (h0, c0))
    #        x_lstm_out, (ht, ct) = self.rnn1(x_lstm, (h0, c0))
        else:
            x_lstm_out, (ht, ct) = self.rnn1_1(x, (h0, c0))
     #       x_lstm_out, (ht, ct) = self.rnn1_1(x_lstm, (h0, c0))
        #x_lstm_final = []
        #print("x_lstm output of rnn",x_lstm_out.size(),flush=True)
        #for k in range(step):
         #   x_lstm_unit = x_lstm_out[k * scene * 6:(k + 1) * scene * 6, self.step_size - 1, :]
         #   print("x_lstm output unit",x_lstm_unit.size(),flush=True)
          #  x_lstm_final.append(x_lstm_unit)
        #x = torch.cat(x_lstm_final, dim=0)
        x = x_lstm_out
        #print("x_lstm output",x.size(),flush=True)
        #x = x.view(-1,self.fc_num)
        #print("x_lstm reshaped output",x.size(),flush=True)
        x = x.view(scene, 6, step, self.fc_num)
  #      print("reshape1 lstm output",x.size(),flush=True)
        x = x.transpose(1, 2).contiguous()
   #     print("reshape2 lstm output",x.size(),flush=True)
        x = x.view(scene * step, self.fc_num * 6)
    #    print("reshape3 lstm output",x.size(),flush=True)
     #   print("--------- batch lstm  --------",flush=True)
        return x

    def forward(self, x, detection_target=None):
        # (S,B,18,H,W)
        scene = x.size(0)
        step = x.size(1)
        x = x.view(-1, 3, 128, 160)
      #  print("scence",scene,"step",step,"x",x.size(),flush=True)
        output_list = self.efficientNet(x)
     #   print("output of efficient net",np.array(output_list).shape,flush=True)
        x = output_list[3].view(output_list[3].size(0), 2, -1)
        x = x.view(x.size(0), 2, -1)
      #  print("x shape before vae encoder",x.size(),flush=True)
        mu = x[:, 0, :]
        logvar = x[:, 1, :]
        x = self.reparameterise(mu, logvar)
        x = x.view(scene, step, 6, self.latent)
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, step, self.latent)
       # print("roadmap: x shape before lstm",x.size(),flush=True)
        x1 = self.batch_lstm(x, scene, step, 1)
        #print("roadmap: lstm output x1",x1.size(),flush=True)
        x1 = self.fc2(x1)
        #print("x1 shape after fc",x1.size(),flush=True)
        x1 = x1.view(x1.size(0), -1, 25, 25)
        #print("x1 shape roadmap input shape",x1.size(),flush=True)
        x1 = self.conv0(x1)
        x1 = self.deconv0(x1)
        x1 = self.conv1(x1)
        x1 = self.deconv1(x1)
        x1 = self.conv2(x1)
        x1 = self.deconv2(x1)
        x1 = self.conv3(x1)
        x1 = self.deconv3(x1)
        x1 = self.convfinal(x1)
        #print("roadmap output x1 shape", x1.size(),flush=True)
        #print("yolo: x shape before lstm",x.size(),flush=True)
        x2 = self.batch_lstm(x, scene, step, 2)
        #print("yolo: lstm output x2",x2.size(),flush=True)
        x2 = self.fc2_1(x2)
        x2 = x2.view(x2.size(0), -1, 25, 25)
        #print("x2 shape yolo input shape",x2.size(),flush=True)
        x2 = self.conv0_1(x2)
        detect_output0 = self.conv0_1_detect(x2)
        detect_output0 = self.convfinal_0(detect_output0)
        detect_output0, detect_loss0 = self.yolo0(detect_output0, detection_target, 800)
        #print("yolo0 output shape",detect_output0.size(),flush=True)
        x2 = self.deconv0_1(x2)
        x2 = self.conv1_1(x2)
        detect_output1 = self.conv1_1_detect(x2)
        detect_output1 = self.convfinal_1(detect_output1)
        detect_output1, detect_loss1 = self.yolo1(detect_output1, detection_target, 800)
        #print("yolo1 output shape",detect_output1.size(),flush=True)

        total_loss = 0.6 * detect_loss0 + 0.4 * detect_loss1
        return nn.LogSoftmax(dim=1)(x1), detect_output0, detect_output1, total_loss


def trainModel(device, anchors, detection_classes=9, batch_size=2, step_size=2, freeze=False):
    return AutoNet(batch_size, step_size, anchors, detection_classes, freeze=freeze, device=device)
