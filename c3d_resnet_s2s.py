# coding: utf-8
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model import Encoder, Decoder, Seq2Seq


def conv3x3(in_planes, out_planes, stride=1):
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


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class Lipreading(nn.Module):
    def __init__(self, mode, inputDim=512, hiddenDim=1024, embedSize=512, nClasses=30):
        super(Lipreading, self).__init__()
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.embedSize = embedSize
        self.nLayers = 3
        # frontend3D
        self.frontend3D = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                )
        # resnet
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        # backend_conv
        self.backend_conv1 = nn.Sequential(
                nn.Conv1d(self.inputDim, 2*self.inputDim, 5, 2, 0, bias=False),
                nn.BatchNorm1d(2*self.inputDim),
                nn.ReLU(True),
                nn.MaxPool1d(2, 2),
                nn.Conv1d(2*self.inputDim, 4*self.inputDim, 5, 2, 0, bias=False),
                nn.BatchNorm1d(4*self.inputDim),
                nn.ReLU(True),
                )
        self.backend_conv2 = nn.Sequential(
                nn.Linear(4*self.inputDim, self.inputDim),
                nn.BatchNorm1d(self.inputDim),
                nn.ReLU(True),
                nn.Linear(self.inputDim, self.nClasses)
                )
                
        # backend_gru        
        self.encoder = Encoder(self.inputDim, self.hiddenDim, self.nLayers)
        self.decoder = Decoder(self.embedSize, 
            self.hiddenDim, 
            self.nClasses,
            self.nLayers)
        self.seq2seq = Seq2Seq(self.encoder, self.decoder)
        # initialize        
        self._initialize_weights()
    
    
    def frontend_forward(self, x): 
        print('c3d in:',x.size())   
        x = self.frontend3D(x)
        print('c3d out:',x.size())
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x)
        return x
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        frameLen = x.size(2)
        if('finetune' in self.mode):
            with torch.no_grad():
                x = self.frontend_forward(x)
        else:
            x = self.frontend_forward(x)
        x = F.dropout(x, p=0.5)        
        if self.mode == 'temporalConv' or self.mode == 'fintuneTemporalConv':
            x = x.view(-1, frameLen, self.inputDim)
            x = x.transpose(1, 2)
            x = self.backend_conv1(x)
            x = torch.mean(x, 2)
            x = self.backend_conv2(x)
            return x
        elif self.mode == 'backendGRU' or self.mode == 'finetuneGRU':
            x = x.view(-1, frameLen, self.inputDim)
            outputs = self.seq2seq(x.transpose(0, 1).contiguous(), 
                target.transpose(0, 1).contiguous() if not target is None else None, 
                teacher_forcing_ratio=teacher_forcing_ratio)
            return outputs.transpose(0, 1).contiguous()
        else:
            raise Exception('No model is selected')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def lipreading(mode, inputDim=512, hiddenDim=1024, nClasses=30):
    model = Lipreading(mode, inputDim=inputDim, hiddenDim=hiddenDim, nClasses=nClasses)
    return model

import options as opt
import os
import cv2
from data import MyDataset
from cvtransforms import *
from torch.utils.data import Dataset, DataLoader

model = lipreading("backendGRU")
model = model.cuda()
if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

img_folder = '/home/yangshuang/luomingshuang/seq2seq_baidu/test'
img_files = os.listdir(img_folder)
img_files = sorted(img_files, key=lambda x:int(str(x.split('_')[0])))
images = [cv2.imread(os.path.join(img_folder, file)) for file in img_files]
images = [cv2.resize(img, (96, 96)) for img in images]
print(len(images))
for i in range(75-len(images)):
        images.append(np.zeros((96, 96, 3), dtype=np.uint8))
images = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                for img in images], axis=0) / 255.0

images = CenterCrop(images, (88, 88))
images = ColorNormalize(images)
input = torch.FloatTensor(images[np.newaxis,...])
txt = 'li shi ji lu'
label = MyDataset.text2tensor(txt, 200, SOS=True)
label = torch.LongTensor(label)
print(input.size())
print(label)
data = {'encoder_tensor':input, 'decoder_tensor':label}
# label = label.unsqueeze(0)
# input = torch.randn(1,75,88,88)
input = input.unsqueeze(0)
# for idx,batch in enumerate(loader):
#     outputs = model(batch['encoder_tensor'])
outputs = model(input.cuda())
predict_txt = MyDataset.tensor2text(outputs.argmax(-1))
print(predict_txt)