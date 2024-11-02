import torch  
import torch.nn as nn  
import numpy as np
import torchvision.models as models  
import torch.nn.functional as F 

class Block(nn.Module):
    def __init__(self, in_planes, planes, stride, track):
        super(Block, self).__init__()
        norm1=nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
        norm2=nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        self.norm1=norm1
        self.conv1=nn.Conv2d(in_planes, planes, 3, padding=1, stride=stride, bias=False)
        self.norm2=norm2
        self.conv2=nn.Conv2d(planes, planes, 3, padding=1, stride=1, bias=False)
        self.shortcut=nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut=nn.Conv2d(in_planes, planes, 1, stride, bias=False)
            
    def forward(self, x):
        out = F.relu(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.norm2(out)))
        out += shortcut
        return out

class Bottleneck(nn.Module):
    expansion=4
    def __init__(self, in_planes, planes, stride, track):
        super(Bottleneck, self).__init__()
        norm1=nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
        norm2=nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        norm3=nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        self.norm1=norm1
        self.conv1=nn.Conv2d(in_planes, planes, 3, padding=1, stride=stride, bias=False)
        self.norm2=norm2
        self.conv2=nn.Conv2d(planes, planes, 3, padding=1, stride=1, bias=False)
        self.norm3=norm3
        self.conv3=nn.Conv2d(planes, planes*self.expansion, 1, stride=1, bias=False)

        
        if stride != 1 or in_planes != planes*self.expansion:
            self.shortcut=nn.Conv2d(in_planes, planes*self.expansion, 1, stride, bias=False)
    
    def forward(self, x):
        out=F.relu(self.norm1(x))
        shortcut=self.shortcut(out) if hasattr(self, 'shortcut') else x
        out=self.conv1(out)
        out=self.conv2(F.relu(self.norm2(out)))
        out=self.conv3(F.relu(self.norm3(out)))
        out+=shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, block, args, model_rate, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(ResNet, self).__init__()
        pre_hidden_size=[64, 128, 256, 512]
        hidden_size=[int(np.ceil(i*model_rate))  for i in pre_hidden_size]
        self.in_planes = hidden_size[0]
        self.conv1=nn.Conv2d(3, hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1=self._make_layer(block, hidden_size[0], num_blocks[0], stride=1, track=False)
        self.layer2=self._make_layer(block, hidden_size[1], num_blocks[1], stride=2, track=False)
        self.layer3=self._make_layer(block, hidden_size[2], num_blocks[2], stride=2, track=False)
        self.layer4=self._make_layer(block, hidden_size[3], num_blocks[3], stride=2, track=False)
        self.output=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_size[-1], num_classes)
        )
        #self.linear=nn.Linear(hidden_size[-1], num_classes)
        for i in range(1, len(args.model_level)):
            projector_name=f'orthogonal_projector{i}'
            setattr(self, projector_name, nn.utils.parametrizations.orthogonal(
                nn.Linear(pre_hidden_size[3], int(np.ceil(args.model_level[i]*pre_hidden_size[3])))))
            projector_name=f'linear_projector{i}'
            setattr(self, projector_name, nn.Linear(pre_hidden_size[3], int(np.ceil(args.model_level[i]*pre_hidden_size[3]))))
            
    def _make_layer(self, block, planes, num_blocks, stride, track):
        strides = [stride] + [1]*(num_blocks-1)
        layers = list()
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, track))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def extract_feature(self, x):
        x=self.conv1(x)
        feat1=self.layer1(x)
        feat2=self.layer2(feat1)
        feat3=self.layer3(feat2)
        feat4=self.layer4(feat3)
        out=self.output(feat4)
        feat1=self.layer2[0].norm1(feat1)
        feat2=self.layer3[0].norm1(feat2)
        feat3=self.layer4[0].norm1(feat3)
        return [feat1, feat2, feat3, feat4], out
    
    def forward_feature(self, x):
        out=self.conv1(x)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        #print(out.shape)

        return out
    
    def forward_head(self, x):
        out=self.output(x)
        return out 
    
    def forward(self, x):
        out=self.forward_feature(x)
        out=self.forward_head(out)

        return out
    
def ResNet18Cifar(args, model_rate):
    model = ResNet(Block, args, model_rate, num_blocks=[2, 2, 2, 2], num_classes=10)
    return model
    
def ResNet34Cifar(args, model_rate):
    model = ResNet(Block, args, model_rate, num_blocks=[3, 4, 6, 3], num_classes=10)
    return model

def ResNet50Cifar(args, model_rate):
    model = ResNet(Bottleneck, args, model_rate, num_blocks=[3, 4, 6, 3], num_classes=10)
    return model

def ResNet101Cifar(args, model_rate):
    model = ResNet(Bottleneck, args, model_rate, num_blocks=[3, 4, 23, 3], num_classes=10)
    return model

def ResNet152Cifar(args, model_rate):
    model = ResNet(Bottleneck, args, model_rate, num_blocks=[3, 8, 36, 3], num_classes=10)
    return model

def ResNetCifar(args, model_rate):
    if args.model=='resnet18':
        return ResNet18Cifar(args, model_rate)
    elif args.model=='resnet34':
        return ResNet34Cifar(args, model_rate)
    elif args.model=='resnet50':
        return ResNet50Cifar(args, model_rate)
    elif args.model=='resnet101':
        return ResNet101Cifar(args, model_rate)
    elif args.model=='resnet152':
        return ResNet152Cifar(args, model_rate)
    else:
        return None