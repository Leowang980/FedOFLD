import torch  
import torch.nn as nn  
import numpy as np
import torchvision.models as models  
import torch.nn.functional as F 

class CNNCifar(nn.Module):
    def __init__(self, model_rate, args):
        super(CNNCifar, self).__init__()

        pre_hidden_size = [64, 128, 256, 512]
        hidden_size=[int(np.ceil(i*model_rate))  for i in pre_hidden_size]
        self.hidden_size=hidden_size

        self.block1=self._make_block(0)
        self.block2=self._make_block(1)
        self.block3=self._make_block(2)
        self.block4=self._make_block(3)
        self.output=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_size[-1], 10)
        )

        #self.logit_projector=nn.utils.parametrizations.orthogonal(nn.Linear(10, 10))
        for i in range(1, len(args.model_level)):
            
            projector_name=f'orthogonal_projector{i}'
            setattr(self, projector_name, nn.utils.parametrizations.orthogonal(
                nn.Linear(pre_hidden_size[3], int(np.ceil(args.model_level[i]*pre_hidden_size[3])))))
            projector_name=f'linear_projector{i}'
            setattr(self, projector_name, nn.Linear(pre_hidden_size[3], int(np.ceil(args.model_level[i]*pre_hidden_size[3]))))

    def _make_block(self, layer_idx):
        layers=list()
        if(layer_idx == 0):
            layers.append(nn.Conv2d(3, self.hidden_size[0], 3, 1, 1))
        else:
            layers.append(nn.Conv2d(self.hidden_size[layer_idx-1], self.hidden_size[layer_idx], 3, 1, 1))
        layers.append(nn.BatchNorm2d(self.hidden_size[layer_idx], momentum=None, track_running_stats=False))
        layers.append(nn.ReLU(inplace=True))
        if(layer_idx != 3):
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)
    
    def forward_feature(self, x):
        out=self.block1(x)
        out=self.block2(out)
        out=self.block3(out)
        out=self.block4(out)
        #print(out.shape)
        return out
    
    def forward_head(self, x):
        out=self.output(x)
        return out
    
    def forward(self, x):
        out=self.forward_feature(x)
        #print(out.shape)
        out=self.forward_head(out)
        return out