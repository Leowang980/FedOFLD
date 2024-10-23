import torch
import torch.nn as nn
import copy

from alg.local import Local
from nets.cnn import CNNCifar
from nets.resnets import ResNetCifar

class Client(object):
    def __init__(self, args, dataloader_train, dataloader_test):
        self.args=args
        self.dataloader_train=dataloader_train
        self.dataloader_test=dataloader_test
        if 'resnet' in args.model:
            self.model=ResNetCifar(args, 1).to(args.device)
        else:
            self.model=CNNCifar(1, args).to(args.device)
        self.local=Local(args, dataloader_train, dataloader_test, self.model)

    def train(self, global_weight):
        weight, loss=self.local.local_train(global_weight)
        return weight, loss
    
    def test(self, global_weight):
        acc=self.local.local_test(global_weight)
        return acc
    
    def get_prob(self, images, global_weight):
        self.model.load_state_dict(global_weight)
        self.model.eval()
        with torch.no_grad():
            feature=self.model.forward_feature(images)
            prob=self.model.forward_head(feature)
        return prob, feature

class HeteroClient(object):
    def __init__(self, args, dataloader_train, dataloader_test, model_rate):
        self.args=args
        self.dataloader_train=dataloader_train
        self.dataloader_test=dataloader_test
        self.model_rate=model_rate
        if 'resnet' in args.model:
            self.model=ResNetCifar(args, model_rate).to(args.device)
        else:
            self.model=CNNCifar(model_rate, args).to(args.device)
        self.local=Local(args, dataloader_train, dataloader_test, self.model)

    def train(self, global_weight):
        weight, loss=self.local.local_train(global_weight)
        return weight, loss
    
    def test(self, global_weight):
        acc=self.local.local_test(global_weight)
        return acc
    
    def get_prob(self, images, global_weight):
        self.model.load_state_dict(global_weight)
        self.model.eval()
        with torch.no_grad():
            prob=self.model(images)
            cur_feature=self.model.forward_feature(images)
            b, c, h, w=cur_feature.shape
            feature=cur_feature.view(b, c, h*w).mean(-1)
            prob=self.model.forward_head(cur_feature)
        return prob, feature