import torch
from torch import nn
#import matplotlib.pyplot as plt

class Local(object):
    def __init__(self, args, dataloader_train, dataloader_test, model):
        self.args=args
        self.dataloader_train=dataloader_train
        self.dataloader_test=dataloader_test
        self.model=model

    def local_train(self, global_weight):

        self.model.load_state_dict(global_weight)
        self.model.train()
        self.model.to(self.args.device)
        if self.args.local_optimizer=='sgd':
            optimizer=torch.optim.SGD(self.model.parameters(), lr= self.args.local_learning_rate, momentum=self.args.local_momentum, weight_decay=self.args.local_weight_decay)
        else:
            optimizer=torch.optim.Adam(self.model.parameters(), lr=self.args.local_learning_rate, weight_decay=self.args.local_weight_decay, amsgrad=True)
        loss_func=nn.CrossEntropyLoss().to(self.args.device)
        epoch_loss=list()
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.local_epoch/2, eta_min=0)

        for idx in range(self.args.local_epoch):

            batch_loss=list()

            for batch_idx, (images, labels) in enumerate(self.dataloader_train):

                images, labels= images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                
                output=self.model(images)
                loss= loss_func(output, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            scheduler.step()
        
        return self.model.state_dict(), sum(epoch_loss)/len(epoch_loss)
    

    def local_test(self, global_weight):
        self.model.load_state_dict(global_weight)
        self.model.eval()
        
        with torch.no_grad():

            correct=0
            total=0

            for batch_idx, (images, labels) in enumerate(self.dataloader_test):
                images, labels= images.to(self.args.device), labels.to(self.args.device)
                output=self.model(images)
                _, predicted=torch.max(output, 1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()

            return 100*correct/total