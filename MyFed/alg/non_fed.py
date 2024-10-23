import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from alg.utils import write_result, global_test, make_checkpoint

class Non_Fed(object):
    def __init__(self, args, dataloader_train, dataloader_test, model):
        self.args=args
        self.dataloader_train=dataloader_train
        self.dataloader_test=dataloader_test
        self.model=model
    
    def train(self):
        if self.args.local_optimizer=='sgd':
            optimizer=torch.optim.SGD(self.model.parameters(), lr= self.args.local_learning_rate, momentum=self.args.local_momentum, weight_decay=self.args.local_weight_decay)
        else:
            optimizer=torch.optim.Adam(self.model.parameters(), lr=self.args.local_learning_rate, weight_decay=self.args.local_weight_decay, amsgrad=True)
        loss_func=nn.CrossEntropyLoss().to(self.args.device)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.local_epoch/2, eta_min=0)
        all_acc=list()
        all_loss=list()
        all_time=list()
        top_acc=0.0
        start_epoch=0
        if self.args.resume:
            check_point=torch.load(self.args.path_checkpoint)
            self.model.load_state_dict(check_point['model'])
            start_epoch=check_point['communication_round']
            optimizer.load_state_dict(check_point['optimizer'])
            scheduler.load_state_dict(check_point['scheduler'])
        
        for epoch in range(start_epoch, self.args.local_epoch):
            start_time=time.time()
            batch_loss=list()
            for batch_idx, (images, labels) in enumerate(self.dataloader_train):
                images, labels= images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                output=self.model(images)
                loss= loss_func(output, labels)
                #print(type(loss))
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            all_loss.append(sum(batch_loss)/len(batch_loss))
            scheduler.step()
            acc=global_test(self.args, self.model, self.dataloader_test)
            all_acc.append(acc)
            all_time.append(time.time()-start_time)
            '''for k, v in self.model.state_dict().items():
                print(k)'''
            if acc>top_acc:
                make_checkpoint(self.args, self.model, epoch, optimizer, scheduler)
                top_acc=acc

            write_result(self.args, epoch, start_epoch, all_loss, all_acc, all_time)