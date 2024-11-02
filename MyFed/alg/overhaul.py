import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from alg.utils import write_result, global_test, make_checkpoint, make_distill_optimizer, make_distill_scheduler

def distillation_loss(global_feature, local_feature):
    local_feature=F.relu(local_feature)
    loss=F.mse_loss(global_feature, local_feature, reduction="none")
    loss=loss*((global_feature>local_feature)|(local_feature>0)).float
    return loss.sum()

def overhaul_feature_distillation(args, global_model, model_rate, total_num, 
                        client_list, local_weight, global_weight, 
                        selected_client, train_len_dict, dataloader_distill, connectors):
    optimizer=make_distill_optimizer(args, global_model)
    criterion=torch.nn.KLDivLoss(reduction='batchmean')
    scheduler=make_distill_scheduler(args, optimizer)

    for idx in range(args.distill_epoch):
        total_loss=0.0
        for batch_idx, (images, labels) in enumerate(dataloader_distill):
            T=args.temperature
            optimizer.zero_grad()
            images=images.to(args.device)
            global_model.load_state_dict(global_weight)
            global_model.train()
            global_features, _=global_model.extract_feature(images)
            loss_distill=0
            local_features=list()
            for client_idx in range(len(selected_client)):
                feature, _=client_list[selected_client[client_idx]].extract_feature(images, local_weight[client_idx])
                for i in range(len(feature)):
                    feature[i]=connectors[selected_client[client_idx]][i](feature[i])
                
                local_features.append([i * train_len_dict[selected_client[client_idx]] for i in feature])
            for i in range(len(local_features[0])):
                for j in range(1,local_features):
                    local_features[0][i]=local_features[0][i]+local_features[j][i]
            
            local_feature=copy.deepcopy(local_features[0])
            local_feature=[i/total_num for i in local_feature]

            loss_distill = 0
            for i in range(len(local_feature)):
                loss_distill+=distillation_loss(global_features[i], local_feature[i].detach())/2**(len(local_feature)-i-1)
            total_loss+=loss_distill.item()
            loss_distill.backward()
            optimizer.step()
        if args.distill_scheduler == 'ReduceLROnPlateau':
            scheduler.step(total_loss)
        else:
            scheduler.step()
    
    return global_model.state_dict()  

def overhaul_feature_logit_distillation(args, global_model, model_rate, total_num, 
                        client_list, local_weight, global_weight, 
                        selected_client, train_len_dict, dataloader_distill, connectors):

    optimizer=make_distill_optimizer(args, global_model)
    criterion=torch.nn.KLDivLoss(reduction='batchmean')
    scheduler=make_distill_scheduler(args, optimizer)

    for idx in range(args.distill_epoch):
        total_loss=0.0
        for batch_idx, (images, labels) in enumerate(dataloader_distill):
            T=args.temperature
            optimizer.zero_grad()
            images=images.to(args.device)
            global_model.load_state_dict(global_weight)
            global_model.train()
            global_features, global_prob=global_model.extract_feature(images)
            loss_distill=0
            local_features=list()
            local_probs=list()
            for client_idx in range(len(selected_client)):
                feature, prob=client_list[selected_client[client_idx]].extract_feature(images, local_weight[client_idx])
                for i in range(len(feature)):
                    feature[i]=connectors[selected_client[client_idx]][i](feature[i])
                
                local_features.append([i * train_len_dict[selected_client[client_idx]] for i in feature])
                local_probs.append(train_len_dict[selected_client[client_idx]]*prob)
            for i in range(len(local_features[0])):
                for j in range(1,local_features):
                    local_features[0][i]=local_features[0][i]+local_features[j][i]
            
            local_feature=copy.deepcopy(local_features[0])
            local_feature=[i/total_num for i in local_feature]
            local_prob=sum(local_probs)/total_num

            loss_distill = 0
            for i in range(len(local_feature)):
                loss_distill+=distillation_loss(global_features[i], local_feature[i].detach())/2**(len(local_feature)-i-1)
            loss=loss_distill/args.distill_batch_size/1000 + criterion(global_prob, local_prob)
            loss_distill.backward()
            optimizer.step()
        if args.distill_scheduler == 'ReduceLROnPlateau':
            scheduler.step(total_loss)
        else:
            scheduler.step()
    
    return global_model.state_dict() 