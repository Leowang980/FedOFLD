import copy
import random
import numpy as np
import torch
import time
import csv
import torch.utils
from alg.utils import write_result, global_test, make_checkpoint, make_distill_optimizer, make_distill_scheduler

#only logit distillation(used for FedDF_homo, FedDF_hetero)
def ensemble_distillation(args, global_model, client_list, local_weight, 
                        global_weight, selected_client, train_len_dict, dataloader_distill):
        
        optimizer=make_distill_optimizer(args, global_model)
        criterion=torch.nn.KLDivLoss(reduction='batchmean')
        scheduler=make_distill_scheduler(args, optimizer)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.distill_epoch, eta_min=0)
        for idx in range(args.distill_epoch):
            total_loss=0.0
            for batch_idx, (images, labels) in enumerate(dataloader_distill):
                prob=list()
                optimizer.zero_grad()
                total_num=0
                images=images.to(args.device)
                for i in range(len(selected_client)):
                    cur_prob, _=client_list[selected_client[i]].get_prob(images, local_weight[i])
                    #cur_prob=self.client_list[selected_client[i]].output(local_weight[i], images)
                    total_num+= train_len_dict[selected_client[i]]
                    prob.append(cur_prob * train_len_dict[selected_client[i]])
                local_prob=sum(prob)/total_num
                
                T=args.temperature
                global_model.load_state_dict(global_weight)
                global_model.train()
                global_prob=global_model(images)
                global_prob=torch.nn.functional.log_softmax(global_prob/T, dim=1)
                local_prob=torch.nn.functional.softmax(local_prob/T, dim=1)

                loss=(T**2)*criterion(global_prob, local_prob)
                total_loss+=loss.item()
                loss.backward()
                optimizer.step()
            if args.distill_scheduler == 'ReduceLROnPlateau':
                scheduler.step(total_loss)
            else:
                scheduler.step()
        
        return global_model.state_dict()    

#feature distillation for homo(used for FedFD_homo)
def homo_feature_distillaton(args, global_model, total_num, 
                            client_list, local_weight, global_weight, 
                            selected_client, train_len_dict, dataloader_distill):
        optimizer=make_distill_optimizer(args, global_model)
        criterion=torch.nn.KLDivLoss(reduction='batchmean')
        scheduler=make_distill_scheduler(args, optimizer)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.distill_epoch, eta_min=0)
        for idx in range(args.distill_epoch):
            total_loss=0.0
            for batch_idx, (images, labels) in enumerate(dataloader_distill):
                feature=list()
                optimizer.zero_grad()
                total_num=0
                images=images.to(args.device)
                for i in range(len(selected_client)):
                    _, cur_feature=client_list[selected_client[i]].get_prob(images, local_weight[i])
                    #cur_prob=self.client_list[selected_client[i]].output(local_weight[i], images)
                    total_num+= train_len_dict[selected_client[i]]
                    feature.append(cur_feature * train_len_dict[selected_client[i]])
                local_feature=sum(feature)/total_num
                
                T=args.temperature
                global_model.load_state_dict(global_weight)
                global_model.train()
                global_feature=global_model.forward_feature(images)
                global_feature=torch.nn.functional.log_softmax(global_feature/T, dim=1)
                local_feature=torch.nn.functional.softmax(local_feature/T, dim=1)

                loss=(T**2)*criterion(global_feature, local_feature)
                total_loss+=loss.item()
                loss.backward()
                optimizer.step()
            if args.distill_scheduler == 'ReduceLROnPlateau':
                scheduler.step(total_loss)
            else:
                scheduler.step()
        
        return global_model.state_dict()  


#linear and orthogonal feature distillation(used for FedLFD_hetero and FedOFL_hetero)
def hetero_feature_distillation(args, global_model, model_rate, total_num, 
                                        client_list, local_weight, global_weight, 
                                        selected_client, train_len_dict, dataloader_distill):
    
    optimizer=make_distill_optimizer(args, global_model)
    criterion=torch.nn.KLDivLoss(reduction='batchmean')

    scheduler=make_distill_scheduler(args, optimizer)
    for idx in range(args.distill_epoch):
        total_loss=0.0
        for batch_idx, (images, labels) in enumerate(dataloader_distill):
            #start_time=time.time()
            T=args.temperature
            optimizer.zero_grad()
            images=images.to(args.device)
            global_model.load_state_dict(global_weight)
            global_model.train()
            cur_global_feature=global_model.forward_feature(images)
            b, c, h, w=cur_global_feature.shape
            global_feature=cur_global_feature.view(b, c, h*w).mean(-1)
            global_prob=global_model.forward_head(cur_global_feature)
            global_feature_dict=dict()
            global_feature_dict[args.model_level[0]]=torch.nn.functional.log_softmax(global_feature/T, dim=1)
            '''for k, v in global_feature_dict.items():
                print(v.shape)'''
            '''projector=dict()
            projector[args.model_level[0]]=torch.nn.Identity()
            for i in range(1, len(args.model_level)):
                if args.method == 'FedLFD_hetero':
                    projector_name=f'linear_projector{i}' 
                elif args.method == 'FedOFD_hetero':
                    projector_name=f'orthogonal_projector{i}'
                projector[args.model_level[i]]=getattr(global_model, projector_name)

            _, fir_feature=client_list[selected_client[0]].get_prob(images, local_weight[0])
            fir_feature=torch.nn.functional.softmax(fir_feature/T, dim=1)
            loss=(T**2) * criterion(global_feature, projector[client_list[selected_client[0]].model_rate](fir_feature))

            for i in range(1, len(selected_client)):
                _, cur_feature=client_list[selected_client[i]].get_prob(images, local_weight[i])
                cur_feature=torch.nn.functional.softmax(cur_feature/T, dim=1)
                loss=loss+ (T**2) * criterion(global_feature, projector[client_list[selected_client[i]].model_rate](cur_feature))'''

            for i in range(1, len(args.model_level)):
                if args.method == 'FedLFD_hetero':
                    projector_name=f'linear_projector{i}' 
                elif args.method == 'FedOFD_hetero':
                    projector_name=f'orthogonal_projector{i}'
                projector=getattr(global_model, projector_name)
                new_feature=projector(global_feature)
                new_feature=torch.nn.functional.log_softmax(new_feature/T, dim=1)
                global_feature_dict[args.model_level[i]]=new_feature

            _, fir_feature=client_list[selected_client[0]].get_prob(images, local_weight[0])

            fir_feature=torch.nn.functional.softmax(fir_feature/T, dim=1)
            loss=(T**2) * criterion(global_feature_dict[client_list[selected_client[0]].model_rate], fir_feature)
            
            for i in range(1, len(selected_client)):
                _, cur_feature=client_list[selected_client[i]].get_prob(images, local_weight[i])
                cur_feature=torch.nn.functional.softmax(cur_feature/T, dim=1)
                loss=loss+ (T**2) * criterion(global_feature_dict[client_list[selected_client[i]].model_rate], cur_feature)
            
            total_loss+=loss.item()
            '''time1=time.time()-start_time'''
            
            loss.backward()
            '''time2=time.time()-start_time
            print('time1:', time1, 'time2:', time2)'''
            optimizer.step()
        if args.distill_scheduler == 'ReduceLROnPlateau':
            scheduler.step(total_loss)
        else:
            scheduler.step() 
    return global_model.state_dict()

#orthogonal feature distillation + logit distillation(used for FedOFLD_hetero)
def hetero_orthogonal_feature_logit_distillation(args, global_model, model_rate, total_num, 
                                                client_list, local_weight, global_weight, 
                                                selected_client, train_len_dict, dataloader_distill):
    optimizer=make_distill_optimizer(args, global_model)
    criterion=torch.nn.KLDivLoss(reduction='batchmean')
    scheduler=make_distill_scheduler(args, optimizer)
    for idx in range(args.distill_epoch):
        total_loss=0.0
        for batch_idx, (images, labels) in enumerate(dataloader_distill):
            #start_time=time.time()
            T=args.temperature
            optimizer.zero_grad()
            images=images.to(args.device)
            global_model.load_state_dict(global_weight)
            global_model.train()
            cur_global_feature=global_model.forward_feature(images)
            b, c, h, w=cur_global_feature.shape
            global_feature=cur_global_feature.view(b, c, h*w).mean(-1)
            global_prob=global_model.forward_head(cur_global_feature)
            global_prob=torch.nn.functional.log_softmax(global_prob/T, dim=1)
            global_feature_dict=dict()
            global_feature_dict[args.model_level[0]]=torch.nn.functional.log_softmax(global_feature/T, dim=1)

            for i in range(1, len(args.model_level)):
                if args.method == 'FedOFLD_hetero':
                    projector_name=f'orthogonal_projector{i}'
                elif args.method == 'FedLFLD_hetero':
                    projector_name=f'linear_projector{i}'
                projector=getattr(global_model, projector_name)
                new_feature=projector(global_feature)
                new_feature=torch.nn.functional.log_softmax(new_feature/T, dim=1)
                global_feature_dict[args.model_level[i]]=new_feature

            fir_prob, fir_feature=client_list[selected_client[0]].get_prob(images, local_weight[0])
            fir_prob=torch.nn.functional.softmax(fir_prob/T, dim=1)
            fir_feature=torch.nn.functional.softmax(fir_feature/T, dim=1)
            logit_loss=(T**2) * criterion(global_prob, fir_prob)
            feature_loss=(T**2) * criterion(global_feature_dict[client_list[selected_client[0]].model_rate], fir_feature)
            
            for i in range(1, len(selected_client)):
                cur_prob, cur_feature=client_list[selected_client[i]].get_prob(images, local_weight[i])
                cur_prob=torch.nn.functional.softmax(cur_prob/T, dim=1)
                cur_feature=torch.nn.functional.softmax(cur_feature/T, dim=1)
                logit_loss=logit_loss+(T**2) * criterion(global_prob, cur_prob)
                feature_loss=feature_loss+(T**2) * criterion(global_feature_dict[client_list[selected_client[i]].model_rate], cur_feature)

            loss=logit_loss+feature_loss
            #time1=time.time()-start_time
            total_loss+=loss.item()
            loss.backward()
            #time2=time.time()-start_time
            #print('time1:', time1, 'time2:', time2)
            optimizer.step()
        if args.distill_scheduler == 'ReduceLROnPlateau':
            scheduler.step(total_loss)
        else:
            scheduler.step() 
    return global_model.state_dict()

#Hetero feature distillation using hetero  (used for HeteroHetero)
def hetero_hetero_feature_distillation(args, global_model, model_rate, total_num, 
                                        client_list, local_weight, global_weight, 
                                        selected_client, train_len_dict, dataloader_distill):
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
            cur_global_feature=global_model.forward_feature(images)
            b, c, h, w=cur_global_feature.shape
            global_feature=cur_global_feature.view(b, c, h*w).mean(-1)
            global_prob=global_model.forward_head(cur_global_feature)
            global_feature=torch.nn.functional.log_softmax(global_feature/T, dim=1)
            local_feature=torch.zeros(global_feature.size(), dtype=global_feature.dtype, device=args.device)
            count=torch.zeros(global_feature.size(), dtype=global_feature.dtype, device=args.device)

            for i in range(len(selected_client)):
                _, cur_feature=client_list[selected_client[i]].get_prob(images, local_weight[i])
                w, h=cur_feature.shape
                input_idx=torch.arange(w, device=args.device)
                output_idx=torch.arange(h, device=args.device)

                local_feature[torch.meshgrid(input_idx, output_idx)]+=cur_feature
                count[torch.meshgrid(input_idx, output_idx)]+=1
            
            local_feature[count>0]=torch.div(local_feature[count>0], count[count>0])
            local_feature=torch.nn.functional.softmax(local_feature/T, dim=1)
            loss=(T**2) * criterion(global_feature, local_feature)
            total_loss+=loss.item()
            loss.backward()
            optimizer.step()
        if args.distill_scheduler == 'ReduceLROnPlateau':
            scheduler.step(total_loss)
        else:
            scheduler.step() 
    return global_model.state_dict()