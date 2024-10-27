import copy
import random
import numpy as np
import torch
import time
import csv
import torch.utils
from alg.client import Client, HeteroClient
from alg.distill import ensemble_distillation, homo_feature_distillaton, hetero_feature_distillation
from alg.distill import hetero_orthogonal_feature_logit_distillation, hetero_hetero_feature_distillation
from alg.utils import write_result, global_test, Avg, setup_client, setup_hetero_client
from alg.utils import make_checkpoint, make_distill_optimizer, make_distill_scheduler
from alg.hetero import make_model_rate, distribute, combine
from collections import OrderedDict

class FedAvg(object):
    def __init__(self, args, model, dataloader_train_dict, dataloader_test_dict, 
                dataloader_test_global, train_len_dict, test_len_dict):
        self.args=args
        self.dataloader_train_dict=dataloader_train_dict
        self.dataloader_test_dict=dataloader_test_dict
        self.train_len_dict=train_len_dict
        self.test_len_dict=test_len_dict
        self.dataloader_test_global=dataloader_test_global
        self.model=model
        self.client_list=setup_client(self.args, self.dataloader_train_dict, self.dataloader_test_dict, self.model)

    def train(self):
        all_loss=list()
        all_acc=list()
        all_time=list()
        top_acc=0.0
        start_round=0
        if self.args.resume:
            check_point=torch.load(self.args.path_checkpoint)
            self.model.load_state_dict(check_point['model'])
            start_round=check_point['communication_round']

        for round_idx in range(start_round, self.args.communication_round):
            start_time=time.time()

            selected_client=np.random.choice(self.args.all_client, self.args.each_client, replace=False)
            #print(selected_client)
            
            global_weight= self.model.state_dict()
            local_weight= list()
            loss=list()
            total_num=0
            for client_idx in selected_client:
                cur_local_weight, cur_loss=self.client_list[client_idx].train(global_weight)
                local_weight.append(cur_local_weight)
                loss.append(cur_loss)
                total_num+=self.train_len_dict[client_idx]

            avg_loss=sum(loss)/len(loss)

            avg_weight=Avg(local_weight, self.train_len_dict, total_num, selected_client)
            self.model.load_state_dict(avg_weight)
            acc= global_test(self.args, self.model, self.dataloader_test_global)

            
            end_time=time.time()
            longing_time=end_time-start_time
            all_acc.append(acc)
            all_loss.append(avg_loss)
            all_time.append(longing_time)
            write_result(self.args, round_idx, start_round, all_loss, all_acc, all_time)
            if acc>top_acc:
                make_checkpoint(self.args, self.model, round_idx)
                top_acc=acc

            #print('communication_round:{},loss:{},acc:{},time:{}'.format(round_idx, avg_loss, acc, longing_time))
        
class HeteroFL(object):
    def __init__(self, args, global_model, dataloader_train_dict, dataloader_test_dict, 
                dataloader_test_global, train_len_dict, test_len_dict):
        self.args=args
        self.dataloader_train_dict=dataloader_train_dict
        self.dataloader_test_dict=dataloader_test_dict
        self.dataloader_test_global=dataloader_test_global
        self.train_len_dict=train_len_dict
        self.test_len_dict=test_len_dict
        self.global_model=global_model
        self.model_rate=make_model_rate(args)
        self.client_list=setup_hetero_client(self.args, self.dataloader_train_dict, self.dataloader_test_dict, self.model_rate)
    
    def train(self):
        all_loss=list()
        all_acc=list()
        all_time=list()
        top_acc=0.0
        start_round=0

        if self.args.resume:
            check_point=torch.load(self.args.path_checkpoint)
            self.global_model.load_state_dict(check_point['model'])
            start_round=check_point['communication_round']

        '''for k, v in self.global_model.state_dict().items():
            print(k, v.size())
        for k, v in self.client_list[0].model.state_dict().items():
            print(k, v.size())
        for k, v in self.client_list[1].model.state_dict().items():
            print(k, v.size())'''
        for round_idx in range(start_round, self.args.communication_round):
            start_time=time.time()
            selected_client=np.random.choice(self.args.all_client, self.args.each_client, replace=False)
            global_weight= self.global_model.state_dict()
            local_param, param_idx=distribute(self.args, selected_client, global_weight, self.model_rate)
            loss=list()
            total_num=0
            for client_idx in range(len(selected_client)):
                local_param[client_idx], cur_loss=self.client_list[selected_client[client_idx]].train(local_param[client_idx])
                loss.append(cur_loss)
                total_num+=self.train_len_dict[selected_client[client_idx]]

            avg_loss=sum(loss)/len(loss)
            global_weight=combine(self.args, global_weight, local_param, param_idx, selected_client)
            #print(global_weight)
            self.global_model.load_state_dict(global_weight)
            acc= global_test(self.args, self.global_model, self.dataloader_test_global)
            end_time=time.time()
            longing_time=end_time-start_time
            all_acc.append(acc)
            all_loss.append(avg_loss)
            all_time.append(longing_time)
            if acc>top_acc:
                make_checkpoint(self.args, self.global_model, round_idx)
                top_acc=acc

            write_result(self.args, round_idx, start_round, all_loss, all_acc, all_time)
            #print('communication_round:{},loss:{},acc:{},time:{}'.format(round_idx, avg_loss, acc, longing_time))

class Fed_Distill_homo(object):
    def __init__(self, args, global_model, dataloader_train_dict, dataloader_test_dict, dataloader_test_global, 
                train_len_dict, test_len_dict, dataloader_distill):
        self.args=args
        self.dataloader_train_dict=dataloader_train_dict
        self.dataloader_test_dict=dataloader_test_dict
        self.dataloader_test_global=dataloader_test_global
        self.train_len_dict=train_len_dict
        self.test_len_dict=test_len_dict
        self.global_model=global_model
        self.dataloader_distill=dataloader_distill
        self.client_list=setup_client(self.args, self.dataloader_train_dict, self.dataloader_test_dict, self.global_model)
    
    def train(self):
        m=2
        all_loss=list()
        all_acc=list()
        all_time=list()
        top_acc=0.0
        start_round=0
        
        if self.args.resume:
            check_point=torch.load(self.args.path_checkpoint)
            self.global_model.load_state_dict(check_point['model'])
            start_round=check_point['communication_round']

        for round_idx in range(start_round, self.args.communication_round):
            start_time=time.time()

            selected_client=np.random.choice(self.args.all_client, self.args.each_client, replace=False)
        
            global_weight= self.global_model.state_dict()
            local_weight= list()
            loss=list()
            total_num=0
            for client_idx in selected_client:
                cur_local_weight, cur_loss=self.client_list[client_idx].train(global_weight)
                local_weight.append(cur_local_weight)
                loss.append(cur_loss)
                total_num+=self.train_len_dict[client_idx]
            
            if m==2:
                print('开始预热')
                m-=1
            avg_loss=sum(loss)/len(loss)
            
            avg_weight = Avg(local_weight, self.train_len_dict, total_num, selected_client)
            
            if round_idx >= self.args.warmup_round:
                if m==1:
                    print('开始蒸馏')
                    m-=1
                if self.args.method == 'FedFD_homo':
                    avg_weight=homo_feature_distillaton(self.args, self.global_model, total_num, self.client_list, local_weight,
                                                        avg_weight, selected_client, self.train_len_dict, self.dataloader_distill)
                elif self.args.method == 'FedDF_homo':
                    avg_weight=ensemble_distillation(self.args, self.global_model, self.client_list, local_weight, avg_weight, 
                                                    selected_client, self.train_len_dict, self.dataloader_distill)
                
            self.global_model.load_state_dict(avg_weight)
            acc= global_test(self.args, self.global_model, self.dataloader_test_global)
            end_time=time.time()
            longing_time=end_time-start_time
            all_acc.append(acc)
            all_loss.append(avg_loss)
            all_time.append(longing_time)

            if acc>top_acc:
                make_checkpoint(self.args, self.global_model, round_idx)
                top_acc=acc

            write_result(self.args, round_idx, start_round, all_loss, all_acc, all_time)

class Fed_Distill_hetero(object):
    def __init__(self, args, global_model, dataloader_train_dict, dataloader_test_dict, 
                dataloader_test_global, train_len_dict, test_len_dict, dataloader_distill):
        self.args=args
        self.dataloader_train_dict=dataloader_train_dict
        self.dataloader_test_dict=dataloader_test_dict
        self.dataloader_test_global=dataloader_test_global
        self.train_len_dict=train_len_dict
        self.test_len_dict=test_len_dict
        self.global_model=global_model
        self.dataloader_distill=dataloader_distill
        self.model_rate=make_model_rate(args)
        self.client_list=setup_hetero_client(self.args, self.dataloader_train_dict, self.dataloader_test_dict, self.model_rate)
    
    def train(self):
        m=2
        all_loss=list()
        all_acc=list()
        all_time=list()
        top_acc=0.0
        start_round=0
        if self.args.resume:
            check_point=torch.load(self.args.path_checkpoint)
            self.model.load_state_dict(check_point['model'])
            start_round=check_point['communication_round']
        
        if self.args.communication_round <= self.args.warmup_round:
                exit('error:warmup_round must be smaller than communication_round')
        
        for round_idx in range(start_round, self.args.communication_round):
            
            start_time=time.time()
            selected_client=np.random.choice(self.args.all_client, self.args.each_client, replace=False)
            global_weight= self.global_model.state_dict()
            local_param, param_idx=distribute(self.args, selected_client, global_weight, self.model_rate)
            loss=list()
            total_num=0
            for client_idx in range(len(selected_client)):
                local_param[client_idx], cur_loss=self.client_list[selected_client[client_idx]].train(local_param[client_idx])
                loss.append(cur_loss)
                total_num+=self.train_len_dict[selected_client[client_idx]]

            avg_loss=sum(loss)/len(loss)

            if m==2:
                print('开始预热')
                m-=1
            
            avg_weight=combine(self.args, global_weight, local_param, param_idx, selected_client)
            self.global_model.load_state_dict(global_weight)
            print('预热时间', time.time()-start_time)
            if round_idx >= self.args.warmup_round:
                if m==1:
                    print('开始蒸馏')
                    m-=1
                if self.args.method in ['FedLFD_hetero', 'FedOFD_hetero']:
                    avg_weight=hetero_feature_distillation(self.args, self.global_model, self.model_rate, total_num, self.client_list, local_param,
                                                        avg_weight, selected_client, self.train_len_dict, self.dataloader_distill)
                elif self.args.method in ['FedOFLD_hetero', 'FedLFLD_hetero']:
                    avg_weight=avg_weight=hetero_orthogonal_feature_logit_distillation(self.args, self.global_model, self.model_rate, 
                                                                                    total_num, self.client_list, local_param,
                                                                                    avg_weight, selected_client, self.train_len_dict, self.dataloader_distill)
                elif self.args.method == 'FedDF_hetero':
                    avg_weight=ensemble_distillation(self.args, self.global_model, self.client_list, local_param, avg_weight, 
                                                    selected_client, self.train_len_dict, self.dataloader_distill)
                elif self.args.method == 'HeteroHetero':
                    avg_weight=hetero_hetero_feature_distillation(self.args, self.global_model, self.model_rate, total_num, self.client_list, local_param, 
                                                                avg_weight, selected_client, self.train_len_dict, self.dataloader_distill)
            print('蒸馏时间', time.time()-start_time)
            self.global_model.load_state_dict(avg_weight)
            acc= global_test(self.args, self.global_model, self.dataloader_test_global)
            end_time=time.time()
            longing_time=end_time-start_time
            print('所有时间', longing_time)
            all_acc.append(acc)
            all_loss.append(avg_loss)
            all_time.append(longing_time)

            if acc>top_acc:
                make_checkpoint(self.args, self.global_model, round_idx)
                top_acc=acc

            write_result(self.args, round_idx, start_round, all_loss, all_acc, all_time)