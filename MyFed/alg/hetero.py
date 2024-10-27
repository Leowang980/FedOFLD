import copy
import random
import numpy as np
import torch
import time
import csv
import torch.utils
from collections import OrderedDict

def split_model(args, user_idx, global_weight, model_rate):
    hidden_size=[64, 128, 256, 512]
    if args.model=='cnn':
        idx_i=[None for _ in range(len(user_idx))]
        idx=[OrderedDict() for _ in range(len(user_idx))]

        output_weight_name = [k for k in global_weight.keys() if 'output' in k and 'weight' in k][0]#输出层权重名
        output_bias_name = [k for k in global_weight.keys() if 'output' in k and 'bias' in k][0]#输出层偏置名

        for k, v in global_weight.items():
            param_type=k.split('.')[-1]

            for m in range(len(user_idx)):
                if 'projector' in k:
                    if v.dim()>1:
                        input_size=v.size(1)
                        output_size=v.size(0)
                        input_idx=torch.arange(input_size, device=args.device)
                        output_idx=torch.arange(output_size, device=args.device)
                        idx[m][k]= output_idx, input_idx
                    else:
                        input_size=v.size(0)
                        input_idx=torch.arange(input_size, device=args.device)
                        idx[m][k]=input_idx
                elif 'weight' in param_type or 'bias' in param_type:
                    if param_type=='weight':
                        if v.dim()>1:
                            input_size=v.size(1)
                            output_size=v.size(0)
                            if idx_i[m] is None:
                                idx_i[m]=torch.arange(input_size, device=args.device)
                            input_idx=idx_i[m]
                            if k==output_weight_name:
                                output_idx=torch.arange(output_size, device=args.device)
                            else:
                                local_output_size=int(np.ceil(output_size*model_rate[user_idx[m]]))
                                output_idx=torch.arange(local_output_size, device=args.device)
                            idx[m][k]=output_idx, input_idx
                            idx_i[m]=output_idx
                        else:
                            input_idx=idx_i[m]
                            idx[m][k]=input_idx
                    else:
                        input_idx=idx_i[m]
                        idx[m][k]=input_idx
                else:
                    pass
    elif 'resnet' in args.model:
        idx_i=[None for _ in range(len(user_idx))]
        idx=[OrderedDict() for _ in range(len(user_idx))]
        for k, v in global_weight.items():
            param_type=k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'projector' in k:
                    if v.dim()>1:
                        input_size=v.size(1)
                        output_size=v.size(0)
                        input_idx=torch.arange(input_size, device=args.device)
                        output_idx=torch.arange(output_size, device=args.device)
                        idx[m][k]= output_idx, input_idx
                    else:
                        input_size=v.size(0)
                        input_idx=torch.arange(input_size, device=args.device)
                        idx[m][k]=input_idx
                elif 'weight' in param_type or 'bias' in param_type:
                    if param_type=='weight':
                        if v.dim()>1:
                            input_size=v.size(1)
                            output_size=v.size(0)
                            if 'conv1' in k or 'conv2' in k or 'conv3' in k:
                                if idx_i[m] is None:
                                    idx_i[m]=torch.arange(input_size, device=args.device)
                                input_idx=idx_i[m]
                                local_output_size=int(np.ceil(output_size*model_rate[user_idx[m]]))
                                output_idx=torch.arange(local_output_size, device=args.device)
                                idx_i[m]=output_idx
                            elif 'shortcut' in k:
                                input_idx=idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx=idx_i[m]
                            elif 'output' in k:
                                input_idx=idx_i[m]
                                output_idx=torch.arange(output_size, device=args.device)
                            else:
                                print(k)
                                raise ValueError('error')
                            idx[m][k]=output_idx, input_idx
                        else:
                            input_idx=idx_i[m]
                            idx[m][k]=input_idx
                    else:
                        input_size=v.size(0)
                        if 'output' in k:
                            input_idx=torch.arange(input_size, device=args.device)
                            idx[m][k]=input_idx
                        else:
                            input_idx=idx_i[m]
                            idx[m][k]=input_idx
                else:
                    pass

    '''for k, v in global_weight.items():
        print(k, v.shape)
        '''
    return idx

def make_model_rate(args):
    model_rate=list()
    total_num=args.all_client
    for i in range(args.model_num-1):
        cur_num=int(np.ceil(args.model_proportion[i]*args.all_client))
        total_num-=cur_num
        for j in range(cur_num):
            model_rate.append(args.model_level[i])
    for i in range(total_num):
        model_rate.append(args.model_level[-1])
    random.shuffle(model_rate)
    return model_rate

def distribute(args, user_idx, global_weight, model_rate):
    param_idx=split_model(args, user_idx, global_weight, model_rate)
    local_param=[OrderedDict() for _ in range(len(user_idx))]
    for k, v in global_weight.items():
        #param_type=k.split('.')[-1]
        for m in range(len(user_idx)):
            if 'weight' in k or 'bias' in k:
                if 'weight' in k:
                    if v.dim()>1:
                        local_param[m][k]=copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                    else:
                        local_param[m][k]=copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_param[m][k]=copy.deepcopy(v[param_idx[m][k]])
            else:
                local_param[m][k]=copy.deepcopy(v)
    return local_param, param_idx

def combine(args, global_param, local_param, param_idx, user_idx):
    count=OrderedDict()
    output_weight_name=[k for k in local_param[0].keys() if 'weight' in k][-1]
    output_bias_name=[k for k in local_param[0].keys() if 'bias' in k][-1]

    for k, v in global_param.items():
        if 'projector' in k:
            continue
        count[k]=v.new_zeros(v.size(), dtype=torch.float32)
        tmp_v=v.new_zeros(v.size(), dtype=torch.float32)

        for m in range(len(user_idx)):
            if 'weight' in k or 'bias' in k:
                if 'weight' in k:
                    #print('weight')
                    if v.dim()>1:
                        tmp_v[torch.meshgrid(param_idx[m][k])]+=local_param[m][k]
                        count[k][torch.meshgrid(param_idx[m][k])]+=1
                    else:
                        tmp_v[param_idx[m][k]]+=local_param[m][k]
                        count[k][param_idx[m][k]]+=1
                else:
                    #print('bias')
                    tmp_v[param_idx[m][k]]=local_param[m][k]
                    count[k][param_idx[m][k]]=+1
            else:
                tmp_v+=local_param[m][k]
                count[k]+=1
        tmp_v[count[k]>0]=torch.div(tmp_v[count[k]>0], count[k][count[k]>0])
        v[count[k]>0]=tmp_v[count[k]>0].to(v.dtype)
        #print('combine success')
    return global_param
