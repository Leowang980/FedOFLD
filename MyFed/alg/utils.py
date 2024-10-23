import csv
import torch
import copy
from alg.client import Client, HeteroClient

def Avg(local_weight, train_len_dict, total_num, selected_client):

    avg_weight = copy.deepcopy(local_weight[0])
    for k in avg_weight.keys():
        avg_weight[k]=avg_weight[k]*train_len_dict[selected_client[0]]
        for i in range(1, len(local_weight)):
            avg_weight[k] = avg_weight[k]+local_weight[i][k] * train_len_dict[selected_client[i]]
        avg_weight[k] = torch.div(avg_weight[k], total_num)
    return avg_weight

def setup_client(args, dataloader_train_dict, dataloader_test_dict, model):

    client_list=list()
    for idx in range(args.all_client):
        c=Client(args, dataloader_train_dict[idx], dataloader_test_dict[idx])
        client_list.append(c)
    return client_list

def setup_hetero_client(args, dataloader_train_dict, dataloader_test_dict, model_rate):
    
    client_list=list()
    for idx in range(args.all_client):
        c=HeteroClient(args, dataloader_train_dict[idx], dataloader_test_dict[idx], model_rate[idx])
        client_list.append(c)

    return client_list

def write_result(args, round_idx, start, all_loss, all_acc, all_time):

    file_name='result/result_'+args.method+'_'+args.model+'.csv'
    with open(file_name, mode='w', newline='') as file:
                writer=csv.writer(file)
                writer.writerow(['communication_round', 'Loss', 'Accuracy', 'Time'])
                for idx in range(round_idx-start+1):
                    writer.writerow([idx+start+1, all_loss[idx], all_acc[idx], all_time[idx]])

def global_test(args, model, dataloader):
    with torch.no_grad():

        correct=0
        total=0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels= images.to(args.device), labels.to(args.device)
            output=model(images)
            _, predicted=torch.max(output, 1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()

        return 100*correct/total
    
def make_checkpoint(args, model, communication_round, optimizer=None, scheduler=None):
    checkpoint={
        'communication_round': communication_round,
        'model': model.state_dict(),
    }
    if optimizer is not None:
        checkpoint['optimizer']=optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler']=scheduler.state_dict()
    torch.save(checkpoint, args.path_checkpoint)

def make_distill_optimizer(args, model):
    if args.distill_optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.distill_learning_rate,
                                    momentum=args.distill_momentum,
                                    weight_decay=args.distill_weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.distill_learning_rate,
                                    weight_decay=args.distill_weight_decay,
                                    amsgrad=True)
    
    return optimizer

def make_distill_scheduler(args, optimizer):
    if args.distill_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=args.distill_epoch, 
                                                                eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                'min',
                                                                factor=0.2, 
                                                                patience=5, 
                                                                verbose=True)
    return scheduler