import torch
import time
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.utils import formataddr
from email import encoders
import smtplib
from utils.sampling import cifar10_noiid, cifar100_distill, cifar10_global
from utils.options import args_parser
from nets.resnets import ResNetCifar
from nets.cnn import CNNCifar
from alg.fed import FedAvg, HeteroFL, Fed_Distill_hetero, Fed_Distill_homo
from alg.non_fed import Non_Fed
if __name__=='__main__':

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
    args=args_parser()
    args.device=torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.path_checkpoint="checkpoint/"+args.method+'_'+args.model+'_'+str(args.alpha)+".pth.tar"
    print(args.path_checkpoint)
    dataloader_train_dict, dataloader_test_dict, train_len_dict, test_len_dict=cifar10_noiid(args=args,root='../../data/cifar10')
    dataloader_distill=cifar100_distill(args=args,root='../../data/cifar100')
    dataloader_train_global, dataloader_test_global=cifar10_global(args=args, root='../../data/cifar10')

    
    if 'resnet' in args.model:
        model=ResNetCifar(args, 1).to(args.device)
        model1=ResNetCifar(args, 1).to(args.device)
        model2=ResNetCifar(args, 1).to(args.device)
    elif args.model == 'cnn':
        model=CNNCifar(1, args).to(args.device)
        model1=CNNCifar(0.7, args).to(args.device)
        model2=CNNCifar(0.4, args).to(args.device)
    else:
        exit('Error: unrecognized model')


    #print(model)
    p=0
    if args.method == 'test':

        for k, v in model.state_dict().items():
            print(k, v)
            break
        '''for k, v in model1.state_dict().items():
            print(k, v)
        for k, v in model2.state_dict().items():
            print(k, v)'''
        exit()
    if args.method == 'FedAvg':
        fed=FedAvg(args, model, dataloader_train_dict, dataloader_test_dict, 
                dataloader_test_global, train_len_dict, test_len_dict)
    elif args.method in ['FedDF_homo', 'FedFD_homo']:
        fed=Fed_Distill_homo(args, model, dataloader_train_dict, dataloader_test_dict, 
                    dataloader_test_global, train_len_dict, test_len_dict, dataloader_distill)
    elif args.method == 'HeteroFL':
        fed=HeteroFL(args, model, dataloader_train_dict, dataloader_test_dict, 
                    dataloader_test_global, train_len_dict, test_len_dict) 
    elif args.method in ['FedLFD_hetero', 'FedOFD_hetero', 'FedLFLD_hetero', 'FedOFLD_hetero', 'FedDF_hetero', 'HeteroHetero']:
        fed=Fed_Distill_hetero(args, model, dataloader_train_dict, dataloader_test_dict, 
                        dataloader_test_global, train_len_dict, test_len_dict, dataloader_distill)
    elif args.method == "Non_Fed":
        fed=Non_Fed(args, dataloader_train_global, dataloader_test_global, model)
    fed.train()

    smtp_sever = 'smtp.163.com'
    from_addr = '18681229122@163.com'
    password = 'SFwAyWgsbycYCmEY'  
    to_addr = 'leowang980@outlook.com'
    
    msg=MIMEMultipart()
    msg['From']=formataddr(['个人电脑',from_addr])
    msg['To']=formataddr(['Leowang980',to_addr])
    msg['Subject']='模型训练完成，结果见附件'
    pth='result/result_'+args.method+'_'+args.model+'_'+str(args.alpha)+'.csv'
    filename='result_'+args.method+'_'+args.model+'_'+str(args.alpha)+'.csv'
    with open(pth, 'rb') as f:
        base=MIMEBase('结果','pdf')
        base.set_payload(f.read())
        encoders.encode_base64(base)
        base.add_header('Content-Disposition', 'attachment', filename=filename)
        msg.attach(base)
    
    sever = smtplib.SMTP(smtp_sever,25)
    sever.login(from_addr,password)
    
    sever.sendmail(from_addr,to_addr,msg.as_string())
    sever.quit()
    print('邮件发送成功')