import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import torchvision
transform_train=transforms.Compose([  
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),    
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])  
transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def cifar10_global(args,root):
    dataset_train=datasets.CIFAR10(root, train=True, transform= transform_train, download=True)
    dataset_test=datasets.CIFAR10(root, train=False, transform= transform_test, download=True)
    dataloader_train=data.DataLoader(dataset=dataset_train, batch_size=args.local_batch_size, shuffle=True)
    dataloader_test=data.DataLoader(dataset=dataset_test, batch_size=args.local_batch_size, shuffle=False)
    return dataloader_train, dataloader_test

def cifar10_noiid(args,root):

    dataset_train=datasets.CIFAR10(root, train=True, transform=transform_train, download=True)
    dataset_test=datasets.CIFAR10(root, train=False, transform=transform_test, download=True)

    #print(type(dataset_train.data),type(dataset_train.targets))
    x_train, y_train=dataset_train.data, np.array(dataset_train.targets)
    x_test, y_test=dataset_test.data, np.array(dataset_test.targets)
    #print('cifar10训练和测试数据大小:',x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    #image, label = dataset_train[0]

    #print(image)
    data_train_dict=set_split(args=args,y=y_train,train=True)
    data_test_dict=set_split(args=args,y=y_test,train=False)
    train_len_dict=dict()
    test_len_dict=dict()
    
    #print(type(y_train),'2')
    dataloader_train_dict, dataloader_test_dict=dict(), dict()
    for idx in range(args.all_client):

        # 获取训练和测试数据的索引列表
        train_indices = data_train_dict[idx]  # 例如: [0, 1, 2, 3]
        test_indices = data_test_dict[idx]    # 例如: [0, 1, 2, 3]
        train_len_dict[idx]=len(train_indices)
        test_len_dict[idx]=len(test_indices)

        # 使用 Subset 创建新的数据集
        train_subset = data.Subset(dataset_train, train_indices)
        test_subset = data.Subset(dataset_test, test_indices)

        dataloader_train_local=data.DataLoader(train_subset, batch_size=args.local_batch_size, shuffle=True, num_workers=0)
        dataloader_test_local=data.DataLoader(test_subset, batch_size=args.local_batch_size, shuffle=True, num_workers=0)
        '''traindata=list()
        testdata=list()

        for batch_idx, (images, labels) in enumerate(dataloader_train_local):
            traindata.append((images, labels))
        
        for batch_idx, (images, labels) in enumerate(dataloader_test_local):
            testdata.append((images, labels))'''
        
        dataloader_train_dict[idx]=dataloader_train_local
        dataloader_test_dict[idx]=dataloader_test_local
        #for batch_idx, (batched_x, batched_y) in enumerate(dataloader_train_local):
        #    print(type(batched_x),type(batched_y))

    
    #print('本地训练数据分配成功')
    return dataloader_train_dict, dataloader_test_dict, train_len_dict, test_len_dict


def cifar100_distill(args, root):

    distill_dataset=datasets.CIFAR100(root=root, transform=transform_test, train=False, download=True)
    distill_dataloader=data.DataLoader(dataset=distill_dataset, batch_size=args.distill_batch_size, shuffle=True)
    '''distill_data=list()
    for idx, (images, labels) in enumerate(distill_dataloader):
        distill_data.append((images, labels))'''
    
    #print('蒸馏数据完成')
    return distill_dataloader


def set_split(args,y,train=True):
    #进行non-iid分配

    min_size=0
    K=10
    N=y.shape[0]
    data_train_dict=dict()
    
    cur_q=0
    while min_size<128:
        cur_q+=1
        #print('第{}次分配数据：'.format(cur_q))
        idx_batch=[[] for _ in range(args.all_client)]

        for k in range(K):

            idx_k=np.where(y==k)[0]
            np.random.shuffle(idx_k)
            proportions=np.random.dirichlet(np.repeat(args.alpha, args.all_client))
            proportions=np.array(proportions)
            #proportions = np.array([p * (len(idx_j) < N/args.all_client) for p, idx_j in zip(proportions, idx_batch)])
            proportions=(np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch=[idx_i+idx.tolist() for idx_i, idx in zip(idx_batch, np.split(idx_k,proportions))]

            min_size=min([len(idx_i) for idx_i in idx_batch])

    sum_num=0
    for i in range(args.all_client):
        np.random.shuffle(idx_batch[i])
        data_train_dict[i]=idx_batch[i]

        sum_num+=len(data_train_dict[i])
        
        #print('第{}个客户端分配到{}个{}数据'.format(i,len(data_train_dict[i]),('train' if train else 'test')))
    #print(sum_num)
    #print('non-iid数据分配完成,alpha:{}'.format(args.alpha))

    return data_train_dict
