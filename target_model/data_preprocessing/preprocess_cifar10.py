'''
Author: Ruijun Deng
Date: 2023-09-03 19:29:00
LastEditTime: 2024-05-07 16:59:39
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/data_preprocessing/preprocess_cifar10.py
Description: 
'''
# 导包
import torch

# 导包
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.data import Subset, random_split

from .dataset import ListDataset

# 构建 train1，train2，test三组数据集
def get_cifar10_normalize_two_train(batch_size = 1, split_ratio=0.5):
    mu = torch.tensor([0.5,0.5,0.5],dtype=torch.float32)
    sigma = torch.tensor([0.5,0.5,0.5],dtype=torch.float32)

    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mu,sigma)
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root='/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar10/',train=True,
                                            download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar10/',train=False,
                                           download=False, transform=transform)
    # 使得两个trainset长度相等                                       
    half_length = len(trainset) // 2                                     
    trainset_seen = Subset(trainset, range(half_length))
    trainset_unseen = Subset(trainset, range(half_length, len(trainset)))
    
    if len(trainset_seen) != len(trainset_unseen):
        if len(trainset_seen) > len(trainset_unseen):
            trainset_seen = Subset(trainset_seen, range(len(trainset_unseen)))
        else:
            trainset_unseen = Subset(trainset_unseen, range(len(trainset_seen)))

    trainloader1 = torch.utils.data.DataLoader(trainset_seen,batch_size=batch_size,shuffle=False, num_workers=4)
    trainloader2 = torch.utils.data.DataLoader(trainset_unseen,batch_size=batch_size,shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False, num_workers=4)
    
    return trainloader1,trainloader2,testloader
    # return trainloader1,trainloader2



def get_cifar10_normalize(batch_size = 1):
    #  数据集 CIFAR
    # 图像归一化
    mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    sigma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    transform = transforms.Compose(
        [
            transforms.ToTensor(), # 数据中的像素值转换到0～1之间
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 接近+-1？ 从[0,1] 不是从[0,255]
            # transforms.Normalize(mu.tolist(), sigma.tolist())
        # ])

    # 数据集加载：
    # 测试数据集
    trainset = torchvision.datasets.CIFAR10(root='/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar10/', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar10/', train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
    
    return trainloader,testloader


def get_cifar10_preprocess(batch_size = 1):
    #  数据集 CIFAR
    mu = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
    sigma = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)
    Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
    Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
    tsf = {
        'train': transforms.Compose(
        [
        transforms.ToTensor(),
        Normalize
        ]),
        'test': transforms.Compose(
        [
        transforms.ToTensor(),
        Normalize
        ])
    }
    trainset = torchvision.datasets.CIFAR10(root='/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar10/', train = True,
                                    download=False, transform = tsf['train'])
    testset = torchvision.datasets.CIFAR10(root='/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar10/', train = False,
                                    download=False, transform = tsf['test'])


    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                        shuffle = False, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                        shuffle = False, num_workers = 1)
    return trainloader,testloader

# 取dataloader的前batch_size个数据，成为一个数据集loader,只有一组数据，bs=batch_size
def get_one_data(dataloader,batch_size = 1): # 得到一个dataloader中第一个数据构造的dataloader
    # trainloader, testloader = get_cifar10_normalize(batch_size=batch_size)
    testIter = iter(dataloader)

    # first = testIter.next()
    # inverse_data_list = [(first[0],first[1])]
    inverse_data_list = []
    for i in range(batch_size):
        first = testIter.next()
        inverse_data_list.append((first[0],first[1]))

    dataset = ListDataset(inverse_data_list)
    inverseloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 1)
    return inverseloader

# mnist和cifar的预处理？ （类似transform）
def preprocess_cifar10(data):

    size = data.shape
    NChannels = size[-1]
    assert NChannels == 1 or NChannels == 3
    if NChannels == 1:
        mu = 0.5
        sigma = 0.5
    elif NChannels == 3:
        mu = [0.485, 0.456, 0.406]
        sigma = [0.229, 0.224, 0.225]
    data = (data - mu) / sigma

    assert data.shape == size
    return data

# 输入一张图片 detransform
# 目前只是一个去normalization的工作，得到的图片会是[0,1]之间的(totensor之后的)
def deprocess(data): # CIFAR10

    assert len(data.size()) == 4

    BatchSize = data.size()[0]
    assert BatchSize == 1 # 如果大于1张图片就不行

    NChannels = data.size()[1] # 通道
    if NChannels == 1:
        mu = torch.tensor([0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5], dtype=torch.float32)
    elif NChannels == 3:
        # normalize
        mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

        # process
        # mu = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
        # sigma = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)
    else:
        print("Unsupported image in deprocess()")
        exit(1)

    Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
    return Unnormalize(data)
    # return Unnormalize(data[0,:,:,:]).unsqueeze(0)
    # return clip(Unnormalize(data[0,:,:,:]).unsqueeze(0))
