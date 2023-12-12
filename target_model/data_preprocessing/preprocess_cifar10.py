'''
Author: Ruijun Deng
Date: 2023-09-03 19:29:00
LastEditTime: 2023-12-12 20:21:15
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
    trainset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)
    
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
    trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train = True,
                                    download=False, transform = tsf['train'])
    testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train = False,
                                    download=False, transform = tsf['test'])


    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                        shuffle = False, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                        shuffle = False, num_workers = 1)
    return trainloader,testloader

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


# 构造数据集 CIFAR10适用
class ListDataset(Dataset):
    def __init__(self, data_list) -> None:
        super().__init__()
        self.x = [t[0] for t in data_list]
        self.y = [t[1] for t in data_list]

    def __getitem__(self, index):  # 张量压缩一个维度
        x = torch.squeeze(self.x[index], 0)
        y = torch.squeeze(self.y[index], 0)
        return x, y

    def __len__(self):
        return len(self.x)
