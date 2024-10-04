'''
Author: Ruijun Deng
Date: 2024-10-02 07:17:42
LastEditTime: 2024-10-02 07:47:20
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/data_preprocessing/preprocess_cifar100.py
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

from .dataset import ListDataset, split_trainset



# 用0.5来normalize的
def get_cifar100_normalize(batch_size = 1, test_bs = None, mu = None, sigma=None):
    if test_bs == None:
        test_bs = batch_size
    #  数据集 CIFAR
    # 图像归一化
    if mu == None:
        mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    if sigma == None:
        sigma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        
    transform = transforms.Compose(
        [
            transforms.ToTensor(), # 数据中的像素值转换到0～1之间
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 接近+-1？ 从[0,1] 不是从[0,255]
            # transforms.Normalize(mu.tolist(), sigma.tolist())
        ])

    # 数据集加载：
    trainset = torchvision.datasets.CIFAR100(root='/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar100/', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root='/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar100/', train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs,
                                            shuffle=False, num_workers=4)
    
    return trainloader,testloader