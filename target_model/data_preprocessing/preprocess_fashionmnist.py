'''
Author: Ruijun Deng
Date: 2024-09-28 01:37:51
LastEditTime: 2024-09-28 02:02:17
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/data_preprocessing/preprocess_fashionmnist.py
Description: 
'''

# 导包
import torch

# 导包
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset



def get_fmnist_normalize(batch_size = 1,test_bs = None, mu = None, sigma=None):
    if test_bs == None:
        test_bs = batch_size
    # 图像归一化
    if mu == None:
        mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    if sigma == None:
        sigma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        
    transform = transforms.Compose(
        [
            transforms.ToTensor(), # 数据中的像素值转换到0～1之间
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 接近+-1？ 从[0,1] 不是从[0,255]
            # transforms.Normalize(mu.tolist(), sigma.tolist())
        # ])

    # 数据集加载：
    trainset = torchvision.datasets.FashionMNIST(root='/home/dengruijun/data/project/data/fmnist/', train=True,
                                            download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=False, num_workers=4)

    testset = torchvision.datasets.FashionMNIST(root='/home/dengruijun/data/project/data/fmnist/', train=False,
                                        download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs,shuffle=False, num_workers=4)
    
    return trainloader,testloader