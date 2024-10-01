'''
Author: Ruijun Deng
Date: 2024-09-28 01:37:27
LastEditTime: 2024-09-29 05:18:50
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/data_preprocessing/preprocess_mnist.py
Description: 
'''

# 导包
import torch

# 导包
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset



def get_mnist_normalize(batch_size = 1,test_bs = None, mu = None, sigma=None):
    if test_bs == None:
        test_bs = batch_size
    # 图像归一化
    if mu == None:
        mu = torch.tensor([0.5], dtype=torch.float32)
    if sigma == None:
        sigma = torch.tensor([0.5], dtype=torch.float32)
        
    transform = transforms.Compose(
        [
            transforms.ToTensor(), # 数据中的像素值转换到0～1之间
            transforms.Normalize((0.5), (0.5))]) # 接近+-1？ 从[0,1] 不是从[0,255]
            # transforms.Normalize(mu.tolist(), sigma.tolist())
        # ])

    # 数据集加载：
    trainset = torchvision.datasets.MNIST(root='/home/dengruijun/data/project/data/mnist/', train=True,
                                            download=False, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=False, num_workers=4)

    testset = torchvision.datasets.MNIST(root='/home/dengruijun/data/project/data/mnist/', train=False,
                                        download=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs,shuffle=False, num_workers=4)
    
    return trainloader,testloader


def deprocess_mnist(data): # CIFAR10

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