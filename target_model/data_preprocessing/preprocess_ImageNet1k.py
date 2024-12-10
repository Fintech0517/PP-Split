'''
Author: Ruijun Deng
Date: 2024-12-08 04:13:43
LastEditTime: 2024-12-08 06:21:40
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/data_preprocessing/preprocess_ImageNet1k.py
Description: 
'''
# 导包
import torch

# 导包
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.models import ViT_B_16_Weights


from torchvision.datasets import ImageNet

imagenet_data_dir = '/home/dengruijun/data/project/data/imageNet1k/ok/'

def get_ImageNet1k_valLoader(batch_size = 1, test_bs = None):
    if test_bs is None:
        test_bs = batch_size

    transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    dataset = ImageNet(root=imagenet_data_dir , split='val', transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=test_bs,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
    return dataloader,dataloader
