'''
Author: Ruijun Deng
Date: 2024-03-07 16:22:11
LastEditTime: 2024-03-13 16:35:27
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/HFL/DLG.py
Description: DLG论文
'''
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, datasets, transforms
# print(torch.__version__, torchvision.__version__)

class DLGAttack():
    def __init__(self,gpu=True) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
        self.tt = transforms.ToTensor()
        self.tp = transforms.ToPILImage()

    def _label_to_onehot(self, target, num_classes=100):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target

    def _cross_entropy_for_onehot(self, pred, target):
        return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
    
    def reconstruct(self, net, data, NEpoch=300): # data 是一个样本
        criterion = self._cross_entropy_for_onehot
        net.to(self.device)

        # 数据处理
        gt_data = data[0].to(self.device) # org_img
        gt_label = torch.Tensor([data[1]]).long().to(self.device) # label获取 本来只是一个数字

        gt_data = gt_data.view(1, *gt_data.size()) # 拉平
        gt_label = gt_label.view(1, ) # 拉平
        gt_onehot_label = self._label_to_onehot(gt_label) # onehot 为啥要用 onehot 为了优化 # 84位为0，其他为1

        # plt.imshow(tp(gt_data[0].cpu())) # 展示图片

        # 原始梯度计算
        # compute original gradient 
        pred = net(gt_data)
        y = criterion(pred, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        # 哑数据&哑label
        # generate dummy data and label
        dummy_data = torch.randn(gt_data.size()).to(self.device).requires_grad_(True)
        dummy_label = torch.randn(gt_onehot_label.size()).to(self.device).requires_grad_(True)

        # plt.imshow(tp(dummy_data[0].cpu())) # 展示哑数据

        # 优化
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
        history = []
        for iters in range(NEpoch): # 先定义300个迭代 # 默认300个迭代
            def closure():
                optimizer.zero_grad()

                dummy_pred = net(dummy_data) 
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
                    grad_diff += ((gx - gy) ** 2).sum() # loss 
                grad_diff.backward()
                
                return grad_diff
            
            optimizer.step(closure)
            if iters % 10 == 0: 
                current_loss = closure()
                print("Iters: ",iters, " Current Loss: %.4f" % current_loss.item())
                history.append(self.tp(dummy_data[0].cpu()))
        return [dummy_data, dummy_label], history

    def show_optimization_process(self,history):
        # 画图，画出优化后的图
        plt.figure(figsize=(12, 8))
        for i in range(30):
            plt.subplot(3, 10, i + 1)
            plt.imshow(history[i])
            plt.title("iter=%d" % (i * 10))
            plt.axis('off')

        plt.show()


