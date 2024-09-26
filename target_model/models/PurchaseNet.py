'''
Author: yjr && 949804347@qq.com
Date: 2023-11-19 15:05:05
LastEditTime: 2024-09-26 05:34:45
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/models/PurchaseNet.py
Description: 
'''
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
import sys
import urllib
import pickle
import tarfile
import collections

# split_layer_list = [0,1,2,3,4,5,6,7,8]
split_layer_list = [1,3,5,7]

purchase_cfg =[
    ('D',600,1024), # 0
    ('Tanh',), #1 
    ('D',1024,512), #2
    ('Tanh',), #3 
    ('D',512,256), #4
    ('Tanh',), #5 
    ('D',256,128), #6
    ('Tanh',), #7 
    ('D',128,100), #8
]

class PurchaseClassifierClient(nn.Module):
    def __init__(self,num_classes=100,layer=9, input_dim = 600):
        super(PurchaseClassifierClient, self).__init__()
        self.layerDict = collections.OrderedDict()
    
        self.linear1=nn.Linear(600,1024)
        self.Tanh1=nn.Tanh()
        self.linear2=nn.Linear(1024,512)
        self.Tanh2=nn.Tanh() # cut
        
        self.layerDict[0]=self.linear1
        self.layerDict[1]=self.Tanh1
        self.layerDict[2]=self.linear2
        self.layerDict[3]=self.Tanh2 # cut


    def forward(self,x):
        # hidden_out = self.features(x)
        # return self.classifier(hidden_out)
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x            
    

class PurchaseClassifier1(nn.Module):
    def __init__(self, input_dim=63, output_dim=1, layer=8, noise_scale=0) -> None:
        super().__init__()
        # self.layers = nn.Sequential()
        linear_idx, lr_idx = 1,1
        assert layer < len(purchase_cfg)
        for i, component in enumerate(purchase_cfg):
            if i > layer:
                break
            if component[0]=='D':
                in_,out_ = component[1],component[2]
                self.add_module(f"linear{linear_idx}",torch.nn.Linear(in_,out_))
                linear_idx+=1
            elif component[0]=='Tanh':
                self.add_module(f"Tanh{lr_idx}",torch.nn.Tanh())
                lr_idx+=1
        self.noise_scale = noise_scale

    def forward(self,x):
        in_ = x
        for layer in self.children():
            in_ = layer(in_)
        if self.noise_scale!=0: # 需要加laplace noise   
            # self._noise = torch.distributions.Laplace(0.0, self.noise_scale)
            # return in_+self._noise.sample(in_.size()).to(in_.device)
            self._noise= torch.randn_like(in_) * self.noise_scale
            return in_+self._noise
        return in_
    

class PurchaseClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(PurchaseClassifier, self).__init__()
        
        # self.features = nn.Sequential(
        #     nn.Linear(600,1024),
        #     nn.Tanh(),
        #     nn.Linear(1024,512),
        #     nn.Tanh(),
        #     nn.Linear(512,256),
        #     nn.Tanh(),
        #     nn.Linear(256,128),
        #     nn.Tanh(),
        # )
        # self.classifier = nn.Linear(128,num_classes)

        self.layerDict = collections.OrderedDict()

        self.linear1=nn.Linear(600,1024)
        self.Tanh1=nn.Tanh()
        self.linear2=nn.Linear(1024,512)
        self.Tanh2=nn.Tanh() # cut
        self.linear3=nn.Linear(512,256)
        self.Tanh3=nn.Tanh()
        self.linear4=nn.Linear(256,128)
        self.Tanh4=nn.Tanh()
        self.linear5=nn.Linear(128,num_classes)
        
        self.layerDict[0]=self.linear1
        self.layerDict[1]=self.Tanh1
        self.layerDict[2]=self.linear2
        self.layerDict[3]=self.Tanh2 # cut
        self.layerDict[4]=self.linear3
        self.layerDict[5]=self.Tanh3
        self.layerDict[6]=self.linear4
        self.layerDict[7]=self.Tanh4
        self.layerDict[8]=self.linear5

    def forward(self,x):
        # hidden_out = self.features(x)
        # return self.classifier(hidden_out)
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x            
    
    def getLayerOutput(self, x, targetLayer):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
            if layer == targetLayer:
                return x

# blackbox 网络  从layer2 重建的网络
class PurchaseDecoder(nn.Module):  # cifar10 relu22 decoder网络 （目前结构就完全是和edge net的相反的层）
    def __init__(self):
        super(PurchaseDecoder, self).__init__()

        self.layerDict = collections.OrderedDict()

        self.delinear1 = nn.Linear(512, 1024)
        self.layerDict['delinear1'] = self.delinear1

        self.Tanh1 = nn.Tanh()
        self.layerDict['Tanh1'] = self.Tanh1

        self.delinear2 = nn.Linear(1024, 600)
        self.layerDict['delinear2'] = self.delinear2

        self.Tanh2 = nn.Tanh()
        self.layerDict['Tanh2'] = self.Tanh2

    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x


class PurchaseDecoder1(nn.Module):  # cifar10 relu22 decoder网络 （目前结构就完全是和edge net的相反的层）
    def __init__(self,layer):
        purchase_inv_cfg = purchase_cfg[:layer+1][::-1]
        print(purchase_inv_cfg)
        super().__init__()
        linear_idx = 1
        assert layer < len(purchase_cfg)
        for i, component in enumerate(purchase_inv_cfg):
            if component[0]=='D':
                out_,in_ = component[1],component[2]
                self.add_module(f"delinear{linear_idx}",torch.nn.Linear(in_,out_))
                self.add_module(f"Tanh{linear_idx}",torch.nn.Tanh())
                linear_idx+=1

    def forward(self, x):
        in_ = x
        for layer in self.children():
            in_ = layer(in_)
        return in_


