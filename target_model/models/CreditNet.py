'''
Author: yjr && 949804347@qq.com
Date: 2023-09-26 20:45:13
LastEditTime: 2024-01-08 19:56:43
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/models/CreditNet.py
Description:
'''

import torch
import torch.nn as nn
import collections
import torch.optim as optim
import torch.nn.init as init

split_layer_list = [0,3,6,9]

credit_cfg = [
    ('D',250,512), # 0
    ('BN',512), #1 
    ('LR',), #2 
    ('D',512,128), #3
    ('BN',128), #4 
    ('LR',), #5 
    ('D',128,32), #6
    ('BN',32), #7 
    ('LR',), #8
    ('D',32,1), #9
]


# edge network
class CreditNet(nn.Module):
    def __init__(self, input_dim=250, output_dim=1): # zhongyuan 38维度+1 is_default
        super(CreditNet, self).__init__()
        self.layerDict = collections.OrderedDict()

        # 先随便写的网络 # dnn，输入为37维度的特征向量
        self.linear1 = nn.Linear(input_dim, 512)
        self.layerDict['linear1'] = self.linear1

        self.batch_norm1 = nn.BatchNorm1d(512)
        self.layerDict['1BN'] = self.batch_norm1

        self.ReLU1 = nn.LeakyReLU()
        self.layerDict['ReLU1'] = self.ReLU1

        self.linear2 = nn.Linear(512, 128)
        self.layerDict['linear2'] = self.linear2

        self.batch_norm2 = nn.BatchNorm1d(128)
        self.layerDict['2BN'] = self.batch_norm2

        self.ReLU2 = nn.LeakyReLU()
        self.layerDict['ReLU2'] = self.ReLU2

        self.linear3 = nn.Linear(128, 32)
        self.layerDict['linear3'] = self.linear3

        self.batch_norm3 = nn.BatchNorm1d(32)
        self.layerDict['3BN'] = self.batch_norm3

        self.ReLU3 = nn.LeakyReLU()
        self.layerDict['ReLU3'] = self.ReLU3

        self.linear4 = nn.Linear(32, output_dim)
        self.layerDict['linear4'] = self.linear4

        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x

    def getLayerOutput(self, x, targetLayer):
        if targetLayer == 'input':
            return x

        for layer in self.layerDict:
            x = self.layerDict[layer](x)
            if layer == targetLayer:
                return x

class CreditNet1(nn.Module):
    def __init__(self, input_dim=63, output_dim=1, layer=9) -> None:
        super().__init__()
        # self.layers = nn.Sequential()
        linear_idx, lr_idx, bn_idx = 1,1,1
        assert layer < len(credit_cfg)
        for i, component in enumerate(credit_cfg):
            if i > layer:
                break
            if component[0]=='D':
                in_,out_ = component[1],component[2]
                self.add_module(f"linear{linear_idx}",torch.nn.Linear(in_,out_))
                linear_idx+=1
            elif component[0]=='LR':
                self.add_module(f"ReLU{lr_idx}",torch.nn.LeakyReLU())
                lr_idx+=1
            elif component[0]=='BN':
                self.add_module(f'batch_norm{bn_idx}',torch.nn.BatchNorm1d(component[1]))
                bn_idx+=1
        self.apply(self.initialize_weights)

    def forward(self,x):
        in_ = x
        for layer in self.children():
            in_ = layer(in_)
        return in_
    
    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

# blackbox 网络  从layer2 重建的网络
class CreditNetDecoder(nn.Module):  # cifar10 relu22 decoder网络 （目前结构就完全是和edge net的相反的层）
    def __init__(self):
        super(CreditNetDecoder, self).__init__()

        self.layerDict = collections.OrderedDict()

        self.delinear1 = nn.Linear(128, 512)
        self.layerDict['delinear1'] = self.delinear1

        self.ReLU1 = nn.ReLU()
        self.layerDict['ReLU1'] = self.ReLU1

        self.delinear2 = nn.Linear(512, 250)
        self.layerDict['delinear2'] = self.delinear2

    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x


class CreditNetDecoder1(nn.Module):
    def __init__(self,layer):
        credit_inv_cfg = credit_cfg[:layer+1][::-1]
        print(credit_inv_cfg)
        super().__init__()
        linear_idx = 1
        assert layer < len(credit_cfg)
        for i, component in enumerate(credit_inv_cfg):
            if component[0]=='D':
                out_,in_ = component[1],component[2]
                self.add_module(f"delinear{linear_idx}",torch.nn.Linear(in_,out_))
                self.add_module(f"ReLU{linear_idx}",torch.nn.ReLU())
                linear_idx+=1

    def forward(self, x):
        in_ = x
        for layer in self.children():
            in_ = layer(in_)
        return in_