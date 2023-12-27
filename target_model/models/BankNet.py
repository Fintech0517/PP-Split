'''
Author: yjr && 949804347@qq.com
Date: 2023-09-09 20:31:05
LastEditors: Ruijun Deng
LastEditTime: 2023-12-20 14:36:00
FilePath: /PP-Split/target_model/models/BankNet.py
Description: 
'''
import torch
import torch.nn as nn
import collections
import torch.optim as optim

split_layer_list = [0,2,4,6]

# bank_cfg = [(63,128),'LeakyReLU',(128,64),'LeakyReLU',(64,16),'LeakyReLU',(16,1)]
bank_cfg = [
    ('D',63,128), # 0
    ('LR'), #1 
    ('D',128,64), #2 
    ('LR'), # 3
    ('D',64,16), #4
    ('LR'), # 5
    ('D',16,1) # 6
]

# edge network
class BankNet(nn.Module):
    def __init__(self, input_dim=63, output_dim=1, layer=7): # zhongyuan 38维度+1 is_default
        super(BankNet, self).__init__()
        self.layerDict = collections.OrderedDict()

        # 先随便写的网络 # dnn，输入为37维度的特征向量
        self.linear1 = nn.Linear(input_dim, 128)
        self.layerDict['linear1'] = self.linear1 # cut 1

        self.ReLU1 = nn.LeakyReLU()
        self.layerDict['ReLU1'] = self.ReLU1 # 1 # 

        self.linear2 = nn.Linear(128, 64)  # 2
        self.layerDict['linear2'] = self.linear2 # cut 2

        self.ReLU2 = nn.LeakyReLU()
        self.layerDict['ReLU2'] = self.ReLU2 
        self.linear3 = nn.Linear(64, 16)
        self.layerDict['linear3'] = self.linear3 # cut 3

        self.ReLU3 = nn.LeakyReLU()
        self.layerDict['ReLU3'] = self.ReLU3 

        self.linear4 = nn.Linear(16, output_dim)
        self.layerDict['linear4'] = self.linear4 # cut 4

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

class BankNet1(nn.Module):
    def __init__(self, input_dim=63, output_dim=1, layer=6) -> None:
        super().__init__()
        # self.layers = nn.Sequential()
        linear_idx, lr_idx = 1,1
        assert layer < len(bank_cfg)
        for i, component in enumerate(bank_cfg):
            if i > layer:
                break
            if component[0]=='D':
                in_,out_ = component[1],component[2]
                self.add_module(f"linear{linear_idx}",torch.nn.Linear(in_,out_))
                linear_idx+=1
            elif component[0]=='LR':
                self.add_module(f"ReLU{lr_idx}",torch.nn.LeakyReLU())
                lr_idx+=1

        # 删除最后一个激活层
        # if layer == 2:
            # delattr (self,"ReLU2")

    def forward(self,x):
        in_ = x
        for layer in self.children():
            in_ = layer(in_)
        return in_
    

# blackbox 网络  从layer2 重建的网络
class BankNetDecoder(nn.Module):  # cifar10 relu22 decoder网络 （目前结构就完全是和edge net的相反的层）
    def __init__(self):
        super(BankNetDecoder, self).__init__()

        self.layerDict = collections.OrderedDict()

        self.delinear1 = nn.Linear(64, 128)
        self.layerDict['delinear1'] = self.delinear1

        self.ReLU1 = nn.ReLU()
        self.layerDict['ReLU1'] = self.ReLU1

        self.delinear2 = nn.Linear(128, 63)
        self.layerDict['delinear2'] = self.delinear2

    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x

class BankNetDecoder1(nn.Module):
    def __init__(self,layer):
        bank_inv_cfg = bank_cfg[:layer+1][::-1]
        print(bank_inv_cfg)
        super().__init__()
        linear_idx = 1
        assert layer < len(bank_cfg)
        for i, component in enumerate(bank_inv_cfg):
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

