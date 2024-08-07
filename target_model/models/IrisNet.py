'''
Author: yjr && 949804347@qq.com
Date: 2023-09-09 20:31:05
LastEditors: Ruijun Deng
LastEditTime: 2024-08-06 22:37:57
FilePath: /PP-Split/target_model/models/IrisNet.py
Description: 
'''
import torch
import torch.nn as nn
import collections
import torch.optim as optim

split_layer_list = []

Iris_cfg = [
    ('D',4,128), # 0
    ('Tanh'), # 1
    ('D',128,64), # 2
    ('Tanh'), # 3
    ('D',64,3), # 4
    ('Tanh'), # 5
]
# Iris_cfg = [
#     ('D',4,64), # 0
#     ('Tanh'), # 1
#     ('D',64,64), # 2
#     ('Tanh'), # 3
#     ('D',64,32), # 4
#     ('Tanh'), # 5
#     ('D',32,12), # 6
#     ('Tanh'), # 7
#     ('D',12,3), # 8
#     ('Tanh'), # 9
# ]


class IrisNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=3, layer=5, noise_scale=0):
        super().__init__()
        # self.layers = nn.Sequential()
        linear_idx, lr_idx = 1,1
        assert layer < len(Iris_cfg)
        for i, component in enumerate(Iris_cfg):
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
            self._noise = torch.distributions.Laplace(0.0, self.noise_scale)
            return in_+self._noise.sample(in_.size()).to(in_.device)
        return in_



class IrisNetDecoder(nn.Module):
    def __init__(self,layer):
        Iris_inv_cfg = Iris_cfg[:layer+1][::-1]
        print(Iris_inv_cfg)
        super().__init__()
        linear_idx = 1
        assert layer < len(Iris_cfg)
        for i, component in enumerate(Iris_inv_cfg):
            if component[0]=='D':
            #     out_,in_ = component[1],component[2]
            #     self.add_module(f"delinear{linear_idx}",torch.nn.Linear(in_,out_))
            #     self.add_module(f"ReLU{linear_idx}",torch.nn.ReLU())
            #     linear_idx+=1
                if i==len(Iris_inv_cfg)-1: # 最后一层不要激活函数？
                    out_,in_ = component[1],component[2]
                    self.add_module(f"delinear{linear_idx}",torch.nn.Linear(in_,out_)) 
                else:                  
                    out_,in_ = component[1],component[2]
                    self.add_module(f"delinear{linear_idx}",torch.nn.Linear(in_,out_))
                    self.add_module(f"Tanh{linear_idx}",torch.nn.Tanh())
                linear_idx+=1

    def forward(self, x):
        in_ = x
        for layer in self.children():
            in_ = layer(in_)
        return in_



