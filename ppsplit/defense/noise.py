'''
Author: Ruijun Deng
Date: 2024-04-14 21:06:17
LastEditTime: 2024-04-14 21:08:37
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/defense/noise.py
Description: 
'''
import torch

class Noise:
    def __init__(self,noise_scale=0.1) -> None:
        self.laplace_noise = torch.distributions.Laplace(0.0, self.noise_scale)
    
    def add_noise(self,data, device):
        noise = self.laplace_noise.sample(data.size()).to(device)
        return data+noise