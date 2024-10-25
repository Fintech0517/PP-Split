'''
Author: Ruijun Deng
Date: 2024-03-07 14:19:17
LastEditTime: 2024-09-28 04:01:00
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/models/LeNet.py
Description: DLG那篇文章的源码
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

        
class LeNet_MNIST(nn.Module):
    def __init__(self):
        super(LeNet_MNIST, self).__init__()
        act = nn.Tanh
        self.body = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=0, stride=1),
            act(),
            nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )
        # self._weights_init()
        
    def _weights_init(self):
        for m in self.modules():
            if hasattr(m, "weight"):
                m.weight.data.uniform_(-0.5, 0.5)
            if hasattr(m, "bias"):
                m.bias.data.uniform_(-0.5, 0.5)

# for m in self.modules():
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, 0, 0.01)
#         nn.init.constant_(m.bias, 0)
                        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out

