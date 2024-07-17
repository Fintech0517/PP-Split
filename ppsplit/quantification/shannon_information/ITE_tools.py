'''
Author: Ruijun Deng
Date: 2024-07-15 21:26:40
LastEditTime: 2024-07-15 21:35:09
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/quantification/shannon_information/ITE_tools.py
Description: 
'''
# 导包
import torch
import torch.nn as nn

import sys
sys.path.append('../')
import numpy as np
import pandas as pd

# ite mutual information
sys.path.append('/home/dengruijun/data/software/ITE/ite-in-python') # 这个首先要手动下载一下ite的库
import ite

import os
os.environ['NUMEXPR_MAX_THREADS'] = '48'

from numpy.random import rand

def Shannon_quantity(inputs):
    co1 = ite.cost.BHShannon_KnnK()
    reshaped_x = inputs.flatten(start_dim=1).detach().cpu().numpy()
    entr = co1.estimation(reshaped_x)
    return entr

