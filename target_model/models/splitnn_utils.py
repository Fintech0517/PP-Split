'''
Author: Ruijun Deng
Date: 2023-12-12 20:28:05
LastEditTime: 2024-09-02 21:26:33
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/models/splitnn_utils.py
Description: 
'''
import time
import math
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import tqdm



def split_weights_client(weights,cweights,no_dense=False):
    for key in cweights:
        print(key)
        if no_dense and 'dense' in key:
            continue
        else:
            cweights[key] = weights[key]
    return cweights
