'''
Author: Ruijun Deng
Date: 2023-12-12 20:28:05
LastEditTime: 2023-12-12 20:48:59
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



def split_weights_client(weights,cweights):
    for key in cweights:
        print(key)
        cweights[key] = weights[key]
    return cweights
