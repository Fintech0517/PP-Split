'''
Author: Ruijun Deng
Date: 2023-12-12 20:28:05
LastEditTime: 2024-12-08 03:26:33
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
import collections


# unit_net, client_net
def split_weights_client(weights,cweights,no_dense=False):
    # print('unit_net weights: ', weights.keys())
    # print('client_net cweights: ', cweights.keys())
    # print('len unit_net weights: ', len(weights.keys()))
    # print('len client_net cweights: ', len(cweights.keys()))
    for key in cweights:
        # print(key)
        if no_dense and 'dense' in key:
            continue
        else:
            cweights[key] = weights[key]
    return cweights

def split_weights_server(weights,cweights,sweights):
    # print('unit_net weights: ', weights.keys())
    # print('server_net cweights: ', sweights.keys())
    # print('len unit_net weights: ', len(weights.keys()))
    # print('len server_net cweights: ', len(sweights.keys()))
    ckeys = list(cweights)
    skeys = list(sweights)
    keys = list(weights)

    for i in range(len(skeys)):
        assert sweights[skeys[i]].size() == weights[keys[i + len(ckeys)]].size()
        sweights[skeys[i]] = weights[keys[i + len(ckeys)]]

    return sweights

# 拼合模型
def concat_weights(weights,cweights,sweights):
	concat_dict = collections.OrderedDict()

	ckeys = list(cweights)
	skeys = list(sweights)
	keys = list(weights)

	for i in range(len(ckeys)):
		concat_dict[keys[i]] = cweights[ckeys[i]]

	for i in range(len(skeys)):
		concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]

	return concat_dict


# 路径
def create_dir(dir_route):
    if not os.path.exists(dir_route):
        os.makedirs(dir_route)
    return

