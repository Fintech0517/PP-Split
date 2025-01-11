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

import json

import collections

# unit_net, client_net
def split_weights_client(weights,cweights,no_dense=False):
    # print('client_net weights: ', weights.keys())
    # print('client_net cweights: ', cweights.keys())
    # print('len client_net weights: ', len(weights.keys()))
    # print('len client_net cweights: ', len(cweights.keys()))
    for key in cweights:
        print(key)
        if no_dense and 'dense' in key:
            continue
        else:
            cweights[key] = weights[key]
    return cweights


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
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# Histogram相关
# prob smashed data的分布 输入是张亮
def plot_smashed_distribution(smashed_data,start = -1, end = 1):
    import matplotlib.pyplot as plt
    import numpy as np
    
    data = smashed_data.flatten(1) # 拉平后的数据
    # data_sort = np.sort(data) # 排序后的拉平数据

    plot_array_distribution(data,start, end)

def plot_array_distribution(data,start = -1, end = 1, notes=''):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # counts, buckets = np.histogram(data, bins=100, range=(start, end), density=True) 
    # 也许这个更加适合FCN？
    print("data.size():",np.size(data))
    # counts, buckets = np.histogram(data, bins=100 if 100<np.size(data) else np.size(data), density=True) 
    counts, buckets = np.histogram(data, bins=100, density=True) 

    # 画图 value-probability 图2
    plt.figure()
    counts = counts/np.sum(counts) # 如果start和end不是[0，1]就都要用
    # edges = np.hstack((buckets,np.array([buckets[-1]+(buckets[1]-buckets[0])]))) # linespace
    edges = buckets # histogram
    plt.stairs(counts,edges,fill=True,color='red', alpha=0.5)
    plt.title('smashed data histogram')
    plt.xlabel('value')
    plt.ylabel('probability')

    plt.tight_layout()  # 自动调整子图布局
    plt.show()
    # plt.savefig(f'smashed_data_distribution{time.time()}.png')
    plt.savefig(f'data_distribution{notes}.png')

    # 打印信息
    print("sigma(prob):",np.sum(counts)) # 查看counts的量
    print("counts[0],counts[-1],counts[49]",counts[0],counts[-1],counts[len(counts)//2])


def plot_index_value(smashed_data): 
    import matplotlib.pyplot as plt
    import numpy as np
    
    data = smashed_data.flatten(dim=1) # 拉平后的数据
    data_sort = np.sort(data) # 排序后的拉平数据

    print("data.size():",np.size(data))
    # counts, buckets = np.histogram(data, bins=100, range=(start, end), density=True) # 也许这个更加适合FCN？
    # counts, buckets = np.histogram(data, bins=100 if 100<np.size(data) else np.size(data), density=True) 
    counts, buckets = np.histogram(data, bins=10, density=True) 

    # 画图 index-value 图1
    plt.figure(figsize=(10,3))
    x_axis = np.arange(0, len(data), 1)
    plt.plot(x_axis, data_sort)
    plt.title('smashed data distribution')
    plt.xlabel('index')
    plt.ylabel('value')

    plt.tight_layout()  # 自动调整子图布局
    plt.show()
    # plt.savefig(f'smashed_data_distribution{time.time()}.png')

# json文件的读取
def load_json(file_path):
    """读取已保存的爬取结果"""
    # filename = f'./dblp-results/{author_name}_publications.json'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_json(data_dict,file_path):
    """保存爬取结果到JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
