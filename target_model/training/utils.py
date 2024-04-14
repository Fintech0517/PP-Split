'''
Author: Ruijun Deng
Date: 2024-04-14 15:58:18
LastEditTime: 2024-04-14 16:58:06
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/training/utils.py
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

import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
# from ...ppsplit.utils.similarity_metrics import SimilarityMetrics
from ppsplit.utils.similarity_metrics import SimilarityMetrics

def acc(pred_logits, true_label):
    pred = np.argmax(pred_logits.cpu().detach().numpy(), axis = 1)
    groundTruth = true_label.cpu().detach().numpy()
    return np.mean(pred == groundTruth)

# 评估edge模型测试精度
def evalTest(testloader, net, gpu = True):
    testIter = iter(testloader)
    acc_test = 0.0
    NBatch = 0
    for i, data in enumerate(tqdm.tqdm(testIter, 0)):
        NBatch += 1
        batchX, batchY = data
        if gpu:
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        logits = net.forward(batchX)

        acc_test+=acc(logits, batchY)

    acc_test = acc_test / NBatch
    return acc_test

def evalTest_tab_acc(testloader, net, gpu = True):
    testIter = iter(testloader)
    acc_test = 0.0
    NBatch = 0

    all_preds = []
    all_groundTruth = []

    sim = SimilarityMetrics()
    for i, data in enumerate(tqdm.tqdm(testIter, 0)):
        NBatch += 1
        batchX, batchY = data
        if gpu:
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        logits = net.forward(batchX)

        pred = torch.sigmoid(logits)
        pred = pred.cpu().detach().numpy()
        groundTruth = batchY.cpu().detach().numpy()

        all_preds.extend(pred)
        all_groundTruth.extend(groundTruth)

        acc = sim.accuracy(y_targets=groundTruth, y_prob_preds=pred)

        acc_test += acc

    acc_test = acc_test / NBatch
    auc_score = roc_auc_score(all_groundTruth, all_preds)

    return acc_test, auc_score

# sec'21的accuracy函数
def accuracy_purchase(output, target, topk=(1,)): 
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0) # batch大小

    # 找到k大的？# make sure that the returned k elements are themselves sorted 
    # # pred是他们的 indices？ 找max，看和target是不是一样
    # topk这个东西，每个数据找到前5个
    _, pred = output.topk(maxk, 1, True, True) 
    pred = pred.t() #转秩
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk: # 100个类，选前5个
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def gradient_gaussian_noise_masking(g, ratio):
    g_norm = torch.norm(g, p=2, dim=1)
    max_norm = torch.max(g_norm)
    gaussian_std = ratio * max_norm/torch.sqrt(torch.tensor(g.shape[1], dtype=torch.float))
    gaussian_noise = torch.normal(mean=0.0, std=gaussian_std, size=g.shape).cuda()
    # res = [g[0][0]+gaussian_noise]
    return g + gaussian_noise