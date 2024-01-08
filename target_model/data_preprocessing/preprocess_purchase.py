'''
Author: yjr && 949804347@qq.com
Date: 2023-11-18 14:13:20
LastEditTime: 2024-01-03 16:09:26
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/data_preprocessing/preprocess_purchase.py
Description: 
'''
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
import sys
import urllib
import pickle
import tarfile

# 从路径获取avazu数据集loader


# numpy转为tensor数据集
def tensor_data_create(features, labels):
    tensor_x = torch.stack([torch.FloatTensor(i) for i in features]) # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:,0]
    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    return dataset

# purchase数据集与处理，和那个前面的purchase_defended/undefended里面的功能一样
def preprocess_purchase(data_path='/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/Purchase100/',batch_size=1):
    print('purchase100 dataset processing...')
    # 文件夹路径
    # DATASET_PATH='./datasets/purchase'
    # home_dir = 
    DATASET_PATH=data_path # dir目录
    DATASET_NAME= 'dataset_purchase' # 原始数据压缩文件
    DATASET_NUMPY = 'data.npz' # 简单处理后的numpy存储
    print('datset route:', DATASET_PATH+'/data.npz')

    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    
    DATASET_FILE = os.path.join(DATASET_PATH,DATASET_NAME)
    
    if not os.path.isfile(DATASET_FILE): # 如果数据没下载
        # 下载官网数据集保存到tmp.tgz, 解压
        print('Dowloading the dataset...')
        # simplified and preprocessed version: 197324 rows
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",os.path.join(DATASET_PATH,'tmp.tgz'))
        print('Dataset Dowloaded')
        tar = tarfile.open(os.path.join(DATASET_PATH,'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)

        #读取数据集以numpy的形式存储到data.npz
        print('reading dataset...')
        data_set =np.genfromtxt(DATASET_FILE,delimiter=',') # 从txt构造np array
        print('finish reading!')
        X = data_set[:,1:].astype(np.float64) # 特征
        Y = (data_set[:,0]).astype(np.int32)-1 # label
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y) # 保存数据
    
    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY),allow_pickle=True) # 加载np数据
    X = data['X']
    Y = data['Y']
    len_train =len(X) # 数据样本量
    print("original dataset shape: ", X.shape)

    # 随机来的index提取数据，分割X和Y 【不改遍数据量】
    r = np.load(DATASET_PATH+'random_r_purchase100.npy')
    X=X[r]
    Y=Y[r]
    print('After random selection, dataset shape: ', X.shape)

    # 训练分类启的数据比例，训练attack的分类比例 【改变数据量为0.1】
    # classifier和attack network的区别？
    train_ratio = 0.8
    train_data = X[:int(train_ratio*len_train)]
    test_data = X[int((train_ratio)*len_train):]
    
    train_label = Y[:int(train_ratio*len_train)]
    test_label = Y[int((train_ratio)*len_train):]
    
    np.random.seed(100)
    # train_len = train_data.shape[0]
    print("After split between classifier and attack: ")
    print("training dataset shape: ", train_data.shape) # 0.1 of all
    print("testing dataset shape: ", test_data.shape)  # 0.4 of all

    # 数据在  target model 和shadow model 之间的分割 【一半】
    # # training
    # r = np.arange(train_len)
    # np.random.shuffle(r)
    # shadow_indices = r[:train_len//2]
    # target_indices = r[train_len//2:]

    # shadow_train_data, shadow_train_label = train_data[shadow_indices], train_label[shadow_indices]
    # target_train_data, target_train_label = train_data[target_indices], train_label[target_indices]

    # testing
    # test_len = 1*train_len
    # r = np.arange(test_len)
    # np.random.shuffle(r)
    # shadow_indices = r[:test_len//2] # 后一半
    # target_indices = r[test_len//2:] # 前一半
    
    # shadow_test_data, shadow_test_label = test_data[shadow_indices], test_label[shadow_indices]
    # target_test_data, target_test_label = test_data[target_indices], test_label[target_indices]

    # print("After split between target and shadow, dataset shape: ")
    # print("target training dataset size: ", target_train_data.shape)
    # print("target testing dataset size: ", target_test_data.shape)
    # print("shadow training dataset size: ", shadow_train_data.shape)
    # print("shadow testing dataset size: ", shadow_train_data.shape)


    # numpy数组转为tensor dataset 加入dataloader
    # shadow_train = tensor_data_create(shadow_train_data, shadow_train_label)
    # shadow_train_loader = torch.utils.data.DataLoader(shadow_train, batch_size=batch_size, shuffle=True, num_workers=1)

    # shadow_test = tensor_data_create(shadow_test_data, shadow_test_label)
    # shadow_test_loader = torch.utils.data.DataLoader(shadow_test, batch_size=batch_size, shuffle=True, num_workers=1)

    train = tensor_data_create(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8,drop_last=True)

    test = tensor_data_create(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8,drop_last=True)
    print('Data loading finished')
    
    return train_loader, test_loader

# sec'21的accuracy函数
def accuracy(output, target, topk=(1,)): 
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

def preprocess_purchase_shadow(data_path='/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/Purchase100/', batch_size=100):
    DATASET_PATH=data_path # dir目录
    DATASET_NAME= 'dataset_purchase' # 原始数据压缩文件
    DATASET_NUMPY = 'data.npz' # 简单处理后的numpy存储
    print('datset route:', DATASET_PATH+'/data.npz')

    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    
    DATASET_FILE = os.path.join(DATASET_PATH,DATASET_NAME)
    
    if not os.path.isfile(DATASET_FILE): # 如果数据没下载
        # 下载官网数据集保存到tmp.tgz, 解压
        print('Dowloading the dataset...')
        # simplified and preprocessed version: 197324 rows
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",os.path.join(DATASET_PATH,'tmp.tgz'))
        print('Dataset Dowloaded')
        tar = tarfile.open(os.path.join(DATASET_PATH,'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)

        #读取数据集以numpy的形式存储到data.npz
        print('reading dataset...')
        data_set =np.genfromtxt(DATASET_FILE,delimiter=',') # 从txt构造np array
        print('finish reading!')
        X = data_set[:,1:].astype(np.float64) # 特征
        Y = (data_set[:,0]).astype(np.int32)-1 # label
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y) # 保存数据
    
    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    X = data['X']
    Y = data['Y']
    len_train =len(X)
    r = np.load('./dataset_shuffle/random_r_purchase100.npy')

    X=X[r]
    Y=Y[r]
        
    train_classifier_ratio, train_attack_ratio = 0.1,0.3
    train_data = X[:int(train_classifier_ratio*len_train)]
    test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    
    train_label = Y[:int(train_classifier_ratio*len_train)]
    test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    
    np.random.seed(100)
    train_len = train_data.shape[0]
    r = np.arange(train_len)
    np.random.shuffle(r)
    shadow_indices = r[:train_len//2]
    target_indices = r[train_len//2:]

    shadow_train_data, shadow_train_label = train_data[shadow_indices], train_label[shadow_indices]
    target_train_data, target_train_label = train_data[target_indices], train_label[target_indices]

    test_len = 1*train_len
    r = np.arange(test_len)
    np.random.shuffle(r)
    shadow_indices = r[:test_len//2]
    target_indices = r[test_len//2:]
    
    shadow_test_data, shadow_test_label = test_data[shadow_indices], test_label[shadow_indices]
    target_test_data, target_test_label = test_data[target_indices], test_label[target_indices]

    shadow_train = tensor_data_create(shadow_train_data, shadow_train_label)
    shadow_train_loader = torch.utils.data.DataLoader(shadow_train, batch_size=batch_size, shuffle=True, num_workers=1)

    shadow_test = tensor_data_create(shadow_test_data, shadow_test_label)
    shadow_test_loader = torch.utils.data.DataLoader(shadow_test, batch_size=batch_size, shuffle=True, num_workers=1)

    target_train = tensor_data_create(target_train_data, target_train_label)
    target_train_loader = torch.utils.data.DataLoader(target_train, batch_size=batch_size, shuffle=True, num_workers=1)

    target_test = tensor_data_create(target_test_data, target_test_label)
    target_test_loader = torch.utils.data.DataLoader(target_test, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Data loading finished')
    return shadow_train_loader, shadow_test_loader, target_train_loader, target_test_loader