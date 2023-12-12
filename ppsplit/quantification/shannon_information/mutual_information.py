'''
Author: yjr && 949804347@qq.com
Date: 2023-11-15 20:03:50
LastEditTime: 2023-12-12 10:28:04
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/quantification/shannon_information/mutual_information.py
Description: 在整个test数据集上，没有平均之说。
'''

# 导包
import torch
import torch.nn as nn

import sys
sys.path.append('../')
import numpy as np
import pandas as pd

# ite mutual information
sys.path.insert(1,'/home/dengruijun/data/FinTech/VFL/quantification/ite-in-python') # 这个首先要手动下载一下ite的库
import ite

import os
os.environ['NUMEXPR_MAX_THREADS'] = '48'

class MuInfo():
    def __init__(self) -> None:
        pass
    
    def quantify(self, inputs, outputs): 
        # batch形式的inputs和outputs
        # batchsize > = 8
        reshaped_x = inputs.flatten(start_dim=1).detach().cpu().numpy()
        reshaped_z = outputs.flatten(start_dim=1).detach().cpu().numpy()

        co = ite.cost.MIShannon_DKL()
        ds = np.array([reshaped_x.shape[1], reshaped_z.shape[1]])
        y = np.concatenate((reshaped_x,reshaped_z),axis=1)
        mi = co.estimation(y, ds) 
        return mi

if __name__=="__main__":
    import argparse
    
    # 硬件
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'CIFAR10')
    args = parser.parse_args()


    # 获取数据集
    if args.dataset=='CIFAR10':
        save_img_dir = f'../results/1-6/MI/'
        batch_size = 10000 # 10000个数据一次
        trainloader,testloader = get_cifar10_normalize(batch_size = batch_size)
        # one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据

        # VGG5
        model_path = '../results/VGG5/BN+Tanh/VGG5-0ep.pth' # VGG5-BN+Tanh
        vgg5_unit = VGG('Unit', 'VGG5', len(model_cfg['VGG5'])-1, model_cfg) # 加载模型结构
        vgg5_unit.load_state_dict(torch.load(model_path)) # 加载模型参数
        # vgg5_unit.load_state_dict(torch.load(model_path,map_location=torch.device('cpu'))) # 完整的模型
        split_layer_list = list(range(len(model_cfg['VGG5'])))
    elif args.dataset=='credit':
        save_img_dir = f'../results/1-7/MI/'

        batch_size = 61503
        # batch_size = 8

        model_path = '../results/1-7/credit-20ep.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/home_credit/dataset/application_train.csv'
        train_data, test_data = preprocess_credit(dataPath)
        test_dataset = bank_dataset(test_data)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=8, drop_last=False)
        # one_data_loader = get_one_data(testloader,batch_size = batch_size) #拿到第一个测试数据
        # split_layer_list = ['linear1', 'linear2']
        split_layer_list = [0,3,6,9]
    elif args.dataset=='bank':
        save_img_dir = f'../results/1-8/MI/'

        batch_size=8238
        # batch_size=8

        model_path = '../results/1-8/bank-20ep.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/bank/bank-additional-full.csv'
        
        train_data, test_data = preprocess_bank(dataPath)
        test_dataset = bank_dataset(test_data)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=8, drop_last=False)
        # one_data_loader = get_one_data(testloader,batch_size = batch_size) #拿到第一个测试数据
        split_layer_list = ['linear1', 'linear2']
        split_layer_list = [0,2,4,6]
    elif args.dataset=='purchase':
        save_img_dir = f'../results/1-9/MI/'

        # batchsize = 39465 # test len
        batch_size = 8 # test len
        model_path = '../results/1-9/epoch_train0.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/Purchase100/'

        trainloader, testloader = preprocess_purchase(data_path=dataPath, batch_size=batch_size)
        # one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        split_layer_list = [0,1,2,3,4,5,6,7,8]
    else:
        sys.exit(-1)

    ############# MI计算 ##########
    # 存储
    MI_diff_layer_list = []
    # 循环计算
    # for i in range(len(model_cfg['VGG5'])): # 对每层
    for i in split_layer_list: # 对每一层

        print(f"Layer {i}")

        # 取得client_net
        if args.dataset == 'CIFAR10':
            client_net = get_client_net_weighted(model_name, vgg5_unit, i) # 从vgg5-raw中提取前i层
        elif args.dataset == 'bank':
            client_net = BankNet1(layer=i)
            pweights = torch.load('../results/1-8/bank-20ep.pth').state_dict()
            if i < 6:
                pweights = split_weights_client(pweights,client_net.state_dict())
            client_net.load_state_dict(pweights)
        elif args.dataset == 'credit':
            client_net = CreditNet1(layer=i)
            pweights = torch.load('../results/1-7/credit-20ep.pth').state_dict()
            if i < 9:
                pweights = split_weights_client(pweights,client_net.state_dict())
            client_net.load_state_dict(pweights)
        elif args.dataset == 'purchase':
            # 读取（load）模型
            client_net = PurchaseClassifier1(layer=i)
            pweights  = torch.load(model_path)['state_dict']
            if i < 8: # 
                pweights = split_weights_client(pweights ,client_net.state_dict())
            client_net.load_state_dict(pweights)
        else:
            sys.exit(-1)
        client_net = client_net.to(device)
        client_net.eval()

        MI_same_layer_list = []
        for j, data in enumerate(tqdm.tqdm(testloader)): # 对testloader遍历
        # for j, data in enumerate(tqdm.tqdm(one_data_loader)): # 测试第一个testloader
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                # inference
                # if args.dataset == 'CIFAR10':
                outputs = client_net(images)
                # elif args.dataset == 'credit' or args.dataset == 'bank' or args.dataset=='purchase': # bank, credit
                    # outputs = client_net.getLayerOutput(images,i).cpu().detach()
                # else:
                    # sys.exit(-1)
                inputs = images

                # reshape data
                reshaped_x = images.flatten(start_dim=1).detach().cpu().numpy()
                reshaped_z = outputs.flatten(start_dim=1).detach().cpu().numpy()

                co = ite.cost.MIShannon_DKL()
                ds = np.array([reshaped_x.shape[1], reshaped_z.shape[1]])
                y = np.concatenate((reshaped_x,reshaped_z),axis=1)
                mi = co.estimation(y, ds) 

                MI_same_layer_list.append(mi)
                
        print(f"Layer {i} MI: {mi}")
        MI_diff_layer_list.append(MI_same_layer_list)

    # 保存到csv中
    matrix = np.array(MI_diff_layer_list) # 有点大，x
    transpose = matrix.T # 一行一条数据，一列代表一个layer 
    # pd.DataFrame(data=transpose, columns=[i for i in range (len(model_cfg['VGG5']))]).to_csv(save_img_dir + f'MI-bs{batch_size}.csv',index=False)
    pd.DataFrame(data=transpose, columns=[i for i in split_layer_list]).to_csv(save_img_dir + f'MI-bs{batch_size}.csv',index=False)



# nohup python -u MuInfo.py  --dataset CIFAR10 >> MI-vgg5.out 2>&1  &
# nohup python -u MuInfo.py  --dataset credit >> MI-credit.out 2>&1  &
# nohup python -u MuInfo.py  --dataset bank >> MI-bank.out 2>&1  &
# nohup python -u MuInfo.py  --dataset purchase >> MI-purchase.out 2>&1  &

# credit [3] 20864