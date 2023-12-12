'''
Author: Ruijun Deng
Date: 2023-08-28 14:50:43
LastEditTime: 2023-12-12 10:29:07
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/quantification/fisher_information/dFIL_inverse.py
Description: 一个一个样本计算，没有平均之说
'''
# FIL 计算函数
import torch.autograd.functional as F
import torch
import time
# import logging
# logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
import os
os.environ['NUMEXPR_MAX_THREADS'] = '48'

class dFIL_inverse():
    def __init__(self) -> None:
        pass
    
    
    def quantify(self, model, inputs, outputs = None, sigmas=0.01, with_outputs = True):
        if with_outputs:
            return self._computing_eta_with_outputs(model, inputs, outputs, sigmas).detach().cpu().numpy()
        else:
            return self._computing_eta_without_outputs(model, inputs,  sigmas).detach().cpu().numpy()
        

    # model的smashed data需要在[0,1]之间，才能保证输出的eta也在[0,1]之间?证明？
    def _computing_eta_without_outputs(self, model, inputs,  sigmas): # sigma_square
        inputs.requires_grad_(True) # 需要求导
        outputs = model(inputs)
        
        # 前向传播
        # outputs = outputs + sigma * torch.randn_like(outputs) # 加噪声 (0,1] uniform
    
        # 计算jacobian
        J = F.jacobian(model, inputs)
        J = J.reshape(J.shape[0],outputs.numel(),inputs.numel()) # (batch, out_size, in_size)
        # print(f"J2.shape: {J.shape}, J2.prod: {torch.prod(torch.tensor(list(J.shape)))}")

        # 计算eta
        I = 1.0/(sigmas)*torch.matmul(J[0].t(), J[0])
        # print(f"I.shape: ", I.shape)
        dFIL = I.trace().div(inputs.numel())
        # eta = dFIL
        # print(f"eta: {eta}")
        # print('t2-t1=',t2-t1, 't3-t2', t3-t2)
        return 1.0/dFIL
    
        # model的smashed data需要在[0,1]之间，才能保证输出的eta也在[0,1]之间?证明？
    def _computing_eta_with_outputs(self, model, inputs, outputs, sigmas): # sigma_square
        # 前向传播
        # outputs = outputs + sigma * torch.randn_like(outputs) # 加噪声 (0,1] uniform
    
        # 计算jacobian
        J = F.jacobian(model, inputs)
        J = J.reshape(J.shape[0],outputs.numel(),inputs.numel()) # (batch, out_size, in_size)
        # print(f"J2.shape: {J.shape}, J2.prod: {torch.prod(torch.tensor(list(J.shape)))}")

        # 计算eta
        I = 1.0/(sigmas)*torch.matmul(J[0].t(), J[0])
        # print(f"I.shape: ", I.shape)
        dFIL = I.trace().div(inputs.numel())
        # eta = dFIL
        # print(f"eta: {eta}")
        # print('t2-t1=',t2-t1, 't3-t2', t3-t2)
        return 1.0/dFIL


# 多层、整个数据集上的dFIL
if __name__ == '__main__':
    # 导包
    import torch
    import pandas as pd
    import tqdm
    import sys
    sys.path.append('../')

    from utils.utils import *
    from utils.datasets import *

    import argparse

    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'CIFAR10')
    args = parser.parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")
    print("current device: ", device)
    sigma = 0.01

    # 获取数据集
    if args.dataset=='CIFAR10':
        save_img_dir = f'../results/1-6/dFIL/'
        trainloader,testloader = get_cifar10_normalize(batch_size=1)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据

        # VGG5
        model_path = '../results/VGG5/BN+Tanh/VGG5-0ep.pth' # VGG5-BN+Tanh
        vgg5_unit = VGG('Unit', 'VGG5', len(model_cfg['VGG5'])-1, model_cfg) # 加载模型结构
        vgg5_unit.load_state_dict(torch.load(model_path)) # 加载模型参数
        split_layer_list = list(range(len(model_cfg['VGG5'])))
    elif args.dataset=='credit':
        save_img_dir = f'../results/1-7/dFIL/'
        model_path = '../results/1-7/credit-20ep.pth'

        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/home_credit/dataset/application_train.csv'
        train_data, test_data = preprocess_credit(dataPath)
        test_dataset = bank_dataset(test_data)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                num_workers=8, drop_last=False)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        # split_layer_list = ['linear1', 'linear2']
        split_layer_list = [0,3,6,9]
    elif args.dataset=='bank':
        model_path = '../results/1-8/bank-20ep.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/bank/bank-additional-full.csv'
        save_img_dir = f'../results/1-8/dFIL/'

        train_data, test_data = preprocess_bank(dataPath)
        test_dataset = bank_dataset(test_data)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                num_workers=8, drop_last=False)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        # split_layer_list = ['linear1', 'linear2']
        split_layer_list = [0,2,4,6]
    elif args.dataset=='purchase':
        save_img_dir = f'../results/1-9/dFIL/'
        model_path = '../results/1-9/epoch_train0.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/Purchase100/'

        trainloader, testloader = preprocess_purchase(data_path=dataPath, batch_size=1)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        # split_layer_list = [3]
        # split_layer_list = [0,1,2,3,4,5,6,7,8]
        split_layer_list = [1,3,5,7]
    else:
        sys.exit(-1)


    ##############开始计算dFIL############
    # 加载训练好的 edge 模型
    eta_diff_layer_list = []
    # for i in range (len(model_cfg['VGG5'])): # 对每一层
    for i in split_layer_list: # 对每一层
        print(" Layer: ", i)

        # 获取模型
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
        
        eta_same_layer_list = []
        # 对traingloader遍历计算所有eta？
        for j, data in enumerate(tqdm.tqdm(testloader)):
        # for j, data in enumerate(tqdm.tqdm(one_data_loader)): # 测试第一个testloader
        #     if j < 31705:
        #         continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad_(True) # 需要求导
            
            # inference
            # if args.dataset == 'CIFAR10' or args.dataset == 'purchase' or args.dataset == 'bank':
            outputs = client_net(inputs)
            # elif args.dataset == 'credit': # bank, credit, purchase
                # outputs = client_net.getLayerOutput(inputs,i)
            # else:
                # sys.exit(-1)

            eta = computing_eta(client_net, inputs, outputs, sigma)

            logger.info(str(j)+": "+str(eta.item()))
            eta_same_layer_list.append(eta.detach().cpu().numpy())

        eta_diff_layer_list.append(eta_same_layer_list)

    # 保存到csv中
    # save_img_dir = f'../results/1-8/dFIL/'
    matrix = np.array(eta_diff_layer_list) # 有点大，x
    transpose = matrix.T # 一行一条数据，一列代表一个layer 

    pd.DataFrame(data=transpose, columns=[i for i in split_layer_list]).to_csv(save_img_dir + f'dFIL-1.csv',index=False)


# nohup python -u dFIL.py --dataset CIFAR10 >> dFIL-cifar10.out 2>&1  &
# nohup python -u dFIL.py --dataset bank >> dFIL-bank.out 2>&1  &
# nohup python -u dFIL.py --dataset credit >> dFIL-credit1.out 2>&1  &
# nohup python -u dFIL.py --dataset purchase >> dFIL-purchase.out 2>&1  &


# credit 4layer [1] 27598 [4] 22073
# purchase 1,3,5,7 [2] 3705