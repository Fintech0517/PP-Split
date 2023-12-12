'''
Author: Ruijun Deng
Date: 2023-12-12 12:42:45
LastEditTime: 2023-12-12 20:15:50
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/attacks/model_inversion/inverse_model.py
Description: 
'''

import torch.nn as nn
import torch
import os
import tqdm
import numpy as np
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能是由于是MacOS系统的原因
import pandas as pd
from torch.utils.data import Dataset
torch.multiprocessing.set_sharing_strategy('file_system')

from similarity_metrics import SimilarityMetrics
import torchvision

class InverseModelAttack():
    def __init__(self,gpu=True,decoder_route=None,data_type=0,inverse_dir=None) -> None:
        self.data_type=data_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
        # 储存或加载攻击模型的路径

        self.inverse_dir = inverse_dir if inverse_dir else './inverted/'
        self.decoder_route = decoder_route if decoder_route else './decoder_net.pth'
        if not os.path.exists(self.inverse_dir):
            os.makedirs(self.inverse_dir)

    def train_decoder(self,client_net,decoder_net,
                      train_loader,test_loader,
                      epochs,optimizer=None):
        
        # 打印相关信息
        print("----train decoder----")
        print("client_net: ")
        print(client_net)
        print("decoder_net: ")
        print(decoder_net)

        # 网络搬到设备上
        client_net.to(self.device)
        decoder_net.to(self.device)
        

        # loss function 统一采用MSELoss？
        if not optimizer:
            optimizer = torch.optim.SGD(decoder_net.parameters(), 1e-3)
        criterion = nn.MSELoss()
        

        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            # train and update
            epoch_loss = []
            for i, (trn_X, trn_y) in enumerate(tqdm.tqdm(train_loader)):
                trn_X = trn_X.to(self.device)
                batch_loss = []

                optimizer.zero_grad()

                out = decoder_net(client_net(trn_X))

                loss = criterion(out, trn_X)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))


            print("--- epoch: {0}, train_loss: {1}".format(epoch, epoch_loss))

        # 储存decoder模型
        torch.save(decoder_net, self.decoder_route)
        print("model saved")
        return decoder_net

    def inverse(self,client_net,decoder_net,
                train_loader,test_loader,
                deprocess=None,
                save_fake=True):
        if self.data_type==1 and deprocess==None:
            exit(0)
        
        if self.data_type==0:
            return self._inverse_tab(client_net,decoder_net,train_loader,test_loader,save_fake)
        else:
            return self._inverse_image(client_net,decoder_net,train_loader,test_loader,deprocess,save_fake)
        
    def _inverse_tab(self,client_net,decoder_net,
                train_loader,test_loader,
                save_fake=True):
        
        # 打印相关信息
        print("----train decoder----")
        print("client_net: ")
        print(client_net)
        print("decoder_net: ")
        print(decoder_net)

        # 网络搬到设备上
        client_net.to(self.device)
        decoder_net.to(self.device)
        
        # 记录数据:
        sim_metrics = SimilarityMetrics(type = self.data_type)

        X_fake_list = []
        for i, (trn_X, trn_y) in enumerate(tqdm.tqdm(test_loader)):  # 对testloader遍历
            originData = trn_X.to(self.device)
            smashed_data = client_net(originData)  # edge 推理
            inverted_data = decoder_net(smashed_data)  # inverse
            X_fake_list.append(inverted_data.cpu().detach().squeeze().numpy())


            cos = sim_metrics.cosine_similarity(inverted_data, originData).item()
            sim_metrics.sim_metric_dict['cos'].append(cos)
            euc = sim_metrics.euclidean_distance(inverted_data, originData).item()
            sim_metrics.sim_metric_dict['euc'].append(euc)
            mse = sim_metrics.mse_loss(inverted_data, originData).item()
            sim_metrics.sim_metric_dict['mse'].append(mse)

        print(f"cosine: {np.mean(sim_metrics.sim_metric_dict['cos'])}, 
              Euclidean: {np.mean(sim_metrics.sim_metric_dict['euc'])},
              MSE:{np.mean(sim_metrics.sim_metric_dict['mse'])}")

        # 存储数据
        # self.inverse_dir = f'../results/1-8/inverted/{split_layer}/' # 每层一个文件夹
        # 储存similairty相关文件
        pd.DataFrame({'cos': sim_metrics.sim_metric_dict['cos'],
                        'euc': sim_metrics.sim_metric_dict['euc'],
                        'mse':sim_metrics.sim_metric_dict['mse']}).to_csv(self.inverse_dir + f'inv-sim.csv', index = False)

        # 存储inverse data
        if save_fake:
            pd.DataFrame(X_fake_list).to_csv(self.inverse_dir+f'inv-X.csv', index = False)
    
    def _inverse_image(self,client_net,decoder_net,
                train_loader,test_loader,
                deprocess, # unormalize的函数
                save_fake=True):
        
        # 打印相关信息
        print("----train decoder----")
        print("client_net: ")
        print(client_net)
        print("decoder_net: ")
        print(decoder_net)

        # 网络搬到设备上
        client_net.to(self.device)
        decoder_net.to(self.device)
        
        # 记录数据:
        sim_metrics = SimilarityMetrics(type = self.data_type)

        X_fake_list = []
        for i, (trn_X, trn_y) in enumerate(tqdm.tqdm(test_loader)):  # 对testloader遍历
            raw_input = trn_X.to(self.device)
            smashed_data = client_net(raw_input)  # edge 推理
            inverted_input = decoder_net(smashed_data)  # inverse
            deprocessImg_raw = deprocess(raw_input.clone()) # x_n
            deprocessImg_inversed = deprocess(inverted_input.clone()) # s_n

            
            X_fake_list.append(inverted_input.cpu().detach().squeeze().numpy())

            ssim = sim_metrics.ssim_metric(deprocessImg_raw, deprocessImg_inversed).item()
            sim_metrics.sim_metric_dict['ssim'].append(ssim)
            mse = sim_metrics.mse_loss(inverted_input, raw_input).item()
            sim_metrics.sim_metric_dict['mse'].append(mse)

            # 保存图片
            if save_fake == True: # 储存原始图像+inv图像
                torchvision.utils.save_image(deprocessImg_raw, self.inverse_dir + str(i) + '-ref.png')
                torchvision.utils.save_image(deprocessImg_inversed, self.inverse_dir + str(i) + '-inv.png')
            
        print(f"SSIM: {np.mean(sim_metrics.sim_metric_dict['ssim'])},
              MSE:{np.mean(sim_metrics.sim_metric_dict['mse'])}")

        # 储存similairty相关文件
        pd.DataFrame({'ssim': sim_metrics.sim_metric_dict['ssim'],
                        'mse':sim_metrics.sim_metric_dict['mse']}).to_csv(self.inverse_dir + f'inv-sim.csv', index = False)

        
def train_decoder(net, device, BatchSize, learningRate, split_layer):

    print("################################ Load Data ############################")
    dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/bank/bank-additional-full.csv'
    decoder_mode = f'../results/1-8/inverted/32bs/decoder-layer{split_layer}.pth'



    train_data, test_data = preprocess_bank(dataPath)

    # dataloader
    train_dataset = bank_dataset(train_data)
    test_dataset = bank_dataset(test_data)

    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True,
                                                num_workers=8, drop_last=False)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=BatchSize, shuffle=False,
                                               num_workers=8, drop_last=False)

    print("################################ Set Federated Models, optimizer, loss ############################")

    decoder = BankNetDecoder1(layer=split_layer).to(device)
    optimizer = torch.optim.SGD(decoder.parameters(), learningRate=1e-3)

    # loss function
    criterion = nn.MSELoss()

    # 加载decoder model
    if os.path.isfile(decoder_mode):
        print("=> loading decoder mode '{}'".format(decoder_mode))
        decoder = torch.load(decoder_mode, map_location=device)
        return decoder

    print("################################ Train Decoder Models ############################")
    for epoch in range(0, 20):
        # train and update
        epoch_loss = []
        for step, (trn_X, trn_y) in enumerate(train_dataset):
            trn_X = trn_X.to(device)
            batch_loss = []

            optimizer.zero_grad()

            out = decoder(net(trn_X))

            loss = criterion(out, trn_X)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))


        print(
            "--- epoch: {0}, train_loss: {1}"
            .format(epoch, epoch_loss))

    torch.save(decoder, decoder_mode)
    print("model saved")
    return decoder


def inverse(DATASET='CIFAR10', network='VGG5', NEpochs=200,
          BatchSize=32, learningRate=1e-3, NDecreaseLR=20, eps=1e-3,
          AMSGrad=True, model_dir="", model_name="", gpu=True, split_layer=2):

    dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/bank/bank-additional-full.csv'

    tabinfo = {
        'onehot': { # 前面是类别列
            'job': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'marital': [12, 13, 14, 15],
             'education': [16, 17, 18, 19, 20, 21, 22, 23], 'default': [24, 25, 26], 'housing': [27, 28, 29],
             'loan': [30, 31, 32], 'contact': [33, 34], 'month': [35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
             'day_of_week': [45, 46, 47, 48, 49], 'poutcome': [50, 51, 52]},
        'numList': [i for i in range(53, 63)] # 后面是数值列
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data, test_data = preprocess_bank(dataPath)

    # dataloader
    train_dataset = bank_dataset(train_data)
    test_dataset = bank_dataset(test_data)

    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True,
                                              num_workers=8, drop_last=False)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                             num_workers=8, drop_last=False)

    # 读取（load）模型
    net = BankNet1(layer=split_layer)
    pweights = torch.load(model_dir + model_name).state_dict()
    if split_layer < 6:
        pweights = split_weights_client(pweights,net.state_dict())
    net.load_state_dict(pweights)
    net = net.to(device)
    net.eval()
    # accTest = evalTest_bank(test_dataset, net, gpu=gpu)  # 测试模型精度

    decoder = train_decoder(net, device, BatchSize, learningRate, split_layer).to(device)
    print(decoder)

    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    euclidean_distance = torch.nn.PairwiseDistance(p=2,eps=1e-6)
    mse_loss = nn.MSELoss().to(device)

    cosine_list = []
    Euclidean_list = []
    acc_list = []
    onehot_acc_list = []
    num_acc_list = []
    mse_list = []

    X_fake_list = []
    for i, (trn_X, trn_y) in enumerate(tqdm.tqdm(test_dataset)):  # 对testloader遍历

        originData = trn_X.to(device)
        # out = decoder(net.getLayerOutput(trn_X, 'linear2'))
        protocolData = net(originData)  # edge 推理
        xGen = decoder(protocolData)  # inverse
        X_fake_list.append(xGen.cpu().detach().squeeze().numpy())
        # X_raw_list.append(trn_X.cpu().detach().squeeze().numpy())


        # 后续数据处理
        acc, onehot_acc, num_acc = tabRebuildAcc(originData, xGen, tabinfo)
        acc_list.append(acc)
        onehot_acc_list.append(onehot_acc)
        num_acc_list.append(num_acc)
        # print("acc:", acc)
        # print("onehot_acc:", onehot_acc)
        # print("num_acc:", num_acc)

        cos = cosine_similarity(xGen, originData).item()
        cosine_list.append(cos)
        euc = euclidean_distance(xGen, originData).item()
        Euclidean_list.append(euc)
        mse = mse_loss(xGen, originData).item()
        mse_list.append(mse)
        # similarity = Similarity(xGen, originData)
        # euclidean_dist = torch.mean(torch.nn.functional.pairwise_distance(xGen, originData)).item()
        # print(f"Similarity: {similarity}")
        # print(f"euclidean_dist: {euclidean_dist}")
    print(f"split_layer: {split_layer}, cosine: {np.mean(cosine_list)}, Euclidean: {np.mean(Euclidean_list)}, \
            acc: {np.mean(acc_list)}, onehot_acc: {np.mean(onehot_acc_list)}, num_acc: {np.mean(num_acc_list)}")
    # 存储数据
    save_img_dir = f'../results/1-8/inverted/{split_layer}/' # 每层一个文件夹
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    # 储存similairty相关文件
    pd.DataFrame({'cos': cosine_list,
                    'euc': Euclidean_list,
                    'acc': acc_list,
                    'onehot_acc': onehot_acc_list,
                    'num_acc': num_acc_list}).to_csv(save_img_dir + f'inv-sim.csv', index = False)
    # 存储inverse data
    pd.DataFrame(X_fake_list,columns=onehot_columns+num_columns).to_csv(save_img_dir+f'inv-X.csv', index = False)
    # pd.DataFrame(X_raw_list,columns=onehot_columns+num_columns).to_csv(save_img_dir+f'raw-X.csv', index = False)




