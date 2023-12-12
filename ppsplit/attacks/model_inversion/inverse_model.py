'''
Author: Ruijun Deng
Date: 2023-12-12 12:42:45
LastEditTime: 2023-12-12 21:22:41
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

from .similarity_metrics import SimilarityMetrics
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

        print(f"cosine: {np.mean(sim_metrics.sim_metric_dict['cos'])}, \
              Euclidean: {np.mean(sim_metrics.sim_metric_dict['euc'])},\
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
            
        print(f"SSIM: {np.mean(sim_metrics.sim_metric_dict['ssim'])},\
              MSE:{np.mean(sim_metrics.sim_metric_dict['mse'])}")

        # 储存similairty相关文件
        pd.DataFrame({'ssim': sim_metrics.sim_metric_dict['ssim'],
                        'mse':sim_metrics.sim_metric_dict['mse']}).to_csv(self.inverse_dir + f'inv-sim.csv', index = False)


