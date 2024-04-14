import time
import math
import os
import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn


from tqdm import tqdm
import pickle

from ...utils.similarity_metrics import SimilarityMetrics


class rMLE_Attack():
    def __init__(self,gpu=True,inverse_dir=None,data_type = 0) -> None:
        self.data_type = data_type # 0 是表格数据集，1是图像数据集
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
        self.inverse_dir = inverse_dir if inverse_dir else './inverted/'
        if not os.path.exists(self.inverse_dir+'/images/'):
            os.makedirs(self.inverse_dir+'/images/')
    
    def inverse(self,client_net,
                dataloader,
                Niters=20000,
                save_fake = True,
                deprocess=None):
        if self.data_type==1 and deprocess==None: # 图像数据集没给deprocess方法
            print("图像数据没给deprocess 函数")
            exit(0)
        # 打印模型信息
        print("client_net: ")
        print(client_net)

        # 网络搬到设备上
        client_net.to(self.device)
        client_net.eval()

        # 记录数据:
        sim_metrics = SimilarityMetrics(type = self.data_type)

        X_fake_list = []
        time_list = []
        infer_time_list = []
        mseloss_list = []
        for i, (trn_X, trn_y) in enumerate(tqdm(dataloader)):  # 对dataloader遍历
            originData = trn_X.to(self.device)
            start_infer = time.time()
       
            inverted_data = torch.zeros(originData.size()).to(self.device)
            inverted_data.requires_grad = True

            # 优化器
            optimizer = optim.Adam(params=[inverted_data])

            smashed_data_raw = client_net(originData)
            # 迭代优化
            mseloss_single_list = []
            start_iter = time.time()
            for j in range(Niters):
                optimizer.zero_grad()
                
                smashed_data_fake = client_net(inverted_data)
                mseloss = ((smashed_data_fake-smashed_data_raw) ** 2).mean()

                mseloss_single_list.append(mseloss.cpu().detach().numpy())
                mseloss.backward(retain_graph=True)

                optimizer.step()
            
            time_list.append(time.time()-start_iter)
            mseloss_list.append(mseloss_single_list)

            # 收集重构数据与原始数据相似度
            if self.data_type == 0: # 表格数据
                sim_metrics.collect_sim_tab(originData,inverted_data)
                X_fake_list.append(inverted_data.cpu().detach().squeeze().numpy())
            elif self.data_type == 1: # 图像数据
                sim_metrics.collect_sim_img(originData,inverted_data, deprocess)
                # 保存图片
                if save_fake == True: # 储存原始图像+inv图像
                    torchvision.utils.save_image(deprocess(originData.clone()), self.inverse_dir + '/images/' + str(i) + '-ref.png')
                    torchvision.utils.save_image(deprocess(inverted_data.clone()), self.inverse_dir + '/images/' + str(i) + '-inv.png')
                
            
        # 报告相似度, 存储相似度
        sim_metrics.report_similarity()
        sim_metrics.store_similarity(inverse_route=self.inverse_dir+f'inv-sim.csv')

        # 储存inverse data (表格数据)
        if save_fake and self.data_type == 0:
            pd.DataFrame(X_fake_list).to_csv(self.inverse_dir+f'inv-X.csv', index = False)
        
