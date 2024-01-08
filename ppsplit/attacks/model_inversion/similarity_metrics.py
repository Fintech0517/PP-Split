'''
Author: Ruijun Deng
Date: 2023-12-12 16:00:55
LastEditTime: 2024-01-08 17:02:40
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/attacks/model_inversion/similarity_metrics.py
Description: 
'''

import torch
import torch.nn as nn
from skimage import metrics # 测量SSIM
import numpy as np

class SimilarityMetrics():
    def __init__(self,gpu=True, type=0) -> None:
        # 0 表格, 1 图像
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.euclidean_distance = torch.nn.PairwiseDistance(p=2,eps=1e-6)
        self.mse_loss = nn.MSELoss()

        if type == 0: # tabular
            self.sim_metric_dict = {'cos':[],'euc':[],'mse':[]}
        elif type == 1: # image
            self.sim_metric_dict = {'ssim':[],'mse':[]}

    def ssim_metric(self,deprocessImg_raw,deprocessImg_inversed):
        # ssim = measure.compare_ssim(
        # X=np.moveaxis(ref_img, 0, -1),  # 把dim=0的维度移动到最后
        # Y=np.moveaxis(inv_img, 0, -1),  
        # data_range = inv_img.max() - inv_img.min(), # 为什么是inv_img的呢
        # multichannel=True)
        ref_img = deprocessImg_raw.detach().cpu().numpy().squeeze()
        inv_img = deprocessImg_inversed.detach().cpu().numpy().squeeze()
        ssim = metrics.structural_similarity(np.moveaxis(inv_img,0,-1), np.moveaxis(ref_img,0,-1), 
                                                data_range=inv_img.max() - inv_img.min(), channel_axis=-1)
        return ssim





