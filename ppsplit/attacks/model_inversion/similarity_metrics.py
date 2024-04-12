'''
Author: Ruijun Deng
Date: 2023-12-12 16:00:55
LastEditTime: 2024-03-18 15:01:16
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

        # 5种相似度计算方法
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.euclidean_distance = torch.nn.PairwiseDistance(p=2,eps=1e-6)
        self.mse_loss = nn.MSELoss()
        self.ssim = self.ssim_metric
        self.accuracy = self._accuracy

        
        if type == 0: # tabular
            self.sim_metric_dict = {'cos':[],'euc':[],'mse':[],'acc':[]}
        elif type == 1: # image
            self.sim_metric_dict = {'ssim':[],'mse':[]}

    def ssim_metric(self,deprocessImg_raw,deprocessImg_inversed):
        # ssim = measure.compare_ssim(
        # X=np.moveaxis(ref_img, 0, -1),  # 把dim=0的维度移动到最后
        # Y=np.moveaxis(inv_img, 0, -1),  
        # data_range = inv_img.max() - inv_img.min(), # 为什么是inv_img的呢
        # multichannel=True)
        ref_img = deprocessImg_raw.clone().detach().cpu().numpy().squeeze()
        inv_img = deprocessImg_inversed.clone().detach().cpu().numpy().squeeze()
        ssim = metrics.structural_similarity(np.moveaxis(inv_img,0,-1), np.moveaxis(ref_img,0,-1), 
                                                data_range=inv_img.max() - inv_img.min(), channel_axis=-1)
        return ssim

    # 将输出的0-1之间的值根据0.5为分界线分为两类,同时统计正确率
    def _compute_correct_prediction(self, y_targets, y_prob_preds, threshold=0.5):
        y_hat_lbls = []
        pred_pos_count = 0
        pred_neg_count = 0
        correct_count = 0
        for y_prob, y_t in zip(y_prob_preds, y_targets):
            if y_prob <= threshold: # 小于threshold
                pred_neg_count += 1
                y_hat_lbl = 0
            else: # 大于threshold
                pred_pos_count += 1
                y_hat_lbl = 1
            y_hat_lbls.append(y_hat_lbl)
            if y_hat_lbl == y_t:
                correct_count += 1

        return np.array(y_hat_lbls), [pred_pos_count, pred_neg_count, correct_count]

    def _accuracy(self, y_targets, y_prob_preds, threshold=0.5):
        # 计算正确率
        _, ans = self._compute_correct_prediction(
            y_targets=y_targets, y_prob_preds=y_prob_preds, threshold=threshold)
        pred_pos_count, pred_neg_count, correct_count = ans

        return correct_count/len(y_targets)


