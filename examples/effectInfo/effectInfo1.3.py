# %% [markdown]
# # 1. 基础设置

# %%
'''
Author: Ruijun Deng
Date: 2024-08-14 16:59:47
LastEditTime: 2024-09-08 07:11:25
LastEditors: Ruijun Deng
FilePath: /PP-Split/examples/effectInfo/effectInfo1.3.py
Description: 
'''
# 导包
import torch
import os
import argparse
import pandas as pd
import tqdm
import numpy as np
# os.environ['NUMEXPR_MAX_THREADS'] = '48'

# 导入各个指标
import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from ppsplit.quantification.distance_correlation.distCor import distCorMetric
from ppsplit.quantification.fisher_information.dFIL_inverse import dFILInverseMetric
from ppsplit.quantification.shannon_information.mutual_information import MuInfoMetric
from ppsplit.quantification.shannon_information.ULoss import ULossMetric
from ppsplit.quantification.rep_reading.rep_reader import PCA_Reader
from ppsplit.quantification.shannon_information.ITE_tools import Shannon_quantity

from target_model.task_select import get_dataloader_and_model,get_dataloader_and_model, get_dataloader,get_models,get_infotopo_para

# utils
from ppsplit.utils.utils import create_dir

# nohup python -u effectInfo.py > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.3/effectInfo1.3-pool4-layer11.log 2>&1 &
# nohup python -u effectInfo.py > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.3/effectInfo1.3-pool4-layer6.log 2>&1 &
# %%
args = {
        # 'device':torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        'device':torch.device("cpu"),
        'dataset':'CIFAR10',
        # 'dataset':'bank',
        # 'dataset':'credit',
        # 'dataset':'purchase',
        # 'dataset':'Iris',
        # 'model': 'ResNet18',
        'model': 'VGG5',
        # 'result_dir': '20240702-FIL/',
        'result_dir': '20240702-effectiveInfo/',
        'oneData_bs': 500,
        'test_bs': 1,
        'train_bs': 1,
        'noise_scale': 0, # 防护措施
        'split_layer': 6,
        # 'test_num': 'invdFIL', # MI, invdFIL, distCor, ULoss,  # split layer [2,3,5,7,9,11] for ResNet18
        'test_num': 'effectiveInfo1.3',
        'no_dense':True,
        }
print(args['device'])
print(args)


# %%
data_msg = get_dataloader(args)
model_msg = get_models(args)
infotopo_msg = get_infotopo_para(args)
msg = {**model_msg,**data_msg,**infotopo_msg}

# 数据集
one_data_loader,trainloader,testloader = data_msg['one_data_loader'],data_msg['trainloader'], data_msg['testloader']

# effectEntropy Infotopo参数
nb_of_values = msg['nb_of_values']
conv = msg['conv']
print("infotopo: nb_of_values: ",nb_of_values)

# 模型
client_net,decoder_net = model_msg['client_net'],model_msg['decoder_net']
decoder_route = model_msg['decoder_route']
image_deprocess = model_msg['image_deprocess']

# 路径
results_dir = model_msg['results_dir']
inverse_dir = results_dir + 'layer'+str(args['split_layer'])+'/'
data_type = 1 if args['dataset'] == 'CIFAR10' else 0
split_layer = args['split_layer']

print('results_dir:',results_dir)
print('inverse_dir:',inverse_dir)
print('decoder_route:',decoder_route)

create_dir(results_dir)

# client_net使用
client_net = client_net.to(args['device'])
client_net.eval()

# for n, p in client_net.named_parameters():
#     print(n, p.shape)

# %% [markdown]
# # 5. effective information

# %% [markdown]
# ## 5.1 effect Fisher

# %%
# effective fisher 计算函数
import torch.autograd.functional as F
import torch
import time

# nips23
from torch.autograd.functional import jvp
import random
import math


import pandas as pd

# 用diag 来化简
def computing_diag_det_with_outputs(model, inputs, outputs, sigmas): # sigma_square
        # batchsize:
        batch_size = inputs.shape[0] # 一个batch的样本数目
        output_size = outputs[0].numel() # 一个样本的outputs长度
        input_size = inputs[0].numel() # 一个样本的outputs长度
        effect_fisher_sum = 0.0

        # avg
        I_diagonal_batch_avg = torch.zeros(input_size).to(args['device']) # batch上做平均
        print("I_diagonal_batch_avg: ",I_diagonal_batch_avg.shape)
        f2_2_avg_outer = torch.tensor(0.0).to(args['device'])
        f2_avg_outer = torch.tensor(0.0).to(args['device'])
        
        # effecti_fisher第一部分
        f1 = input_size * torch.log(2*torch.pi*torch.exp(torch.tensor(1.0)))
        # print('f1: ',f1)

        # f2需要求平均？
        # 遍历单个样本: 换数据
        for i in range(batch_size): # 对每个样本
            input_i = inputs[i].unsqueeze(0)

            # 计算jacobian
            J = F.jacobian(model, input_i)
            # J = J.reshape(J.shape[0],outputs.numel(),inputs.numel()) # (batch, out_size, in_size)
            J = J.reshape(output_size, input_size) # (batch, out_size, in_size)
            # print(f"J2.shape: {J.shape}, J2.prod: {torch.prod(torch.tensor(list(J.shape)))}")
            # 计算eta
            JtJ = torch.matmul(J.t(), J)
            I = 1.0/(sigmas)*JtJ
            # print("I: ", I)
            # diagonal fisher information matrix (approximation)
            I_diagonal = torch.diagonal(I,dim1=0,dim2=1) # vector
            # print("I_diagonal: ",I_diagonal.shape)

            I_diag = torch.diag_embed(I_diagonal) # matrix

            # batch的平均
            I_diagonal_batch_avg += I_diagonal / (batch_size)

            # 储存I
            # I_np = I.cpu().detach().numpy()
            # df = pd.DataFrame(I_np)
            # df.to_csv(f'{i}.csv',index=False,header=False)

            # print("I: ", I)
            # w = torch.det(I)
            # print('det I: ', I.det().log())

            f2 = torch.logdet(I) # 直接用torch计算
            # f2_1 = torch.logdet(I_diag)
            f2_2 = torch.sum(torch.log(I_diagonal+1e-10)) # /I_diagonal.numel() # diagonal后计算

            f2_2_avg_outer += f2_2 / batch_size
            f2_avg_outer += f2 / batch_size

            # print('log det I: ',f2 )
            # print('f1: ',f1)
            print('f2: ',f2)
            # print('f2_1: ',f2_1)
            print('f2_2: ',f2_2)

        f2_2_avg_inner = torch.sum(torch.log(I_diagonal_batch_avg+1e-10)) # 用平均后的diagonal 计算

        print('f2_2_avg_outer: ',f2_2_avg_outer)
        print('f2_2_avg_inner: ',f2_2_avg_inner)
        print('f2_avg_outer: ',f2_avg_outer)

        # effect_fisher = 0.5 * (f1 - f2_2_avg_inner)
        effect_fisher = 0.5 * (f1 - f2_2_avg_outer)
        # effect_fisher = 0.5 * (f1 - f2_avg_outer)
        # effect_fisher_sum+=effect_fisher

        # print("effect_fisher: ",effect_fisher)
        
        # effect_fisher_mean = effect_fisher_sum / batch_size
        return effect_fisher.cpu().detach().numpy()

# %% [markdown]
# ## 5.2 effect Entropy

# %%
# effect entropy 计算函数
import math
import numpy as np
def shannon_entropy_pyent(time_series):
    """Calculate Shannon Entropy of the sample data.

    Parameters
    ----------
    time_series: np.ndarray | list[str]

    Returns
    -------
    ent: float
        The Shannon Entropy as float value
    """

    # Calculate frequency counts
    _, counts = np.unique(time_series, return_counts=True)
    total_count = len(time_series)
    # print('counts: ', counts)
    # print("total_count: ",total_count)

    # Calculate frequencies and Shannon entropy
    frequencies = counts / total_count
    # print("freq: ",frequencies)
    ent = -np.sum(frequencies * np.log(frequencies))

    return ent

# import infotopo
import ppsplit.quantification.shannon_information.infotopo as infotopo
from torch.nn.functional import avg_pool2d
def shannon_entropy_infotopo(x, conv = False):
    information_top = infotopo.infotopo(dimension_max = x.shape[1],
                                        dimension_tot = x.shape[1],
                                        sample_size = x.shape[0],
                                        # nb_of_values = nb_of_values, # 不是很懂这个意思，为什么iris对应9？
                                        nb_of_values = 17, # 不是很懂这个意思，为什么iris对应9？
                                        # forward_computation_mode = True,
                                        )
    if conv:
        images_convol = information_top.convolutional_patchs(x)
        print('images_convol: ',images_convol.shape)
        x = images_convol

    # 计算联合分布的概率？（全排列）
    # joint_prob = information_top._compute_probability(x)
    # print('joint_prob: ',joint_prob)
    
    # 计算联合熵（全排列的）
    joint_prob_ent = information_top.simplicial_entropies_decomposition(x) # log2
    new_joint_prob_ent = {key: value * np.log(2) for key, value in joint_prob_ent.items()} #ln 转2为底 成 e为底
    print("joint_entropy: ",new_joint_prob_ent)
    # ent = information_top._compute_forward_entropies(x)
    # information_top.entropy_simplicial_lanscape(joint_prob_ent) # 画图
    # ent = _entropy(np.array(list(new_joint_prob_ent.values())))

    joint_entropy_final = list(new_joint_prob_ent.values())[-1]
    return joint_entropy_final


# %% [markdown]
# ## 5.3 effect information

# %%

effecInfo_diff_layer_list = []
effecInfo_same_layer_list = []
EntropyMetric = ULossMetric()
Fishermetric = dFILInverseMetric()

InversedFIL_same_layer_list = []
# for j, data in enumerate(tqdm.tqdm(testloader)): # 对testloader遍历
for j, data in enumerate(tqdm.tqdm(one_data_loader)): # 测试第一个testloader
    images, labels = data
    images, labels = images.to(args['device']), labels.to(args['device'])
    with torch.no_grad():
        # effect entropy 
        # effecEntro= EntropyMetric._entropy_prob_batch(images) # H(x)
        
        # infotopo
        if conv:
            print('images: ', images.shape)
            images= avg_pool2d(images,kernel_size=4)
            print('images_pooled: ',images.shape)

        effectEntro = shannon_entropy_infotopo(images.flatten(start_dim=1).detach().cpu().numpy(), conv)
        
        # PyEntropy
        effectEntro_pyent = 0.0
        for i in range(len(images[0])): # 对每个维度
            effectEntro_pyent  += shannon_entropy_pyent(images[:,i].flatten().detach().cpu().numpy())
            # print('effectEntro_pyent: ',effectEntro_pyent)
        # effectEntro_pyent = shannon_entropy(images.flatten(start_dim=1).detach().cpu().numpy())
        print('effectEntro_pyent: ',effectEntro_pyent)

        
        # effecit fisher
            # inference
        outputs = client_net(images).clone().detach()
        # inverse_dFIL = Fishermetric.quantify(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01, with_outputs=True)
        # effectFisher = computing_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)
        # effectFisher = computing_diag_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)
        effectFisher = computing_diag_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)

        # 存储
        effecInfo_same_layer_list.append(effectEntro-effectFisher)
        # InversedFIL_same_layer_list.append(inverse_dFIL)

        # 打印一下
        print("effecEntro: ", effectEntro)
        print("effecFisher: ",effectFisher)
        # print("inverse_dFIL: ",inverse_dFIL)

print(f"Layer {args['split_layer']} effecInfo: {sum(effecInfo_same_layer_list)/len(effecInfo_same_layer_list)}") # 在多个batch上再求平均，这里有点问题。
# print(f"Layer {args['split_layer']} InversedFIL: {sum(InversedFIL_same_layer_list)/len(InversedFIL_same_layer_list)}")
effecInfo_diff_layer_list.append(effecInfo_same_layer_list)


# 保存到csv中
matrix = np.array(effecInfo_diff_layer_list) # 有点大，x
transpose = matrix.T # 一行一条数据，一列代表一个layer 
# pd.DataFrame(data=transpose, columns=[i for i in split_layer_list]).to_csv(results_dir + f'effecInfo-bs{batch_size}.csv',index=False)
save_route = results_dir + f'effecInfo-120.csv'
if os.path.exists(save_route):
    df = pd.read_csv(save_route)
    df[args['split_layer']] = transpose
    df.to_csv(save_route,index=False)
else:
    pd.DataFrame(data=transpose, columns=[args['split_layer']]).to_csv(save_route,index=False)








