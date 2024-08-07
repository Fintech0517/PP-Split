# %% [markdown]
# # 1. 基础设置

# %%
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

from target_model.task_select import get_dataloader_and_model,get_dataloader_and_model, get_dataloader,get_models

# utils
from ppsplit.utils.utils import create_dir

# %%
args = {
        # 'device':torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        'device':torch.device("cpu"),
        # 'dataset':'CIFAR10',
        # 'dataset':'bank',
        # 'dataset':'credit',
        # 'dataset':'purchase',
        'dataset':'Iris',
        # 'result_dir': '20240702-FIL/',
        'result_dir': '20240702-effectiveInfo/',
        'oneData_bs': 120,
        'test_bs': 1,
        'train_bs': 1,
        'noise_scale': 0, # 防护措施
        'OneData':True,
        'split_layer': 5,
        # 'test_num': 'invdFIL', # MI, invdFIL, distCor, ULoss, 
        'test_num': 'effectiveInfo1.1'
        }
# print(args['device'])
print(args)

# %%
data_msg = get_dataloader(args)
model_msg = get_models(args)
msg = {**model_msg,**data_msg}

# 数据集
one_data_loader,trainloader,testloader = data_msg['one_data_loader'],data_msg['trainloader'], data_msg['testloader']

# 模型和路径
client_net,decoder_net = model_msg['client_net'],model_msg['decoder_net']
decoder_route = model_msg['decoder_route']
image_deprocess = model_msg['image_deprocess']

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

# 自己实现的、规规矩矩的
def computing_det_with_outputs(model, inputs, outputs, sigmas): # sigma_square
        # batchsize:
        batch_size = inputs.shape[0] # 一个batch的样本数目
        output_size = outputs[0].numel() # 一个样本的outputs长度
        input_size = inputs[0].numel() # 一个样本的outputs长度
        effect_fisher_sum = 0.0

        # 遍历单个样本: 换数据
        for i in range(batch_size):
            input_i = inputs[i].unsqueeze(0)

            # 计算jacobian
            J = F.jacobian(model, input_i)
            # J = J.reshape(J.shape[0],outputs.numel(),inputs.numel()) # (batch, out_size, in_size)
            J = J.reshape(output_size, input_size) # (batch, out_size, in_size)
            # print(f"J2.shape: {J.shape}, J2.prod: {torch.prod(torch.tensor(list(J.shape)))}")
            # 计算eta
            JtJ = torch.matmul(J.t(), J)
            I = 1.0/(sigmas)*JtJ
            # ddFIL  = I.trace().div(input_size*input_size)

            # 储存I
            # I_np = I.cpu().detach().numpy()
            # df = pd.DataFrame(I_np)
            # df.to_csv(f'{i}.csv',index=False,header=False)

            # print("I: ", I)
            # w = torch.det(I)
            # print('det I: ', I.det().log())

            f1 = input_size * torch.log(2*torch.pi*torch.exp(torch.tensor(1.0)))
            f2 = torch.logdet(I)
            # print('log det I: ',f2 )
            print('f1: ',f1)
            print('f2: ',f2)
            effect_fisher = 0.5 * (f1 - f2)
            effect_fisher_sum+=effect_fisher

            print("effect_fisher: ",effect_fisher)

        # print("Jt*J: ", JtJ)
        # print("Jt*J: ", JtJ.shape, JtJ)
        # print("I.shape: ", I.shape)
        # eta = dFIL
        # print(f"eta: {eta}")
        # print('t2-t1=',t2-t1, 't3-t2', t3-t2)
        effect_fisher_mean = effect_fisher_sum / batch_size
        return effect_fisher_mean.cpu().detach().numpy()

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
        for i in range(batch_size): # 对每个batch
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

            f2 = torch.logdet(I)
            # f2_1 = torch.logdet(I_diag)
            f2_2 = torch.sum(torch.log(I_diagonal+1e-10)) # /I_diagonal.numel()

            f2_2_avg_outer += f2_2 / batch_size
            f2_avg_outer += f2 / batch_size

            # print('log det I: ',f2 )
            # print('f1: ',f1)
            print('f2: ',f2)
            # print('f2_1: ',f2_1)
            print('f2_2: ',f2_2)

        f2_2_avg_inner = torch.sum(torch.log(I_diagonal_batch_avg+1e-10))

        print('f2_2_avg_outer: ',f2_2_avg_outer)
        print('f2_2_avg_inner: ',f2_2_avg_inner)

        # effect_fisher = 0.5 * (f1 - f2_2_avg_inner)
        effect_fisher = 0.5 * (f1 - f2_2_avg_outer)
        # effect_fisher = 0.5 * (f1 - f2_avg_outer)
        # effect_fisher_sum+=effect_fisher

        # print("effect_fisher: ",effect_fisher)
        
        # effect_fisher_mean = effect_fisher_sum / batch_size
        return effect_fisher.cpu().detach().numpy()

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

import infotopo

def shannon_entropy_infotopo(x):
    information_top = infotopo.infotopo(dimension_max = x.shape[1],
                                        dimension_tot = x.shape[1],
                                        sample_size = x.shape[0],
                                        nb_of_values = 9, # 不是很懂这个意思，为什么iris对应9？
                                        )
    # 计算联合分布的概率？（全排列）
    joint_prob = information_top._compute_probability(x)
    print('joint_prob: ',joint_prob)
    # 计算联合熵（全排列的）
    joint_prob_ent = information_top.simplicial_entropies_decomposition(x) # log2
    new_joint_prob_ent = {key: value * np.log(2) for key, value in joint_prob_ent.items()} #ln 转2为底 成 e为底
    print("joint_entropy: ",new_joint_prob_ent)
    # ent = information_top._compute_forward_entropies(x)
    # information_top.entropy_simplicial_lanscape(joint_prob) # 画图
    # ent = _entropy(np.array(list(new_joint_prob_ent.values())))

    joint_entropy_final = list(new_joint_prob_ent.values())[-1]
    return joint_entropy_final


# %%
# effect Information 指标计算

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
        # inference
        outputs = client_net(images).clone().detach()
        # effect entropy 
        # effecEntro= EntropyMetric._entropy_prob_batch(images) # H(x)
        effectEntro = shannon_entropy_infotopo(images.flatten(start_dim=1).detach().cpu().numpy())

        # effecit fisher
        # outputs = client_net(images)
        inverse_dFIL = Fishermetric.quantify(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01, with_outputs=True)
        effectFisher = computing_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)

        # 存储
        effecInfo_same_layer_list.append(effectEntro-effectFisher)
        InversedFIL_same_layer_list.append(inverse_dFIL)

        # 打印一下
        print("effecEntro: ", effectEntro)
        print("inverse_dFIL: ",inverse_dFIL)

print(f"Layer {args['split_layer']} effecInfo: {sum(effecInfo_same_layer_list)/len(effecInfo_same_layer_list)}") # 在多个batch上再求平均，这里有点问题。
print(f"Layer {args['split_layer']} InversedFIL: {sum(InversedFIL_same_layer_list)/len(InversedFIL_same_layer_list)}")
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

