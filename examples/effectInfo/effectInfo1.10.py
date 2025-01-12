# %% [markdown]
# # 1. 基础设置

# %%
'''
Author: Ruijun Deng
Date: 2024-08-14 16:59:47
LastEditTime: 2024-12-08 06:41:07
LastEditors: Ruijun Deng
FilePath: /PP-Split/examples/effectInfo/effectInfo1.10.py
Description: 
'''
# 导包
import torch
import os
import argparse
import pandas as pd
import tqdm
import numpy as np
from torch.nn.functional import avg_pool2d
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

from target_model.task_select import get_dataloader_and_model,get_dataloader_and_model, \
    get_dataloader,get_models,get_infotopo_para

# utils
from ppsplit.utils import create_dir

# %%
# %%
# nohup python -u effectInfo1.8.py > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.8/effectInfo1.8-pool4-layer11-gpu.log 2>&1 &
# nohup python -u effectInfo1.8.py > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.8/effectInfo1.8-pool4-layer6-gpu.log 2>&1 &

parser = argparse.ArgumentParser(description='PP-Split')
parser.add_argument('--device', type=str, default="cuda:0", help='device')
parser.add_argument('--dataset', type=str, default="CIFAR10", help='dataset') # 'bank', 'credit', 'purchase', 'Iris',
parser.add_argument('--model', type=str, default="ResNet18", help='model')  # 'ResNet18'
parser.add_argument('--result_dir', type=str, default="20240702-effectiveInfo/", help='result_dir')
parser.add_argument('--oneData_bs', type=int, default=5, help='oneData_bs')
parser.add_argument('--test_bs', type=int, default=500, help='test_bs')
parser.add_argument('--train_bs', type=int, default=1, help='train_bs')
parser.add_argument('--noise_scale', type=float, default=0, help='noise_scale')
parser.add_argument('--split_layer', type=int, default=2, help='split_layer')
parser.add_argument('--test_num', type=str, default='effectiveInfo1.10', help='test_num')
parser.add_argument('--no_dense', action='store_true', help='no_dense')
parser.add_argument('--ep', type=int, help='epochs', default=-1)

args_python = parser.parse_args()
args = vars(args_python)



# args = {
#         'device':torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#         # 'device':torch.device("cpu"),
#         'dataset':'CIFAR10',
#         # 'dataset':'bank',
#         # 'dataset':'credit',
#         # 'dataset':'purchase',
#         # 'dataset':'Iris',
#         # 'model': 'ResNet18',
#         'model': 'VGG5',
#         # 'result_dir': '20240702-FIL/',
#         'result_dir': '20240702-effectiveInfo/',
#         'oneData_bs': 1,
#         'test_bs': 500,
#         'train_bs': 1,
#         'noise_scale': 0, # 防护措施
#         'split_layer': 2,
#         # 'test_num': 'invdFIL', # MI, invdFIL, distCor, ULoss,  # split layer [2,3,5,7,9,11] for ResNet18
#         'test_num': 'effectiveInfo1.8.1',
#         'no_dense':True,
#         }
# print(args['device'])
# print(args)



# %%
data_msg = get_dataloader(args)
model_msg = get_models(args)
infotopo_msg = get_infotopo_para(args)
msg = {**model_msg,**data_msg,**infotopo_msg}

# 数据集
one_data_loader,trainloader,testloader = data_msg['one_data_loader'],data_msg['trainloader'], data_msg['testloader']
data_interval = data_msg['data_interval']
data_type = msg['data_type']

# effectEntropy Infotopo参数
nb_of_values = msg['nb_of_values']

conv = msg['conv']
pool_size = msg['pool_size']
# conv = False
print("infotopo: nb_of_values: ",nb_of_values)

# 模型
client_net,decoder_net = model_msg['client_net'],model_msg['decoder_net']
decoder_route = model_msg['decoder_route']
image_deprocess = model_msg['image_deprocess']

# 路径
results_dir = model_msg['results_dir']
inverse_dir = results_dir + 'layer' + str(args['split_layer'])+'/'
# data_type = 1 if args['dataset'] == 'CIFAR10' else 0
split_layer = args['split_layer']

print('results_dir:', results_dir)
print('inverse_dir:', inverse_dir)
print('decoder_route:', decoder_route)

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
def computing_diag_det_with_outputs(model, inputs, outputs, sigmas=1.0): # sigma_square
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

        # I = JtJ
        # print("I: ", I)
        # diagonal fisher information matrix (approximation)
        I_diagonal = torch.diagonal(I,dim1=0,dim2=1) # vector
        # print("I_diagonal: ",I_diagonal.shape)

        I_diag = torch.diag_embed(I_diagonal) # matrix
        # print('drj trace: ',torch.trace(I_diag))
        
        # batch的平均
        I_diagonal_batch_avg += I_diagonal / (batch_size)

        # # 储存I
        # I_np = I.cpu().detach().numpy()
        # df = pd.DataFrame(I_np)
        # df.to_csv(f'{i}.csv',index=False,header=False)

        # print("I: ", I)
        # w = torch.det(I)
        # print('det I: ', I.det().log())
        
        try:
            s,f2 = torch.slogdet(I) # 直接用torch计算
            if s <= 0:
                raise RuntimeError("sign <=0 ")
            print('f2: ', f2)
        except RuntimeError as e:
            print("logdet计算报错")
        # f2_1 = torch.logdet(I_diag) # 和后面的是一样的
        f2_2 = torch.sum(torch.log(I_diagonal+1e-10)) # /I_diagonal.numel() # diagonal后计算

        f2_2_avg_outer += f2_2 / batch_size
        # f2_avg_outer += f2 / batch_size

        # print('log det I: ', f2)
        # print('f1: ' , f1)
        # print('f2: ', f2)
        # print('f2_1: ', f2_1)
        print('f2_2: ', f2_2)

    f2_2_avg_inner = torch.sum(torch.log(I_diagonal_batch_avg+1e-10)) # 用平均后的diagonal 计算

    print('f2_avg_outer: ',f2_avg_outer)
    print('f2_2_avg_outer: ',f2_2_avg_outer)
    # print('f2_2_avg_inner: ',f2_2_avg_inner)
    print('f1: ',f1)

    # effect_fisher = 0.5 * (f1 - f2_2_avg_inner)
    effect_fisher = 0.5 * (f1 - f2_2_avg_outer)
    # effect_fisher = 0.5 * (f1 - f2_avg_outer)
    # effect_fisher_sum+=effect_fisher

    # print("effect_fisher: ",effect_fisher)
    
    # effect_fisher_mean = effect_fisher_sum / batch_size
    return effect_fisher.cpu().detach().numpy()


# %% [markdown]
# ## 5.2 effect uniform

# %%
# Effect uniform
import numpy as np
import torch
def calculate_effect_normalize(input_vector,interval=(-1.0,1.0)):
    interval_len = interval[1] - interval[0]
    # 确定每个维度的取值范围
    a = torch.tensor(interval_len)
    # 计算每个维度的熵
    entropy_per_dimension = torch.log(a)
    # 总熵是每个维度的熵的总和
    size = input_vector.numel()
    total_entropy = size * entropy_per_dimension
    return total_entropy


def calculate_effect_normalize_hetero(input_vector, interval=(1.0,-1.0)):
    size = input_vector.numel()
    input_flattened = input_vector.reshape(-1)
    total_entropy_single = 0.0
    for i in range(size):
        l = 2*torch.min(torch.abs(input_flattened[i]-torch.tensor(interval[0])),torch.abs(input_flattened[i]-torch.tensor(interval[1])))
        total_entropy_single += torch.log(l+1e-10)
    print(f"entropy for single_input: {total_entropy_single}")
    return total_entropy_single 


def calculate_effect_normalize_hetero_batch(inputs, interval=(1.0,-1.0)):
    # batchsize:
    batch_size = inputs.shape[0] # 一个batch的样本数目
    total_entropy = 0.0

    for i in range(batch_size):
        input_i = inputs[i].unsqueeze(0)
        total_entropy += calculate_effect_normalize_hetero(input_i,interval)
    
    return total_entropy/batch_size


# %% [markdown]
# ## 5.3 effect Entropy

# %%
# effect entropy 计算函数
import math
import numpy as np
def shannon_entropy_pyent(time_series): # 这个甚至不适合连续值吧
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
                                        # nb_of_values = 17, # 不是很懂这个意思，为什么iris对应9？
                                        nb_of_values = nb_of_values, # 不是很懂这个意思，为什么iris对应9？
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
    
    # print("joint_entropy: ",new_joint_prob_ent)
    # ent = information_top._compute_forward_entropies(x)
    # information_top.entropy_simplicial_lanscape(joint_prob_ent) # 画图
    # ent = _entropy(np.array(list(new_joint_prob_ent.values())))

    joint_entropy_final = list(new_joint_prob_ent.values())[-1]
    return joint_entropy_final


# %% [markdown]
# ## 5.3 effectInfo

# %%
effecInfo_diff_layer_list = []
effecInfo_same_layer_list = []
EntropyMetric = ULossMetric()
Fishermetric = dFILInverseMetric()


InversedFIL_same_layer_list = []
NBatch = len(testloader)
image_dimension = -1

for j, data in enumerate(tqdm.tqdm(testloader)): # 对testloader遍历
# for j, data in enumerate(tqdm.tqdm(one_data_loader)): # 测试第一个testloader
    images, labels = data
    images, labels = images.to(args['device']), labels.to(args['device'])
    with torch.no_grad():
        print('images: ', images.shape)
        
        if conv:
            images= avg_pool2d(images,kernel_size=pool_size)
            print('images_pooled: ',images.shape)
        if image_dimension ==-1:
            image_dimension = images[0].numel()

        # ITE
        effectEntro = Shannon_quantity(images)
        print("effectEntro_ite: ",effectEntro)

        # effect fisher
            # inference
        outputs = client_net(images).clone().detach()
        # inverse_dFIL = Fishermetric.quantify(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01, with_outputs=True)
        # effectFisher = computing_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)
        # effectFisher = computing_diag_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)
        # effectFisher = computing_diag_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)
        effectFisher = computing_diag_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)

        # effect Uniform 
        effectUniform = calculate_effect_normalize_hetero_batch(images,data_interval)
        # uniform interval
        effectUniform_interval = calculate_effect_normalize(images[0],data_interval) # 用第一张图片就可

        # 存储
        # effecInfo_same_layer_list.append(effectEntro-effectFisher)
        effectInfo = (effectEntro-effectFisher)
        effecInfo_same_layer_list.append(effectInfo)
        # InversedFIL_same_layer_list.append(inverse_dFIL)

        # 打印一下
        print("effecEntro: ", effectEntro)
        print("effecFisher: ", effectFisher)
        print("effectUniform: ",effectUniform)
        print('effectUniform_interval: ',effectUniform_interval)
        # print("inverse_dFIL: ",inverse_dFIL)

avg_effectInfo = sum(effecInfo_same_layer_list)/len(effecInfo_same_layer_list)
avg_d_effectInfo = avg_effectInfo/image_dimension
print(f"Layer {args['split_layer']} effecInfo: {avg_effectInfo}") # 在多个batch上再求平均，这里有点问题。
print(f"Layer {args['split_layer']} effecInfo_avg_d: {avg_d_effectInfo}") # 在多个batch上再求平均，这里有点问题。
# print(f"Layer {args['split_layer']} InversedFIL: {sum(InversedFIL_same_layer_list)/len(InversedFIL_same_layer_list)}")
effecInfo_diff_layer_list.append(effecInfo_same_layer_list)

# 保存到csv中
matrix = np.array(effecInfo_diff_layer_list) # 有点大，x
transpose = matrix.T # 一行一条数据，一列代表一个layer 
# pd.DataFrame(data=transpose, columns=[i for i in split_layer_list]).to_csv(results_dir + f'effecInfo-bs{batch_size}.csv',index=False)
save_route = results_dir + f'effecInfo-10000.csv'
if os.path.exists(save_route):
    df = pd.read_csv(save_route)
    df[args['split_layer']] = transpose
    df.to_csv(save_route,index=False)
else:
    pd.DataFrame(data=transpose, columns=[args['split_layer']]).to_csv(save_route,index=False)

