# %% [markdown]
# # 1. 基础设置

# %%
'''
Author: Ruijun Deng
Date: 2024-08-14 16:59:47
LastEditTime: 2024-09-16 23:29:42
LastEditors: Ruijun Deng
FilePath: /PP-Split/examples/effectInfo/effectInfo1.6.py
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
from ppsplit.utils.utils import create_dir

# %%
# %%
# nohup python -u effectInfo1.6.py > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.6/effectInfo1.6-pool4-layer11-gpu.log 2>&1 &
# nohup python -u effectInfo1.6.py > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.6/effectInfo1.6-pool4-layer7-gpu.log 2>&1 &
args = {
        'device':torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        # 'device':torch.device("cpu"),
        'dataset':'CIFAR10',
        # 'dataset':'bank',
        # 'dataset':'credit',
        # 'dataset':'purchase',
        # 'dataset':'Iris',
        'model': 'ResNet18',
        # 'model': 'VGG5',
        # 'result_dir': '20240702-FIL/',
        'result_dir': '20240702-effectiveInfo/',
        'oneData_bs': 500,
        'test_bs': 1,
        'train_bs': 1,
        'noise_scale': 0, # 防护措施
        'split_layer': 11,
        # 'test_num': 'invdFIL', # MI, invdFIL, distCor, ULoss,  # split layer [2,3,5,7,9,11] for ResNet18
        'test_num': 'effectiveInfo1.6',
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
        
        f2 = torch.logdet(I) # 直接用torch计算
        # f2_1 = torch.logdet(I_diag) # 和后面的是一样的
        f2_2 = torch.sum(torch.log(I_diagonal+1e-10)) # /I_diagonal.numel() # diagonal后计算

        f2_2_avg_outer += f2_2 / batch_size
        f2_avg_outer += f2 / batch_size

        # print('log det I: ', f2)
        # print('f1: ' , f1)
        print('f2: ', f2)
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

# arxiv'21 迁移学习领域的log det fisher 计算

# nips'23 fisher trace 计算
def calc_tr(net, x, device, sigmas=0.01, subsample=-1, jvp_parallelism=1): # nips'23 源码
    # 并行粒度=1 意思是，每次只处理一个维度

    print(f'x.shape: {x.shape}')
    
    # 定义一个局部函数 jvp_func**：这个函数接受两个参数 x 和 tgt，并返回 net.forward_first 方法的雅可比向量积（JVP）。
    # 这意味着 jvp_func 用于计算网络对于输入 x 在方向 tgt 上的一阶导数
    # tgt 计算雅各比向量积的向量
    def jvp_func(x, tgt): 
        # return jvp(net.forward_first, (x,), (tgt,)) #返回 outputs, jacobian product
        return jvp(net.forward, (x,), (tgt,)) #返回 outputs, jacobian product

    # 获取一个batch中第一个数据的维度？d代表的是批次中第一个数据点展平后的特征数量，即输入数据的维度。
    d = x[0].flatten().shape[0] # 把一个batch的x展平，获取input dim

    # 用于存储每个输入数据点的迹，求迹的和。
    tr = torch.zeros(x.shape[0], dtype=x.dtype).to(device)
    print(f'tr.shape: {tr.shape}')

    samples = range(d)

    for j in range(math.ceil(d)): # 对于每个数据块 # 每个数据块包含不同的维度
        tgts = []

        # 遍历每个数据块中的每个维度
        '''
        在这个函数中，tgt 是用于计算雅可比向量积（JVP）的向量。具体来说，tgt 的作用如下：
        构建雅可比向量积的向量：tgt 是一个与输入 x 形状相同的张量，但它的元素大部分为零，只有一个特定位置的元素为 1。这个特定位置对应于我们在计算迹时关注的特征维度。
        计算 JVP：在 helper 函数中，tgt 被传递给 jvp_func，用于计算网络对于输入 x 在方向 tgt 上的一阶导数。具体来说，jvp_func 计算的是网络输出相对于输入 x 的雅可比矩阵与 tgt 的乘积。
        估计迹：通过在不同的特征维度上重复上述过程，可以估计网络对于输入数据的迹。迹的计算涉及到对所有特征维度的导数进行求和，而 tgt 的作用就是在每次计算时只关注一个特征维度。
        简而言之，tgt 是一个用于选择特定特征维度的向量，通过它可以逐个计算每个特征维度的导数，从而最终估计整个输入数据的迹。
        '''
        # 对于每一列，构建tgt， 形状和x一样，但是只有一列是1，其他是0
        for k in samples[j:(j+1)]: # 提取整个batch中每个数据的特定维度
            tgt = torch.zeros_like(x).reshape(x.shape[0], -1) # 按照batch 排列？# 雅各比向量积的
            # 除了当前样本索引 k 对应的元素设置为 1。这相当于在计算迹时，每次只关注一个特征维度。
            tgt[:, k] = 1. # 提取tgt所有的样本的k的特征 计算雅各比向量积的向量，可用于计算trace，所有行的特定几列有1值
            tgt = tgt.reshape(x.shape) # 又变回x的形状
            # print(f'tgt.shape: {tgt.shape}')
            tgts.append(tgt) 
        tgts = torch.stack(tgts) # 把多个维度的tgt vstack，一行一行拼接起来，一行是一个维度。


        # 定义一个辅助函数 helper，该函数接受一个目标张量 tgt并返回一个迹的张量和一个值的张量。
        # jvp wrapper，遍历每个batchsize
        def helper(tgt,x=x): # x是一个batch的数据
            batch_size = x.shape[0]
            grads_list = []
            for i in range(batch_size): # 对每个样本
                _, grad = jvp_func(x[i].unsqueeze(0), tgt[i].unsqueeze(0))  # 对每个批次元素调用jvp_func
                grads_list.append(grad)
            # 将结果列表转换为张量, 多个batch的给stack起来
            grad = torch.stack(grads_list)

            # print('grad.shape: ',grad.shape)
            # print('grad: ',grad)

            # grad.reshape(sum(list(x.shape)),-1)
            # I_np = grad.cpu().detach().numpy()
            # df = pd.DataFrame(I_np)
            # df.to_csv(f'{time.time()}.csv',index=False,header=False)

            # print('grad*grad: ',grad*grad)
            # vals, grad = vmap(jvp_func, randomness='same')(x, tgt)
            
            # print('grad shape: ', grad.shape)
            # 因此，矩阵平方的迹和迹的平方通常是不相等的。
            # 先求平方再求迹
            # range(1, len(grad.shape)) 生成一个从 1 到 len(grad.shape) - 1 的整数序列。
            # torch.sum 函数对张量的指定维度进行求和。
            # 这里，它对 grad * grad 沿着 tuple(range(1, len(grad.shape))) 指定的维度进行求和。
            # ？为什么呢？--- 前面有个unsqueeze？
            return torch.sum(grad * grad, dim=tuple(range(1, len(grad.shape))))

        # vmap被替换
        # 遍历每个数据块
        trs,vals = [],[]
        for item in tgts: # 对每个维度
            trs_ = helper(item,x)
            trs.append(trs_) # 每个batch对应一个向量
            # print('trs_: ',trs_.shape)
        trs= torch.stack(trs) 
        trs = torch.log(trs+1e-10) # 为了求 f2 logdet
        # print('trs: ',trs.shape, trs)

        # 对数据，的每个维度的迹求和
        tr += trs.sum(dim=0) 
    print('tr: ',tr)

    return tr  # squeeze removes one dimension jvp puts

def f2_trace(net,x,device):
    tr = calc_tr(net, x, device, sigmas=0.01, subsample=-1, jvp_parallelism=1)
    # f2 = torch.log(tr)
    return tr

# %% [markdown]
# ## 5.2 effect uniform

# %%
# Effect uniform
import numpy as np
import torch
def calculate_effect_normalize(input_vector):
    # 确定每个维度的取值范围
    a = torch.tensor(2.0)
    # 计算每个维度的熵
    entropy_per_dimension = torch.log(a)
    # 总熵是每个维度的熵的总和
    size = input_vector.numel()
    total_entropy = size * entropy_per_dimension
    return total_entropy


def calculate_effect_normalize_hetero(input_vector):
    size = input_vector.numel()
    input_flattened = input_vector.reshape(-1)
    total_entropy_single = 0.0
    for i in range(size):
        l = 2*torch.min(torch.abs(input_flattened[i]-torch.tensor(-1.0)),torch.abs(input_flattened[i]-torch.tensor(1.0)))
        total_entropy_single += torch.log(l+1e-10)
    print(f"total_entropy_single: {total_entropy_single}")
    return total_entropy_single 


def calculate_effect_normalize_batch(inputs):
    # batchsize:
    batch_size = inputs.shape[0] # 一个batch的样本数目
    total_entropy = 0.0

    for i in range(batch_size):
        input_i = inputs[i].unsqueeze(0)
        total_entropy += calculate_effect_normalize_hetero(input_i)
    
    return total_entropy/batch_size

# %% [markdown]
# ## 5.3 effect Entropy

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
# ## 5.3 effectInfo

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

        # effectEntro = shannon_entropy_infotopo(images.flatten(start_dim=1).detach().cpu().numpy(), conv)
        
        # PyEntropy
        effectEntro_pyent = 0.0
        for i in range(len(images[0])): # 对每个维度
            effectEntro_pyent  += shannon_entropy_pyent(images[:,i].flatten().detach().cpu().numpy())
            # print('effectEntro_pyent: ',effectEntro_pyent)
        # effectEntro_pyent = shannon_entropy(images.flatten(start_dim=1).detach().cpu().numpy())
        print('effectEntro_pyent: ',effectEntro_pyent)

        # effect uniform
        # one_image = images[0]
        # effectUniform = calculate_effect_normalize(one_image.flatten())
        effectUniform = calculate_effect_normalize_batch(images)

        # effecit fisher
            # inference
        outputs = client_net(images).clone().detach()
        # inverse_dFIL = Fishermetric.quantify(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01, with_outputs=True)
        # effectFisher = computing_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)
        # effectFisher = computing_diag_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)
        # effectFisher = computing_diag_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)
        effectFisher = computing_diag_det_with_outputs(model=client_net, inputs=images, outputs=outputs,sigmas = 0.01)

        # 存储
        # effecInfo_same_layer_list.append(effectEntro-effectFisher)
        effectInfo = -effectFisher+effectUniform
        effecInfo_same_layer_list.append(effectInfo.detach().cpu().numpy())
        # InversedFIL_same_layer_list.append(inverse_dFIL)

        # 打印一下
        # print("effecEntro: ", effectEntro)
        print("effecFisher: ", effectFisher)
        print("effectUniform: ",effectUniform)
        # print("inverse_dFIL: ",inverse_dFIL)


print(f"Layer {args['split_layer']} effecInfo: {sum(effecInfo_same_layer_list)/len(effecInfo_same_layer_list)}") # 在多个batch上再求平均，这里有点问题。
# print(f"Layer {args['split_layer']} InversedFIL: {sum(InversedFIL_same_layer_list)/len(InversedFIL_same_layer_list)}")
effecInfo_diff_layer_list.append(effecInfo_same_layer_list)

# 保存到csv中
matrix = np.array(effecInfo_diff_layer_list) # 有点大，x
transpose = matrix.T # 一行一条数据，一列代表一个layer 
# pd.DataFrame(data=transpose, columns=[i for i in split_layer_list]).to_csv(results_dir + f'effecInfo-bs{batch_size}.csv',index=False)
save_route = results_dir + f'effecInfo-500.csv'
if os.path.exists(save_route):
    df = pd.read_csv(save_route)
    df[args['split_layer']] = transpose
    df.to_csv(save_route,index=False)
else:
    pd.DataFrame(data=transpose, columns=[args['split_layer']]).to_csv(save_route,index=False)
