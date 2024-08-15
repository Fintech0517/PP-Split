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


args = {
        'device':torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        # 'device':torch.device("cpu"),
        'dataset':'CIFAR10',
        # 'dataset':'bank',
        # 'dataset':'credit',
        # 'dataset':'purchase',
        # 'dataset':'Iris',
        # 'result_dir': '20240702-FIL/',
        'result_dir': '20240702-effectiveInfo/',
        'oneData_bs': 500,
        'test_bs': 1,
        'train_bs': 1,
        'noise_scale': 0, # 防护措施
        'split_layer': 2,
        # 'test_num': 'invdFIL', # MI, invdFIL, distCor, ULoss, 
        # 'test_num': 'effectiveInfo1.1'
        'test_num': 'effectEntropy1.1'
        }
print(args['device'])



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
    print("joint_entropy: ",new_joint_prob_ent)
    # ent = information_top._compute_forward_entropies(x)
    # information_top.entropy_simplicial_lanscape(joint_prob_ent) # 画图
    # ent = _entropy(np.array(list(new_joint_prob_ent.values())))

    joint_entropy_final = list(new_joint_prob_ent.values())[-1]
    return joint_entropy_final




# effective entorpy

from pyentrp import entropy as ent

for j, data in enumerate(tqdm.tqdm(one_data_loader)): # 测试第一个testloader
# for j, data in enumerate(tqdm.tqdm(testloader)): 
    images, labels = data
    images, labels = images.to(args['device']), labels.to(args['device'])
    # print(images)
    with torch.no_grad():
        # ITE
        # effectEntro = Shannon_quantity(images)
        # print("effectEntro_ite: ",effectEntro)

        # PyEntropy
        effectEntro_pyent = 0.0
        for i in range(len(images[0])): # 对每个维度
            effectEntro_pyent  += shannon_entropy_pyent(images[:,i].flatten().detach().cpu().numpy())
            # print('effectEntro_pyent: ',effectEntro_pyent)
        # effectEntro_pyent = shannon_entropy(images.flatten(start_dim=1).detach().cpu().numpy())
        print('effectEntro_pyent: ',effectEntro_pyent)

        # infotopo
        # if conv:
        #     print('images: ', images.shape)
        #     images = avg_pool2d(images,kernel_size=4)
        #     print('images_pooled: ',images.shape)
            
        images_flattened = images.flatten(start_dim=1).detach().cpu().numpy()
        
        effectEntro_infotopo = shannon_entropy_infotopo(images_flattened,conv=conv)
        # print('effectEntro_infotopo: ',effectEntro_infotopo)


# nohup python -u effectEntropy.py >> effectEntropy.log 2>&1 & #[1] 2068