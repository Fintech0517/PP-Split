'''
Author: Ruijun Deng
Date: 2024-08-14 16:59:47
LastEditTime: 2024-08-25 00:47:50
LastEditors: Ruijun Deng
FilePath: /PP-Split/examples/effectInfo/effectInfo.ipynb
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
from ppsplit.quantification.shannon_information.FMInfo import FMInfoMetric
from ppsplit.quantification.shannon_information.ITE_tools import Shannon_quantity

from target_model.task_select import get_dataloader,get_models,get_infotopo_para,get_decoder

# utils
from ppsplit.utils import create_dir
# %%
# %%
# nohup python -u effectInfo1.8.py > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.8/effectInfo1.8-pool4-layer11-gpu.log 2>&1 &
# nohup python -u effectInfo1.8.py > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.8/effectInfo1.8-pool4-layer6-gpu.log 2>&1 &

parser = argparse.ArgumentParser(description='PP-Split')
parser.add_argument('--device', type=str, default="cuda:1", help='device')
parser.add_argument('--dataset', type=str, default="ImageNet1k", help='dataset') # 'bank', 'credit', 'purchase', 'Iris',
parser.add_argument('--model', type=str, default="ViTb_16", help='model')  # 'ResNet18'
parser.add_argument('--result_dir', type=str, default="20240702-effectiveInfo/", help='result_dir')
parser.add_argument('--oneData_bs', type=int, default=1, help='oneData_bs')
parser.add_argument('--test_bs', type=int, default=500, help='test_bs')
parser.add_argument('--train_bs', type=int, default=1, help='train_bs')
parser.add_argument('--noise_scale', type=float, default=0, help='noise_scale')
parser.add_argument('--split_layer', type=int, default=2, help='split_layer')
parser.add_argument('--test_num', type=str, default='effectiveInfo1.11', help='test_num')
parser.add_argument('--no_dense', action='store_true', help='no_dense')
parser.add_argument('--ep', type=int, help='epochs', default=-1)
parser.add_argument('--no_pool', action='store_true', help='no_pool', default=False)

args_python = parser.parse_args()
args = vars(args_python)

print(args)


# %%
data_msg = get_dataloader(args)
model_msg = get_models(args)
decoder_msg = get_decoder(args)
infotopo_msg = get_infotopo_para(args)


msg = {**model_msg,**data_msg,**infotopo_msg,**decoder_msg}

# 数据集
one_data_loader,trainloader,testloader = data_msg['one_data_loader'],data_msg['trainloader'], data_msg['testloader']
data_interval = data_msg['data_interval']
data_type = msg['data_type']
conv = msg['conv']
pool_size = msg['pool_size']

# effectEntropy Infotopo参数
nb_of_values = msg['nb_of_values']
print("infotopo: nb_of_values: ",nb_of_values)


# 模型
client_net,decoder_net = msg['client_net'],msg['decoder_net']
decoder_route = msg['decoder_route']
image_deprocess = msg['image_deprocess']

# 路径
results_dir = msg['results_dir']
inverse_dir = results_dir + 'layer' + str(args['split_layer'])+'/'
split_layer = args['split_layer']

print('results_dir:', results_dir)
print('inverse_dir:', inverse_dir)
print('decoder_route:', decoder_route)

create_dir(results_dir)

# client_net使用
client_net = client_net.to(args['device'])
client_net.eval()


# %% [markdown]
# ## 5.3 effectInfo

# 用定义的类
FMInfo_metric = FMInfoMetric(sigma = 0.01,device = args['device'])
FMInfo_diff_layer_list = []
FMInfo_same_layer_list = []
FMInfo_same_avg_d_layer_list = []

for j, data in enumerate(tqdm.tqdm(testloader)): # 对testloader遍历
# for j, data in enumerate(tqdm.tqdm(one_data_loader)): # 测试第一个testloader
    inputs, labels = data
    inputs, labels = inputs.to(args['device']), labels.to(args['device'])
    with torch.no_grad():
        print('inputs shape: ',inputs.shape)
        if conv:
            inputs = avg_pool2d(inputs,kernel_size=pool_size)
            print('inputs pooled shape: ',inputs.shape)

        FMInfo,avg_d_FMI = FMInfo_metric.quantify(inputs = inputs, client_net = client_net)

        FMInfo_same_layer_list.append(FMInfo)
        FMInfo_same_avg_d_layer_list.append(avg_d_FMI)

avg_FMInfo = sum(FMInfo_same_layer_list)/len(FMInfo_same_layer_list)
avg_d_FMInfo = sum(FMInfo_same_avg_d_layer_list)/len(FMInfo_same_avg_d_layer_list)

print(f"Layer {args['split_layer']} FMInfo: {avg_FMInfo}") # 在多个batch上再求平均，这里有点问题。
print(f"Layer {args['split_layer']} FMInfo_avg_d: {avg_d_FMInfo}") # 在多个batch上再求平均，这里有点问题。
FMInfo_diff_layer_list.append(FMInfo_same_layer_list)



# 保存到csv中
matrix = np.array(FMInfo_diff_layer_list) # 有点大，x
transpose = matrix.T # 一行一条数据，一列代表一个layer 
# pd.DataFrame(data=transpose, columns=[i for i in split_layer_list]).to_csv(results_dir + f'effecInfo-bs{batch_size}.csv',index=False)
save_route = results_dir + f'FMInfo-10000.csv'
if os.path.exists(save_route):
    df = pd.read_csv(save_route)
    df[args['split_layer']] = transpose
    df.to_csv(save_route,index=False)
else:
    pd.DataFrame(data=transpose, columns=[args['split_layer']]).to_csv(save_route,index=False)





