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

from target_model.task_select import get_dataloader_and_model,get_dataloader_and_model, \
    get_dataloader,get_models,get_infotopo_para
# utils
from ppsplit.utils.utils import create_dir

# %%
# %%
# nohup python -u effectInfo.py > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.3/effectInfo1.3-pool4.log 2>&1 &


import argparse
# parser
parser = argparse.ArgumentParser(description='PP-Split')
parser.add_argument('--device', type=str, default="cuda:0", help='device')
parser.add_argument('--dataset', type=str, default="CIFAR10", help='dataset') # 'bank', 'credit', 'purchase', 'Iris',
parser.add_argument('--model', type=str, default="ResNet18", help='model')  # 'ResNet18',' VGG5'
parser.add_argument('--result_dir', type=str, default="20240904-fisher/", help='result_dir')
parser.add_argument('--oneData_bs', type=int, default=1, help='oneData_bs')
parser.add_argument('--test_bs', type=int, default=1, help='test_bs')
parser.add_argument('--train_bs', type=int, default=1, help='train_bs')
parser.add_argument('--noise_scale', type=int, default=0, help='noise_scale')
parser.add_argument('--split_layer', type=int, default=2, help='split_layer')
parser.add_argument('--test_num', type=str, default='drjCodeFIL', help='test_num')
parser.add_argument('--no_dense', action='store_true', help='no_dense')

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
#         'result_dir': '20240904-fisher/',
#         'oneData_bs': 1,
#         'test_bs': 1,
#         'train_bs': 1,
#         'noise_scale': 0, # 防护措施
#         'split_layer': 5,
#         # 'test_num': 'invdFIL', # MI, invdFIL, distCor, ULoss,  # split layer [2,3,5,7,9,11] for ResNet18
#         'test_num': 'drjCodeFIL',
#         'no_dense':True,
#         }

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

print('results_dir:' ,results_dir)
print('inverse_dir:' ,inverse_dir)
print('decoder_route:' ,decoder_route)

create_dir(results_dir)

# client_net使用
client_net = client_net.to(args['device'])
client_net.eval()

# for n, p in client_net.named_parameters():
#     print(n, p.shape)


# %%
# 我实现的：
# dFIL inverse指标计算

eta_same_layer_list = []
eta_diff_layer_list=[]

metric = dFILInverseMetric()
# 对traingloader遍历计算所有 inverse dFIL
# for j, data in enumerate(tqdm.tqdm(testloader)):
for j, data in enumerate(tqdm.tqdm(one_data_loader)): # 测试第一个testloader
    # if j < 31705:
        # continue
    inputs, labels = data
    inputs, labels = inputs.to(args['device']), labels.to(args['device'])
    
    # inference
    outputs = client_net(inputs)

    eta = metric.quantify(model=client_net, inputs=inputs, outputs=outputs, with_outputs=True)
    # 打印
    # print(str(j)+": "+str(eta.item()))
    eta_same_layer_list.append(eta)
eta_diff_layer_list.append(eta_same_layer_list)

# 结果储存到csv中
# matrix = np.array(eta_diff_layer_list) # 有点大
# transpose = matrix.T # 一行一条数据，一列代表一个layer 
# pd.DataFrame(data=transpose, columns=[split_layer]).to_csv(results_dir + f'inv_dFIL{split_layer}.csv',index=False)

# print(matrix)

# 保存到csv中
matrix = np.array(eta_diff_layer_list) # 有点大，x
transpose = matrix.T # 一行一条数据，一列代表一个layer 
# pd.DataFrame(data=transpose, columns=[i for i in split_layer_list]).to_csv(results_dir + f'effecInfo-bs{batch_size}.csv',index=False)
save_route = results_dir + f'inverse_dFIL.csv'
if os.path.exists(save_route):
    df = pd.read_csv(save_route)
    df[args['split_layer']] = transpose
    df.to_csv(save_route,index=False)
else:
    pd.DataFrame(data=transpose, columns=[args['split_layer']]).to_csv(save_route,index=False)

