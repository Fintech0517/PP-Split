# %% [markdown]
# 

# %%
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

# task select
from target_model.task_select import get_dataloader_and_model,get_dataloader_and_model, \
    get_dataloader,get_models,get_infotopo_para

# utils
from ppsplit.utils import concat_weights, create_dir, load_json, save_json

# defense:
from ppsplit.defense.obfuscation.scheduler import Scheduler


# %%
config = load_json('./config/nopeek.json')
# config = load_json('./config/shredder.json')
# config = load_json('./config/cloak.json')
# config = load_json('./config/uniform_noise.json')

# 加入脚本
import argparse
# parser
parser = argparse.ArgumentParser(description='PP-Split')
parser.add_argument('--result_dir', type=str, default="20241228-defense/", help='result_dir')
parser.add_argument('--test_num', type=str, default='nopeek', help='test_num')
parser.add_argument('--device', type=str, default="cuda:0", help='device')
parser.add_argument('--dataset', type=str, default="CIFAR10", help='dataset') # 'bank', 'credit', 'purchase', 'Iris',
parser.add_argument('--oneData_bs', type=int, default=1, help='oneData_bs')
parser.add_argument('--train_bs', type=int, default=32, help='train_bs')
parser.add_argument('--test_bs', type=int, default=64, help='test_bs')
parser.add_argument('--model', type=str, default="ResNet18", help='model')  # 'ResNet18',' VGG5'
parser.add_argument('--split_layer', type=int, default=2, help='split_layer')
parser.add_argument('--ep', type=int, help='epochs', default=-1)
parser.add_argument('--no_dense', action='store_true', help='no_dense')
parser.add_argument('--noise_scale', type=float, default=0, help='noise_scale')

args_python = parser.parse_args()
args_python = vars(args_python)

# 更新config中的general
config['general'] = args_python 
args = config['general']
config['defense']['device']=args['device']

print(config)


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
server_net,unit_net = model_msg['server_net'], model_msg['unit_net']
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

create_dir(inverse_dir)

# net使用
client_net = client_net.to(args['device'])
server_net = server_net.to(args['device'])
unit_net = unit_net.to(args['device'])


# %%
# 防御
config['defense']["results_dir"] = results_dir
# config["1"] = results_dir
defense_scheduler = Scheduler(config)
if config['defense']['method']=='cloak':
    client_net = None
    server_net = unit_net

# 初始化和run存储模型
defense_scheduler.initialize(train_loader = trainloader, test_loader = testloader, client_model = client_net, server_model = server_net)
client_net,server_net = defense_scheduler.run_job()

# 拼接模型并保存
if client_net: # 如果有client_net，则拼接
    new_weights_unit = concat_weights(unit_net.state_dict(),client_net.state_dict(),server_net.state_dict())
else: # 如果没有client_net，则直接保存server_net
    new_weights_unit = server_net.state_dict()

unit_net.load_state_dict(new_weights_unit)
torch.save(unit_net.state_dict(), inverse_dir + 'unit_net_defensed.pth')

print("model saved in ",inverse_dir + 'unit_net_defensed.pth')

# %%



