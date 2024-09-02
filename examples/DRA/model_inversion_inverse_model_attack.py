'''
Author: Ruijun Deng
Date: 2024-08-24 00:41:30
LastEditTime: 2024-09-02 01:15:35
LastEditors: Ruijun Deng
FilePath: /PP-Split/examples/DRA/model_inversion_inverse_model_attack.py
Description: 
'''
# 导包
import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from ppsplit.attacks.model_inversion.inverse_model import InverseModelAttack
from ppsplit.utils.utils import create_dir
import torch
import os

# 防护措施
from ppsplit.defense.noise import Noise

# 模型、数据集获取
from target_model.task_select import get_dataloader_and_model, get_dataloader,get_models

# nohup python -u model_inversion_inverse_model_attack.py > ../../results/inverse-model-results-20240414//VGG5/InverseModelAttack/layer6.log 2>&1 &
# 超参数
args = {
        'device':torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        # 'device':torch.device("cpu"),
        'dataset':'CIFAR10',
        # 'dataset':'bank',
        # 'dataset':'credit',
        # 'dataset':'purchase',
        # 'dataset':'Iris',
        # 'model': 'ResNet18',
        'model': 'VGG5',
        'result_dir': 'inverse-model-results-20240414/',
        'oneData_bs': 1,
        'test_bs': 1,
        'train_bs': 32,
        'noise_scale':0, # 防护措施
        'split_layer': 6,
        'test_num': 'InverseModelAttack'
        # 'num_pairs': 10000, # RepE # 这个要另外准备
        }

print(args['device'])
print(args)

# 获取模型和数据集
# msg = get_dataloader_and_model(**args)

model_msg = get_models(args)

# one_data_loader,trainloader,testloader = model_msg['one_data_loader'],model_msg['trainloader'], model_msg['testloader']
client_net,decoder_net = model_msg['client_net'],model_msg['decoder_net']
decoder_route = model_msg['decoder_route']
image_deprocess = model_msg['image_deprocess']

results_dir = model_msg['results_dir']
inverse_dir = results_dir + 'layer'+str(args['split_layer'])+'/'
data_type = 1 if args['dataset'] == 'CIFAR10' else 0

print('results_dir:',results_dir)
print('inverse_dir:',inverse_dir)
print('decoder_route:',decoder_route)


# 准备inverse_model attack使用到的东西
# 创建Inverse Model Attack对象
im_attack = InverseModelAttack(decoder_route=decoder_route,data_type=data_type,inverse_dir=inverse_dir)

# 加载decoder模型
if not os.path.isfile(decoder_route): # 如果没有训练decoder
    # 训练decoder
    args['train_bs']=32
    args['test_bs']=32
    msg_data = get_dataloader(args)
    # trainloader,testloader = get_cifar10_normalize(batch_size=32)
    decoder_net= im_attack.train_decoder(client_net=client_net,decoder_net=decoder_net,
                            train_loader=msg_data['trainloader'],test_loader=msg_data['testloader'],
                            epochs=20)
else:
    print("Load decoder model from:",decoder_route)

print(decoder_net)


# 实现攻击,恢复testloader中所有图片
# trainloader,testloader = get_cifar10_normalize(batch_size=1)
args['train_bs']=1
args['test_bs']=1
msg_data = get_dataloader(args)

im_attack.inverse(client_net=client_net,decoder_net=decoder_net,
                  train_loader=msg_data['trainloader'],test_loader=msg_data['testloader'],
                  deprocess=image_deprocess,
                  save_fake=True,
                  tab=msg_data['tabinfo'])







