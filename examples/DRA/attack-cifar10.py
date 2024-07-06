'''
Author: Ruijun Deng
Date: 2024-07-02 16:14:03
LastEditTime: 2024-07-02 16:14:04
LastEditors: Ruijun Deng
FilePath: /PP-Split/examples/attack-cifar10.py
Description: 
'''
# %%
# 这个notebook 介绍了 如何对split learning 发起 inverse-model attack攻击

# %%
# 导包
import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from ppsplit.attacks.model_inversion.inverse_model import InverseModelAttack
from ppsplit.utils.utils import create_dir
import torch
import os


# %% [markdown]
# # cifar10 （图像多分类）

# %%
# 导包和超参数设置
from target_model.data_preprocessing.preprocess_cifar10 import get_cifar10_normalize,get_one_data,deprocess

from target_model.models.VGG import VGG,VGG5Decoder,model_cfg
from target_model.models.splitnn_utils import split_weights_client

test_num = 9 # 测试编号（对应结果文件夹名称）
split_layer = 6 # 模型切割点 （split point）在该层之前的层（含），作为client的模型，之后的层作为server的模型

# 重要路径设置
unit_net_route = '/home/dengruijun/data/project/Inverse_efficacy/results/VGG5/BN+Tanh/2-20240101/VGG5-params-19ep.pth'
results_dir = f'../../results/VGG5/{test_num}/'
inverse_dir = results_dir + 'layer'+str(split_layer)+'/'
decoder_net_route = results_dir + f'Decoder-layer{split_layer}.pth' # 攻击的decoder net存储位置


# %%
# 准备基本模型client net
# split_layer_list = list(range(len(model_cfg['VGG5']))) # 可能的切割点

# 创建对应文件夹
create_dir(results_dir)
create_dir(inverse_dir)


# 把unit模型切割成client-server 的模型pair
client_net = VGG('Client','VGG5',split_layer,model_cfg)
pweights = torch.load(unit_net_route)
if split_layer < len(model_cfg['VGG5']):
    pweights = split_weights_client(pweights,client_net.state_dict())
client_net.load_state_dict(pweights)


# %%
# 准备inverse_model attack使用到的东西
# 创建Inverse Model Attack对象
im_attack = InverseModelAttack(decoder_route=decoder_net_route,data_type=1,inverse_dir=inverse_dir)

# 加载decoder模型
if os.path.isfile(decoder_net_route): # 如果已经训练好了
    print("=> loading decoder model '{}'".format(decoder_net_route))
    decoder_net = torch.load(decoder_net_route)
else: # 如果没有
    print("train decoder model...")
    decoder_net = VGG5Decoder(split_layer=split_layer)
    # 训练decoder
    trainloader,testloader = get_cifar10_normalize(batch_size=32)

    decoder_net= im_attack.train_decoder(client_net=client_net,decoder_net=decoder_net,
                            train_loader=trainloader,test_loader=testloader,
                            epochs=20)


# %%
# 实现攻击,恢复testloader中所有图片
trainloader,testloader = get_cifar10_normalize(batch_size=1)

im_attack.inverse(client_net=client_net,decoder_net=decoder_net,
                  train_loader=trainloader,test_loader=testloader,
                  deprocess=deprocess,
                  save_fake=True)

# nohup python -u attack-cifar10.py >> cifar-3-0.out 2>&1  &
# nohup python -u attack-cifar10.py >> cifar-4-1.out 2>&1  &
# nohup python -u attack-cifar10.py >> cifar-5-2.out 2>&1  &
# nohup python -u attack-cifar10.py >> cifar-6-3.out 2>&1  &
# nohup python -u attack-cifar10.py >> cifar-7-4.out 2>&1  &
# nohup python -u attack-cifar10.py >> cifar-8-5.out 2>&1  &
# nohup python -u attack-cifar10.py >> cifar-9-6.out 2>&1  &
