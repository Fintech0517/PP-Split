'''
Author: Ruijun Deng
Date: 2024-01-28 21:34:55
LastEditTime: 2024-07-02 16:14:25
LastEditors: Ruijun Deng
FilePath: /PP-Split/examples/attack-purchase.py
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
# # Purchase100 数据集 （表格数据多分类）

# %%
# 导包和超参数设置
from target_model.data_preprocessing.preprocess_purchase import preprocess_purchase

from target_model.models.PurchaseNet import PurchaseClassifier1,PurchaseDecoder1,purchase_cfg
from target_model.models.splitnn_utils import split_weights_client

test_num = 6 # 测试编号（对应结果文件夹名称）
split_layer = 8 # 模型切割点 （split point）在该层之前的层（含），作为client的模型，之后的层作为server的模型

# 重要路径设置
unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Purchase100/Purchase_bestmodel_param.pth'
results_dir = f'../../results/Purchase/{test_num}/'
inverse_dir = results_dir + 'layer'+str(split_layer)+'/'
decoder_net_route = results_dir + f'Decoder-layer{split_layer}.pth' # 攻击的decoder net存储位置


# %%
# 准备target model的 client net（对模型进行切割）
create_dir(results_dir)
create_dir(inverse_dir)

client_net = PurchaseClassifier1(layer=split_layer)
pweights = torch.load(unit_net_route)
if split_layer < len(purchase_cfg):
    pweights = split_weights_client(pweights,client_net.state_dict())
client_net.load_state_dict(pweights)


# %%
# 准备inverse_model attack使用到的东西
# 创建Inverse Model Attack对象
im_attack = InverseModelAttack(decoder_route=decoder_net_route,data_type=0,inverse_dir=inverse_dir)

# 加载decoder模型
if os.path.isfile(decoder_net_route): # 如果已经训练好了
    print("=> loading decoder model '{}'".format(decoder_net_route))
    decoder_net = torch.load(decoder_net_route)
else: # 如果没有
    print("train decoder model...")
    decoder_net = PurchaseDecoder1(layer=split_layer)
    # 训练decoder
    trainloader,testloader = preprocess_purchase(batch_size=32)

    decoder_net= im_attack.train_decoder(client_net=client_net,decoder_net=decoder_net,
                            train_loader=trainloader,test_loader=testloader,
                            epochs=20)

# %%
# 实现攻击,恢复testloader中所有表格数据行
trainloader,testloader = preprocess_purchase(batch_size=1)

im_attack.inverse(client_net=client_net,decoder_net=decoder_net,
                  train_loader=trainloader,test_loader=testloader,
                  save_fake=True)

# nohup python -u attack-purchase.py >> Purchase-6-8.out 2>&1  &