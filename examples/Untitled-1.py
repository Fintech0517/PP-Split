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
# # Credit 数据集 （表格数据二分类）

# %%
# 导包和超参数设置
from target_model.data_preprocessing.preprocess_credit import preprocess_credit

from target_model.models.CreditNet import CreditNet1,CreditNetDecoder1,credit_cfg
from target_model.models.splitnn_utils import split_weights_client

test_num = 3 # 测试编号（对应结果文件夹名称）
split_layer = 3 # 模型切割点 （split point）在该层之前的层（含），作为client的模型，之后的层作为server的模型

# 重要路径设置
unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/credit/credit-20ep_params.pth'
results_dir = f'../results/Credit/{test_num}/'
inverse_dir = results_dir + 'layer'+str(split_layer)+'/' # 储存
decoder_net_route = results_dir + f'Decoder-layer{split_layer}.pth' # 攻击的decoder net存储位置

# %%
# 准备target model的 client net（对模型进行切割）
create_dir(results_dir)
create_dir(inverse_dir)

client_net = CreditNet1(layer=split_layer)
pweights = torch.load(unit_net_route)
if split_layer < len(credit_cfg):
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
    decoder_net = CreditNetDecoder1(layer=split_layer)
    # optimizer = torch.optim.SGD(decoder_net.parameters(), 1e-3)

    # 训练decoder
    trainloader,testloader = preprocess_credit(batch_size=32)

    decoder_net= im_attack.train_decoder(client_net=client_net,decoder_net=decoder_net,
                            train_loader=trainloader,test_loader=testloader,
                            epochs=120,
                            # optimizer=optimizer
                            )

# %%
# 实现攻击,恢复testloader中所有表格数据行
trainloader,testloader = preprocess_credit(batch_size=1)

im_attack.inverse(client_net=client_net,decoder_net=decoder_net,
                  train_loader=trainloader,test_loader=testloader,
                  save_fake=True)

# [1] 42117