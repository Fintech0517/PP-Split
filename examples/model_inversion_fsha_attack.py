# %%
# 这个notebook 发起了 fsha 数据重构攻击

# %%
# 导包
import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from ppsplit.attacks.model_inversion.fsha import FSHA_Attack, discriminator_net
from ppsplit.utils.utils import create_dir
import torch
import os
from torch.utils.data import DataLoader

# 导入各个baseline模型及其数据集预处理方法
# 模型
from target_model.models.splitnn_utils import split_weights_client
from target_model.models.VGG import VGG,VGG5Decoder,model_cfg
from target_model.models.BankNet import BankNet1,BankNetDecoder1,bank_cfg
from target_model.models.CreditNet import CreditNet1,CreditNetDecoder1,credit_cfg
from target_model.models.PurchaseNet import PurchaseClassifier1,PurchaseDecoder1,purchase_cfg
# 数据预处理方法
from target_model.data_preprocessing.preprocess_cifar10 import get_cifar10_normalize,get_one_data,deprocess
from target_model.data_preprocessing.preprocess_bank import bank_dataset,preprocess_bank
from target_model.data_preprocessing.preprocess_credit import preprocess_credit
from target_model.data_preprocessing.preprocess_purchase import preprocess_purchase


# %%
# 一些超参数
args = {
        'device':torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        # 'device':torch.device("cpu"),
        'dataset':'CIFAR10',
        # 'dataset':'bank',
        # 'dataset':'credit',
        # 'dataset':'purchase',
        'batch_size':1
        }
print(args['device'])
args['data_type']= 1 if args['dataset']=='CIFAR10' else 0 # 区分图像数据（1）和表格数据（0）
print(args['dataset'])

# %% [markdown]
# ## 数据集和模型加载

# %%
# 数据集和模型加载
# 加载模型和数据集，并从unit模型中切割出client_model
if args['dataset']=='CIFAR10':
    # 超参数
    testset_len = 10000 # 10000个数据一次 整个测试集合的长度
    # split_layer_list = list(range(len(model_cfg['VGG5'])))
    split_layer = 2 # 定成3吧？
    test_num = 1 # 试验序号
    
    # 关键路径
    unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG5/BN+Tanh/VGG5-params-20ep.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
    results_dir  = f'../results/rMLE-results-20240413/VGG5/{test_num}/'
    decoder_route = f"../results/VGG5/{test_num}/Decoder-layer{split_layer}.pth"

    # 数据集加载
    trainloader,testloader = get_cifar10_normalize(batch_size=1)
    one_data_loader = get_one_data(testloader,batch_size = args['batch_size']) #拿到第一个测试数据

    # 切割成client model
    client_net = VGG('Client','VGG5',split_layer,model_cfg)
    pweights = torch.load(unit_net_route)
    if split_layer < len(model_cfg['VGG5']):
        pweights = split_weights_client(pweights,client_net.state_dict())
    client_net.load_state_dict(pweights)
    
    # 其他fsha要用到的网络
    shadow_net = VGG('Client','VGG5',split_layer,model_cfg)
    decoder_net = VGG5Decoder(split_layer=split_layer)

elif args['dataset']=='bank':
    # 超参数
    test_num = 1 # 试验序号
    testset_len=8238
    # split_layer_list = ['linear1', 'linear2']
    split_layer_list = [0,2,4,6]
    split_layer = 2

    # 关键路径
    results_dir  = f'../results/rMLE-results-20240413/Bank/{test_num}/'
    unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Bank/bank-20ep_params.pth'
    decoder_route = f"../results/Bank/{test_num}/Decoder-layer{split_layer}.pth"

    # 数据集加载
    trainloader,testloader = preprocess_bank(batch_size=1)
    one_data_loader = get_one_data(testloader,batch_size = args['batch_size']) #拿到第一个测试数据 

    # 模型加载
    client_net = BankNet1(layer=split_layer)
    pweights = torch.load(unit_net_route)
    if split_layer < len(bank_cfg):
        pweights = split_weights_client(pweights,client_net.state_dict())
    client_net.load_state_dict(pweights)

    # 其他fsha要用到的网络
    shadow_net = BankNet1(layer=split_layer)
    decoder_net = BankNetDecoder1(layer=split_layer)

elif args['dataset']=='credit':
    # 超参数
    test_num = 1 # 试验序号
    testset_len = 61503 # for the mutual information
    split_layer_list = [0,3,6,9]
    split_layer = 3

    # 关键路径
    results_dir  = f'../results/rMLE-results-20240413/Credit/{test_num}/'
    unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/credit/credit-20ep_params.pth'
    decoder_route = f"../results/Credit/{test_num}/Decoder-layer{split_layer}.pth"

    # 数据集加载
    trainloader,testloader = preprocess_credit(batch_size=1)
    one_data_loader = get_one_data(testloader,batch_size = args['batch_size']) #拿到第一个测试数据

    # client模型切割加载
    client_net = CreditNet1(layer=split_layer)
    pweights = torch.load(unit_net_route)
    if split_layer < len(credit_cfg):
        pweights = split_weights_client(pweights,client_net.state_dict())
    client_net.load_state_dict(pweights)

    # 其他fsha要用到的网络
    shadow_net = CreditNet1(layer=split_layer)
    decoder_net = CreditNetDecoder1(layer=split_layer)

elif args['dataset']=='purchase':
    # 超参数
    test_num = 1 # 试验序号
    testset_len = 39465 # test len
    # split_layer_list = [0,1,2,3,4,5,6,7,8]
    split_layer = 3

    # 关键路径
    results_dir = f'../results/rMLE-results-20240413/Purchase/{test_num}/'
    unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Purchase100/Purchase_bestmodel_param.pth'
    decoder_route = f"../results/Purchase/{test_num}/Decoder-layer{split_layer}.pth"
    
    # 数据集加载
    trainloader,testloader = preprocess_purchase(batch_size=1)
    one_data_loader = get_one_data(testloader,batch_size = args['batch_size']) #拿到第一个测试数据

    # 模型加载
    client_net = PurchaseClassifier1(layer=split_layer)
    # pweights = torch.load(unit_net_route,map_location=torch.device('cpu'))
    pweights = torch.load(unit_net_route)
    if split_layer < len(purchase_cfg):
        pweights = split_weights_client(pweights,client_net.state_dict())
    client_net.load_state_dict(pweights)

    # 其他fsha要用到的网络
    shadow_net = PurchaseClassifier1(layer=split_layer)
    decoder_net = PurchaseDecoder1(layer=split_layer)

else:
    exit(-1)

discriminator_net = discriminator_net()

# %%
# 创建储存结果的文件夹
inverse_dir = results_dir + 'layer'+str(split_layer)+'/' # 储存逆向结果的dir
create_dir(results_dir)
create_dir(inverse_dir)

# 准备好攻击所需的模型的路径
shadow_net_route = results_dir+'/shadow_net.pth'
# shadow_net_route = unit_net_route # 直接用client net的参数
discriminator_net_route = results_dir+'discriminator_net.pth'
decoder_net_route = results_dir+'decoder_net.pth'
client_net_route = results_dir+'client_net.pth'

# client_net调整模式
client_net = client_net.to(args['device'])



# %% [markdown]
# ## 训练攻击模型

# %%
fsha_attack = FSHA_Attack(gpu=True,
                          client_route=client_net_route,
                          shadow_route=shadow_net_route,
                          decoder_route=decoder_net_route,
                          discriminator_route=discriminator_net_route,
                          inverse_dir=inverse_dir)


# %%
# 训练攻击模型
if os.path.isfile(decoder_net_route): # 如果已经训练好了 直接加载模型
    print("=> loading decoder model '{}'".format(decoder_net_route))
    # shadow_net.load_state_dict(pweights) # 加载client_net 参数
    client_net = torch.load(client_net_route)
    shadow_net = torch.load(shadow_net_route)
    decoder_net = torch.load(decoder_net_route)
    discriminator_net = torch.load(discriminator_net_route)


else: # 如果没有, 就训练一个
    print("train decoder model...")
    # 创建新的batch_size为1的DataLoader 
    shadow_net.load_state_dict(client_net.state_dict())
    new_trainloader = DataLoader(trainloader.dataset, batch_size=32)
    fsha_attack.train_decoder(client_net=client_net,
                              shadow_net=shadow_net,
                              decoder_net=decoder_net,
                              discriminator_net=discriminator_net,
                              private_loader=new_trainloader,public_loader=new_trainloader,
                              epochs=25)

# %% [markdown]
# ## 进行重构攻击并评估结果

# %%
fsha_attack.inverse(client_net=client_net,
                    decoder_net=decoder_net,
                    train_loader=trainloader,test_loader=testloader,
                    deprocess=None if args['data_type']==0 else deprocess,
                    save_fake=True)

# %%
# 另外评估ML_Efficacy指标





# nohup python -u model_inversion_fsha_attack.py >> temp.out 2>&1 &
# [1] 31609