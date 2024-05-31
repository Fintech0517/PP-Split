# 就是ipynb迁移过来
# MLLeaksAttack
# 导包
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
import sys
import urllib
import pickle
import argparse

sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from ppsplit.attacks.membership_inference.Mentr_attack import MentrAttack # 包含了4种方法的攻击类
from ppsplit.attacks.membership_inference.ML_Leaks_attack import MLLeaksAttack # 包含了ML Leaks攻击类

# 模型
from target_model.models.splitnn_utils import split_weights_client
from target_model.models.VGG import VGG,VGG5Decoder,model_cfg
from target_model.models.BankNet import BankNet1,bank_cfg
from target_model.models.CreditNet import CreditNet1,credit_cfg
from target_model.models.PurchaseNet import PurchaseClassifier1,purchase_cfg
# 数据预处理方法
from target_model.data_preprocessing.preprocess_cifar10 import get_cifar10_normalize,get_one_data,deprocess
from target_model.data_preprocessing.preprocess_bank import bank_dataset,preprocess_bank
from target_model.data_preprocessing.preprocess_credit import preprocess_credit
from target_model.data_preprocessing.preprocess_purchase import preprocess_purchase


from target_model.models.splitnn_utils import split_weights_client
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# utils
from ppsplit.utils.utils import create_dir

args = {
        'device':torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        # 'device':torch.device("cpu"),
        'dataset':'CIFAR10',
        # 'dataset':'bank',
        # 'dataset':'credit',
        # 'dataset':'purchase',
        # 'result_dir': 'InvMetric-202403',
        'result_dir': 'MIA/',
        'batch_size':32,
        'noise_scale':0, # 防护措施 
        'num_pairs': 200, # RepE
        'topk':10, # smashed data的size
        }
print(args['device'])

# 加载模型和数据集，并从unit模型中切割出client_model
if args['dataset']=='CIFAR10':
    # 超参数
    testset_len = 10000 # 10000个数据一次 整个测试集合的长度
    # split_layer_list = list(range(len(model_cfg['VGG5'])))
    split_layer = 5 # 定成2吧？
    test_num = 3 # 试验序号
# nohup python -u MIA.py >> split5.out 2>&1 &
    # 关键路径
    # 此时是为了repE
    # unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG5/BN+Tanh/VGG5-params-20ep.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
    unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG5/20240429-RepE/VGG5-params-19ep.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
    results_dir  = f"../results/{args['result_dir']}/VGG5/layer{split_layer}/"
    decoder_route = f"../results/{args['result_dir']}/VGG5/{test_num}/Decoder-layer{split_layer}.pth"

    # 数据集加载
    # trainloader,testloader = get_cifar10_normalize(batch_size = args['batch_size'])
    # one_data_loader = get_one_data(testloader,batch_size = args['batch_size']) #拿到第一个测试数据
    shadow_train_loader, shadow_test_loader = get_cifar10_normalize(batch_size = args['batch_size'])
    target_train_loader, target_test_loader = get_cifar10_normalize(batch_size = args['batch_size'])

    # 切割成client model
    # vgg5_unit.load_state_dict(torch.load(unit_net_route,map_location=torch.device('cpu'))) # 完整的模型
    client_net = VGG('Client','VGG5',split_layer,model_cfg,noise_scale=args['noise_scale'])
    pweights = torch.load(unit_net_route)
    if split_layer < len(model_cfg['VGG5']):
        pweights = split_weights_client(pweights,client_net.state_dict())
    client_net.load_state_dict(pweights)

    class_num = 10

elif args['dataset']=='purchase':
    # 设置一些超参数
    batch_size = 100 # 批大小
    # unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Purchase100/Purchase_bestmodel_param.pth' # 待检测模型
    unit_net_route = '/home/dengruijun/data/FinTech/VFL/MIAs/membership_inference_evaluation/adv_reg/training_code/models/purchase_undefended/model_best.pth.tar' # 待检测模型
    split_layer = 8 # 切隔层
    # purchase 数据集 和 模型 导入
    from target_model.models.PurchaseNet import PurchaseClassifier1, purchase_cfg
        
    class PurchaseClassifier(nn.Module):
        def __init__(self,num_classes=100):
            super(PurchaseClassifier, self).__init__()

            self.features = nn.Sequential(
                nn.Linear(600,1024),
                nn.Tanh(),
                nn.Linear(1024,512),
                nn.Tanh(),
                nn.Linear(512,256),
                nn.Tanh(),
                nn.Linear(256,128),
                nn.Tanh(),
            )
            self.classifier = nn.Linear(128,num_classes)
            
        def forward(self,x):
            hidden_out = self.features(x)
            return self.classifier(hidden_out)

    from target_model.data_preprocessing.preprocess_purchase import preprocess_purchase_shadow
    class_num = 100 # Purchase的分类类别数目 # 源论文默认100

    # 模型加载并切割：
    # client_net = PurchaseClassifier1(layer=split_layer)
    # pweights = torch.load(unit_net_route,map_location=device)
    # if split_layer < len(purchase_cfg):
    #     pweights = split_weights_client(pweights,client_net.state_dict())
    # client_net.load_state_dict(pweights)

    client_net = PurchaseClassifier(num_classes=100)
    client_net = torch.nn.DataParallel(client_net).cuda()
    checkpoint = torch.load(unit_net_route)
    client_net.load_state_dict(checkpoint['state_dict'])
    client_net.eval()

    # model = PurchaseClassifier1()
    # model = torch.nn.DataParallel(model).cuda()
    # checkpoint = torch.load(target_model_path)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()

    # 加载数据集
    shadow_train_loader, shadow_test_loader,\
        target_train_loader, target_test_loader = preprocess_purchase_shadow(batch_size=batch_size)
    
else:
    exit(-1)

# 创建文件夹
create_dir(results_dir)

# client_net使用
client_net = client_net.to(args['device'])
client_net.eval()


# 攻击 对象实例化
# ('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), # 0
# ('M', 32, 32, 2, 32*16*16, 0),  # 1
# ('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), #2
# ('M', 64, 64, 2, 64*8*8, 0), # 3
# ('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), # 4
# ('D', 8*8*64, 128, 1, 64, 128*8*8*64), # 5 
# ('D', 128, 10, 1, 10, 128*10)], # 6


# 数据集
from target_model.data_preprocessing.dataset import ListDataset
from target_model.data_preprocessing.preprocess_cifar10 import get_cifar10_normalize_two_train
from torch.utils.data import DataLoader

seen_loader,unseen_loader,test_loader = get_cifar10_normalize_two_train(batch_size=1)

print("seen data length: ",len(seen_loader.dataset))
print("unseen data length: ", len(unseen_loader.dataset))
print("test data length: ", len(test_loader.dataset))

x = iter(seen_loader).next()
print(x[0].shape)
print(x[1].shape)
print(x[1])


# performance准备 准备attacker的 input feature，处理完存储起来
import pickle
import os

dataset_route = f"../results/{args['result_dir']}/VGG5/layer{split_layer}/"

MIA = MLLeaksAttack(smashed_data_size=args['topk']) # top10

if os.path.isfile(dataset_route+'attack_train_member25000.pkl'):
    print(f"=> loading paired dataset from {dataset_route}")
    with open(dataset_route+'attack_train_member25000.pkl','rb') as f:
        smashed_data_list = pickle.load(file=f)
else:
    print(f"=> making paired dataset...")
    smashed_data_list = MIA.prepare_model_performance(client_net,seen_loader,unseen_loader,test_loader)
    with open(dataset_route+'attack_train_member25000.pkl','wb') as f:
        pickle.dump(obj=smashed_data_list, file=f)

smashed_dataset = ListDataset(smashed_data_list)
# print(len(smashed_dataset))
attack_loader = DataLoader(smashed_dataset, batch_size=100)


# 查看attack training数据集：
all_labels = [d[1] for d in smashed_data_list]
all_features = [d[0] for d in smashed_data_list]
print("第一个smashed data形状:",all_features[0].shape)
print("第一个smashed data取值:",all_features[0])
print("所有标签: ",all_labels)

# 第一个数据的shape：
print("第一个数据的shape: ")
(x,y) = iter(attack_loader).next()
print(x.shape)
print(y.shape)

# 训练attack model
attack_loader = DataLoader(smashed_dataset, batch_size=32)

MIA.train_attack_model(attack_loader,optimizer=None,epochs=20)

# 使用 attack model 进行 MIA攻击
data_loader = DataLoader(smashed_dataset, batch_size=1)

MIA.MIA_test(data_loader)

# nohup python -u MIA.py >> split1.out 2>&1 &
