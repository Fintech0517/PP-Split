# 导包
import torch
import os
import argparse
import pandas as pd
import tqdm
import numpy as np
# os.environ['NUMEXPR_MAX_THREADS'] = '48'

# 导入repE相关
import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from ppsplit.quantification.rep_reading.rep_reader import RepE
from target_model.data_preprocessing.dataset import pair_data

# 导入各个baseline模型及其数据集预处理方法
# 模型
from target_model.models.splitnn_utils import split_weights_client
from target_model.models.VGG import VGG,VGG5Decoder,model_cfg
from target_model.models.BankNet import BankNet1,bank_cfg
from target_model.models.CreditNet import CreditNet1,credit_cfg
from target_model.models.PurchaseNet import PurchaseClassifier1,purchase_cfg

# 数据预处理方法
from target_model.data_preprocessing.preprocess_cifar10 import get_cifar10_normalize,deprocess,get_indexed_loader
from target_model.data_preprocessing.preprocess_bank import bank_dataset,preprocess_bank
from target_model.data_preprocessing.preprocess_credit import preprocess_credit
from target_model.data_preprocessing.preprocess_purchase import preprocess_purchase
from target_model.data_preprocessing.dataset import get_one_data

# utils
from ppsplit.utils.utils import create_dir

# 基本参数：
args = {
        'device':torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        # 'device':torch.device("cpu"),
        'dataset':'CIFAR10',
        # 'dataset':'bank',
        # 'dataset':'credit',
        # 'dataset':'purchase',
        # 'result_dir': 'InvMetric-202403',
        'result_dir': '20240428-Rep-quantify/',
        'batch_size':5,
        'noise_scale':0, # 防护措施
        'num_pairs': 10000, # RepE
        }
print(args['device'])

# 加载模型和数据集，并从unit模型中切割出client_model
if args['dataset']=='CIFAR10':
    # 超参数
    testset_len = 10000 # 10000个数据一次 整个测试集合的长度
    # split_layer_list = list(range(len(model_cfg['VGG5'])))
    split_layer = 2 # 定成2吧？
    test_num = 1 # 试验序号
    #  nohup python -u repE.py >> 200-split6-top10.out 2>&1  &
    # 关键路径
    # unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG5/BN+Tanh/VGG5-params-20ep.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
    unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG5/20240429-RepE/VGG5-params-19ep.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
    results_dir  = f"../results/{args['result_dir']}/VGG5/quantification/{test_num}/"

    # 数据集加载
    # trainloader,testloader = get_cifar10_normalize(batch_size = 5)
    # one_data_loader = get_one_data(testloader,batch_size = args['batch_size']) #拿到第一个测试数据

    # 切割成client model
    # vgg5_unit.load_state_dict(torch.load(unit_net_route,map_location=torch.device('cpu'))) # 完整的模型
    client_net = VGG('Client','VGG5',split_layer,model_cfg,noise_scale=args['noise_scale'])
    pweights = torch.load(unit_net_route)
    if split_layer < len(model_cfg['VGG5']):
        pweights = split_weights_client(pweights,client_net.state_dict())
    client_net.load_state_dict(pweights)

elif args['dataset']=='bank':
    # 超参数
    test_num = 1 # 试验序号
    testset_len=8238
    # split_layer_list = ['linear1', 'linear2']
    split_layer_list = [0,2,4,6]
    split_layer = 2

    # 关键路径
    results_dir  = f"../results/{args['result_dir']}/Bank/quantification/{test_num}/"
    unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Bank/bank-20ep_params.pth'
    decoder_route = f"../results/{args['result_dir']}/Bank/{test_num}/Decoder-layer{split_layer}.pth"

    # 数据集加载
    trainloader,testloader = preprocess_bank(batch_size=1)
    one_data_loader = get_one_data(testloader,batch_size = args['batch_size']) #拿到第一个测试数据 

    # 模型加载
    client_net = BankNet1(layer=split_layer,noise_scale=args['noise_scale'])
    pweights = torch.load(unit_net_route)
    if split_layer < len(bank_cfg):
        pweights = split_weights_client(pweights,client_net.state_dict())
    client_net.load_state_dict(pweights)

elif args['dataset']=='credit':
    # 超参数
    test_num = 1 # 试验序号
    testset_len = 61503 # for the mutual information
    split_layer_list = [0,3,6,9]
    split_layer = 3
    # split_layer_list = ['linear1', 'linear2']

    # 关键路径
    results_dir  = f"../results/{args['result_dir']}/Credit/quantification/{test_num}/"
    unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/credit/credit-20ep_params.pth'
    decoder_route = f"../results/{args['result_dir']}/Credit/{test_num}/Decoder-layer{split_layer}.pth"

    # 数据集加载
    trainloader,testloader = preprocess_credit(batch_size=1)
    one_data_loader = get_one_data(testloader,batch_size = args['batch_size']) #拿到第一个测试数据

    # client模型切割加载
    client_net = CreditNet1(layer=split_layer,noise_scale=args['noise_scale'])
    pweights = torch.load(unit_net_route)
    if split_layer < len(credit_cfg):
        pweights = split_weights_client(pweights,client_net.state_dict())
    client_net.load_state_dict(pweights)

elif args['dataset']=='purchase':
    # 超参数
    test_num = 1 # 试验序号
    testset_len = 39465 # test len
    # split_layer_list = [0,1,2,3,4,5,6,7,8]
    split_layer = 3

    # 关键路径
    results_dir = f"../results/{args['result_dir']}/Purchase/quantification/{test_num}/"
    unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Purchase100/Purchase_bestmodel_param.pth'
    decoder_route = f"../results/{args['result_dir']}/Purchase/{test_num}/Decoder-layer{split_layer}.pth"
    
    # 数据集加载
    trainloader,testloader = preprocess_purchase(batch_size=1)
    one_data_loader = get_one_data(testloader,batch_size = args['batch_size']) #拿到第一个测试数据

    # 模型加载
    client_net = PurchaseClassifier1(layer=split_layer,noise_scale=args['noise_scale'])
    # pweights = torch.load(unit_net_route,map_location=torch.device('cpu'))
    pweights = torch.load(unit_net_route)
    if split_layer < len(purchase_cfg):
        pweights = split_weights_client(pweights,client_net.state_dict())
    client_net.load_state_dict(pweights)

else:
    exit(-1)


# 提取数据
import pandas as pd
import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from ppsplit.utils.utils import plot_array_distribution

# cifar10 layer2
# fisher
fisher_data_path = '/home/dengruijun/data/project/AISecurity/Inverse_efficacy/results/2-3-VGG5/quantification/1/6.2/dFIL.csv'
# mse [euc,mse,ssim]
mse_data_path = '/home/dengruijun/data/FinTech/PP-Split/results/inverse-model-results-20240414/VGG5/1/layer2/inv-sim.csv'

# inv_data_path = '/home/dengruijun/data/FinTech/PP-Split/results/Purchase/1/layer3/inv-X.csv'
# sim_data_path = '/home/dengruijun/data/FinTech/PP-Split/results/Purchase/1/layer3/inv-sim.csv'

sim_df = pd.read_csv(mse_data_path,sep=',',quotechar='"')
# sim_df = pd.read_csv(fisher_data_path,sep=',',quotechar='"')

# 打印文件基本信息
print('inv_sim_columns: ', sim_df.columns)


# 打印指标分布，观察指标情况
# print 指标分布
mse_df = sim_df['mse'].to_numpy() # mse 
print("mse_df.shape: ",mse_df.shape)
plot_array_distribution(mse_df,start=-1, end=1, notes=split_layer)


# 提取出cos<0.4和cos>0.6的样本，生成两个df 
import numpy as np
# 1.找索引
index_low = mse_df<0.1
index_high = mse_df>0.1

print('小组中的数据索引：', index_low)
print('大组中的数据索引：', index_high)

print('小组数据的数目: ', sum(index_low))
print('大组数据的数目: ', sum(index_high))
# index = np.where(index_low)
# print(len(index[0]))
# print(index[0])
# for i in range(len(index[0])):
#     print(index[0][i])
#     print('ok')
# 2. 提取出cifar10中对应的数据，形成两个数据集dataloader
low_loader = get_indexed_loader(index_low)
high_loader = get_indexed_loader(index_high)
min_len = min(sum(index_low),sum(index_high))

# 3. 构建pair data
# low 的是true（seen），high的是false（unseen）
train_loader,train_labels,test_loader,test_labels = pair_data(low_loader,
                                                                    high_loader,
                                                                    num_pairs=min_len)
print(train_labels[0])
print(test_labels[0].index(1))

# repE过程
repE_agent = RepE()

# 2. collecting neural activity
# 收集所有smashed data
train_smashed_data_list,diff_data = repE_agent.collect_neural_activity(train_loader,client_net)
print("diff_data.shape: ", diff_data.shape)

# 3. constructing a linear model
# 训练direction finder
directions,signs = repE_agent.construct_linear_model(diff_data,train_smashed_data_list,train_labels)
print('direction shape of first layer: ', directions.shape)
print('signs of first layer: ', signs)

# 4. 测试
acc = repE_agent.eval_MIA_acc(test_loader,test_labels,client_net)
print(f"quantified accuracy(privacy lekage): {acc} ")



#  nohup python -u repE.py >> min-split2-first10.out 2>&1  &
