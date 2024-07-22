

# 导入各个baseline模型及其数据集预处理方法
# 模型
from .models.splitnn_utils import split_weights_client
from .models.VGG import VGG,VGG5Decoder,model_cfg
from .models.BankNet import BankNet1,bank_cfg
from .models.CreditNet import CreditNet1,credit_cfg
from .models.PurchaseNet import PurchaseClassifier1,purchase_cfg
from .models.IrisNet import IrisNet,Iris_cfg

# 数据预处理方法
from .data_preprocessing.preprocess_cifar10 import get_cifar10_normalize,deprocess
from .data_preprocessing.preprocess_bank import bank_dataset,preprocess_bank,preprocess_bank_dataset
from .data_preprocessing.preprocess_credit import preprocess_credit
from .data_preprocessing.preprocess_purchase import preprocess_purchase
from .data_preprocessing.preprocess_Iris import preprocess_Iris
from .data_preprocessing.dataset import get_one_data

# utils
from .utils import create_dir

import torch



def get_dataloader_and_model(dataset='CIFAR10', loader_bs=1, oneData_bs=1, noise_scale=0.1, result_dir='1-1', OneData=False, device='cpu',split_layer=-1):
    if OneData:
        loader_bs=1
    result_ws = result_dir

    # 加载模型和数据集，并从unit模型中切割出client_model
    if dataset=='CIFAR10':
        # 超参数
        testset_len = 10000 # 10000个数据一次 整个测试集合的长度
        # split_layer_list = list(range(len(model_cfg['VGG5'])))
        split_layer = 1 if split_layer==-1 else split_layer # 定成3吧？
        test_num = 2 # 试验序号

        # 关键路径
        unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG5/BN+Tanh/VGG5-params-20ep.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
        results_dir  = f"../../results/{result_ws}/VGG5/{test_num}/"
        decoder_route = f"../../results/{result_ws}/VGG5/{test_num}/Decoder-layer{split_layer}.pth"

        # 数据集加载
        trainloader,testloader = get_cifar10_normalize(batch_size = loader_bs)
        one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据

        # 切割成client model
        # vgg5_unit.load_state_dict(torch.load(unit_net_route,map_location=torch.device('cpu'))) # 完整的模型
        client_net = VGG('Client','VGG5',split_layer,model_cfg,noise_scale=noise_scale)
        pweights = torch.load(unit_net_route)
        if split_layer < len(model_cfg['VGG5']):
            pweights = split_weights_client(pweights,client_net.state_dict())
        client_net.load_state_dict(pweights)

    elif dataset=='credit':
        # 超参数
        test_num = 1 # 试验序号
        testset_len = 61503 # for the mutual information
        split_layer_list = [0,3,6,9]
        split_layer = 3 if split_layer==-1 else split_layer
        # split_layer_list = ['linear1', 'linear2']

        # 关键路径
        unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/credit/credit-20ep_params.pth'
        results_dir  = f"../results/{result_ws}/Credit/{test_num}/"
        decoder_route = f"../results/{result_ws}/Credit/{test_num}/Decoder-layer{split_layer}.pth"

        # 数据集加载
        trainloader,testloader = preprocess_credit(batch_size = loader_bs)
        one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据

        # client模型切割加载
        client_net = CreditNet1(layer=split_layer,noise_scale=noise_scale)
        pweights = torch.load(unit_net_route)
        if split_layer < len(credit_cfg):
            pweights = split_weights_client(pweights,client_net.state_dict())
        client_net.load_state_dict(pweights)

    elif dataset=='bank':
        # 超参数
        test_num = 1 # 试验序号
        testset_len=8238
        # split_layer_list = ['linear1', 'linear2']
        split_layer_list = [0,2,4,6]
        split_layer = 2 if split_layer==-1 else split_layer

        # 关键路径
        unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Bank/bank-20ep_params.pth'
        results_dir  = f"../results/{result_ws}/Bank/{test_num}/"
        decoder_route = f"../results/{result_ws}/Bank/{test_num}/Decoder-layer{split_layer}.pth"
    
        # 数据集加载
        trainloader,testloader = preprocess_bank(batch_size = loader_bs)
        # one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据 

        # 模型加载
        client_net = BankNet1(layer=split_layer,noise_scale=noise_scale)
        pweights = torch.load(unit_net_route)
        if split_layer < len(bank_cfg):
            pweights = split_weights_client(pweights,client_net.state_dict())
        client_net.load_state_dict(pweights)    
        
    elif dataset=='Iris':
        # 超参数
        test_num = 1 # 试验序号
        testset_len=30
        # split_layer_list = ['linear1', 'linear2']
        # split_layer_list = [0,2,4,6]
        # split_layer = 2

        # 关键路径
        unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Iris/1/Iris-100ep.pth'
        results_dir  = f"../results/{result_ws}/Iris/{test_num}/"
        # decoder_route = f"../results/{result_ws}/Iris/{test_num}/Decoder-layer{split_layer}.pth"
        decoder_route = None
    
        # 数据集加载
        trainloader,testloader = preprocess_Iris(batch_size = loader_bs)
        one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据 

        # # 模型加载
        client_net = IrisNet(layer=split_layer,noise_scale=noise_scale)
        pweights = torch.load(unit_net_route)
        if split_layer < len(bank_cfg):
            pweights = split_weights_client(pweights,client_net.state_dict())
        client_net.load_state_dict(pweights)    
        # client_net = None

    elif dataset=='purchase':
        # 超参数
        test_num = 1 # 试验序号
        testset_len = 39465 # test len
        # split_layer_list = [0,1,2,3,4,5,6,7,8]
        split_layer = 3

        # 关键路径
        unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Purchase100/Purchase_bestmodel_param.pth'
        results_dir = f"../../results/{result_ws}/Purchase/{test_num}/"
        decoder_route = f"../../results/{result_ws}/Purchase/{test_num}/Decoder-layer{split_layer}.pth"

        # 数据集加载
        trainloader,testloader = preprocess_purchase(batch_size = loader_bs)
        one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据

        # 模型加载
        client_net = PurchaseClassifier1(layer=split_layer,noise_scale=noise_scale)
        # pweights = torch.load(unit_net_route,map_location=torch.device('cpu'))
        pweights = torch.load(unit_net_route)
        if split_layer < len(purchase_cfg):
            pweights = split_weights_client(pweights,client_net.state_dict())
        client_net.load_state_dict(pweights)

    else:
        exit(-1)

    client_net.to(device)
    create_dir(results_dir)

    if OneData:
        return one_data_loader,client_net,results_dir,decoder_route
    else:
        return trainloader,testloader,client_net,results_dir,decoder_route
