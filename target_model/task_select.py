

# 导入各个baseline模型及其数据集预处理方法
# 模型
from target_model.models.splitnn_utils import split_weights_client
from target_model.models.VGG import VGG,VGG5Decoder,model_cfg
from target_model.models.BankNet import BankNet1,BankNetDecoder1,bank_cfg
from target_model.models.CreditNet import CreditNet1,CreditNetDecoder1,credit_cfg
from target_model.models.PurchaseNet import PurchaseClassifier1,PurchaseDecoder1,purchase_cfg
from target_model.models.IrisNet import IrisNet,IrisNetDecoder,Iris_cfg

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
import os

def get_dataloader(dataset='CIFAR10',
                   train_bs=1,
                   test_bs=1,
                   oneData_bs=1,):

    # 加载模型和数据集，并从unit模型中切割出client_model
    if dataset=='CIFAR10':
        # 超参数
        testset_len = 10000 # 10000个数据一次 整个测试集合的长度

        # 数据集加载
        trainloader,testloader = get_cifar10_normalize(batch_size = loader_bs)
        one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据

    elif dataset=='credit':
        # 超参数
        testset_len = 61503 # for the mutual information

        # 数据集加载
        trainloader,testloader = preprocess_credit(batch_size = loader_bs)
        one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据

    elif dataset=='bank':
        # 超参数
        testset_len=8238

        # 数据集加载
        trainloader,testloader = preprocess_bank(batch_size = loader_bs)
        # one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据 

    elif dataset=='Iris':
        # 超参数
        testset_len=30

        # 数据集加载
        trainloader,testloader = preprocess_Iris(batch_size = loader_bs) # 只针对train data，testbs = 1
        one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据 

    elif dataset=='purchase':
        # 超参数
        testset_len = 39465 # test len

        # 数据集加载
        trainloader,testloader = preprocess_purchase(batch_size = loader_bs)
        one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据


    else:
        exit(-1)
    
    return trainloader,testloader,one_data_loader


def get_dataloader_and_model(dataset='CIFAR10', 
                             loader_bs=1, 
                             oneData_bs=1, 
                             noise_scale=0.1, 
                             result_dir='1-1', 
                             OneData=False, 
                             device='cpu',
                             split_layer=-1):
    if OneData:
        loader_bs=1
    result_ws = result_dir
    image_deprocess = None

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

        # decoder net
        if os.path.isfile(decoder_route): # 如果已经训练好了
            print("=> loading decoder model '{}'".format(decoder_route))
            decoder_net = torch.load(decoder_route)
        else: # 如果没有,加载一个
            print("train decoder model...")
            decoder_net = VGG5Decoder(split_layer=split_layer)

        image_deprocess = deprocess

    elif dataset=='credit':
        # 超参数
        test_num = 1 # 试验序号
        testset_len = 61503 # for the mutual information
        split_layer_list = [0,3,6,9]
        split_layer = 3 if split_layer==-1 else split_layer
        # split_layer_list = ['linear1', 'linear2']

        # 关键路径
        unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/credit/credit-20ep_params.pth'
        results_dir  = f"../../results/{result_ws}/Credit/{test_num}/"
        decoder_route = f"../../results/{result_ws}/Credit/{test_num}/Decoder-layer{split_layer}.pth"

        # 数据集加载
        trainloader,testloader = preprocess_credit(batch_size = loader_bs)
        one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据

        # client模型切割加载
        client_net = CreditNet1(layer=split_layer,noise_scale=noise_scale)
        pweights = torch.load(unit_net_route)
        if split_layer < len(credit_cfg):
            pweights = split_weights_client(pweights,client_net.state_dict())
        client_net.load_state_dict(pweights)

        # decoder net
        if os.path.isfile(decoder_route): # 如果已经训练好了
            print("=> loading decoder model '{}'".format(decoder_route))
            decoder_net = torch.load(decoder_route)
        else: # 如果没有,加载一个
            print("train decoder model...")
            decoder_net = CreditNetDecoder1(split_layer=split_layer)

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

        # decoder net
        if os.path.isfile(decoder_route): # 如果已经训练好了
            print("=> loading decoder model '{}'".format(decoder_route))
            decoder_net = torch.load(decoder_route)
        else: # 如果没有,加载一个
            print("train decoder model...")
            decoder_net = BankNetDecoder1(split_layer=split_layer)


    elif dataset=='Iris':
        # 超参数
        test_num = 1 # 试验序号
        testset_len=30
        # split_layer_list = ['linear1', 'linear2']
        # split_layer_list = [0,2,4,6]
        # split_layer = 2

        # 关键路径
        unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Iris/1/Iris-100ep.pth'
        results_dir  = f"../../results/{result_ws}/Iris/{test_num}/"
        decoder_route = result_dir+"/Decoder-layer{split_layer}.pth"
        decoder_route = None
    
        # 数据集加载
        trainloader,testloader = preprocess_Iris(batch_size = loader_bs) # 只针对train data，testbs = 1
        one_data_loader = get_one_data(testloader,batch_size = oneData_bs) #拿到第一个测试数据 

        # # 模型加载
        client_net = IrisNet(layer=split_layer,noise_scale=noise_scale)
        pweights = torch.load(unit_net_route)
        if split_layer < len(bank_cfg):
            pweights = split_weights_client(pweights,client_net.state_dict())
        client_net.load_state_dict(pweights)    

        # decoder net
        if os.path.isfile(decoder_route): # 如果已经训练好了
            print("=> loading decoder model '{}'".format(decoder_route))
            decoder_net = torch.load(decoder_route)
        else: # 如果没有,加载一个
            print("train decoder model...")
            decoder_net = IrisNetDecoder(split_layer=split_layer)

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

        # decoder net
        if os.path.isfile(decoder_route): # 如果已经训练好了
            print("=> loading decoder model '{}'".format(decoder_route))
            decoder_net = torch.load(decoder_route)
        else: # 如果没有,加载一个
            print("train decoder model...")
            decoder_net = PurchaseDecoder1(split_layer=split_layer)

    else:
        exit(-1)

    client_net.to(device)
    create_dir(results_dir)

    msg = {}
    if OneData:
        msg['one_data_loader'] = one_data_loader
        msg['client_net'] = client_net
        msg['results_dir'] = results_dir
        msg['decoder_net'] = decoder_net
        msg['decoder_route'] = decoder_route

        # return one_data_loader,client_net,results_dir,decoder_route
    else:
        msg['trainloader'] = trainloader
        msg['testloader'] = testloader
        msg['client_net'] = client_net
        msg['results_dir'] = results_dir
        msg['decoder_net'] = decoder_net
        msg['decoder_route'] = decoder_route

        # return trainloader,testloader,client_net,results_dir,decoder_route
    msg['image_deprocess'] = image_deprocess
    return msg
