

# 导入各个baseline模型及其数据集预处理方法
# 模型
from target_model.models.splitnn_utils import split_weights_client
from target_model.models.VGG import VGG,VGG5Decoder,model_cfg
from target_model.models.BankNet import BankNet1,BankNetDecoder1,bank_cfg
from target_model.models.CreditNet import CreditNet1,CreditNetDecoder1,credit_cfg
from target_model.models.PurchaseNet import PurchaseClassifier1,PurchaseDecoder1,purchase_cfg
from target_model.models.IrisNet import IrisNet,IrisNetDecoder,Iris_cfg
from target_model.models.ResNet import resnet18,resnet_model_cfg,InversionNet



# 数据预处理方法
from .data_preprocessing.preprocess_cifar10 import get_cifar10_normalize,deprocess,get_cifar10_fisher_normalize
from .data_preprocessing.preprocess_bank import bank_dataset,preprocess_bank,preprocess_bank_dataset,tabinfo_bank
from .data_preprocessing.preprocess_credit import preprocess_credit,tabinfo_credit
from .data_preprocessing.preprocess_purchase import preprocess_purchase,tabinfo_purchase
from .data_preprocessing.preprocess_Iris import preprocess_Iris, tabinfo_Iris
from .data_preprocessing.dataset import get_one_data

# utils
from .utils import create_dir

import torch
import os
from torch.utils.data import DataLoader, Dataset

def get_infotopo_para(args):
    # 提取参数
    dataset = args['dataset']
    # dataset,train_bs,test_bs,oneData_bs=args['dataset'],args['train_bs'],args['test_bs'],args['oneData_bs']
    # 加载模型和数据集，并从unit模型中切割出client_model

    if dataset=='CIFAR10':
        nb_of_values=36 # nb_of_values-1=bins?
        conv = True
    elif dataset=='credit':
        pass
    elif dataset=='bank':
        pass
    elif dataset=='Iris':
        nb_of_values = 9
    elif dataset=='purchase':
        pass
    else:
        exit(-1)
    
    
    
    # msg
    msg = {}

    # 填充
    msg['nb_of_values']=nb_of_values
    msg['conv']=conv

    return msg



def get_dataloader(args):
    # 提取参数
    dataset,train_bs,test_bs,oneData_bs=args['dataset'],args['train_bs'],args['test_bs'],args['oneData_bs']
    model = args['model']

    # 加载模型和数据集，并从unit模型中切割出client_model
    if dataset=='CIFAR10':
        # 超参数
        testset_len = 10000 # 10000个数据一次 整个测试集合的长度
        tab_info=None

        # 数据集加载
        trainloader,testloader = get_cifar10_normalize(batch_size = train_bs, test_bs=test_bs)

    elif dataset=='credit':
        # 超参数
        testset_len = 61503 # for the mutual information

        # 数据集加载
        trainloader,testloader = preprocess_credit(batch_size = train_bs, test_bs=test_bs)
        tab_info=tabinfo_credit

    elif dataset=='bank':
        # 超参数
        testset_len=8238

        # 数据集加载
        trainloader,testloader = preprocess_bank(batch_size = train_bs, test_bs=test_bs)
        tab_info=tabinfo_bank

    elif dataset=='Iris':
        # 超参数
        testset_len=30

        # 数据集加载
        trainloader,testloader = preprocess_Iris(batch_size = train_bs, test_bs=test_bs) # 只针对train data，testbs = 1
        tab_info=tabinfo_Iris

    elif dataset=='purchase':
        # 超参数
        testset_len = 39465 # test len

        # 数据集加载
        trainloader,testloader = preprocess_purchase(batch_size = train_bs, test_bs=test_bs)
        tab_info=tabinfo_purchase

    else:
        exit(-1)
    
    # one loader
    one_bs_testloader = DataLoader(testloader.dataset, batch_size=1, shuffle=False, num_workers=4)
    # one_bs_testloader = DataLoader(trainloader.dataset, batch_size=1, shuffle=False, num_workers=4)
    one_data_loader = get_one_data(one_bs_testloader,batch_size = oneData_bs) #拿到第一个测试数据
    
    # msg
    msg = {}
    msg['tabinfo']= tab_info
    msg['trainloader'] = trainloader
    msg['testloader'] = testloader
    msg['one_data_loader'] = one_data_loader

    return msg

def get_models(args):
    # 参数提取
    dataset=args['dataset']
    model=args['model'] # 新增，数据集和模型可能不是一一对应的
    noise_scale=args['noise_scale']
    result_dir=args['result_dir']
    device=args['device']
    split_layer=args['split_layer']
    result_ws = result_dir
    image_deprocess = None
    test_num = args['test_num']

    # 加载模型和数据集，并从unit模型中切割出client_model
    if dataset=='CIFAR10':
        image_deprocess = deprocess
        if model == 'VGG5':
            # 超参数
            # split_layer_list = list(range(len(model_cfg['VGG5'])))
            split_layer = 2 if split_layer==-1 else split_layer # 定成3吧？

            # 关键路径
            unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG5/BN+Tanh/VGG5-params-20ep.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
            results_dir  = f"../../results/{result_ws}/VGG5/{test_num}/"
            decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

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

            
        elif model == 'ResNet18':
            split_layer_list=[2,3,5,7,9,11]

            # 超参数
            # split_layer_list = list(range(len(model_cfg['VGG5'])))
            split_layer = 7 if split_layer==-1 else split_layer # 定成3吧？

            # 关键路径
            unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/CIFAR10-models/ResNet18/32bs-ep20-relu-max-adam/resnet18-drj.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
            results_dir  = f"../../results/{result_ws}/Resnet18/{test_num}/"
            decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

            # 切割成client model
            client_net = resnet18(pretrained=False, split_layer=split_layer, bottleneck_dim=-1, num_classes=10, activation='gelu', pooling='avg')
            # client_net = VGG('Client','VGG5',split_layer,model_cfg,noise_scale=noise_scale)
            pweights = torch.load(unit_net_route)
            # if split_layer < len(resnet_model_cfg['resenet18']):
                # pweights = split_weights_client(pweights,client_net.state_dict())
            client_net.load_state_dict(pweights,strict=False) 

            # decoder net
            if os.path.isfile(decoder_route): # 如果已经训练好了
                print("=> loading decoder model '{}'".format(decoder_route))
                decoder_net = torch.load(decoder_route)
            else: # 如果没有,加载一个
                print("train decoder model...")
                decoder_net = InversionNet(split_layer=split_layer)

        else:
            exit(-1)

    elif dataset=='credit':
        # 超参数
        # test_num = 1 # 试验序号
        testset_len = 61503 # for the mutual information
        split_layer_list = [0,3,6,9]
        split_layer = 3 if split_layer==-1 else split_layer
        # split_layer_list = ['linear1', 'linear2']

        # 关键路径
        unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/credit/credit-20ep_params.pth'
        results_dir  = f"../../results/{result_ws}/Credit/{test_num}/"
        decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

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
        # test_num = 1 # 试验序号
        testset_len=8238
        # split_layer_list = ['linear1', 'linear2']
        split_layer_list = [0,2,4,6]
        split_layer = 2 if split_layer==-1 else split_layer

        # 关键路径
        unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Bank/bank-20ep_params.pth'
        results_dir  = f"../../results/{result_ws}/Bank/{test_num}/"
        decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

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
        # test_num = 1 # 试验序号
        testset_len=30

        # 关键路径
        unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Iris/1/Iris-100ep.pth'
        results_dir  = f"../../results/{result_ws}/Iris/{test_num}/"
        decoder_route = results_dir+f"/Decoder-layer{split_layer}.pth"

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
            decoder_net = IrisNetDecoder(layer=split_layer)

    elif dataset=='purchase':
        # 超参数
        # test_num = 1 # 试验序号
        testset_len = 39465 # test len
        # split_layer_list = [0,1,2,3,4,5,6,7,8]
        split_layer = 3

        # 关键路径
        unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/Purchase100/Purchase_bestmodel_param.pth'
        results_dir = f"../../results/{result_ws}/Purchase/{test_num}/"
        decoder_route = results_dir + f"Decoder-layer{split_layer}.pth"

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

    # 返回值
    msg={}

    msg['image_deprocess'] = image_deprocess
    msg['client_net'] = client_net
    msg['results_dir'] = results_dir
    msg['decoder_net'] = decoder_net
    msg['decoder_route'] = decoder_route

    return msg

def get_dataloader_and_model(args):

    # 模型和路径:
    msg = get_models(args)

    # 数据集:
    msg_data =  get_dataloader(args['dataset'],
                                args['train_bs'],
                                args['test_bs'],
                                args['oneData_bs'])

    msg.update(msg_data)
    # 返回值
    # msg={}

    # msg['trainloader'] = trainloader
    # msg['testloader'] = testloader
    # msg['one_data_loader'] = one_data_loader
    # msg['image_deprocess'] = image_deprocess
    # msg['client_net'] = client_net
    # msg['results_dir'] = results_dir
    # msg['decoder_net'] = decoder_net
    # msg['decoder_route'] = decoder_route

    return msg
