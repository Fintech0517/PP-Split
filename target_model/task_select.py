

# 导入各个baseline模型及其数据集预处理方法
# 模型
from target_model.models.splitnn_utils import split_weights_client
from target_model.models.ImageClassification.VGG5_9 import VGG,VGG5Decoder,model_cfg
from target_model.models.TableClassification.BankNet import BankNet1,BankNetDecoder1,bank_cfg
from target_model.models.TableClassification.CreditNet import CreditNet1,CreditNetDecoder1,credit_cfg
from target_model.models.TableClassification.PurchaseNet import PurchaseClassifier1,PurchaseDecoder1,purchase_cfg
from target_model.models.TableClassification.IrisNet import IrisNet,IrisNetDecoder,Iris_cfg
from target_model.models.ImageClassification.ResNet import resnet18,resnet34,resnet50,resnet_model_cfg,InversionNet

# 数据预处理方法
from .data_preprocessing.preprocess_cifar10 import get_cifar10_normalize,deprocess,get_cifar10_fisher_normalize
from .data_preprocessing.preprocess_bank import bank_dataset,preprocess_bank,preprocess_bank_dataset,tabinfo_bank
from .data_preprocessing.preprocess_credit import preprocess_credit,tabinfo_credit
from .data_preprocessing.preprocess_purchase import preprocess_purchase,tabinfo_purchase
from .data_preprocessing.preprocess_Iris import preprocess_Iris, tabinfo_Iris
from .data_preprocessing.preprocess_mnist import get_mnist_normalize
from .data_preprocessing.preprocess_cifar100 import get_cifar100_normalize
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
        pool_size = 4
    elif dataset=='CIFAR100': # 和cifar10一样的
        nb_of_values=36 # nb_of_values-1=bins?
        conv = True
        pool_size = 4
    elif dataset=='credit':
        pass
    elif dataset=='bank':
        pass
    elif dataset=='Iris':
        nb_of_values = 9
        conv = False
    elif dataset=='purchase':
        nb_of_values =2
        conv = False
    elif dataset=='MNIST':
        nb_of_values = 2
        conv = True
        pool_size = 2
    else:
        raise ValueError('dataset error')
    
    
    
    # msg
    msg = {}

    # 填充
    msg['nb_of_values']=nb_of_values
    msg['conv']=conv
    msg['pool_size']=pool_size

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
        data_interval = (-1.0,1.0) # [-1,1]
        data_type=1
    
    elif dataset=='CIFAR100':
        # 超参数
        testset_len = 10000
        tab_info = None

        # 数据集加载
        trainloader,testloader = get_cifar100_normalize(batch_size = train_bs, test_bs=test_bs)
        data_interval = (-1.0,1.0) # [-1,1]
        data_type=1
    elif dataset=='credit':
        # 超参数
        testset_len = 61503 # for the mutual information

        # 数据集加载
        trainloader,testloader = preprocess_credit(batch_size = train_bs, test_bs=test_bs)
        tab_info=tabinfo_credit
        data_type=0
    elif dataset=='bank':
        # 超参数
        testset_len=8238

        # 数据集加载
        trainloader,testloader = preprocess_bank(batch_size = train_bs, test_bs=test_bs)
        tab_info=tabinfo_bank
        data_type=0
    elif dataset=='Iris':
        # 超参数
        testset_len=30

        # 数据集加载
        trainloader,testloader = preprocess_Iris(batch_size = train_bs, test_bs=test_bs) # 只针对train data，testbs = 1
        tab_info=tabinfo_Iris
        data_type=1
    elif dataset=='purchase':
        # 超参数
        testset_len = 39465 # test len

        # 数据集加载
        trainloader,testloader = preprocess_purchase(batch_size = train_bs, test_bs=test_bs)
        tab_info=tabinfo_purchase

        data_interval = (0.0,1.0) # [0,1]
        data_type=1
    elif dataset=='MNIST':
        # 超参数
        testset_len = 10000
        tab_info=None

        # 数据集加载
        trainloader,testloader = get_mnist_normalize(batch_size = train_bs, test_bs=test_bs)
        data_interval = (-1.0,1.0)
        data_type=1
    else:
        print('get_dataloader error')
        raise ValueError('dataset error')
    
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
    msg['data_interval'] = data_interval
    msg['data_type'] = data_type
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
    no_dense = args['no_dense'] # 默认为False
    ep = args['ep']

    # 加载模型和数据集，并从unit模型中切割出client_model
    if dataset=='CIFAR10':

        image_deprocess = deprocess
        if model == 'VGG5':
            # 超参数
            # split_layer_list = list(range(len(model_cfg['VGG5'])))
            split_layer = 2 if split_layer==-1 else split_layer # 定成3吧？

            # 关键路径
            if ep==-1:
                # vgg5 (20ep)
                unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG5/BN+Tanh/VGG5-params-20ep.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
                results_dir  = f"../../results/{result_ws}/VGG5/{test_num}/"
            else:
                # 0ep
                unit_net_route = f'/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG5/CIFAR10/VGG5-CIFAR10-{ep}epoch.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
                results_dir  = f"../../results/{result_ws}/VGG5/VGG5_{ep}ep/{test_num}/"

            decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

            # 切割成client model
            # vgg5_unit.load_state_dict(torch.load(unit_net_route,map_location=torch.device('cpu'))) # 完整的模型
            client_net = VGG('Client','VGG5',split_layer,model_cfg,noise_scale=noise_scale)
            pweights = torch.load(unit_net_route)
            if split_layer < len(model_cfg['VGG5']):
                pweights = split_weights_client(pweights,client_net.state_dict(),no_dense=no_dense)
            client_net.load_state_dict(pweights)

            # decoder net
            if os.path.isfile(decoder_route): # 如果已经训练好了
                print("=> loading decoder model '{}'".format(decoder_route))
                decoder_net = torch.load(decoder_route)
            else: # 如果没有,加载一个
                print("train decoder model...")
                decoder_net = VGG5Decoder(split_layer=split_layer)

        elif model == 'VGG9':
            # 超参数
            # split_layer_list = list(range(len(model_cfg['VGG5'])))
            split_layer = 4 if split_layer==-1 else split_layer # 定成3吧？

            # 关键路径
            if ep==-1:
                # vgg9 (20ep)
                unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG9/CIFAR10/VGG9-CIFAR10-20ep.pth' # VGG9-BN+Tanh # 存储的是模型参数，不包括模型结构
                results_dir  = f"../../results/{result_ws}/VGG9/{test_num}/"
            else:
                # 0ep
                unit_net_route = f'/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG9/CIFAR10/VGG9-CIFAR10-20epoch.pth-{ep}.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
                results_dir  = f"../../results/{result_ws}/VGG9/VGG9_{ep}ep/{test_num}/"

            decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"
            # 切割成client model
            # vgg5_unit.load_state_dict(torch.load(unit_net_route,map_location=torch.device('cpu'))) # 完整的模型
            client_net = VGG('Client','VGG9',split_layer,model_cfg,noise_scale=noise_scale)
            pweights = torch.load(unit_net_route)
            if split_layer < len(model_cfg['VGG9']):
                pweights = split_weights_client(pweights,client_net.state_dict(),no_dense=no_dense)
            client_net.load_state_dict(pweights)

            # decoder net
            if os.path.isfile(decoder_route): # 如果已经训练好了
                print("=> loading decoder model '{}'".format(decoder_route))
                decoder_net = torch.load(decoder_route)
            else: # 如果没有,加载一个
                print("train decoder model...")
                decoder_net = VGG5Decoder(split_layer=split_layer,network='VGG9')

            
        elif model == 'ResNet18':
            split_layer_list=[2,3,5,7,9,11]

            # 超参数
            # split_layer_list = list(range(len(model_cfg['VGG5'])))
            split_layer = 7 if split_layer==-1 else split_layer # 定成3吧？

            # 关键路径
            # xs，原来根本没有加载参数
            # 100ep
            # unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/CIFAR10-models/ResNet18/32bs-ep20-relu-max-adam/resnet18-drj-small.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
            # 20 ep
            # unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/ResNet/resnet18/CIFAR10/resnet18-drj-align.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
            # 20 ep narrow
            # unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/ResNet/resnet18_narrow/CIFAR10/resnet18-drj-align.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
            # 20 ep wide
            # unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/ResNet/resnet18_wide/CIFAR10/resnet18-drj-align.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
            # 20 ep 2narrow
            unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/ResNet/resnet18_2narrow/CIFAR10/resnet18-drj-align.pth'

            # results_dir  = f"../../results/{result_ws}/Resnet18/{test_num}/"
            # results_dir  = f"../../results/{result_ws}/Resnet18/Resnet18_20ep_org/{test_num}/"
            # results_dir  = f"../../results/{result_ws}/Resnet18/Resnet18_20ep_narrow/{test_num}/"
            # results_dir  = f"../../results/{result_ws}/Resnet18/Resnet18_20ep_wide/{test_num}/"
            results_dir  = f"../../results/{result_ws}/Resnet18/Resnet18_20ep_2narrow/{test_num}/"
            decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

            # 切割成client model
            client_net = resnet18(pretrained=False, split_layer=split_layer, bottleneck_dim=-1, num_classes=10, activation='gelu', pooling='avg')
            # client_net = VGG('Client','VGG5',split_layer,model_cfg,noise_scale=noise_scale)
            pweights = torch.load(unit_net_route)
            
            # 不需要划分模型参数？
            if split_layer < len(resnet_model_cfg['resnet18']):
                pweights = split_weights_client(pweights,client_net.state_dict())
            client_net.load_state_dict(pweights,strict=False) 

            # decoder net
            if os.path.isfile(decoder_route): # 如果已经训练好了
                print("=> loading decoder model '{}'".format(decoder_route))
                decoder_net = torch.load(decoder_route)
            else: # 如果没有,加载一个
                print("train decoder model...")
                decoder_net = InversionNet(split_layer=split_layer)

        elif model == 'ResNet34':

            # 超参数
            # split_layer_list = list(range(len(model_cfg['VGG5'])))
            split_layer = 10 if split_layer==-1 else split_layer # 定成3吧？

            unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/ResNet/resnet34/CIFAR10/resnet34-drj-align.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构

            results_dir  = f"../../results/{result_ws}/Resnet34/{test_num}/"
            decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

            # 切割成client model
            client_net = resnet34(pretrained=False, split_layer=split_layer, bottleneck_dim=-1, num_classes=10, activation='gelu', pooling='avg')
            # client_net = VGG('Client','VGG5',split_layer,model_cfg,noise_scale=noise_scale)
            pweights = torch.load(unit_net_route)
            
            # 不需要划分模型参数？
            if split_layer < 22:
                pweights = split_weights_client(pweights,client_net.state_dict())
            client_net.load_state_dict(pweights,strict=False) 

            # decoder net
            if os.path.isfile(decoder_route): # 如果已经训练好了
                print("=> loading decoder model '{}'".format(decoder_route))
                decoder_net = torch.load(decoder_route)
            else: # 如果没有,加载一个
                print("train decoder model...")
                decoder_net = InversionNet(split_layer=7) # 手动设定，之后要改的 7 for split point  =10
            # decoder_net = None
           
        else:
            raise ValueError('model error')

    elif dataset=='CIFAR100':
        image_deprocess = deprocess

        if model == 'ResNet18':
            split_layer_list=[2,3,5,7,9,11]

            # 超参数
            # split_layer_list = list(range(len(model_cfg['VGG5'])))
            split_layer = 7 if split_layer==-1 else split_layer # 定成3吧？

            # 关键路径
            # xs，原来根本没有加载参数
            unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/ResNet/resnet18/CIFAR100/resnet18-drj-align.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
            results_dir  = f"../../results/{result_ws}/Resnet18_CIFAR100/{test_num}/"
            decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

            # 切割成client model
            client_net = resnet18(pretrained=False, split_layer=split_layer, bottleneck_dim=-1, num_classes=100, activation='gelu', pooling='avg')
            # client_net = VGG('Client','VGG5',split_layer,model_cfg,noise_scale=noise_scale)
            pweights = torch.load(unit_net_route)
            
            # 不需要划分模型参数？
            if split_layer < len(resnet_model_cfg['resnet18']):
                pweights = split_weights_client(pweights,client_net.state_dict())
            client_net.load_state_dict(pweights,strict=False) 

            # decoder net  # 先用cifar10的吧，但估计是错的。
            if os.path.isfile(decoder_route): # 如果已经训练好了
                print("=> loading decoder model '{}'".format(decoder_route))
                decoder_net = torch.load(decoder_route)
            else: # 如果没有,加载一个
                print("train decoder model...")
                decoder_net = InversionNet(split_layer=split_layer)

        elif model == 'ResNet34':
            split_layer_list=[2,3,5,7,9,11]

            # 超参数
            # split_layer_list = list(range(len(model_cfg['VGG5'])))
            split_layer = 7 if split_layer==-1 else split_layer # 定成3吧？

            # 关键路径
            # xs，原来根本没有加载参数
            unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/ResNet/resnet34/CIFAR100/resnet34-drj-align.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
            results_dir  = f"../../results/{result_ws}/Resnet34_CIFAR100/{test_num}/"
            decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

            # 切割成client model
            client_net = resnet34(pretrained=False, split_layer=split_layer, bottleneck_dim=-1, num_classes=100, activation='gelu', pooling='avg')
            # client_net = VGG('Client','VGG5',split_layer,model_cfg,noise_scale=noise_scale)
            pweights = torch.load(unit_net_route)
            
            # 不需要划分模型参数？
            if split_layer < 22:
                pweights = split_weights_client(pweights,client_net.state_dict())
            client_net.load_state_dict(pweights,strict=False) 

            # decoder net  # 先用cifar10的吧，但估计是错的。
            if os.path.isfile(decoder_route): # 如果已经训练好了
                print("=> loading decoder model '{}'".format(decoder_route))
                decoder_net = torch.load(decoder_route)
            else: # 如果没有,加载一个
                print("train decoder model...")
                decoder_net = InversionNet(split_layer=split_layer)
        
        elif model == 'ResNet50':
            split_layer_list=[2,3,5,7,9,11]

            # 超参数
            # split_layer_list = list(range(len(model_cfg['VGG5'])))
            split_layer = 7 if split_layer==-1 else split_layer # 定成3吧？

            # 关键路径
            # xs，原来根本没有加载参数
            unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/ResNet/resnet50/CIFAR100/resnet18-drj-align.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
            results_dir  = f"../../results/{result_ws}/Resnet50_CIFAR100/{test_num}/"
            decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

            # 切割成client model
            client_net = resnet50(pretrained=False, split_layer=split_layer, bottleneck_dim=-1, num_classes=100, activation='gelu', pooling='avg')
            # client_net = VGG('Client','VGG5',split_layer,model_cfg,noise_scale=noise_scale)
            pweights = torch.load(unit_net_route)
            
            # 不需要划分模型参数？
            if split_layer < 22:
                pweights = split_weights_client(pweights,client_net.state_dict())
            client_net.load_state_dict(pweights,strict=False) 

            # decoder net  # 先用cifar10的吧，但估计是错的。
            if os.path.isfile(decoder_route): # 如果已经训练好了
                print("=> loading decoder model '{}'".format(decoder_route))
                decoder_net = torch.load(decoder_route)
            else: # 如果没有,加载一个
                print("train decoder model...")
                decoder_net = InversionNet(split_layer=split_layer)        
        else:
            raise ValueError('model error')

    elif dataset =='MNIST':
        image_deprocess = deprocess
        if model == 'VGG5':
            # 超参数
            # split_layer_list = list(range(len(model_cfg['VGG5'])))
            split_layer = 2 if split_layer==-1 else split_layer # 定成3吧？

            # 关键路径
            if ep==-1:
                # vgg5 20ep
                unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG5/MNIST/VGG5-MNIST-20ep.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
                results_dir  = f"../../results/{result_ws}/VGG5_MNIST/{test_num}/"
            else:
                # 0ep
                unit_net_route = f'/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG5/MNIST/VGG5_MNIST-MNIST-20epoch.pth-{ep}.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
                results_dir  = f"../../results/{result_ws}/VGG5_MNIST/VGG5_{ep}ep/{test_num}/"



            decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

            # 切割成client model
            # vgg5_unit.load_state_dict(torch.load(unit_net_route,map_location=torch.device('cpu'))) # 完整的模型
            client_net = VGG('Client','VGG5_MNIST',split_layer,model_cfg,noise_scale=noise_scale)
            pweights = torch.load(unit_net_route)
            if split_layer < len(model_cfg['VGG5_MNIST']):
                pweights = split_weights_client(pweights,client_net.state_dict(),no_dense=no_dense)
            client_net.load_state_dict(pweights)

            # decoder net
            if os.path.isfile(decoder_route): # 如果已经训练好了
                print("=> loading decoder model '{}'".format(decoder_route))
                decoder_net = torch.load(decoder_route)
            else: # 如果没有,加载一个
                print("train decoder model...")
                decoder_net = VGG5Decoder(split_layer=split_layer,network='VGG5_MNIST')
                # print("decoder_net:",decoder_net)

        elif model == 'VGG9':
            # 超参数
            # split_layer_list = list(range(len(model_cfg['VGG5'])))
            split_layer = 4 if split_layer==-1 else split_layer # 定成3吧？

            # 关键路径

            if ep==-1:
                # vgg9 20ep
                unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG9/MNIST/VGG9-MNIST-20ep.pth' # VGG9-BN+Tanh # 存储的是模型参数，不包括模型结构
                results_dir  = f"../../results/{result_ws}/VGG9_MNIST/{test_num}/"
            else:
                # 0ep
                unit_net_route = f'/home/dengruijun/data/FinTech/PP-Split/results/trained_models/VGG9/MNIST/VGG9_MNIST-MNIST-20epoch.pth-{ep}.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
                results_dir  = f"../../results/{result_ws}/VGG9_MNIST/VGG9_{ep}ep/{test_num}/"



            decoder_route = results_dir + f"/Decoder-layer{split_layer}.pth"

            # 切割成client model
            # vgg5_unit.load_state_dict(torch.load(unit_net_route,map_location=torch.device('cpu'))) # 完整的模型
            client_net = VGG('Client','VGG9_MNIST',split_layer,model_cfg,noise_scale=noise_scale)
            pweights = torch.load(unit_net_route)
            if split_layer < len(model_cfg['VGG9_MNIST']):
                pweights = split_weights_client(pweights,client_net.state_dict(),no_dense=no_dense)
            client_net.load_state_dict(pweights)

            # decoder net
            if os.path.isfile(decoder_route): # 如果已经训练好了
                print("=> loading decoder model '{}'".format(decoder_route))
                decoder_net = torch.load(decoder_route)
            else: # 如果没有,加载一个
                print("train decoder model...")
                decoder_net = VGG5Decoder(split_layer=split_layer,network='VGG9_MNIST')


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
        print("get_models error")
        raise ValueError('model error')

    client_net.to(device)
    create_dir(results_dir)

    # 返回值
    msg={}

    msg['image_deprocess'] = image_deprocess
    msg['client_net'] = client_net
    msg['results_dir'] = results_dir
    msg['decoder_net'] = decoder_net
    msg['decoder_route'] = decoder_route

    print('unit_net_route:',unit_net_route)

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
