'''
Author: Ruijun Deng
Date: 2023-08-28 14:50:08
LastEditTime: 2024-01-13 22:24:27
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/quantification/shannon_information/ULoss.py
Description:  目前这个版本只适应 1 sample，还不能适应batch, 这个是可以求平均的
'''

# 导包
import torch
from torch.nn.functional import softmax, sigmoid
import numpy as np
import tqdm

# scipy的entropy计算
from scipy.stats import entropy as entropy_scipy # scipy的entropy的normalization就是普通的归一化，x/(x+y)


class ULossMetric():
    def __init__(self) -> None:
        pass

    def quantify(self, output, decoder_net):
        return self.inverse_confidence(output,decoder_net)

    # 对数公式
    def _lnyx(self, input, base = 2):
        return np.log(input) / np.log(base)

    # 基数为e的熵
    # TODO:目前的熵函数都是针对batchsize=1的
    def _entropy(self, x,base='e'): # 此时x应该是一个概率分布（离散的)
        x.flatten()
        if base=='e': 
            y = x * np.log(x+1e-9) # y<=0 # log是以e为底的
        else:
            y = x * self._lnyx(x+1e-9,base=base)
        # print("x.shape: ",x.shape)
        # print("y.shape: ",y.shape)
        return -np.sum(y) # 熵

    # 基于分布图的normalize后的熵（yjr）
    def _entropy_prob(self, x):
        x = x.flatten()
        # 概率直方图，bins=256，范围[-1,1] （为什么只统计这一块）
        hist, bin_edges = np.histogram(x, bins=100, density=True) 
        # hist, bin_edges = np.histogram(x, bins=100 if 100<np.size(x) else np.size(x), density=True) 

        # 归一化
        hist_sum = np.sum(hist)# 类似于概率和
        hist = hist / hist_sum # 概率[0,1]
        # print("hist.sum(): ",hist.sum())
        
        # plt.stairs(hist, bin_edges, fill=True, color='red', alpha=0.5)
        # 这时候和softmax差不多了 -> 不一样，虽然也是一种归一化手段，但是得到的分布是不一样的。
        # eps = np.finfo(float).eps # 机器精度? 2.220446049250313e-16
        return self._entropy(hist)

    # branchy_net的entropy（经过softmax归一化）
    def _entropy_branchy(self, x): 
        x = x.flatten()
        if isinstance(x, np.ndarray): # numpy ndarray
            x = softmax(torch.tensor(x),dim=0)
        else: # tensor
            x = softmax(x,dim=0) # 按行softmax # 一版输入的x 会是啥样shape的？
        
        # plot_smashed_distribution(x.numpy())
        # print("*", np.count_nonzero(x.numpy()<0))
        # print('sum(x): ',x.sum())
        y = self.entropy(x.numpy())
        return y

    # DDCC的entropy（经过softmax归一化+除以一个C）# normalized softmax
    # eb = entropy(F.softmax(b))/np.log(b.shape[1]) # 先经过一个softmax，再除以这个logC
    def _entropy_ddcc(self, x): # DDCC的entropy
        x.flatten()
        C = torch.tensor(x).numel()
        # print("C= ",C)
        if isinstance(x, np.ndarray):
            x = softmax(torch.tensor(x),dim=0)
        else:
            x = softmax(x,dim=0)
        y = self.entropy(x.numpy())/np.log(C) # 都是以e为底
        # 当你的x每层输出的神经元个数不同时，分别class数目就是神经元个数
        return y

    def _entropy_sigmoid(self, x): # 感觉不太有用了直接使用sigmoid，没有归一化的normalization
        x = x.flatten()
        if isinstance(x, np.ndarray):
            x = sigmoid(torch.tensor(x))
        else:
            x = sigmoid(x)
        y = self.entropy(x.numpy().flatten())
        return y

    # normalized sigmoid # 并不符合sigma(x)=1 
    def _entropy_FedEntropy(self, x):
        x.flatten()
        K = torch.tensor(x).numel()
        # K = 10
        if isinstance(x, np.ndarray):
            x = sigmoid(torch.tensor(x))
        else:
            x = sigmoid(x)
        # hist, bin_edges = np.histogram(x, bins=500, density=True) # 要有sigma(x)=1
        # x = hist / np.sum(hist) # 概率[0,1]
        y = self.entropy(x.flatten(),base=K)/K # 其实是不需要进行sigma(x)=1的归一化的
        return y

    def inverse_confidence(self, x,decoderNet): # adversaries' confidence
        # 加载decoder网络：
        inversed_input = decoderNet(x)
        inverse_entropy = self._entropy_prob(inversed_input.cpu().detach().numpy())
        return inverse_entropy # np的值？
    

if __name__=='__main__':
    import pandas as pd

    # 硬件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', type = str, default = '1-6')
    parser.add_argument('--network', type = str, default = 'VGG5')
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--dataset', type = str, default = 'CIFAR10')
    args = parser.parse_args()
    
       # 获取数据集
    if args.dataset=='CIFAR10':
        trainloader,testloader = get_cifar10_normalize()
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据

        # VGG5
        model_path = '../results/VGG5/BN+Tanh/VGG5-0ep.pth' # VGG5-BN+Tanh
        vgg5_unit = VGG('Unit', 'VGG5', len(model_cfg['VGG5'])-1, model_cfg) # 加载模型结构
        vgg5_unit.load_state_dict(torch.load(model_path)) # 加载模型参数
        split_layer_list = list(range(len(model_cfg['VGG5'])))
    elif args.dataset=='credit':
        model_path = '../results/1-7/credit-20ep.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/home_credit/dataset/application_train.csv'
        train_data, test_data = preprocess_credit(dataPath)
        test_dataset = bank_dataset(test_data)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                num_workers=8, drop_last=False)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        split_layer_list = ['linear1', 'linear2']
    elif args.dataset=='bank':
        model_path = '../results/1-8/bank-20ep.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/bank/bank-additional-full.csv'
        
        train_data, test_data = preprocess_bank(dataPath)
        test_dataset = bank_dataset(test_data)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                num_workers=8, drop_last=False)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        split_layer_list = ['linear1', 'linear2']
    elif args.dataset=='purchase':
        model_path = '../results/1-9/epoch_train0.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/Purchase100/'
        trainloader, testloader = preprocess_purchase(data_path=dataPath, batch_size=1)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        split_layer_list = [3]
    else:
        sys.exit(-1)
    NBatch = len(testloader) # 多少个batch？

    # 多层存储
    InvEnt_diff_layer_list = []
    # 循环计算
    for i in split_layer_list: # 对每层
    # for i in range(1): # 对第一层 # 第0层啊。。。。这是
        print(f"Layer {i}")

        # 获取模型
        if args.dataset == 'CIFAR10':
            client_net = get_client_net_weighted(model_name, vgg5_unit, i) # 从vgg5-raw中提取前i层
            decoder_route = "../results/" + str(args.serial) + '/' + args.network+'-raw_layer'+ str(i) +'_decoder-0ep'+ '.pth'  # decoder模型
            decoderNet = torch.load(decoder_route)
        elif args.dataset == 'credit':
            client_net = torch.load(model_path)
            decoder_route = '../results/credit/decoder-120.pth'
            decoderNet = torch.load(decoder_route)
        elif args.dataset == 'bank':
            client_net = torch.load(model_path)
            decoder_route = '../results/1-8/decoder-layer2-120.pth'
            decoderNet = torch.load(decoder_route)
        elif args.dataset == 'purchase':
            # 读取（load）模型
            client_net = PurchaseClassifier()
            model_parameters = torch.load(model_path)['state_dict']
            client_net.load_state_dict(model_parameters)
            decoder_route = '../results/1-9/decoder-5ep-mse.pth'
            decoderNet = torch.load(decoder_route)
        else:
            sys.exit(-1)
        
        client_net = client_net.to(device)
        decoderNet = decoderNet.to(device)
        client_net.eval()
        decoderNet.eval()

        # 输出
        InvEnt_same_layer_list = []
        one_layer_InvEnt_metric = 0.0
        for j, data in enumerate(tqdm.tqdm(testloader)): # 对testloader遍历
        # for j, data in enumerate(tqdm.tqdm(one_data_loader)): # 测试第一个testloader
            tab, labels = data
            tab, labels = tab.to(device), labels.to(device)
            with torch.no_grad():
                pred = client_net.getLayerOutput(tab,i).cpu().detach()

                InvEnt_metric = inverse_confidence(pred,decoderNet,device) # ? 数据集上的
                
                one_layer_InvEnt_metric += InvEnt_metric/NBatch # 计算本层平均inverse entropy
                InvEnt_same_layer_list.append(InvEnt_metric.item())

        print(f"Layer {i}: avg InvEnt=",one_layer_InvEnt_metric.item())
        InvEnt_diff_layer_list.append(InvEnt_same_layer_list)

    # 保存到csv中
    # list转numpy
    save_img_dir = f'../results/1-9/ULoss/'
    matrix = np.array(InvEnt_diff_layer_list) # 有点大，x
    transpose = matrix.T # 一行一条数据，一列代表一个layer 
    
    pd.DataFrame(data=transpose, columns=[i for i in split_layer_list]).to_csv(save_img_dir + f'ULoss-bs{args.batch_size}.csv',index=False)

