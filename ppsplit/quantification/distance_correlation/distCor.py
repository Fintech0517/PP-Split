'''
Author: Ruijun Deng
Date: 2023-09-07 10:18:31
LastEditTime: 2023-12-06 16:00:59
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/quantification/distance_correlation/distCor.py
Description: 
# 距离相关性是一种非参数的距离度量，用于衡量两个数据集之间的相关性，不需要对数据的分布进行假设。
# 它的值范围在0到1之间，其中0表示没有相关性，1表示完全相关性。
# 这个函数的目的是计算输入数据集 X 和 Y 之间的距离相关性，以衡量它们之间的相关性程度。
这个不需要再求平均，discor函数里面已经计算每个batch内的平均了
'''
# 导包
import torch
import numpy as np
import pandas as pd




class distCorMetric():
    def __init__(self) -> None:
        super().__init__()

    def quantify(self,inputs,outputs):
        # 返回的是一个numpy array
        assert (inputs.shape[0] == outputs.shape[0]), 'inputs.shape[0]!= outputs.shape[0]'
        batch_size = inputs.shape[0]
        x = inputs.detach().reshape(batch_size,-1)
        y = outputs.detach().reshape(batch_size,-1)
        distCor=self.dist_corr(x,y)
        return distCor.detach().cpu().numpy()

    # 计算每对点之间的平方欧几里得距离
    def pairwise_dist(self,X):
        # from "NoPeek: Information leakage reduction to share activations in distributed deep learning" Vepakomma et al.
        r = torch.sum(X*X, dim=1) # 
        r = r.view(-1, 1)
        D = torch.maximum(r - 2*torch.matmul(X, X.t()) + r.t(), torch.tensor(1e-7))
        D = torch.sqrt(D)
        return D

    def dist_corr(self,X, Y):
        # print(f"X.shape={X.shape},Y.shape={Y.shape}")
        X = X.view(X.size(0), -1) # flatten # 相对于batch的 flatten 1个batch应该也可以吧？
        Y = Y.view(Y.size(0), -1) # flatten
        n = float(X.size(0)) # batchsize
        a = self.pairwise_dist(X)
        b = self.pairwise_dist(Y)

        A = a - torch.mean(a, dim=1) - torch.unsqueeze(torch.mean(a, dim=0), dim=1) + torch.mean(a)
        B = b - torch.mean(b, dim=1) - torch.unsqueeze(torch.mean(b, dim=0), dim=1) + torch.mean(b)

        dCovXY = torch.sqrt(torch.sum(A*B) / (n ** 2))
        dVarXX = torch.sqrt(torch.sum(A*A) / (n ** 2))
        dVarYY = torch.sqrt(torch.sum(B*B) / (n ** 2))

        if dVarXX * dVarYY == 0: # X 和 Y independent？
            dCorXY = torch.tensor(0.0,dtype=torch.float32)
        else:
            dCorXY = dCovXY / torch.sqrt(dVarXX * dVarYY)
        return dCorXY

if __name__=="__main__":
    import argparse

    # 硬件
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'CIFAR10')
    args = parser.parse_args()

    # batchsize (分布采样)
    batch_size = 1000

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
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=8, drop_last=False)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        split_layer_list = ['linear1', 'linear2']
    elif args.dataset=='bank':
        model_path = '../results/1-8/bank-20ep.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/bank/bank-additional-full.csv'
        
        train_data, test_data = preprocess_bank(dataPath)
        test_dataset = bank_dataset(test_data)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=8, drop_last=False)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        split_layer_list = ['linear1', 'linear2']
    elif args.dataset=='purchase':
        model_path = '../results/1-9/epoch_train0.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/Purchase100/'

        trainloader, testloader = preprocess_purchase(data_path=dataPath, batch_size=batch_size)
        # one_data_loader = get_one_data(testloader,batch_size = 5) #拿到第一个测试数据
        split_layer_list = [0,1,2,3,4,5,6,7,8]
    else:
        sys.exit(-1)
    NBatch = len(testloader) # 多少个batch？

    # 存储
    distCorr_diff_layer_list = []
    # 循环计算
    for i in split_layer_list: # 对每层
    # for i in range(1): # 对第一层
        print(f"Layer {i}")
        # 获取模型
        if args.dataset == 'CIFAR10':
            client_net = get_client_net_weighted(model_name, vgg5_unit, i) # 从vgg5-raw中提取前i层
        elif args.dataset == 'credit' or args.dataset == 'bank':
            client_net = torch.load(model_path)
        elif args.dataset == 'purchase':
            # 读取（load）模型
            client_net = PurchaseClassifier()
            model_parameters = torch.load(model_path)['state_dict']
            client_net.load_state_dict(model_parameters)
        else:
            sys.exit(-1)
        
        client_net = client_net.to(device)
        client_net.eval()

        distCorr_same_layer_list = []
        one_layer_distCorr=0.0
        for j, data in enumerate(tqdm.tqdm(testloader)): # 对testloader遍历
        # for j, data in enumerate(tqdm.tqdm(one_data_loader)): # 测试第一个testloader
            tab, labels = data
            tab, labels = tab.to(device), labels.to(device)
            with torch.no_grad():
                pred = client_net.getLayerOutput(tab,i).cpu().detach()
                inputs = tab.cpu().detach()

                distCorr = dist_corr(inputs,pred) # x,z
                distCorr_same_layer_list.append(distCorr.detach().cpu().numpy())
                
                one_layer_distCorr += distCorr/NBatch # 计算本层平均distCorr
                
        print(f"Layer {i} Avg distCorr: {distCorr.item()}")
        distCorr_diff_layer_list.append(distCorr_same_layer_list)

    # 保存到csv中
    # list转numpy
    save_img_dir = f'../results/1-9/DLoss/'
    # 如果文件不存在就创建


    matrix = np.array(distCorr_diff_layer_list) # 有点大，x
    transpose = matrix.T # 一行一条数据，一列代表一个layer 
    pd.DataFrame(data=transpose, columns=[i for i in range (len(split_layer_list))]).to_csv(save_img_dir + f'DLoss-bs{batch_size}.csv',index=False)

# python distCor.py
# nohup python -u distCor-bank-all.py  --dataset CIFAR10 >> dCor-vgg5.out 2>&1  &
# nohup python -u distCor-bank-all.py  --dataset credit >> dCor-credit.out 2>&1  &
# nohup python -u distCor-bank-all.py  --dataset bank >> dCor-bank.out 2>&1  &
# nohup python -u distCor-bank-all.py  --dataset purchase >> dCor-purchase.out 2>&1  &



