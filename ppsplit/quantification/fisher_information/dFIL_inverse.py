'''
Author: Ruijun Deng
Date: 2023-08-28 14:50:43
LastEditTime: 2024-09-04 23:26:40
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/quantification/fisher_information/dFIL_inverse.py
Description: 一个一个样本计算，没有平均之说
'''
# FIL 计算函数
import torch.autograd.functional as F
import torch
import time

# nips23
from torch.autograd.functional import jvp
import random
import math

# import logging
# logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
import os
os.environ['NUMEXPR_MAX_THREADS'] = '48'

class dFILInverseMetric():
    def __init__(self) -> None:
        pass
    
    def quantify(self, model, inputs, outputs = None, sigmas=0.01, with_outputs = True):
        if with_outputs:
            return self._computing_eta_with_outputs(model, inputs, outputs, sigmas).detach().cpu().numpy()
        else:
            return self._computing_eta_without_outputs(model, inputs,  sigmas).detach().cpu().numpy()

    # model的smashed data需要在[0,1]之间，才能保证输出的eta也在[0,1]之间?证明？
    def _computing_eta_without_outputs(self, model, inputs,  sigmas): # sigma_square
        inputs.requires_grad_(True) # 需要求导
        outputs = model(inputs)
        
        # 前向传播
        # outputs = outputs + sigma * torch.randn_like(outputs) # 加噪声 (0,1] uniform
    
        # 计算jacobian
        J = F.jacobian(model, inputs)
        # J = J.reshape(J.shape[0],outputs.numel(),inputs.numel()) # (batch, out_size, in_size)
        J = J.reshape(outputs.numel(),inputs.numel()) # (batch, out_size, in_size)
        # print(f"J2.shape: {J.shape}, J2.prod: {torch.prod(torch.tensor(list(J.shape)))}")

        # 计算eta
        I = 1.0/(sigmas)*torch.matmul(J[0].t(), J[0])
        # print(f"I.shape: ", I.shape)
        dFIL = I.trace().div(inputs.numel())
        # eta = dFIL
        # print(f"eta: {eta}")
        # print('t2-t1=',t2-t1, 't3-t2', t3-t2)
        return 1.0/dFIL
    
        # model的smashed data需要在[0,1]之间，才能保证输出的eta也在[0,1]之间?证明？
        
    def _computing_eta_with_outputs(self, model, inputs, outputs, sigmas): # sigma_square
        '''
        可以直接处理batch的数据
        '''
        # 前向传播
        # outputs = outputs + sigma * torch.randn_like(outputs) # 加噪声 (0,1] uniform
    
        # 计算jacobian
        J = F.jacobian(model, inputs)
        # J = J.reshape(J.shape[0],outputs.numel(),inputs.numel()) # (batch, out_size, in_size)
        J = J.reshape(outputs.numel(),inputs.numel()) # (batch, out_size, in_size)
        # print(f"J2.shape: {J.shape}, J2.prod: {torch.prod(torch.tensor(list(J.shape)))}")

        # 计算eta
        JtJ = torch.matmul(J.t(), J)
        # print("Jt*J: ", JtJ)
        # print("Jt*J: ", JtJ.shape, JtJ)
        I = 1.0/(sigmas)*JtJ
        # print("I.shape: ", I.shape)
        dFIL = I.trace().div(inputs.numel())
        # eta = dFIL
        # print(f"eta: {eta}")
        # print('t2-t1=',t2-t1, 't3-t2', t3-t2)
        return 1.0/dFIL

# TODO: 未完成
    def _computing_det_with_outputs(self, model, inputs, outputs, sigmas): # sigma_square
        # batchsize:
        batch_size = inputs.shape[0] # 一个batch的样本数目
        output_size = outputs[0].numel() # 一个样本的outputs长度
        input_size = inputs[0].numel() # 一个样本的outputs长度
        effect_fisher_sum = 0.0

        # 遍历单个样本:
        for i in range(batch_size):
            input_i = inputs[i].unsqueeze(0)

            # 计算jacobian
            J = F.jacobian(model, input_i)
            # J = J.reshape(J.shape[0],outputs.numel(),inputs.numel()) # (batch, out_size, in_size)
            J = J.reshape(output_size, input_size) # (batch, out_size, in_size)
            # print(f"J2.shape: {J.shape}, J2.prod: {torch.prod(torch.tensor(list(J.shape)))}")
            # 计算eta
            JtJ = torch.matmul(J.t(), J)
            I = 1.0/(sigmas)*JtJ
            effect_fisher = 0.5 * (input_size * torch.log(2*torch.pi*torch.exp(torch.tensor(1.0))) - torch.logdet(I))
            effect_fisher_sum+=effect_fisher

        return effect_fisher_sum.cpu().detach().numpy()
    
    def calc_tr(self, net, x, device, sigmas=0.01, subsample=-1, jvp_parallelism=1): # nips'23 源码
        '''
        calc_tr 函数利用雅可比向量积（JVP）来估计网络对于输入数据的迹，
        这在分析网络的灵敏度或稳定性时非常有用。
        此外，通过支持子采样和并行处理，该函数还提供了一种在保持计算效率的同时估计迹的方法。
        返回的也是inverse_dFIL
        '''
        print(f'x.shape: {x.shape}')
        
        # 定义一个局部函数 jvp_func**：这个函数接受两个参数 x 和 tgt，并返回 net.forward_first 方法的雅可比向量积（JVP）。
        # 这意味着 jvp_func 用于计算网络对于输入 x 在方向 tgt 上的一阶导数
        # tgt 计算雅各比向量积的向量
        def jvp_func(x, tgt): 
            # return jvp(net.forward_first, (x,), (tgt,)) #返回 outputs, jacobian product
            return jvp(net.forward, (x,), (tgt,)) #返回 outputs, jacobian product

        # 获取一个batch中第一个数据的维度？d代表的是批次中第一个数据点展平后的特征数量，即输入数据的维度。
        d = x[0].flatten().shape[0] # 把一个batch的x展平，获取input dim

        # 用于存储每个输入数据点的迹，求迹的和。
        tr = torch.zeros(x.shape[0], dtype=x.dtype).to(device)
        #print(f'd: {d}, {x.shape}')

        # 加速，矩阵降维，但是这个损伤精度，或许改成特征提取更好点？ # 也是用了矩阵降维度
        # Randomly subsample pixels for faster execution
        if subsample > 0: 
            samples = random.sample(range(d), min(d, subsample)) # 选取了部分维度
        else:
            samples = range(d)

        # print(x.shape, d, samples)
        # jvp parallelism是数据并行的粒度？
        # 函数通过分批处理样本来计算迹，每批处理 jvp_parallelism 个样本
        for j in range(math.ceil(len(samples) / jvp_parallelism)): # 对于每个数据块 # 每个数据块包含不同的维度
            tgts = []

            # 遍历每个数据块中的每个维度
            '''
            在这个函数中，tgt 是用于计算雅可比向量积（JVP）的向量。具体来说，tgt 的作用如下：
            构建雅可比向量积的向量：tgt 是一个与输入 x 形状相同的张量，但它的元素大部分为零，只有一个特定位置的元素为 1。这个特定位置对应于我们在计算迹时关注的特征维度。
            计算 JVP：在 helper 函数中，tgt 被传递给 jvp_func，用于计算网络对于输入 x 在方向 tgt 上的一阶导数。具体来说，jvp_func 计算的是网络输出相对于输入 x 的雅可比矩阵与 tgt 的乘积。
            估计迹：通过在不同的特征维度上重复上述过程，可以估计网络对于输入数据的迹。迹的计算涉及到对所有特征维度的导数进行求和，而 tgt 的作用就是在每次计算时只关注一个特征维度。
            简而言之，tgt 是一个用于选择特定特征维度的向量，通过它可以逐个计算每个特征维度的导数，从而最终估计整个输入数据的迹。
            '''
            for k in samples[j*jvp_parallelism:(j+1)*jvp_parallelism]: # 提取整个batch中每个数据的特定维度
                tgt = torch.zeros_like(x).reshape(x.shape[0], -1) # 按照batch 排列？# 雅各比向量积的
                # 除了当前样本索引 k 对应的元素设置为 1。这相当于在计算迹时，每次只关注一个特征维度。
                tgt[:, k] = 1. # 提取tgt所有的样本的k的特征 计算雅各比向量积的向量，可用于计算trace，所有行的特定几列有1值
                tgt = tgt.reshape(x.shape) # 又变回x的形状
                tgts.append(tgt)
            tgts = torch.stack(tgts) # 把多个维度的tgt vstack，一行一行拼接起来


            # 定义一个辅助函数 helper，该函数接受一个目标张量 tgt并返回一个迹的张量和一个值的张量。
            # jvp wrapper，遍历每个batchsize
            def helper(tgt):
                batch_size = x.shape[0]
                vals_list = []
                grads_list = []
                for i in range(batch_size): # 对每个样本
                    val, grad = jvp_func(x[i], tgt[i])  # 对每个批次元素调用jvp_func
                    vals_list.append(val)
                    grads_list.append(grad)
                # 将结果列表转换为张量, 多个batch的给stack起来
                vals = torch.stack(vals_list)
                grad = torch.stack(grads_list)

                # vals, grad = vmap(jvp_func, randomness='same')(x, tgt)
                
                # print('grad shape: ', grad.shape)
                # 因此，矩阵平方的迹和迹的平方通常是不相等的。
                # 先求平方再求迹
                return torch.sum(grad * grad, dim=tuple(range(1, len(grad.shape)))), vals 

            # vmap被替换，
            # 遍历每个数据块
            trs,vals = [],[]
            for item in tgts:
                trs_, vals_ = helper(item)
                trs.append(trs_) # 每个batch对应一个向量
                vals.append(vals_)
            trs,vals = torch.stack(trs),torch.stack(vals)

            # trs, vals = vmap(helper, randomness='same')(tgts) # randomness for randomness control of dropout
            # vals are stacked results that are repeated by d (should be all the same)

            tr += trs.sum(dim=0) # 对每个数据块的迹求和

        # Scale if subsampled
        if subsample > 0:
            tr *= d / len(samples)

        # 1/dFIL = d/tr(I)
        tr = tr/(d*1.0)
        tr = 1.0/tr*sigmas

        # print('tr: ',tr.shape, tr)
        return tr.cpu().item(), vals[0].squeeze(1)  # squeeze removes one dimension jvp puts

        # nips'23 fisher trace 计算
    def calc_tr(net, x, device, sigmas=0.01, subsample=-1, jvp_parallelism=1): # nips'23 源码
        '''
        FMInfo中改过的
        '''
        # 并行粒度=1 意思是，每次只处理一个维度

        print(f'x.shape: {x.shape}')
        
        # 定义一个局部函数 jvp_func**：这个函数接受两个参数 x 和 tgt，并返回 net.forward_first 方法的雅可比向量积（JVP）。
        # 这意味着 jvp_func 用于计算网络对于输入 x 在方向 tgt 上的一阶导数
        # tgt 计算雅各比向量积的向量
        def jvp_func(x, tgt): 
            # return jvp(net.forward_first, (x,), (tgt,)) #返回 outputs, jacobian product
            return jvp(net.forward, (x,), (tgt,)) #返回 outputs, jacobian product

        # 获取一个batch中第一个数据的维度？d代表的是批次中第一个数据点展平后的特征数量，即输入数据的维度。
        d = x[0].flatten().shape[0] # 把一个batch的x展平，获取input dim

        # 用于存储每个输入数据点的迹，求迹的和。
        tr = torch.zeros(x.shape[0], dtype=x.dtype).to(device)
        print(f'tr.shape: {tr.shape}')

        samples = range(d)

        for j in range(math.ceil(d)): # 对于每个数据块 # 每个数据块包含不同的维度
            tgts = []

            # 遍历每个数据块中的每个维度
            '''
            在这个函数中，tgt 是用于计算雅可比向量积（JVP）的向量。具体来说，tgt 的作用如下：
            构建雅可比向量积的向量：tgt 是一个与输入 x 形状相同的张量，但它的元素大部分为零，只有一个特定位置的元素为 1。这个特定位置对应于我们在计算迹时关注的特征维度。
            计算 JVP：在 helper 函数中，tgt 被传递给 jvp_func，用于计算网络对于输入 x 在方向 tgt 上的一阶导数。具体来说，jvp_func 计算的是网络输出相对于输入 x 的雅可比矩阵与 tgt 的乘积。
            估计迹：通过在不同的特征维度上重复上述过程，可以估计网络对于输入数据的迹。迹的计算涉及到对所有特征维度的导数进行求和，而 tgt 的作用就是在每次计算时只关注一个特征维度。
            简而言之，tgt 是一个用于选择特定特征维度的向量，通过它可以逐个计算每个特征维度的导数，从而最终估计整个输入数据的迹。
            '''
            # 对于每一列，构建tgt， 形状和x一样，但是只有一列是1，其他是0
            for k in samples[j:(j+1)]: # 提取整个batch中每个数据的特定维度
                tgt = torch.zeros_like(x).reshape(x.shape[0], -1) # 按照batch 排列？# 雅各比向量积的
                # 除了当前样本索引 k 对应的元素设置为 1。这相当于在计算迹时，每次只关注一个特征维度。
                tgt[:, k] = 1. # 提取tgt所有的样本的k的特征 计算雅各比向量积的向量，可用于计算trace，所有行的特定几列有1值
                tgt = tgt.reshape(x.shape) # 又变回x的形状
                # print(f'tgt.shape: {tgt.shape}')
                tgts.append(tgt) 
            tgts = torch.stack(tgts) # 把多个维度的tgt vstack，一行一行拼接起来，一行是一个维度。


            # 定义一个辅助函数 helper，该函数接受一个目标张量 tgt并返回一个迹的张量和一个值的张量。
            # jvp wrapper，遍历每个batchsize
            def helper(tgt,x=x): # x是一个batch的数据
                batch_size = x.shape[0]
                grads_list = []
                for i in range(batch_size): # 对每个样本
                    _, grad = jvp_func(x[i].unsqueeze(0), tgt[i].unsqueeze(0))  # 对每个批次元素调用jvp_func
                    grads_list.append(grad)
                # 将结果列表转换为张量, 多个batch的给stack起来
                grad = torch.stack(grads_list)

                # print('grad.shape: ',grad.shape)
                # print('grad: ',grad)

                # grad.reshape(sum(list(x.shape)),-1)
                # I_np = grad.cpu().detach().numpy()
                # df = pd.DataFrame(I_np)
                # df.to_csv(f'{time.time()}.csv',index=False,header=False)

                # print('grad*grad: ',grad*grad)
                # vals, grad = vmap(jvp_func, randomness='same')(x, tgt)
                
                # print('grad shape: ', grad.shape)
                # 因此，矩阵平方的迹和迹的平方通常是不相等的。
                # 先求平方再求迹
                # range(1, len(grad.shape)) 生成一个从 1 到 len(grad.shape) - 1 的整数序列。
                # torch.sum 函数对张量的指定维度进行求和。
                # 这里，它对 grad * grad 沿着 tuple(range(1, len(grad.shape))) 指定的维度进行求和。
                # ？为什么呢？--- 前面有个unsqueeze？
                return torch.sum(grad * grad, dim=tuple(range(1, len(grad.shape))))

            # vmap被替换
            # 遍历每个数据块
            trs,vals = [],[]
            for item in tgts: # 对每个维度
                trs_ = helper(item,x)
                trs.append(trs_) # 每个batch对应一个向量
                # print('trs_: ',trs_.shape)
            trs= torch.stack(trs) 
            trs = torch.log(trs+1e-10) # 为了求 f2 logdet
            # print('trs: ',trs.shape, trs)

            # 对数据，的每个维度的迹求和
            tr += trs.sum(dim=0) 
        print('tr: ',tr)

        return tr  # squeeze removes one dimension jvp puts


# 多层、整个数据集上的dFIL
if __name__ == '__main__':
    # 导包
    import torch
    import pandas as pd
    import tqdm
    import sys
    sys.path.append('../')

    from utils.utils import *
    from utils.datasets import *

    import argparse

    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'CIFAR10')
    args = parser.parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")
    print("current device: ", device)
    sigma = 0.01

    # 获取数据集
    if args.dataset=='CIFAR10':
        save_img_dir = f'../results/1-6/dFIL/'
        trainloader,testloader = get_cifar10_normalize(batch_size=1)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据

        # VGG5
        model_path = '../results/VGG5/BN+Tanh/VGG5-0ep.pth' # VGG5-BN+Tanh
        vgg5_unit = VGG('Unit', 'VGG5', len(model_cfg['VGG5'])-1, model_cfg) # 加载模型结构
        vgg5_unit.load_state_dict(torch.load(model_path)) # 加载模型参数
        split_layer_list = list(range(len(model_cfg['VGG5'])))
    elif args.dataset=='credit':
        save_img_dir = f'../results/1-7/dFIL/'
        model_path = '../results/1-7/credit-20ep.pth'

        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/home_credit/dataset/application_train.csv'
        train_data, test_data = preprocess_credit(dataPath)
        test_dataset = bank_dataset(test_data)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                num_workers=8, drop_last=False)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        # split_layer_list = ['linear1', 'linear2']
        split_layer_list = [0,3,6,9]
    elif args.dataset=='bank':
        model_path = '../results/1-8/bank-20ep.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/bank/bank-additional-full.csv'
        save_img_dir = f'../results/1-8/dFIL/'

        train_data, test_data = preprocess_bank(dataPath)
        test_dataset = bank_dataset(test_data)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                num_workers=8, drop_last=False)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        # split_layer_list = ['linear1', 'linear2']
        split_layer_list = [0,2,4,6]
    elif args.dataset=='purchase':
        save_img_dir = f'../results/1-9/dFIL/'
        model_path = '../results/1-9/epoch_train0.pth'
        dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/Purchase100/'

        trainloader, testloader = preprocess_purchase(data_path=dataPath, batch_size=1)
        one_data_loader = get_one_data(testloader,batch_size = 1) #拿到第一个测试数据
        # split_layer_list = [3]
        # split_layer_list = [0,1,2,3,4,5,6,7,8]
        split_layer_list = [1,3,5,7]
    else:
        sys.exit(-1)


    ##############开始计算dFIL############
    # 加载训练好的 edge 模型
    eta_diff_layer_list = []
    # for i in range (len(model_cfg['VGG5'])): # 对每一层
    for i in split_layer_list: # 对每一层
        print(" Layer: ", i)

        # 获取模型
        if args.dataset == 'CIFAR10':
            client_net = get_client_net_weighted(model_name, vgg5_unit, i) # 从vgg5-raw中提取前i层
        elif args.dataset == 'bank':
            client_net = BankNet1(layer=i)
            pweights = torch.load('../results/1-8/bank-20ep.pth').state_dict()
            if i < 6:
                pweights = split_weights_client(pweights,client_net.state_dict())
            client_net.load_state_dict(pweights)
        elif args.dataset == 'credit':
            client_net = CreditNet1(layer=i)
            pweights = torch.load('../results/1-7/credit-20ep.pth').state_dict()
            if i < 9:
                pweights = split_weights_client(pweights,client_net.state_dict())
            client_net.load_state_dict(pweights)          
        elif args.dataset == 'purchase':
            # 读取（load）模型
            client_net = PurchaseClassifier1(layer=i)
            pweights  = torch.load(model_path)['state_dict']
            if i < 8: # 
                pweights = split_weights_client(pweights ,client_net.state_dict())
            client_net.load_state_dict(pweights)
        else:
            sys.exit(-1)
        
        client_net = client_net.to(device)
        client_net.eval()
        
        eta_same_layer_list = []
        # 对traingloader遍历计算所有eta？
        for j, data in enumerate(tqdm.tqdm(testloader)):
        # for j, data in enumerate(tqdm.tqdm(one_data_loader)): # 测试第一个testloader
        #     if j < 31705:
        #         continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad_(True) # 需要求导
            
            # inference
            # if args.dataset == 'CIFAR10' or args.dataset == 'purchase' or args.dataset == 'bank':
            outputs = client_net(inputs)
            # elif args.dataset == 'credit': # bank, credit, purchase
                # outputs = client_net.getLayerOutput(inputs,i)
            # else:
                # sys.exit(-1)

            eta = computing_eta(client_net, inputs, outputs, sigma)

            logger.info(str(j)+": "+str(eta.item()))
            eta_same_layer_list.append(eta.detach().cpu().numpy())

        eta_diff_layer_list.append(eta_same_layer_list)

    # 保存到csv中
    # save_img_dir = f'../results/1-8/dFIL/'
    matrix = np.array(eta_diff_layer_list) # 有点大，x
    transpose = matrix.T # 一行一条数据，一列代表一个layer 

    pd.DataFrame(data=transpose, columns=[i for i in split_layer_list]).to_csv(save_img_dir + f'dFIL-1.csv',index=False)

# nohup python -u dFIL.py --dataset CIFAR10 >> dFIL-cifar10.out 2>&1  &
# nohup python -u dFIL.py --dataset bank >> dFIL-bank.out 2>&1  &
# nohup python -u dFIL.py --dataset credit >> dFIL-credit1.out 2>&1  &
# nohup python -u dFIL.py --dataset purchase >> dFIL-purchase.out 2>&1  &

# credit 4layer [1] 27598 [4] 22073
# purchase 1,3,5,7 [2] 3705


