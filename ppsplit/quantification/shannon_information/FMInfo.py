# 导包
import torch
import os
import argparse
import pandas as pd
import tqdm
import numpy as np
from torch.nn.functional import avg_pool2d
import math
import torch.autograd.functional as F

# os.environ['NUMEXPR_MAX_THREADS'] = '48'

class FMInfoMetric():
    def __init__(self, sigma=0.01, device = 'cpu') -> None:
        self.sigma = sigma
        self.device = device

    def quantify(self, inputs, client_net, ):
        '''
        输入一个batch的inputs
        '''
        image_dimension = inputs[0].numel()

        # effectFisher
        outputs = client_net(inputs)
        FMInfo = - self.computing_diag_det_with_outputs(model=client_net, 
                                                            inputs=inputs, 
                                                            outputs=outputs,
                                                            sigmas = self.sigma)
        
        avg_d_FMI = FMInfo / image_dimension

        print("FMInfo: ",FMInfo)
        print("avg_d_FMInfo: ",avg_d_FMI)

        return FMInfo, avg_d_FMI

    # Effect Fisher
    def computing_det_with_outputs(self, model, inputs, outputs, sigmas=1.0): # sigma_square
        '''
        自己实现的、规规矩矩的 jacobian + logdet 全部用torch的函数
        '''
        # batchsize:
        batch_size = inputs.shape[0] # 一个batch的样本数目
        output_size = outputs[0].numel() # 一个样本的outputs长度
        input_size = inputs[0].numel() # 一个样本的outputs长度
        effect_fisher_sum = 0.0

        # 遍历单个样本: 换数据
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
            # ddFIL  = I.trace().div(input_size*input_size)

            # 储存I
            # I_np = I.cpu().detach().numpy()
            # df = pd.DataFrame(I_np)
            # df.to_csv(f'{i}.csv',index=False,header=False)

            # print("I: ", I)
            # w = torch.det(I)
            # print('det I: ', I.det().log())

            f1 = input_size * torch.log(2*torch.pi*torch.exp(torch.tensor(1.0)))
            f2 = torch.logdet(I)
            # print('log det I: ',f2 )
            print('f1: ' ,f1)
            print('f2: ' ,f2)
            effect_fisher = 0.5 * (f1 - f2)
            effect_fisher_sum += effect_fisher

            print("effect_fisher: " , effect_fisher)

        # print("Jt*J: ", JtJ)
        # print("Jt*J: ", JtJ.shape, JtJ)
        # print("I.shape: ", I.shape)
        # eta = dFIL
        # print(f"eta: {eta}")
        # print('t2-t1=',t2-t1, 't3-t2', t3-t2)
        effect_fisher_mean = effect_fisher_sum / batch_size
        return effect_fisher_mean.cpu().detach().numpy()

    def computing_diag_det_with_outputs(self, model, inputs, outputs, sigmas=1.0): # # 用diag 来化简 sigma_square
        # batchsize:
        batch_size = inputs.shape[0] # 一个batch的样本数目
        output_size = outputs[0].numel() # 一个样本的outputs长度
        input_size = inputs[0].numel() # 一个样本的outputs长度
        effect_fisher_sum = 0.0

        # avg
        I_diagonal_batch_avg = torch.zeros(input_size).to(self.device) # batch上做平均
        print("I_diagonal_batch_avg: ",I_diagonal_batch_avg.shape)
        f2_2_avg_outer = torch.tensor(0.0).to(self.device)
        f2_avg_outer = torch.tensor(0.0).to(self.device)
        
        # effecti_fisher第一部分
        f1 = input_size * torch.log(2*torch.pi*torch.exp(torch.tensor(1.0)))

        # f2需要求平均？
        # 遍历单个样本: 换数据
        for i in range(batch_size): # 对每个样本
            input_i = inputs[i].unsqueeze(0)

            # 计算jacobian
            J = F.jacobian(model, input_i)
            # J = J.reshape(J.shape[0],outputs.numel(),inputs.numel()) # (batch, out_size, in_size)
            J = J.reshape(output_size, input_size) # (batch, out_size, in_size)
            # print(f"J2.shape: {J.shape}, J2.prod: {torch.prod(torch.tensor(list(J.shape)))}")
            # 计算eta
            JtJ = torch.matmul(J.t(), J)
            I = 1.0/(sigmas)*JtJ

            # I = JtJ
            # print("I: ", I)
            # diagonal fisher information matrix (approximation)
            I_diagonal = torch.diagonal(I,dim1=0,dim2=1) # vector
            # print("I_diagonal: ",I_diagonal.shape)

            I_diag = torch.diag_embed(I_diagonal) # matrix
            # print('drj trace: ',torch.trace(I_diag))
            
            # batch的平均
            I_diagonal_batch_avg += I_diagonal / (batch_size)

            # # 储存I
            # I_np = I.cpu().detach().numpy()
            # df = pd.DataFrame(I_np)
            # df.to_csv(f'{i}.csv',index=False,header=False)

            # print("I: ", I)
            # w = torch.det(I)
            # print('det I: ', I.det().log())
            
            try:
                s,f2 = torch.slogdet(I) # 直接用torch计算
                if s <= 0:
                    raise RuntimeError("sign <=0 ")
                print('f2: ', f2)
            except RuntimeError as e:
                print("logdet计算报错")
            # f2_1 = torch.logdet(I_diag) # 和后面的是一样的
            f2_2 = torch.sum(torch.log(I_diagonal+1e-10)) # /I_diagonal.numel() # diagonal后计算

            f2_2_avg_outer += f2_2 / batch_size
            # f2_avg_outer += f2 / batch_size

            # print('log det I: ', f2)
            # print('f1: ' , f1)
            # print('f2: ', f2)
            # print('f2_1: ', f2_1)
            print('f2_2: ', f2_2)

        f2_2_avg_inner = torch.sum(torch.log(I_diagonal_batch_avg+1e-10)) # 用平均后的diagonal 计算

        print('f2_avg_outer: ',f2_avg_outer)
        print('f2_2_avg_outer: ',f2_2_avg_outer)
        # print('f2_2_avg_inner: ',f2_2_avg_inner)
        print('f1: ',f1)

        # effect_fisher = 0.5 * (f1 - f2_2_avg_inner)
        effect_fisher = 0.5 * (f1 - f2_2_avg_outer)
        # effect_fisher = 0.5 * (f1 - f2_avg_outer)
        # effect_fisher_sum+=effect_fisher

        # print("effect_fisher: ",effect_fisher)
        
        # effect_fisher_mean = effect_fisher_sum / batch_size
        return effect_fisher.cpu().detach().numpy()

    # Effect Uniform
    def calculate_effect_normalize(self, input_vector,interval=(-1.0,1.0)): # effect uniform
        '''
        直接给参数计算uniform熵
        '''
        interval_len = interval[1] - interval[0]
        # 确定每个维度的取值范围
        a = torch.tensor(interval_len)
        # 计算每个维度的熵
        entropy_per_dimension = torch.log(a)
        # 总熵是每个维度的熵的总和
        size = input_vector.numel()
        total_entropy = size * entropy_per_dimension
        return total_entropy
    
    def calculate_effect_normalize_hetero(self, input_vector, interval=(1.0,-1.0)):
        '''
        给定x的条件下的uniform的熵
        '''
        size = input_vector.numel()
        input_flattened = input_vector.reshape(-1)
        total_entropy_single = 0.0
        for i in range(size):
            l = 2*torch.min(torch.abs(input_flattened[i]-torch.tensor(interval[0])),torch.abs(input_flattened[i]-torch.tensor(interval[1])))
            total_entropy_single += torch.log(l+1e-10)
        print(f"entropy for single_input: {total_entropy_single}")
        return total_entropy_single 

    def calculate_effect_normalize_hetero_batch(self, inputs, interval=(1.0,-1.0)):
        '''
        计算一个batch 的uniform的熵
        '''
        # batchsize:
        batch_size = inputs.shape[0] # 一个batch的样本数目
        total_entropy = 0.0

        for i in range(batch_size):
            input_i = inputs[i].unsqueeze(0)
            total_entropy += self.calculate_effect_normalize_hetero(input_i,interval)
        
        return total_entropy/batch_size
    
    # Effect Entropy
    def shannon_entropy_pyent(self, time_series): # 这个甚至不适合连续值吧
        """
        pyent来计算高维随机变量的熵
        Calculate Shannon Entropy of the sample data.

        Parameters
        ----------
        time_series: np.ndarray | list[str]

        Returns
        -------
        ent: float
            The Shannon Entropy as float value
        """

        # Calculate frequency counts
        _, counts = np.unique(time_series, return_counts=True)
        total_count = len(time_series)
        # print('counts: ', counts)
        # print("total_count: ",total_count)

        # Calculate frequencies and Shannon entropy
        frequencies = counts / total_count
        # print("freq: ",frequencies)
        ent = -np.sum(frequencies * np.log(frequencies))

        return ent
    
    def shannon_entropy_infotopo(self, x, conv = False):
        from . import infotopo
        information_top = infotopo.infotopo(dimension_max = x.shape[1],
                                            dimension_tot = x.shape[1],
                                            sample_size = x.shape[0],
                                            # nb_of_values = nb_of_values, # 不是很懂这个意思，为什么iris对应9？
                                            # nb_of_values = 17, # 不是很懂这个意思，为什么iris对应9？
                                            nb_of_values = nb_of_values, # 不是很懂这个意思，为什么iris对应9？
                                            # forward_computation_mode = True,
                                            )
        if conv:
            images_convol = information_top.convolutional_patchs(x)
            print('images_convol: ',images_convol.shape)
            x = images_convol

        # 计算联合分布的概率？（全排列）
        # joint_prob = information_top._compute_probability(x)
        # print('joint_prob: ',joint_prob)
        
        # 计算联合熵（全排列的）
        joint_prob_ent = information_top.simplicial_entropies_decomposition(x) # log2
        new_joint_prob_ent = {key: value * np.log(2) for key, value in joint_prob_ent.items()} #ln 转2为底 成 e为底
        
        # print("joint_entropy: ",new_joint_prob_ent)
        # ent = information_top._compute_forward_entropies(x)
        # information_top.entropy_simplicial_lanscape(joint_prob_ent) # 画图
        # ent = _entropy(np.array(list(new_joint_prob_ent.values())))

        joint_entropy_final = list(new_joint_prob_ent.values())[-1]
        return joint_entropy_final

    def shannon_entropy_approximation(images):
        '''
        # approximation entropy: 参考 sec24，用高斯分布近似
        '''
        images_flat = images.view(images.size(0), -1)
        # 计算均值和方差
        mean = torch.mean(images_flat, dim=0)
        covariance_matrix = torch.cov(images_flat.T)

        n = covariance_matrix.size(0)
        # det_cov = torch.det(covariance_matrix)
        # 计算熵
        print('covariance_matrix: ',covariance_matrix)

        # try:
        #     s,f2 = torch.slogdet(covariance_matrix) # 直接用torch计算
        #     # print('s: ',s)
        #     if s <= 0:
        #         # print('s=',s)
        #         raise RuntimeError("sign <=0 ")
        #     print('input entropy approximation f2 : ', f2)
        # except RuntimeError as e:
        #     print("logdet计算报错")
        #     raise e

        determinant = torch.det(covariance_matrix)
        print('determinant: ',determinant)
        
        entropy = 0.5 * (n * (np.log(2 * np.pi * np.e)) + torch.log(determinant))
        # entropy = 0.5 * (n * (np.log(2 * np.pi * np.e)) + torch.logdet(covariance_matrix))
        # entropy = 0.5 * (n * (np.log(2 * np.pi * np.e)) + f2)

        # 打印结果
        print(f"均值: {mean.mean().item()}")
        print(f"协方差矩阵的行列式: {torch.det(covariance_matrix).item()}")
        print(f"熵: {entropy.item()}")
        return entropy
    


