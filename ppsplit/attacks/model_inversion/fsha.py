import time
import math
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from torch.utils.data import  DataLoader

from tqdm import tqdm


from.inverse_model import InverseModelAttack

class discriminatorNet(nn.Module):
    # 包含decoder和攻击者的编码器tilde_f
    def __init__(self,input_dim = 16):
        super().__init__()

        self.nnlayers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16,8),
            nn.LeakyReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.nnlayers(x)

class FSHA_Attack(InverseModelAttack):
    def __init__(self,data_type = 0, gpu=True,
                 client_route = None,
                 shadow_route=None,
                   discriminator_route = None,
                   decoder_route=None,
                 inverse_dir=None) -> None:
        self.data_type = data_type # 0 是表格数据集，1是图像数据集
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

        self.inverse_dir = inverse_dir if inverse_dir else './inverted/'
        
        self.client_route = client_route if client_route else './client_net.pth'
        self.decoder_route = decoder_route if decoder_route else './decoder_net.pth'
        self.shadow_route = shadow_route if shadow_route else './shadow_net.pth'
        self.discriminator_route = discriminator_route if discriminator_route else './discriminator_net.pth'
        if not os.path.exists(self.inverse_dir):
            os.makedirs(self.inverse_dir)


    # 训练decoder和攻击者的编码器tilde_f
    # 训练鉴别器discriminator
    # 训练client端的被攻击网络f
    def train_decoder(self, client_net, decoder_net, discriminator_net,shadow_net,
                      private_loader, public_loader,
                      epochs,
                      propertyInfer = False):

        # 打印相关信息
        print("----train fsha----")
        print("client_net: ")
        print(client_net)
        print('shadow_net: ')
        print(shadow_net)
        print("decoder_net: ")
        print(decoder_net)
        print("discriminator_net: ")
        print(discriminator_net)

        # 网络搬到设备上
        client_net.to(self.device)
        shadow_net.to(self.device)
        decoder_net.to(self.device)
        discriminator_net.to(self.device)

        # 确定学习率和优化器
        # decoder_optimizer = optim.Adam(params=decoder_net.parameters(), lr=learningRate, eps=eps, amsgrad=True)
        # shadow_optimizer = optim.Adam(params=shadow_net.parameters(), lr=learningRate * 10, eps=eps, amsgrad=True)
        # client_optimizer = optim.Adam(params=client_net.parameters(), lr=learningRate * 10, eps=eps, amsgrad=True)
        # discriminator_optimizer = optim.Adam(params=discriminator_net.parameters(), lr=learningRate * 10, eps=eps, amsgrad=True) 
        decoder_optimizer = optim.Adam(params=decoder_net.parameters())
        shadow_optimizer = optim.Adam(params=shadow_net.parameters())
        client_optimizer = optim.Adam(params=client_net.parameters())
        discriminator_optimizer = optim.Adam(params=discriminator_net.parameters()) 

        # 损失函数
        if propertyInfer:
            property_criterion = nn.BCELoss()
            property_sigmoid = nn.Sigmoid()
        else:
            MSELossLayer = nn.MSELoss()
        criterion = nn.BCELoss()
        sigmoid = nn.Sigmoid()
        
        # 训练
        for epoch in range(epochs):
            print("Epoch {}".format(epoch))

            for i,(private_data,public_data) in enumerate(tqdm(zip(private_loader,public_loader))):
                # print(len(public_data))
                # print(type(private_data))
                x_public, label_public = public_data
                x_private, label_private = private_data

                x_public, label_public = x_public.to(self.device), label_public.to(self.device)
                x_private, label_private = x_private.to(self.device), label_private.to(self.device)

                # 清除梯度
                decoder_optimizer.zero_grad() 
                shadow_optimizer.zero_grad() 
                client_optimizer.zero_grad() 
                discriminator_optimizer.zero_grad() 

                # ------1.鉴别器和edge端网络---------
                # 这个是他原始的网络训练（最初功能的训练）
                # 用z代表中间层
                z_private = client_net(x_private)
                # print(z_private.shape)
                adv_private_logits = discriminator_net(z_private)

                # 鉴别器真实数据是记作0
                # 这里的训练目标就是要将edge和伪造edge端网络趋同！！
                # 为什么还要有edge loss？
                edge_loss = criterion(sigmoid(adv_private_logits), torch.ones_like(adv_private_logits)) # 区分器
                edge_loss.backward(retain_graph=True)

                # ------2.fsha攻击者网络------
                # 自动编码器训练
                z_public = shadow_net(x_public) # tilde_f 输出
                infer_x_public = decoder_net(z_public)  # 从切割层往后推理？

                if propertyInfer:
                    # 这里只取第一列数据
                    fsha_loss = property_criterion(property_sigmoid(infer_x_public),x_public[:, 0:1])
                else:
                    fsha_loss = MSELossLayer(infer_x_public, x_public) # decoder 功能，这两个网络的损失
                fsha_loss.backward(retain_graph=True)

                # ------3.鉴别器和fsha伪造edge端网络---------
                # 这个是把f训练成tilde_f
                adv_public_logits = discriminator_net(z_public)
                loss_discr_true = criterion(sigmoid(adv_public_logits), torch.ones_like(adv_public_logits))
                # loss_discr_true.backward()
                loss_discr_fake = criterion(sigmoid(adv_private_logits), torch.zeros_like(adv_private_logits))
                # loss_discr_fake.backward()

                # 计算训练损失
                D_loss = (loss_discr_true + loss_discr_fake) / 2 # true和fake的相似度尽量高
                D_loss.backward(retain_graph=True)

                # 更新模型
                decoder_optimizer.step() 
                shadow_optimizer.step() 
                client_optimizer.step() 
                discriminator_optimizer.step() 

            # 打印loss
            print("Epoch ", epoch, "edge Loss: ", edge_loss.cpu().detach().numpy(), "fsha Loss: ",
                fsha_loss.cpu().detach().numpy(), "D Loss: ", D_loss.cpu().detach().numpy())
            # print("--- epoch: {0}, train_loss: {1}".format(epoch, epoch_loss))


        # 储存攻击方模型参数
        torch.save(client_net, self.client_route)
        torch.save(shadow_net, self.shadow_route)
        torch.save(decoder_net, self.decoder_route)
        torch.save(discriminator_net, self.discriminator_route)
        print("model saved")

        # 返回训练好的decoder
        return client_net, shadow_net, decoder_net

