import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
from ppsplit.fedml_core.trainer.splitLearning import SplitNN, SplitNNClient, ISO_SplitNN, MAX_NORM_SplitNN, Marvell_SplitNN
import torch.optim as optim

class NormDirect_Attack:
    def __init__(self) -> None:
        pass
    
    def norm_attack(self,splitnn, dataloader, attack_criterion, device="cpu", marvell=False):
        epoch_labels = []
        epoch_g_norm = []
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            for opt in splitnn.optimizers:
                # 第一次训练需要清空梯度
                opt.zero_grad()

            # 输出第一个模型的结果
            outputs = splitnn(inputs)
            loss = attack_criterion(outputs, labels)

            if marvell:
                splitnn.backward(loss, labels)
            else:
            # iso和max_norm使用
                splitnn.backward(loss)
            # 对应marvell使用下面
            # splitnn.backward(loss, labels)

            grad_from_server = splitnn.clients[0].grad_from_next_client
            # 注pow(2)是对每个元素平方
            # sum是元素相加,dim表示横向相加,也就是128个16维梯度加为128个单维
            # sqrt开平方，综上表示取模
            g_norm = grad_from_server.pow(2).sum(dim=1).sqrt()
            epoch_labels.append(labels)
            epoch_g_norm.append(g_norm)

        epoch_labels = torch.cat(epoch_labels)
        epoch_g_norm = torch.cat(epoch_g_norm)

        # score = roc_auc_score(epoch_labels.cpu(), epoch_g_norm.cpu().view(-1, 1))
        
        # 假设 epoch_labels 是你的标签数据
        unique_classes = np.unique(epoch_labels.cpu())
        # print(unique_classes)
        if len(unique_classes) > 1:
            score = roc_auc_score(epoch_labels.cpu(), epoch_g_norm.cpu().view(-1, 1))
        else:
            # print("只有一个类别存在，无法计算 AUC")
            score = 0.5  # 或者你可以根据情况设定一个默认值
        
        return score


    def direction_attack(self, splitnn, dataloader, attack_criterion, device="cpu", marvell=False):
        epoch_labels = []
        epoch_g_direc = []
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            for opt in splitnn.optimizers:
                # 第一次训练需要清空梯度
                opt.zero_grad()

            # 输出第一个模型的结果
            outputs = splitnn(inputs)
            # print(outputs.shape)
            loss = attack_criterion(outputs, labels)

            if marvell:
                splitnn.backward(loss, labels)
            else:
            # iso和max_norm使用
                splitnn.backward(loss)
            # 对应marvell使用下面
            # splitnn.backward(loss, labels)

            grad_from_server = splitnn.clients[0].grad_from_next_client

            g_direc = torch.split(grad_from_server, 1, 0)

            # 余弦相似度类
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)

            if labels[0].item() > 0.5:
                sig_g_direc = torch.ones([1])
                for i in range(1, len(g_direc)):
                    if cos(g_direc[0], g_direc[i]).item() >= 0:
                        sig_g_direc = torch.cat((sig_g_direc, torch.ones([1])), 0)
                    else:
                        sig_g_direc = torch.cat((sig_g_direc, torch.zeros([1])), 0)
            else:
                sig_g_direc = torch.zeros([1])
                for i in range(1, len(g_direc)):
                    if cos(g_direc[0], g_direc[i]).item() < 0:
                        sig_g_direc = torch.cat((sig_g_direc, torch.ones([1])), 0)
                    else:
                        sig_g_direc = torch.cat((sig_g_direc, torch.zeros([1])), 0)

            epoch_labels.append(labels)
            epoch_g_direc.append(sig_g_direc)

        epoch_labels = torch.cat(epoch_labels)
        epoch_g_direc = torch.cat(epoch_g_direc)

        # score = roc_auc_score(epoch_labels.cpu(), epoch_g_direc.cpu().view(-1, 1))
        
        unique_classes = np.unique(epoch_labels.cpu())
        if len(unique_classes) > 1:
            score = roc_auc_score(epoch_labels.cpu(), epoch_g_direc.cpu().view(-1, 1))
        else:
            # print("只有一个类别存在，无法计算 AUC")
            score = 0.5  # 或者你可以根据情况设定一个默认值
        return score

    def train_and_attack(self, protect_method, 
                         first_net, seconde_net,
                         train_loader,test_loader,
                          device, t = None):
        # 模型
        # 第一层输入貌似只有28个维度 线性层输入后有16个隐藏层
        # 使用relu激活函数
        model_1 = first_net
        # 将模型指定到对应的硬件上
        model_1 = model_1.to(device)
        # 第二层输入是16个维度，输出是1个维度
        # 使用sigmoid激活函数
        model_2 = seconde_net
        model_2 = model_2.to(device)

        # 将所有的浮点类型的参数和缓冲转换为(双浮点)double数据类型.
        model_1.double()
        model_2.double()

        # 用神经网络 对模型的参数 初始化优化器 lr是学习速率
        opt_1 = optim.Adam(model_1.parameters(), lr=1e-3)
        opt_2 = optim.Adam(model_2.parameters(), lr=1e-3)
        optimizers = [opt_1, opt_2]

        # 损失函数 二元交叉熵并且求平均
        criterion = nn.BCELoss()
        client_1 = SplitNNClient(model_1, user_id=0).to(device)
        client_2 = SplitNNClient(model_2, user_id=0).to(device)

        clients = [client_1, client_2]

        if protect_method == "ISO":
            splitnn = ISO_SplitNN(clients, optimizers, t=t)    
        elif protect_method == "MAX_NORM":
            splitnn = MAX_NORM_SplitNN(clients, optimizers)    
        elif protect_method == "Marvell":
            splitnn = Marvell_SplitNN(clients, optimizers)
        else:
            splitnn = SplitNN(clients, optimizers)
        
        

        # -------------------------------------------------
        # 至此搭建模型完成，开始训练
        # -------------------------------------------------

        splitnn.train()
        for epoch in range(128):
            epoch_loss = 0
            epoch_outputs = []
            epoch_labels = []
            for i, data in enumerate(train_loader):
                for opt in optimizers:
                    # 第一次训练需要清空梯度
                    opt.zero_grad()

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = splitnn(inputs)
                loss = criterion(outputs, labels)
                # loss.backward()

                # iso和norm_max保护方法使用：
                # splitnn.backward(loss)
                # marvell算法需要传入标签，所以backward修改如下：
                if protect_method == "Marvell":
                    splitnn.backward(loss, labels)
                else:
                    splitnn.backward(loss)

                epoch_loss += loss.item() / len(train_loader.dataset)

                epoch_outputs.append(outputs.cpu().detach())
                epoch_labels.append(labels.cpu().detach())

                # 更新对两个模型参数
                for opt in optimizers:
                    opt.step()

            print(
                f"epoch={epoch}, loss: {epoch_loss}, auc: {roc_auc_score(torch.cat(epoch_labels), torch.cat(epoch_outputs))}"
            )

        # 攻击1：模攻击
        if protect_method == "Marvell":
            train_leak_auc = self.norm_attack(
                splitnn, train_loader, attack_criterion=nn.BCELoss(), device=device, marvell=True)
            print("norm_attack: train_leak_auc is ", train_leak_auc)
            test_leak_auc = self.norm_attack(
                splitnn, test_loader, attack_criterion=nn.BCELoss(), device=device, marvell=True)
            print("norm_attack: test_leak_auc is ", test_leak_auc)
        else:
            train_leak_auc = self.norm_attack(
                splitnn, train_loader, attack_criterion=nn.BCELoss(), device=device)
            print("norm_attack: train_leak_auc is ", train_leak_auc)
            test_leak_auc = self.norm_attack(
                splitnn, test_loader, attack_criterion=nn.BCELoss(), device=device)
            print("norm_attack: test_leak_auc is ", test_leak_auc)

        # 攻击2：余弦攻击
        if protect_method == "Marvell":
            train_leak_auc = self.direction_attack(
                splitnn, train_loader, attack_criterion=nn.BCELoss(), device=device, marvell=True)
            print("direction_attack: train_leak_auc is ", train_leak_auc)
            test_leak_auc = self.direction_attack(
                splitnn, test_loader, attack_criterion=nn.BCELoss(), device=device, marvell=True)
            print("direction_attack: test_leak_auc is ", test_leak_auc)
        else:
            train_leak_auc = self.direction_attack(
                splitnn, train_loader, attack_criterion=nn.BCELoss(), device=device)
            print("direction_attack: train_leak_auc is ", train_leak_auc)
            test_leak_auc = self.direction_attack(
                splitnn, test_loader, attack_criterion=nn.BCELoss(), device=device)
            print("direction_attack: test_leak_auc is ", test_leak_auc)