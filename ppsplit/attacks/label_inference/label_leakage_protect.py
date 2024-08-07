# 一个是加上了防御的
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from ppsplit.fedml_core.trainer.splitLearning import SplitNN, SplitNNClient, ISO_SplitNN, MAX_NORM_SplitNN, Marvell_SplitNN
# from splitLearning import SplitNN, SplitNNClient, ISO_SplitNN, MAX_NORM_SplitNN, Marvell_SplitNN
# from attack import norm_attack, direction_attack
from ppsplit.attacks.label_inference.attack import NormDirect_Attack

class NumpyDataset(Dataset):
    """This class allows you to convert numpy.array to torch.Dataset
    Args:
        x (np.array):
        y (np.array):
        transform (torch.transform):
    Attriutes
        x (np.array):
        y (np.array):
        transform (torch.transform):
    """

    def __init__(self, x, y=None, transform=None, return_idx=False):
        self.x = x
        self.y = y
        self.transform = transform
        self.return_idx = return_idx

    def __getitem__(self, index):
        x = self.x[index]
        if self.y is not None:
            y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        if not self.return_idx:
            if self.y is not None:
                return x, y
            else:
                return x
        else:
            if self.y is not None:
                return index, x, y
            else:
                return index, x

    def __len__(self):
        """get the number of rows of self.x"""
        return len(self.x)


args = {
    'batch_size': 256,
    'hidden_dim': 16, # 输入貌似只有28个维度 线性层输入后有16个隐藏层
}


class FirstNet(nn.Module):
    def __init__(self, train_features):
        super(FirstNet, self).__init__()
        self.L1 = nn.Linear(train_features.shape[-1], args['hidden_dim'])

    def forward(self, x):
        x = self.L1(x)
        x = nn.functional.relu(x)
        return x


class SecondNet(nn.Module):
    def __init__(self):
        super(SecondNet, self).__init__()
        self.L2 = nn.Linear(args['hidden_dim'], 1)

    def forward(self, x):
        x = self.L2(x)
        x = torch.sigmoid(x)
        return x




def main_protect(data_path="mini_creditcard.csv", protect_method=None, t=None):
    device = "cpu"
    # GPU代码没有改好，就cpu上跑吧
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is ", device)
    raw_df = pd.read_csv(data_path)
    # 将读入的数据分成两类
    raw_df_neg = raw_df[raw_df["Class"] == 0]
    raw_df_pos = raw_df[raw_df["Class"] == 1]

    down_df_neg = raw_df_neg  # .sample(40000)
    # 基于同一轴将多个数据集合并  为啥要拆开再合并呢？
    down_df = pd.concat([down_df_neg, raw_df_pos])

    neg, pos = np.bincount(down_df["Class"])
    total = neg + pos
    print(
        "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
            total, pos, 100 * pos / total
        )
    )

    cleaned_df = down_df.copy()
    # You don't want the `Time` column.
    cleaned_df.pop("Time")
    # The `Amount` column covers a huge range. Convert to log-space.
    # “金额”一栏涵盖的范围很广。转换为log空间。
    eps = 0.001  # 0 => 0.1¢
    # 注:在pop删除的时候同时放回该值
    # 但是这个加 eps是为什么？
    cleaned_df["Log Ammount"] = np.log(cleaned_df.pop("Amount") + eps)

    # Use a utility from sklearn to split and shuffle our dataset.
    # 使用sklearn中的实用程序分割和shuffle数据集。

    train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    # 划分后 train_df 0.64||val_df 0.16|| test_df 0.2

    # Form np arrays of labels and features.
    # 获取对应的标签
    train_labels = np.array(train_df.pop("Class"))
    val_labels = np.array(val_df.pop("Class"))
    test_labels = np.array(test_df.pop("Class"))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    # sklearn.preprocessing.StandardScaler是数据标准化的一个库，可以将现有的数据通过某种关系，映射到某一空间内
    # 常用的标准化方式是,减去平均值，然后通过标准差映射到均至为0的空间内
    # StandardScaler类是一个用来将数据进行归一化和标准化的类。
    scaler = StandardScaler()
    # 使得新的X数据集方差为1，均值为0

    train_features = scaler.fit_transform(train_features)
    # 根据训练集的数据，将剩下的数据进行标准化，所有的数据标准化是一致的
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    # np.clip是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值
    # 最大值直接定为5 ，最小值定为-5
    # 注：由于数据服从标准正态分布：3σ原理，>5或者<5的值都几乎不存在，
    train_features = np.clip(train_features, -5, 5)
    val_features = np.clip(val_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)

    print("Training labels shape:", train_labels.shape)
    print("Validation labels shape:", val_labels.shape)
    print("Test labels shape:", test_labels.shape)
    print("Training features shape:", train_features.shape)
    print("Validation features shape:", val_features.shape)
    print("Test features shape:", test_features.shape)

    # 向将 numpy的数据转化为torch的数据，传入dataloader
    # astype是进行数据装换的将标签变为float64
    # .reshape(-1, 1) -1表示未指定值，1表示一行一个，即转化为一列数据
    train_dataset = NumpyDataset(
        train_features, train_labels.astype(np.float64).reshape(-1, 1)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True
    )
    test_dataset = NumpyDataset(
        test_features, test_labels.astype(np.float64).reshape(-1, 1)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=True
    )

    # -------------------------------------------------
    # 至此完成数据加载，开始搭建模型
    # -------------------------------------------------

    # 第一层输入貌似只有28个维度 线性层输入后有16个隐藏层
    # 使用relu激活函数
    model_1 = FirstNet(train_features)
    # 将模型指定到对应的硬件上
    model_1 = model_1.to(device)
    # 第二层输入是16个维度，输出是1个维度
    # 使用sigmoid激活函数
    model_2 = SecondNet()
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
    client_1 = SplitNNClient(model_1, user_id=0)
    client_2 = SplitNNClient(model_2, user_id=0)

    clients = [client_1, client_2]

    # 不进行保护
    # splitnn = SplitNN(clients, optimizers)

    # 保护方法1 iso 高斯白噪声
    # t 表示高斯噪声的强度
    splitnn = ISO_SplitNN(clients, optimizers, t=0.005)

    # 保护方法2 max_norm
    # splitnn = MAX_NORM_SplitNN(clients, optimizers)

    # 保护方法3 Marvell
    # splitnn = Marvell_SplitNN(clients, optimizers)
    
    # if protect_method is not None:
    
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

            epoch_outputs.append(outputs)
            epoch_labels.append(labels)

            # 更新对两个模型参数
            for opt in optimizers:
                opt.step()

        print(
            f"epoch={epoch}, loss: {epoch_loss}, auc: {roc_auc_score(torch.cat(epoch_labels), torch.cat(epoch_outputs))}"
        )

    attack = NormDirect_Attack
    # 攻击1：模攻击
    if protect_method == "Marvell":
        train_leak_auc = attack.norm_attack(
            splitnn, train_loader, attack_criterion=nn.BCELoss(), device=device, marvell=True)
        print("norm_attack: train_leak_auc is ", train_leak_auc)
        test_leak_auc = attack.norm_attack(
            splitnn, test_loader, attack_criterion=nn.BCELoss(), device=device, marvell=True)
        print("norm_attack: test_leak_auc is ", test_leak_auc)
    else:
        train_leak_auc = attack.norm_attack(
            splitnn, train_loader, attack_criterion=nn.BCELoss(), device=device)
        print("norm_attack: train_leak_auc is ", train_leak_auc)
        test_leak_auc = attack.norm_attack(
            splitnn, test_loader, attack_criterion=nn.BCELoss(), device=device)
        print("norm_attack: test_leak_auc is ", test_leak_auc)

    # 攻击2：余弦攻击
    if protect_method == "Marvell":
        train_leak_auc = attack.direction_attack(
            splitnn, train_loader, attack_criterion=nn.BCELoss(), device=device, marvell=True)
        print("direction_attack: train_leak_auc is ", train_leak_auc)
        test_leak_auc = attack.direction_attack(
            splitnn, test_loader, attack_criterion=nn.BCELoss(), device=device, marvell=True)
        print("direction_attack: test_leak_auc is ", test_leak_auc)
    else:
        train_leak_auc = attack.direction_attack(
            splitnn, train_loader, attack_criterion=nn.BCELoss(), device=device)
        print("direction_attack: train_leak_auc is ", train_leak_auc)
        test_leak_auc = attack.direction_attack(
            splitnn, test_loader, attack_criterion=nn.BCELoss(), device=device)
        print("direction_attack: test_leak_auc is ", test_leak_auc)



if __name__ == "__main__":
    # writer = SummaryWriter("iso-protect")
    # main(writer)
    # writer.close()
    main_protect()
