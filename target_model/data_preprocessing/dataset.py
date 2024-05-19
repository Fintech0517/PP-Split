'''
Author: Ruijun Deng
Date: 2024-01-02 19:39:41
LastEditTime: 2024-05-02 22:20:39
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/data_preprocessing/dataset.py
Description: 
'''
# 导包
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset,ConcatDataset,Subset,TensorDataset
import numpy as np
import torch
import random



# 要多少对训练数据啊？（先拿所有吧）
# def pair_smashed_data(trainset1,trainset2,num_pairs,batch_size=1):
#     datasetC = ConcatDataset([trainset1,trainset2])
#     if num_pairs:
#         datasetC = Subset(datasetC,range(num_pairs))
#     dataloader = DataLoader(datasetC,batch_size=batch_size,shuffle=False,num_workers=4)
#     return dataloader


def pair_smashed_data(trainloader1,trainloader2,num_pairs,batch_size=1):
    # 构造数据集，每个pair是[看过的，没看过的]
    data = [[seen[0],unseen[0]] for seen,unseen in zip(trainloader1,trainloader2)]
    train_data = data[:num_pairs]
    test_data = data[num_pairs:2*num_pairs]

    train_labels = []
    # 此时开始shuffle每个pair的位置并且获取labels
    for d in train_data:
        true_feature = d[0]
        random.shuffle(d)
        train_labels.append([s.equal(true_feature) for s in d])
    
    test_labels = []
    for d in test_data:
        true_feature = d[0]
        random.shuffle(d)
        test_labels.append([s.equal(true_feature) for s in d])

    # 转为可用的dataset
    trainset = [item for sublist in train_data for item in sublist]
    testset = [item for sublist in test_data for item in sublist]

    trainset = TensorDataset(*trainset)
    testset = TensorDataset(*testset)

    train_loader = DataLoader(*trainset,shuffle=False,batch_size=batch_size)
    test_loader = DataLoader(*testset,shuffle=False,batch_size=batch_size)
    # datasetC = ConcatDataset([train_loader1.dataset,train_loader2.dataset])
    # if num_pairs:
    #     datasetC = Subset(datasetC,range(num_pairs))
    # dataloader = DataLoader(datasetC,batch_size=batch_size,shuffle=False,num_workers=4)
    return train_loader,train_labels,test_loader,test_labels


# def diff_pair_data(trainset1,trainset2):
#     tensor_A = torch.stack([trainset1[i][0] for i in range(num_pairs)])
#     tensor_B = torch.stack([trainset2[i][0] for i in range(num_pairs)])
#     # 计算每个元素的差值
#     diff = tensor_A - tensor_B

#     # 创建数据集 
#     dataset_D = TensorDataset(diff)

#     array_D = np.array(dataset_D[i][0].numpy() for i in range(num_pairs))

#     return array_D

def diff_pair_data(smashed_data):
    # smashed_data = np.array(smashed_data)
    # smashed_data = torch.stack(smashed_data).squeeze()
    relative_hidden_states = smashed_data[::2] - smashed_data[1::2]

    return relative_hidden_states


# 构造数据集 CIFAR10适用
class ListDataset(Dataset):
    def __init__(self, data_list) -> None:
        super().__init__()
        self.x = [t[0] for t in data_list]
        self.y = [t[1] for t in data_list]

    def __getitem__(self, index):  # 张量压缩一个维度
        x = torch.squeeze(self.x[index], 0)
        y = torch.squeeze(self.y[index], 0)
        return x, y

    def __len__(self):
        return len(self.x)



def get_one_data(dataloader,batch_size = 1): # 得到一个dataloader中第一个数据构造的dataloader
    # trainloader, testloader = get_cifar10_normalize(batch_size=batch_size)
    testIter = iter(dataloader)

    # first = testIter.next()
    # inverse_data_list = [(first[0],first[1])]
    inverse_data_list = []
    for i in range(batch_size):
        first = testIter.next()
        inverse_data_list.append((first[0],first[1]))

    dataset = ListDataset(inverse_data_list)
    inverseloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 1)
    return inverseloader



# bank数据集
class bank_dataset(Dataset):
    def __init__(self, data):
        # data = data
        self.Xa, self.y = data

    def __getitem__(self, item):
        Xa = self.Xa[item]
        y = self.y[item]

        return np.float32(Xa), np.float32(y)

    def __len__(self):
        return len(self.Xa)
    

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


