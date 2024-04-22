'''
Author: Ruijun Deng
Date: 2024-01-02 19:39:41
LastEditTime: 2024-04-22 10:41:51
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/data_preprocessing/dataset.py
Description: 
'''
# 导包
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

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


