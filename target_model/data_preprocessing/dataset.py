'''
Author: Ruijun Deng
Date: 2024-01-02 19:39:41
LastEditTime: 2024-01-02 19:40:44
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/data_preprocessing/dataset.py
Description: 
'''
# 导包
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

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
    



