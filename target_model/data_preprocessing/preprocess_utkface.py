'''
Author: Ruijun Deng
Date: 2024-10-02 07:17:42
LastEditTime: 2024-10-24 05:29:28
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/data_preprocessing/preprocess_utkface.py
Description: 
'''
# 导包
import torch

# 导包
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.data import Subset, random_split

from .dataset import ListDataset, split_trainset

from PIL import Image
from glob import glob
from abc import abstractmethod


# 用0.5来normalize的
# def get_utkface_normalize(batch_size = 1, test_bs = None, mu = None, sigma=None):
#     if test_bs == None:
#         test_bs = batch_size
#     # 数据集 CIFAR
#     # 图像归一化
#     if mu == None:
#         mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
#     if sigma == None:
#         sigma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        
#     transform = transforms.Compose(
#         [
#             transforms.ToTensor(), # 数据中的像素值转换到0～1之间
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 接近+-1？ 从[0,1] 不是从[0,255]
#             # transforms.Normalize(mu.tolist(), sigma.tolist())
#         ])

#     # 数据集加载：
#     trainset = torchvision.datasets.CIFAR100(root='/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar100/', train=True,
#                                             download=False, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                             shuffle=False, num_workers=4)

#     testset = torchvision.datasets.CIFAR100(root='/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar100/', train=False,
#                                         download=False, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs,
#                                             shuffle=False, num_workers=4)
    
#     return trainloader,testloader



def get_utkface_normalize(batch_size = 1, test_bs = None, mu = None, sigma=None):
    if test_bs == None:
        test_bs = batch_size

        # UTKFace Dataset
    trainTransform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    dataset = UTKFace({"path": "/home/dengruijun/data/project/data/utkface/UTKFace_cropped/kaggle/UTKFace/",
                        "transforms": trainTransform,
                        "format": "jpg",
                        "attribute": "gender"})
    train_dataset, test_dataset = get_split(0.8, dataset)

    train_dataset = MyDataset(train_dataset)
    test_dataset = MyDataset(test_dataset)

    # Data Loader (Input Pipeline)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 分割数据集
def get_split(train_split, dataset):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    np.random.shuffle(indices)

    train_indices, test_indices = indices[:split], indices[split:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return train_dataset, test_dataset

class BaseDataset(Dataset):
    """docstring for BaseDataset"""

    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.format = config["format"]
        self.set_filepaths(config["path"])
        self.transforms = config["transforms"]

    def set_filepaths(self, path):
        filepaths = path + "/*.{}".format(self.format)
        self.filepaths = glob(filepaths)

    def load_image(self, filepath):
        img = Image.open(filepath)
        # img = np.array(img)
        return img

    @staticmethod
    def to_tensor(obj):
        return torch.tensor(obj)

    @abstractmethod
    def load_label(self):
        pass

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        img = self.load_image(filepath)
        img = self.transforms(img)
        label = self.load_label(filepath)
        label = self.to_tensor(label)
        return img, label

    def __len__(self):
        return len(self.filepaths)


class UTKFace(BaseDataset): # UTKFace提取数据集
    """docstring for UTKFace"""

    def __init__(self, config):
        super(UTKFace, self).__init__(config)
        self.attribute = config["attribute"]

    def load_label(self, filepath):
        labels = filepath.split("/")[-1].split("_")
        if self.attribute == "race":
            try:
                label = int(labels[2])
            except:
                print("corrupt label")
                label = np.random.randint(0, 4)
        elif self.attribute == "gender":
            label = int(labels[1])
        elif self.attribute == "age": # 分成4类了？
            # label = float(labels[0])
            if int(labels[0]) < 15:
                label = 0
            elif int(labels[0]) < 30:
                label = 1
            elif int(labels[0]) < 50:
                label = 2
            elif int(labels[0]) < 70:
                label = 3
            else:
                label = 4
            #label = float(label)
        return label


class MyDataset(Dataset):
    def __init__(self, trainset):
        self.set = trainset

    def __getitem__(self, index):
        data, target = self.set[index]
        return data, target, index

    def __len__(self):
        return len(self.set)