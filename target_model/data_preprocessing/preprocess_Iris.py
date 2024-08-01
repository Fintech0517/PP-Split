'''
Author: yjr 949804347@qq.com
Date: 2023-09-09 20:35:31
LastEditors: Ruijun Deng
LastEditTime: 2024-07-22 20:56:38
FilePath: /PP-Split/target_model/data_preprocessing/preprocess_Iris.py
Description: none
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import Dataset
from .dataset import bank_dataset
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


dataPath = '/home/dengruijun/data/project/data/iris/Iris.csv'

num_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
y_columns = ['Species']

tabinfo_Iris = {
    'numList': [i for i in range(4)] # 后面是数值列
}

def preprocess_Iris_dataset(dataPath):
    print("===============processing data===============")

    df1 = pd.read_csv(dataPath)
    # df1.head()

    # label encoding the attributes of the target clumn
    df1['Species'] = df1['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
    df1.drop(['Id'],axis=1,inplace=True)
    X = df1.drop(["Species"],axis=1).values
    y = df1["Species"].values
    # y = df1["Species"].values - 1

    df = pd.read_csv(dataPath, delimiter=';',quotechar='"') # 读取文件生成df
    # df.head()
    # df.info()

    # scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test) # TODO:但其实确实应该是 transform而不是fit_transform，难怪要用标准差？

    # -----------------------划分训练集和测试集-------------------------
    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)
    # print("X_onehot_index:", X_index)

    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    print("===============processing data end===============")

    return [X_train, y_train], [X_test, y_test]

def preprocess_Iris(batch_size = 1):
    train_data, test_data = preprocess_Iris_dataset(dataPath)
    # print("test_data:", test_data[0])
    
    # scalar = MinMaxScaler(feature_range=(0,1))
    # x = scalar.fit_transform(test_data[0])
    # print('x:', x)
    train_dataset = bank_dataset(train_data)
    test_dataset = bank_dataset(test_data)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=8, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=8, drop_last=True)

    return train_loader,test_loader


if __name__ == "__main__":
    dataPath = '/home/dengruijun/data/project/data/iris/Iris.csv'

    # df = pd.read_csv(dataPath, delimiter=';')

    # print(df.sample(10))
    # print(df.shape)
    # print(df.info())
    # print(df.describe())
    # print(df.head())
    # print(df.columns.values)
    [Xa_train, y_train], [Xa_test, y_test] = preprocess_Iris(dataPath)


    # print(Xa_train[10:])
    # print(y_train[10:])