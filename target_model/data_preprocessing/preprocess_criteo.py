'''
Author: Jirui Yang
Date: 2024-04-22 10:52:30
LastEditTime: 2024-04-22 10:55:59
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/data_preprocessing/preprocess_criteo.py
Description: 
'''
import pandas as pd
from .dataset import NumpyDataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def preprocess_criteo(data_path = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/criteo/mini_creditcard.csv',
                      batch_size = 1):
    # 数据集
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
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = NumpyDataset(
        test_features, test_labels.astype(np.float64).reshape(-1, 1)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader, test_loader