'''
Author: yjr 949804347@qq.com
Date: 2023-09-09 20:35:31
LastEditors: Ruijun Deng
LastEditTime: 2024-08-01 21:47:10
FilePath: /PP-Split/target_model/data_preprocessing/preprocess_credit.py
Description: none
'''
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from .dataset import bank_dataset
import torch

dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/home_credit/dataset/application_train.csv'

tabinfo_credit = {
    'onehot': {'NAME_CONTRACT_TYPE': [0, 1], 'CODE_GENDER': [2, 3, 4], 'FLAG_OWN_CAR': [5, 6], 'FLAG_OWN_REALTY': [7, 8],
            'NAME_TYPE_SUITE': [9, 10, 11, 12, 13, 14, 15, 16], 'NAME_INCOME_TYPE': [17, 18, 19, 20, 21, 22, 23, 24],
            'NAME_EDUCATION_TYPE': [25, 26, 27, 28, 29], 'NAME_FAMILY_STATUS': [30, 31, 32, 33, 34, 35],
            'NAME_HOUSING_TYPE': [36, 37, 38, 39, 40, 41],
            'OCCUPATION_TYPE': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            'WEEKDAY_APPR_PROCESS_START': [61, 62, 63, 64, 65, 66, 67],
            'ORGANIZATION_TYPE': [68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
            'FONDKAPREMONT_MODE': [126, 127, 128, 129, 130], 'HOUSETYPE_MODE': [131, 132, 133, 134],
            'WALLSMATERIAL_MODE': [135, 136, 137, 138, 139, 140, 141, 142], 'EMERGENCYSTATE_MODE': [143, 144, 145]},
    'numList': [i for i in range(146, 250)]
}

def to_onehot(df, col_features):
    # 对类别型特征进行one-hot编码,并返回离散特征的索引
    onehot_df = pd.get_dummies(df[col_features])
    onehot_features = onehot_df.columns.values
    discrete_index = {s: [i for i in range(len(onehot_features)) if s in onehot_features[i]] for s in col_features}

    return onehot_df, discrete_index


def preprocess_credit_dataset(dataPath):
    print("===============processing data===============")

    df = pd.read_csv(dataPath)
    # print(df.head())
    # print(df.info())
    # # df.head().to_csv("./head.csv")
    # # df.info().to_csv
    #
    # print(df.columns)

    cate_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # 打印分类和连续特征
    # print("Categorical Features:")
    # print(categorical_features)
    # print(len(categorical_features))
    #
    # print("\nNumeric Features:")
    # print(numeric_features)
    # print(len(numeric_features))

    numeric_features = [f for f in numeric_features if f not in ['SK_ID_CURR', 'TARGET']]

    df = df.fillna(value=-1)

    # sys.exit(0)

    # 处理分类特征
    # cate_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
    #              'poutcome']

    # df['y'] = (df['y'] == 'yes').astype(int)

    df_object_col = cate_cols
    df_num_col = numeric_features
    target = df['TARGET']

    # 连续列缩放到[-1,1]之间
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[df_num_col] = scaler.fit_transform(df[df_num_col])

    # -----------------------划分训练集和测试集-------------------------
    X, X_index = to_onehot(df, df_object_col)
    # print(X_index)
    X = pd.concat([X, df[df_num_col]], axis=1).values


    y = target.values
    y = np.expand_dims(y, axis=1)

    n_train = int(0.8 * X.shape[0])

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]


    pd.DataFrame(X_test).to_csv('raw-X.csv', index = False)

    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)
    # print("X_onehot_index:", X_index)

    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))

    print("===============processing data end===============")

    return [X_train, y_train], [X_test, y_test]

def preprocess_credit(batch_size = 1, test_bs=None):
    if not test_bs:
        test_bs = batch_size

    train_data,test_data = preprocess_credit_dataset(dataPath)
    train_dataset = bank_dataset(train_data)
    test_dataset = bank_dataset(test_data)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=8, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=False,
                                               num_workers=8, drop_last=True)
    return train_loader, test_loader 



if __name__ == "__main__":
    dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/home_credit/dataset/application_train.csv'

    # df = pd.read_csv(dataPath, delimiter=';')

    # print(df.sample(10))
    # print(df.shape)
    # print(df.info())
    # print(df.describe())
    # print(df.head())
    # print(df.columns.values)


    [Xa_train, y_train], [Xa_test, y_test] = preprocess_credit(dataPath)


    # print(Xa_train[10:])
    # print(y_train[10:])