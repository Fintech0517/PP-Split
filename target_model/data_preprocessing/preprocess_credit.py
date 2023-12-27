'''
Author: yjr 949804347@qq.com
Date: 2023-09-09 20:35:31
LastEditors: Ruijun Deng
LastEditTime: 2023-12-20 14:31:20
FilePath: /PP-Split/target_model/data_preprocessing/preprocess_credit.py
Description: none
'''
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

dataPath = 'dataset/bank-additional-full.csv'


def to_onehot(df, col_features):
    # 对类别型特征进行one-hot编码,并返回离散特征的索引
    onehot_df = pd.get_dummies(df[col_features])
    onehot_features = onehot_df.columns.values
    discrete_index = {s: [i for i in range(len(onehot_features)) if s in onehot_features[i]] for s in col_features}

    return onehot_df, discrete_index


def preprocess_credit(dataPath):
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
    scaler = MinMaxScaler(feature_range=(-1, 1))
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

    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)
    # print("X_onehot_index:", X_index)

    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))

    print("===============processing data end===============")

    return [X_train, y_train], [X_test, y_test]


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