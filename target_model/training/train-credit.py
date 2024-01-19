import sys
sys.path.append('../')
from utils.preprocess_credit import *
from models.net import *
from models.CreditNet import *
from utils.utils import evalTest_credit, setLearningRate

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import softmax
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch
import time
import math
import tqdm
import os
import numpy as np
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能是由于是MacOS系统的原因

from torch.utils.data import Dataset

from utils.datasets import bank_dataset
# from torch.utils.tensorboard import SummaryWriter
from utils.utils import accuracy_bank


def train(DATASET='CIFAR10', network='VGG5', NEpochs=200, 
          BatchSize=32, learningRate=1e-3, NDecreaseLR=20, eps=1e-3,
          model_dir="", model_name="", gpu=True):

    dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/home_credit/dataset/application_train.csv'

    train_data, test_data = preprocess_credit(dataPath)

    X_train, y_train = train_data
    # X_test, y_test = test_data

    # dataloader
    train_dataset = bank_dataset(train_data)
    test_dataset = bank_dataset(test_data)

    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True,
                                              num_workers=8, drop_last=False)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=BatchSize, shuffle=False,
                                             num_workers=8, drop_last=False)

    net = CreditNet(input_dim=250, output_dim=1)
    # 打印数据集和网络信息
    # print("edge net: \n", net)

    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    # GPU配置
    if gpu:
        net.cuda()
        criterion.cuda()

    # 优化器配置
    # optimizer = optim.SGD(params = net.parameters(), lr = learningRate, momentum=0.9)
    optimizer = optim.Adam(params=net.parameters(),
                           lr=learningRate, eps=eps, amsgrad=True)

    # batch配置
    NBatch = len(X_train) / BatchSize

    cudnn.benchmark = True

    # trainIter = iter(trainloader)
    # 迭代训练
    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        for i, (batchX, batchY) in enumerate(tqdm.tqdm(train_dataset)):

            if gpu:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            optimizer.zero_grad()
            logits = net(batchX)
            # prob = softmax(logits)
            prob = logits
            loss = criterion(logits, batchY)
            loss.backward()
            optimizer.step()

            # 防止loss nan所以 / 10000
            lossTrain += loss.cpu().detach().numpy() / NBatch \
            # print(lossTrain)
            pred = torch.sigmoid(prob)
            pred = pred.cpu().detach().numpy()
            groundTruth = batchY.cpu().detach().numpy()


            acc = accuracy_bank(y_targets=groundTruth, y_prob_preds=pred)

            accTrain += acc / NBatch

        if (epoch + 1) % NDecreaseLR == 0:
            # learningRate = learningRate / 2.0
            learningRate = learningRate * 0.1
            setLearningRate(optimizer, learningRate)

        print("Epoch: ", epoch, "Loss: ", lossTrain,
              "Train accuracy: ", accTrain)
        

        acc_test, auc_score = evalTest_credit(test_dataset, net, gpu=gpu)  # 测试一下模型精度
        print("Test accuracy: ", acc_test)
        print("Test auc: ",auc_score)

    # 储存训练模型文件夹：“checkpoints/CIFAR10”
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(net, model_dir + model_name)
    print("Model saved")

    # 读取（load）模型
    newNet = torch.load(model_dir + model_name)
    newNet.eval()
    acc_test, auc_score = evalTest_credit(test_dataset, net, gpu=gpu)  # 测试模型精度

    print("Test accuracy: ", acc_test)
    print("Test auc: ", auc_score)


if __name__ == '__main__':

    import argparse
    import sys
    import traceback

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--network', type=str, default='VGG5')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--decrease_LR', type=int, default=10)
    parser.add_argument('--gpu', type=bool, default=True)
    args = parser.parse_args()

    # 训练好的模型存储的dir
    # model_dir = "../results/credit/"
    model_dir = "../trained_models/credit/"
    model_name = f"credit-{args.epochs}ep.pth"

    # 待inverse的模型训练
    train(DATASET=args.dataset,
            network=args.network,
            NEpochs=args.epochs,
            BatchSize=args.batch_size, learningRate=args.learning_rate, NDecreaseLR=args.decrease_LR, eps=args.eps,
            model_dir=model_dir,
            model_name=model_name,
            gpu=args.gpu)




# run : python training.py
# nohup python -u train-credit.py >> train_credt-20ep.out 2>&1 &
# 20ep: 9[1] 9624
