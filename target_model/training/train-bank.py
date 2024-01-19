from utils.preprocess_bank import *
from models.net import *
from models.BankNet import *
from utils.utils import evalTest_bank, setLearningRate

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
import os
import numpy as np
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能是由于是MacOS系统的原因

from torch.utils.data import Dataset
from utils.datasets import bank_dataset
from torch.utils.tensorboard import SummaryWriter
from utils.utils import accuracy_bank



def train(DATASET='CIFAR10', network='VGG5', NEpochs=200,
          BatchSize=32, learningRate=1e-3, NDecreaseLR=20, eps=1e-3,
           model_dir="", model_name="", gpu=True):

    dataPath = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/bank/bank-additional-full.csv'

    train_data, test_data = preprocess_bank(dataPath)


    # dataloader
    train_dataset = bank_dataset(train_data)
    test_dataset = bank_dataset(test_data)

    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True,
                                              num_workers=8, drop_last=False)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=BatchSize, shuffle=False,
                                             num_workers=8, drop_last=False)

    net = BankNet(input_dim=63, output_dim=1)
    # 打印数据集和网络信息
    print("edge net: \n", net)

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
    NBatch = len(train_dataset)
    # NBatch = len(X_train) / BatchSize

    cudnn.benchmark = True

    # 迭代训练
    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        for i, (batchX, batchY) in enumerate(train_dataset):

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

            lossTrain += loss.cpu().detach().numpy() / NBatch
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

        accTest = evalTest_bank(test_dataset, net, gpu=gpu)  # 测试一下模型精度
        print("Test accuracy: ", accTest)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(net, model_dir + model_name)
    print("Model saved")

    # 读取（load）模型
    newNet = torch.load(model_dir + model_name)
    newNet.eval()
    accTest = evalTest_bank(test_dataset, newNet, gpu=gpu)  # 测试模型精度

    print("test Acc:", accTest)
    print("Model restore done")


if __name__ == '__main__':

    import argparse
    import sys
    import traceback

    # 参数解析
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='CIFAR10')
        parser.add_argument('--network', type=str, default='VGG5')
        parser.add_argument('--epochs', type=int, default=0)
        parser.add_argument('--eps', type=float, default=1e-3)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--decrease_LR', type=int, default=10)

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)
        args = parser.parse_args()

        # 训练好的模型存储的dir
        model_dir = "results/bank/"
        model_name = "bank-1ep.pth"

        if args.dataset == 'CIFAR10':  # cifar10 数据集
            imageWidth = 32
            imageHeight = 32
            imageSize = imageWidth * imageHeight
            NChannels = 3
            NClasses = 10
        else:
            print("No Dataset Found")
            exit(0)

        # 待inverse的模型训练
        train(DATASET=args.dataset,
              network=args.network,
              NEpochs=args.epochs,
              BatchSize=args.batch_size, learningRate=args.learning_rate, NDecreaseLR=args.decrease_LR, eps=args.eps,
              model_dir=model_dir,
              model_name=model_name,
              gpu=args.gpu)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


# run : python training.py
# nohup python my.py >> nohup.out 2>&1 &

