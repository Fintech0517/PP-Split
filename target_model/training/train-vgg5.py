# @Author: zechenghe
# @Date:   2019-01-20T16:46:24-05:00
# @Last modified by:   zechenghe
# @Last modified time: 2019-02-01T14:01:19-05:00

import time
import math
import os
import numpy as np
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能是由于是MacOS系统的原因

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from target_model.models.VGG import VGG,model_cfg
from target_model.data_preprocessing.preprocess_cifar10 import get_cifar10_normalize,get_one_data,deprocess
from utils import evalTest 



# from utils.utils import *
# from torch.utils.tensorboard import SummaryWriter

def train(network = 'VGG5', NEpochs = 200, 
        BatchSize = 32, learningRate = 1e-3, NDecreaseLR = 20, 
        model_dir = "", gpu = True):
    
    # 储存训练模型文件夹：“checkpoints/CIFAR10”
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 数据集和模型
    trainloader,testloader = get_cifar10_normalize(batch_size=BatchSize)
    net = VGG('Unit',network,len(model_cfg[network])-1,model_cfg)

    # 数据集装入loader中
    criterion = nn.CrossEntropyLoss(reduction='mean') # 在batch上的平均了
    softmax = nn.Softmax(dim=1)
    # GPU配置
    if gpu:
        net.cuda()
        criterion.cuda()

    # 优化器配置
    # optimizer = optim.SGD(params = net.parameters(), lr = learningRate, momentum=0.9)
    optimizer = optim.Adam(params = net.parameters(), lr = learningRate)

    cudnn.benchmark = True

    NBatch = len(trainloader)
    # 迭代训练
    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        for i, (batchX, batchY) in enumerate(trainloader):
            if gpu:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            optimizer.zero_grad()
            logits = net(batchX)
            # prob = nn.Softmax(logits)
            prob = softmax(logits)
            # prob = logits
            loss = criterion(logits, batchY)
            loss.backward()
            optimizer.step()

            # 计算loss
            lossTrain += loss.cpu().detach().numpy() / NBatch

            # 计算accuracy
            if gpu: # 使用GPU的时候要迁移回cpu
                prob,batchY = prob.cpu(),batchY.cpu()
            pred = np.argmax(prob.detach().numpy(), axis = 1)
            groundTruth = batchY.detach().numpy()

            acc = np.mean(pred == groundTruth)
            accTrain += acc / NBatch

        if (epoch + 1)  % NDecreaseLR == 0: # 调整learning rate
            learningRate = learningRate * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learningRate
            # setLearningRate(optimizer, learningRate)

        print("Epoch: ", epoch, "Loss: ", lossTrain, "Train accuracy: ", accTrain)

        evalTest(testloader, net, gpu = gpu)  # 测试一下模型精度

        model_name = f'VGG5-params-{epoch}ep.pth'
        torch.save(net.state_dict(), model_dir + model_name)
        print(f"Model saved for epoch {epoch}")

    # 读取（load）模型
    # newNet = torch.load(model_dir + model_name)
    # newNet.eval()
    # evalTest(testloader, newNet, gpu = gpu)  # 测试模型精度
    # print("Model restore done")


if __name__ == '__main__':
    
    import argparse
    import traceback

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True

    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type = str, default = 'VGG5')
    parser.add_argument('--epochs', type = int, default = 20)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--decrease_LR', type = int, default = 10)
    parser.add_argument('--nogpu', dest='gpu', action='store_false')
    parser.set_defaults(gpu=True)
    args = parser.parse_args()

    # 训练好的模型存储的dir
    model_dir = "../../results/trained_models/VGG5/20240414/"
    # model_name = "VGG5-20ep.pth"

    # 待inverse的模型训练
    train(
    network = args.network, 
    NEpochs = args.epochs, 
    BatchSize = args.batch_size, learningRate = args.learning_rate, NDecreaseLR = args.decrease_LR,
    model_dir = model_dir,
    gpu = args.gpu)


# run : python training.py
# nohup python my.py >> nohup.out 2>&1 &
# 10 :[1] 3707176
# 20ep : [2] 3755640