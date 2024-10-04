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
from target_model.data_preprocessing.preprocess_cifar10 import get_cifar10_normalize,deprocess,get_cifar10_normalize_two_train
from target_model.data_preprocessing.preprocess_mnist import get_mnist_normalize
from utils import evalTest 

import wandb

def train(args):
    device = args['device']
    train_bs = args['batch_size']
    test_bs = args['batch_size']
    NEpochs = args['epochs']
    learningRate = args['learning_rate']
    model_dir = args['model_dir']
    # model_name = args['model_name']
    dataset = args['dataset']
    NDecreaseLR = args['decrease_LR']
    network = args['network']
    model_name = f"{network}-{dataset}-{NEpochs}epoch.pth"

    # print('NDecreaseLR:',NDecreaseLR)
    # print(args)

    # 可视化
    wandb.init(project='VGG',name=f'{network}-{dataset}')
    wandb.config.update(args)

    # 储存训练模型文件夹：“checkpoints/CIFAR10”
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 数据集和模型
    if dataset == 'CIFAR10':
        trainloader,testloader = get_cifar10_normalize(batch_size = train_bs, test_bs=test_bs)
    elif dataset == 'MNIST':
        trainloader,testloader = get_mnist_normalize(batch_size = train_bs, test_bs=test_bs)
    
    net = VGG('Unit',network,len(model_cfg[network])-1,model_cfg)

    # 数据集装入loader中
    criterion = nn.CrossEntropyLoss(reduction='mean') # 在batch上的平均了
    softmax = nn.Softmax(dim=1)

    # GPU配置
    net.to(device)
    criterion.to(device)

    # 优化器配置
    # optimizer = optim.SGD(params = net.parameters(), lr = learningRate, momentum=0.9)
    optimizer = optim.Adam(params = net.parameters(), lr = learningRate)

    cudnn.benchmark = True

    NBatch = len(trainloader)

    # 最初的模型精度
    acc_test = evalTest(testloader, net, device)  # 测试一下模型精度
    print("Test accuracy: ", acc_test)
    wandb.log({'Test Accuracy': acc_test})
    torch.save(net.state_dict(), model_dir + model_name)
    print(f"Model saved for epoch {0}")

    # 迭代训练
    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        for i, (batchX, batchY) in enumerate(trainloader):
            batchX = batchX.to(device)
            batchY = batchY.to(device)

            optimizer.zero_grad()
            logits = net(batchX)
            prob = softmax(logits)
            loss = criterion(logits, batchY)
            loss.backward()
            optimizer.step()

            # 计算loss
            lossTrain += loss.cpu().detach().numpy() / NBatch

            # 计算accuracy
            pred = np.argmax(prob.cpu().detach().numpy(), axis = 1)
            groundTruth = batchY.cpu().detach().numpy()

            acc = np.mean(pred == groundTruth)
            accTrain += acc / NBatch

        if (epoch + 1)  % NDecreaseLR == 0: # 调整learning rate
            learningRate = learningRate * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learningRate
            # setLearningRate(optimizer, learningRate)

        print("Epoch: ", epoch, "Loss: ", lossTrain, "Train accuracy: ", accTrain)
        acc_test = evalTest(testloader, net, device)  # 测试一下模型精度
        print("Test accuracy: ", acc_test)
        wandb.log({'Train Loss': lossTrain, 'Train Accuracy': accTrain, 'Test Accuracy': acc_test})

        # 储存模型
        torch.save(net.state_dict(), model_dir + model_name + f"-{epoch+1}.pth")
        print(f"Model saved for epoch {epoch}")

    # 读取（load）模型
    newNet = torch.load(model_dir + model_name)
    # newNet.eval()
    # evalTest(testloader, newNet, gpu = gpu)  # 测试模型精度
    print("Model restore done")


if __name__ == '__main__':
    
    import argparse
    import traceback

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True

    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type = str, default = 'VGG9') # VGG5, VGG9, VGG5_MNIST, VGG9_MNIST
    parser.add_argument('--epochs', type = int, default = 20)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--decrease_LR', type = int, default = 20)
    parser.add_argument('--device',type=str,default="cuda:1")
    # parser.add_argument('--model_dir',type=str,default="../../results/trained_models/VGG9/MNIST/")
    parser.add_argument('--model_dir',type=str,default="../../results/trained_models/VGG9/CIFAR10/")
    # parser.add_argument('--model_dir',type=str,default="../../results/trained_models/VGG5/CIFAR10/")
    # parser.add_argument('--model_dir',type=str,default="../../results/trained_models/VGG5/MNIST/")
    # parser.add_argument('--model_name',type=str,default="VGG5") # VGG9-MNIST-20ep.pth
    # parser.add_argument('--dataset',type=str,default="MNIST") # MNIST CIFAR10
    parser.add_argument('--dataset',type=str,default="CIFAR10") # MNIST CIFAR10

    args_parsed = parser.parse_args()
    args = vars(args_parsed)
    print(args)

    # print("decrease_LR:  dd",args['decrease_LR'])
    # 待inverse的模型训练
    train(args)


# run : python training.py
# nohup python my.py >> nohup.out 2>&1 &
# 10 :[1] 3707176
# 20ep : [2] 3755640