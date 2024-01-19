
import torch.optim as optim
import torch.nn as nn
import torch
import os
import numpy as np
import shutil
import errno
import tqdm
import time


from utils.preprocess_bank import *
from utils.preprocess_purchase import preprocess_purchase,accuracy
from models.PurchaseNet import PurchaseClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def train(train_loader, model, criterion, optimizer, epoch, use_cuda,num_batchs=999999):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end = time.time()

    len_t = len(train_loader)

    for ind,(inputs,targets) in enumerate(tqdm.tqdm(train_loader)):
        if ind > num_batchs:
            break

        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)

        # compute output
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size()[0])
        top1.update(prec1, inputs.size()[0])
        top5.update(prec5, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if ind%1000==0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=ind + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))

    return (losses.avg, top1.avg)


def test(test_loader, model, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    len_t = len(test_loader)

    for ind,(inputs, targets) in enumerate(tqdm.tqdm(test_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size()[0])
        top1.update(prec1, inputs.size()[0])
        top5.update(prec5, inputs.size()[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

# 保存模型
def save_checkpoint(state, is_best, checkpoint='./results/purchase/', filename='checkpoint.pth'):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


if __name__ == '__main__':
    import argparse
    
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type = bool, default = True)
    args = parser.parse_args()

    # 训练好的模型存储的dir
    model_dir = "results/purchase/"
    model_name = "purchase-0ep.pth"
    data_dir = '/home/dengruijun/data/FinTech/DATASET/kaggle-dataset/Purchase100/'

    # 公用定义
    best_acc = 0.0
    epochs=50
    batch_size=128
    model = PurchaseClassifier()
    # model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 数据获取
    train_loader, test_loader = preprocess_purchase(data_path=data_dir,batch_size=batch_size)

    # # 进行迭代训练
    # for epoch in range(epochs):
    #     print('\nEpoch: [%d | %d]' % (epoch + 1, epochs))

    #     # 训练
    #     train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.gpu)

    #     # 测试
    #     test_loss, test_acc = test(test_loader, model, criterion, epoch, args.gpu)
    #     print ('test acc',test_acc)

    #     # save model
    #     is_best = test_acc>best_acc
    #     best_acc = max(test_acc, best_acc)
    #     save_checkpoint({
    #             'epoch': epoch + 1,
    #             'state_dict': model.state_dict(),
    #             'acc': test_acc,
    #             'best_acc': best_acc,
    #             'optimizer' : optimizer.state_dict(),
    #         }, is_best,filename='epoch_train%d.pth'%(epoch+1))


# run : python training.py
# nohup python my.py >> nohup.out 2>&1 &

