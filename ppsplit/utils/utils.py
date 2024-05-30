'''
Author: Ruijun Deng
Date: 2024-05-21 15:20:26
LastEditTime: 2024-05-21 15:20:26
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/utils/utils.py
Description: 
'''

import os

def create_dir(dir_route):
    if not os.path.exists(dir_route):
        os.makedirs(dir_route)
    return

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
