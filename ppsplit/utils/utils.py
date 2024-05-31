'''
Author: Ruijun Deng
Date: 2024-05-21 15:20:26
LastEditTime: 2024-05-31 20:28:13
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

# Histogram相关
# prob smashed data的分布 输入是张亮
def plot_smashed_distribution(smashed_data,start = -1, end = 1):
    import matplotlib.pyplot as plt
    import numpy as np
    
    data = smashed_data.flatten(1) # 拉平后的数据
    # data_sort = np.sort(data) # 排序后的拉平数据

    plot_array_distribution(data,start, end)

def plot_array_distribution(data,start = -1, end = 1):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # counts, buckets = np.histogram(data, bins=100, range=(start, end), density=True) 
    # 也许这个更加适合FCN？
    print("data.size():",np.size(data))
    # counts, buckets = np.histogram(data, bins=100 if 100<np.size(data) else np.size(data), density=True) 
    counts, buckets = np.histogram(data, bins=100, density=True) 

    # 画图 value-probability 图2
    plt.figure()
    counts = counts/np.sum(counts) # 如果start和end不是[0，1]就都要用
    # edges = np.hstack((buckets,np.array([buckets[-1]+(buckets[1]-buckets[0])]))) # linespace
    edges = buckets # histogram
    plt.stairs(counts,edges,fill=True,color='red', alpha=0.5)
    plt.title('smashed data histogram')
    plt.xlabel('value')
    plt.ylabel('probability')

    plt.tight_layout()  # 自动调整子图布局
    plt.show()
    # plt.savefig(f'smashed_data_distribution{time.time()}.png')

    # 打印信息
    print("sigma(prob):",np.sum(counts)) # 查看counts的量
    print("counts[0],counts[-1],counts[49]",counts[0],counts[-1],counts[len(counts)//2])


def plot_index_value(smashed_data): 
    import matplotlib.pyplot as plt
    import numpy as np
    
    data = smashed_data.flatten(dim=1) # 拉平后的数据
    data_sort = np.sort(data) # 排序后的拉平数据

    print("data.size():",np.size(data))
    # counts, buckets = np.histogram(data, bins=100, range=(start, end), density=True) # 也许这个更加适合FCN？
    # counts, buckets = np.histogram(data, bins=100 if 100<np.size(data) else np.size(data), density=True) 
    counts, buckets = np.histogram(data, bins=10, density=True) 

    # 画图 index-value 图1
    plt.figure(figsize=(10,3))
    x_axis = np.arange(0, len(data), 1)
    plt.plot(x_axis, data_sort)
    plt.title('smashed data distribution')
    plt.xlabel('index')
    plt.ylabel('value')

    plt.tight_layout()  # 自动调整子图布局
    plt.show()
    # plt.savefig(f'smashed_data_distribution{time.time()}.png')
