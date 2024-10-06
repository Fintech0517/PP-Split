# 导包
import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from ppsplit.attacks.model_inversion.inverse_model import InverseModelAttack
from ppsplit.utils.utils import create_dir
import torch
import os

# 防护措施
from ppsplit.defense.noise import Noise

# 模型、数据集获取
from target_model.task_select import get_dataloader_and_model, get_dataloader,get_models



import argparse
# parser
parser = argparse.ArgumentParser(description='PP-Split')
parser.add_argument('--device', type=str, default="cuda:0", help='device')
parser.add_argument('--dataset', type=str, default="CIFAR10", help='dataset') # 'bank', 'credit', 'purchase', 'Iris',
parser.add_argument('--model', type=str, default="ResNet18", help='model')  # 'ResNet18',' VGG5'
parser.add_argument('--result_dir', type=str, default="inverse-model-results-20240414/", help='result_dir')
parser.add_argument('--oneData_bs', type=int, default=1, help='oneData_bs')
parser.add_argument('--test_bs', type=int, default=1, help='test_bs')
parser.add_argument('--train_bs', type=int, default=32, help='train_bs')
parser.add_argument('--noise_scale', type=int, default=0.1, help='noise_scale')
parser.add_argument('--split_layer', type=int, default=2, help='split_layer')
parser.add_argument('--test_num', type=str, default='InverseModelAttack-defense0.1', help='test_num')
parser.add_argument('--no_dense', action='store_true', help='no_dense')
parser.add_argument('--ep', type=int, help='epochs', default=-1)

args_python = parser.parse_args()
args = vars(args_python)


# # 超参数
# args = {
#         'device':torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#         # 'device':torch.device("cpu"),
#         'dataset':'CIFAR10',
#         # 'dataset':'bank',
#         # 'dataset':'credit',
#         # 'dataset':'purchase',
#         # 'dataset':'Iris',
#         # 'model': 'ResNet18',
#         'model': 'VGG5',
#         # 'result_dir': '20240702-FIL/',
#         'result_dir': 'inverse-model-results-20240414/',
#         'oneData_bs': 1,
#         'test_bs': 1,
#         'train_bs': 32,
#         'noise_scale': 0.1, # 防护措施
#         'split_layer': 0,
#         # 'test_num': 'invdFIL', # MI, invdFIL, distCor, ULoss,  # split layer [2,3,5,7,9,11] for ResNet18
#         'test_num': 'InverseModelAttack-defense0.1',
#         'no_dense':True,
#         }

# print(args['device'])
print(args)


# 获取模型和数据集
# msg = get_dataloader_and_model(**args)

model_msg = get_models(args)

# one_data_loader,trainloader,testloader = model_msg['one_data_loader'],model_msg['trainloader'], model_msg['testloader']
client_net,decoder_net = model_msg['client_net'], model_msg['decoder_net']
decoder_route = model_msg['decoder_route']
image_deprocess = model_msg['image_deprocess']


results_dir = model_msg['results_dir']
inverse_dir = results_dir + 'layer'+str(args['split_layer'])+'/'

data_msg = get_dataloader(args)
data_type = data_msg['data_type']
# data_type = 1 if args['dataset'] in ['CIFAR10','MNIST'] else 0

print('results_dir:',results_dir)
print('inverse_dir:',inverse_dir)
print('decoder_route:',decoder_route)

# 准备inverse_model attack使用到的东西
# 创建Inverse Model Attack对象
im_attack = InverseModelAttack(decoder_route=decoder_route,data_type=data_type,inverse_dir=inverse_dir,device=args['device'])


# 加载decoder模型
if not os.path.isfile(decoder_route): # 如果没有训练decoder
    # 训练decoder
    args['train_bs']=32
    args['test_bs']=32
    msg_data = get_dataloader(args)
    # trainloader,testloader = get_cifar10_normalize(batch_size=32)
    decoder_net= im_attack.train_decoder(client_net=client_net,decoder_net=decoder_net,
                            train_loader=msg_data['trainloader'],test_loader=msg_data['testloader'],
                            epochs=20)
else:
    print("Load decoder model from:",decoder_route)


print(decoder_net)

# 实现攻击,恢复testloader中所有图片
# trainloader,testloader = get_cifar10_normalize(batch_size=1)
args['train_bs']=1
args['test_bs']=1
msg_data = get_dataloader(args)

im_attack.inverse(client_net=client_net,decoder_net=decoder_net,
                  train_loader=msg_data['trainloader'],test_loader=msg_data['testloader'],
                  deprocess=image_deprocess,
                  save_fake=True,
                  tab=msg_data['tabinfo'])

