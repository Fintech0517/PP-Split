
###
 # @Author: Ruijun Deng
 # @Date: 2024-09-26 06:27:40
 # @LastEditTime: 2024-09-26 08:33:58
 # @LastEditors: Ruijun Deng
 # @FilePath: /PP-Split/examples/FIL/fil.sh
 # @Description: 
### 
# Resnet18  
nohup python -u fil.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 2 > ../../results/20240904-fisher/Resnet18/drjCodeFIL/inverse_dFIL-layer2-gpu.log 2>&1 &
nohup python -u fil.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 3 > ../../results/20240904-fisher/Resnet18/drjCodeFIL/inverse_dFIL-layer3-gpu.log 2>&1 &
# nohup python -u fil.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 5 > ../../results/20240904-fisher/Resnet18/drjCodeFIL/inverse_dFIL-layer5-gpu.log 2>&1 &
# nohup python -u fil.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/20240904-fisher/Resnet18/drjCodeFIL/inverse_dFIL-layer7-gpu.log 2>&1 &
# nohup python -u fil.py  --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 9 > ../../results/20240904-fisher/Resnet18/drjCodeFIL/inverse_dFIL-layer9-gpu.log 2>&1 &
# nohup python -u fil.py  --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 11 > ../../results/20240904-fisher/Resnet18/drjCodeFIL/inverse_dFIL-layer11-gpu.log 2>&1 &


# VGG5
nohup python -u fil.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 0  > ../../results/20240904-fisher/VGG5/drjCodeFIL/inverse_dFIL-layer0-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 1  > ../../results/20240904-fisher/VGG5/drjCodeFIL/inverse_dFIL-layer1-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  > ../../results/20240904-fisher/VGG5/drjCodeFIL/inverse_dFIL-layer2-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 3  > ../../results/20240904-fisher/VGG5/drjCodeFIL/inverse_dFIL-layer3-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 4  > ../../results/20240904-fisher/VGG5/drjCodeFIL/inverse_dFIL-layer4-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 5  > ../../results/20240904-fisher/VGG5/drjCodeFIL/inverse_dFIL-layer5-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 6  > ../../results/20240904-fisher/VGG5/drjCodeFIL/inverse_dFIL-layer6-gpu.log 2>&1 &

