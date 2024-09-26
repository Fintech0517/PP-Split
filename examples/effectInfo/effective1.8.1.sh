###
 # @Author: Ruijun Deng
 # @Date: 2024-09-25 06:01:20
 # @LastEditTime: 2024-09-25 07:44:40
 # @LastEditors: Ruijun Deng
 # @FilePath: /PP-Split/examples/effectInfo/effective1.8.1.sh
 # @Description: 
### 

# Resnet18  
# nohup python -u effectInfo1.8.1.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 2 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.8.1/effectInfo1.8-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.8.1.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 3 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.8.1/effectInfo1.8-pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.8.1.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 5 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.8.1/effectInfo1.8-pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.8.1.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.8.1/effectInfo1.8-pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.8.1.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 9 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.8.1/effectInfo1.8-pool4-layer9-gpu.log 2>&1 &
# nohup python -u effectInfo1.8.1.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 11 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.8.1/effectInfo1.8-pool4-layer11-gpu.log 2>&1 &



# VGG5
# nohup python -u effectInfo1.8.1.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 0  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.8.1/effectInfo1.8-pool4-layer0-gpu.log 2>&1 &
# nohup python -u effectInfo1.8.1.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 1  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.8.1/effectInfo1.8-pool4-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.8.1.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.8.1/effectInfo1.8-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.8.1.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 3  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.8.1/effectInfo1.8-pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.8.1.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 4  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.8.1/effectInfo1.8-pool4-layer4-gpu.log 2>&1 &
nohup python -u effectInfo1.8.1.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 5 --no_dense > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.8.1/effectInfo1.8-pool4-layer5-gpu.log 2>&1 &
nohup python -u effectInfo1.8.1.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 6 --no_dense > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.8.1/effectInfo1.8-pool4-layer6-gpu.log 2>&1 &


