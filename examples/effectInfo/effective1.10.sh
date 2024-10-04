###
 # @Author: Ruijun Deng
 # @Date: 2024-09-25 06:01:20
 # @LastEditTime: 2024-10-04 04:15:05
 # @LastEditors: Ruijun Deng
 # @FilePath: /PP-Split/examples/effectInfo/effective1.10.sh
 # @Description: 
### 

# Resnet18  
# nohup python -u effectInfo1.10.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 2 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.10/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 3 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.10/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 5 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.10/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.10/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 9 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.10/pool4-layer9-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 11 > ../../results/20240702-effectiveInfo/Resnet18/effectiveInfo1.10/pool4-layer11-gpu.log 2>&1 &


# VGG5 CIFAR10 
nohup python -u effectInfo1.10.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 0  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.10/0ep/pool4-layer0-gpu.log 2>&1 &
nohup python -u effectInfo1.10.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 1  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.10/0ep/pool4-layer1-gpu.log 2>&1 &
nohup python -u effectInfo1.10.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.10/0ep/pool4-layer2-gpu.log 2>&1 &
nohup python -u effectInfo1.10.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 3  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.10/0ep/pool4-layer3-gpu.log 2>&1 &
nohup python -u effectInfo1.10.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 4  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.10/0ep/pool4-layer4-gpu.log 2>&1 &
nohup python -u effectInfo1.10.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 5 --no_dense > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.10/0ep/pool4-layer5-gpu.log 2>&1 &
nohup python -u effectInfo1.10.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 6 --no_dense > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.10/0ep/pool4-layer6-gpu.log 2>&1 &


# VGG5 MNIST
# nohup python -u effectInfo1.10.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 0  > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.10/pool4-layer0-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 1  > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.10/pool4-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2  > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.10/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 3  > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.10/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 4  > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.10/pool4-layer4-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 5 --no_dense > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.10/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 6 --no_dense > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.10/pool4-layer6-gpu.log 2>&1 &


# Purchase100
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset purchase --split_layer 3  > ../../results/20240702-effectiveInfo/Purchase/effectiveInfo1.10/effectInfo1.10-layer3-gpu.log 2>&1 &


# Iris
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset Iris --split_layer 3  > ../../results/20240702-effectiveInfo/Purchase/effectiveInfo1.10/effectInfo1.10-layer3-gpu.log 2>&1 &



# VGG9 CIFAR10 [1,4,7,9,10,11,12,13]
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 1  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 2  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 3  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer4-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 5  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 6  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer6-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 7  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 9  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer9-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 10  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer10-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 11  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer11-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 12  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer12-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 13  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.10/pool4-layer13-gpu.log 2>&1 &


# VGG9 MNIST
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 1  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.10/pool4-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 2  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.10/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 3  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.10/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.10/pool4-layer4-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 5  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.10/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 6  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.10/pool4-layer6-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 7  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.10/pool4-layer7-gpu.log 2>&1 &


# ResNet18 CIFAR100
# nohup python -u effectInfo1.10.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 2 > ../../results/20240702-effectiveInfo/Resnet18_CIFAR100/effectiveInfo1.10/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 3 > ../../results/20240702-effectiveInfo/Resnet18_CIFAR100/effectiveInfo1.10/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 5 > ../../results/20240702-effectiveInfo/Resnet18_CIFAR100/effectiveInfo1.10/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 7 > ../../results/20240702-effectiveInfo/Resnet18_CIFAR100/effectiveInfo1.10/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 9 > ../../results/20240702-effectiveInfo/Resnet18_CIFAR100/effectiveInfo1.10/pool4-layer9-gpu.log 2>&1 &
# nohup python -u effectInfo1.10.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 11 > ../../results/20240702-effectiveInfo/Resnet18_CIFAR100/effectiveInfo1.10/pool4-layer11-gpu.log 2>&1 &
