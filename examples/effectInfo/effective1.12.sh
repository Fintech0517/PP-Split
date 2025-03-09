###
 # @Author: Ruijun Deng
 # @Date: 2024-09-25 06:01:20
 # @LastEditTime: 2024-12-08 20:11:22
 # @LastEditors: Ruijun Deng
 # @FilePath: /PP-Split/examples/effectInfo/effective1.10.sh
 # @Description: 
### 

# ResNet18  
# nohup python -u effectInfo1.12.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 2 > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 3 > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 5 > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 9 > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/pool4-layer9-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 11 > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/pool4-layer11-gpu.log 2>&1 &


# VGG5 CIFAR10 
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 0  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/0ep/pool4-layer0-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 1  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/0ep/pool4-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/0ep/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 3  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/0ep/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 4  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/0ep/pool4-layer4-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 5 --no_dense > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/0ep/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 6 --no_dense > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/0ep/pool4-layer6-gpu.log 2>&1 &


# VGG5 MNIST
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 0  > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.12/pool4-layer0-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 1  > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.12/pool4-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2  > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 3  > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.12/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 4  > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.12/pool4-layer4-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 5 --no_dense > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.12/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 6 --no_dense > ../../results/20240702-effectiveInfo/VGG5_MNIST/effectiveInfo1.12/pool4-layer6-gpu.log 2>&1 &


# Purchase100
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset purchase --split_layer 0 --no_pool  > ../../results/20240702-effectiveInfo/Purchase/effectiveInfo1.12/effectInfo1.11-layer0-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset purchase --split_layer 1 --no_pool  > ../../results/20240702-effectiveInfo/Purchase/effectiveInfo1.12/effectInfo1.11-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset purchase --split_layer 3 --no_pool  > ../../results/20240702-effectiveInfo/Purchase/effectiveInfo1.12/effectInfo1.11-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset purchase --split_layer 5 --no_pool  > ../../results/20240702-effectiveInfo/Purchase/effectiveInfo1.12/effectInfo1.11-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset purchase --split_layer 7 --no_pool  > ../../results/20240702-effectiveInfo/Purchase/effectiveInfo1.12/effectInfo1.11-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset purchase --split_layer 8 --no_pool  > ../../results/20240702-effectiveInfo/Purchase/effectiveInfo1.12/effectInfo1.11-layer8-gpu.log 2>&1 &


# Iris
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset Iris --split_layer 3  > ../../results/20240702-effectiveInfo/Purchase/effectiveInfo1.12/effectInfo1.11-layer3-gpu.log 2>&1 &



# VGG9 CIFAR10 [1,4,7,9,10,11,12,13]
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 1  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 2  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 3  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer4-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 5  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 6  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer6-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 7  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 9  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer9-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 10  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer10-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 11  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer11-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 12  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer12-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 13  > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/pool4-layer13-gpu.log 2>&1 &


# VGG9 MNIST
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 1  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.12/pool4-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 2  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 3  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.12/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.12/pool4-layer4-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 5  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.12/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 6  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.12/pool4-layer6-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 7  > ../../results/20240702-effectiveInfo/VGG9_MNIST/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &


# ResNet18 CIFAR100
# nohup python -u effectInfo1.12.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 2 > ../../results/20240702-effectiveInfo/ResNet18_CIFAR100/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 3 > ../../results/20240702-effectiveInfo/ResNet18_CIFAR100/effectiveInfo1.12/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 5 > ../../results/20240702-effectiveInfo/ResNet18_CIFAR100/effectiveInfo1.12/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 7 > ../../results/20240702-effectiveInfo/ResNet18_CIFAR100/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 9 > ../../results/20240702-effectiveInfo/ResNet18_CIFAR100/effectiveInfo1.12/pool4-layer9-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 11 > ../../results/20240702-effectiveInfo/ResNet18_CIFAR100/effectiveInfo1.12/pool4-layer11-gpu.log 2>&1 &


# Different width ResNet18 CIFAR10
# nohup python -u effectInfo1.12.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/20240702-effectiveInfo/ResNet18/ResNet18_20ep_org/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/20240702-effectiveInfo/ResNet18/ResNet18_20ep_narrow/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/20240702-effectiveInfo/ResNet18/ResNet18_20ep_wide/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/20240702-effectiveInfo/ResNet18/ResNet18_20ep_2narrow/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &


# Different deepth ResNet18 CIFAR10
# nohup python -u effectInfo1.12.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/20240702-effectiveInfo/ResNet18/ResNet18_20ep_org/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py  --device cuda:0 --dataset CIFAR10 --model ResNet34 --split_layer 10 > ../../results/20240702-effectiveInfo/ResNet34/effectiveInfo1.12/pool4-layer10-gpu.log 2>&1 &


# different epochs
# vgg5 cifar10
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 0 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 1 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/1ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 2 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/2ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 3 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/3ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 4 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/4ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 5 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/5ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 6 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/6ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 7 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/7ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 8 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/8ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 9 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/9ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 10 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/10ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 11 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/11ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 12 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/12ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 13 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/13ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 14 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/14ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 15 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/15ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 16 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/16ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 17 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/17ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 18 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/18ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 19 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/19ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep 20 > ../../results/20240702-effectiveInfo/VGG5/VGG5_0ep/effectiveInfo1.12/20ep-pool4-layer2-gpu.log 2>&1 &


# different epochs
# vgg5 MNIST
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2 --ep 0 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 2 --ep 1 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/1ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 2 --ep 2 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/2ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 2 --ep 3 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/3ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 2 --ep 4 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/4ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 2 --ep 5 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/5ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 2 --ep 6 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/6ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 2 --ep 7 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/7ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 2 --ep 8 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/8ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 2 --ep 9 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/9ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 2 --ep 10 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/10ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2 --ep 11 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/11ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2 --ep 12 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/12ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2 --ep 13 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/13ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2 --ep 14 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/14ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2 --ep 15 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/15ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2 --ep 16 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/16ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2 --ep 17 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/17ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2 --ep 18 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/18ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2 --ep 19 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/19ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2 --ep 20 > ../../results/20240702-effectiveInfo/VGG5_MNIST/VGG5_0ep/effectiveInfo1.12/20ep-pool4-layer2-gpu.log 2>&1 &



# different epochs
# vgg9 MNIST
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG9 --split_layer 4 --ep 0 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG9 --split_layer 4 --ep 1 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/1ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG9 --split_layer 4 --ep 2 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/2ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG9 --split_layer 4 --ep 3 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/3ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG9 --split_layer 4 --ep 4 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/4ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG9 --split_layer 4 --ep 5 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/5ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG9 --split_layer 4 --ep 6 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/6ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG9 --split_layer 4 --ep 7 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/7ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG9 --split_layer 4 --ep 8 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/8ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG9 --split_layer 4 --ep 9 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/9ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset MNIST --model VGG9 --split_layer 4 --ep 10 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/10ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4 --ep 11 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/11ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4 --ep 12 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/12ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4 --ep 13 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/13ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4 --ep 14 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/14ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4 --ep 15 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/15ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4 --ep 16 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/16ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4 --ep 17 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/17ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4 --ep 18 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/18ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4 --ep 19 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/19ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4 --ep 20 > ../../results/20240702-effectiveInfo/VGG9_MNIST/VGG9_0ep/effectiveInfo1.12/20ep-pool4-layer2-gpu.log 2>&1 &


# different epochs
# vgg9 CIFAR10
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 0 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 1 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/1ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 2 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/2ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 3 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/3ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 4 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/4ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 5 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/5ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 6 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/6ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 7 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/7ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 8 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/8ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 9 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/9ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 10 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/10ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 11 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/11ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 12 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/12ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 13 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/13ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 14 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/14ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 15 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/15ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 16 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/16ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 17 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/17ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 18 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/18ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 19 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/19ep-pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep 20 > ../../results/20240702-effectiveInfo/VGG9/VGG9_0ep/effectiveInfo1.12/20ep-pool4-layer2-gpu.log 2>&1 &

# ViTb_8+ CIFAR 10 avgpool4
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 0  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer0-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 1  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 2  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 3  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 4  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer4-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 5 > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 6 > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer6-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 7 > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ViTb_16 --split_layer 8 > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer8-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ViTb_16 --split_layer 9 > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer9-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ViTb_16 --split_layer 10 > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer10-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ViTb_16 --split_layer 11 > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer11-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ViTb_16 --split_layer 12 > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer12-gpu.log 2>&1 &

# ViTb_16 + CIFAR 10 no avgpool0
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 0  --no_pool  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer0-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 1  --no_pool   > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer1-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 2  --no_pool   > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 3  --no_pool   > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer3-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 4  --no_pool   > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer4-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 5  --no_pool  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer5-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 6  --no_pool  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer6-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model ViTb_16 --split_layer 7  --no_pool  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ViTb_16 --split_layer 8  --no_pool  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer8-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ViTb_16 --split_layer 9  --no_pool  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer9-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ViTb_16 --split_layer 10  --no_pool  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer10-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ViTb_16 --split_layer 11  --no_pool  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer11-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ViTb_16 --split_layer 12  --no_pool  > ../../results/20240702-effectiveInfo/ViTb_16_CIFAR10/effectiveInfo1.12/pool4-layer12-gpu.log 2>&1 &
# kill -9 3408233
# kill -9  3409276
# kill -9  3409277
# kill -9  3409278
# kill -9  3409279
# kill -9  3409280
# kill -9  3409281
# kill -9  3409282
# kill -9  3409283
#   kill -9 3409284
#   kill -9 3409285
#   kill -9 3409286
#   kill -9 3409287

# 开防御：
# VGG5+defense gaussian
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.1 --test_num effectiveInfo1.12/Gaussian-0.1 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-0.1.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.2 --test_num effectiveInfo1.12/Gaussian-0.2 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-0.2.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.3 --test_num effectiveInfo1.12/Gaussian-0.3 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-0.3.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.4 --test_num effectiveInfo1.12/Gaussian-0.4 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-0.4.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.5 --test_num effectiveInfo1.12/Gaussian-0.5 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-0.5.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.6 --test_num effectiveInfo1.12/Gaussian-0.6 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-0.6.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.7 --test_num effectiveInfo1.12/Gaussian-0.7 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-0.7.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.8 --test_num effectiveInfo1.12/Gaussian-0.8 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-0.8.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.9 --test_num effectiveInfo1.12/Gaussian-0.9 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-0.9.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 1 --test_num effectiveInfo1.12/Gaussian-1 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-1.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 5 --test_num  effectiveInfo1.12/Gaussian-5   > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-5.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 10 --test_num effectiveInfo1.12/Gaussian-10  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-10.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 15 --test_num effectiveInfo1.12/Gaussian-15  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-15.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 20 --test_num effectiveInfo1.12/Gaussian-20  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-20.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 25 --test_num effectiveInfo1.12/Gaussian-25  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-25.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 30 --test_num effectiveInfo1.12/Gaussian-30  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-30.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 35 --test_num effectiveInfo1.12/Gaussian-35  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-35.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 40 --test_num effectiveInfo1.12/Gaussian-40  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-40.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 45 --test_num effectiveInfo1.12/Gaussian-45  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-45.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 50 --test_num effectiveInfo1.12/Gaussian-50  > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/Gaussian/pool4-layer2-gpu-50.log 2>&1 &


# VGG5 CIFAR10 nopeek
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 0 --ep -2 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/nopeek/pool4-layer0-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 1 --ep -2   > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/nopeek/pool4-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep -2   > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/nopeek/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 3 --ep -2   > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/nopeek/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 4 --ep -2   > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/nopeek/pool4-layer4-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 5 --ep -2  --no_dense > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/nopeek/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 6 --ep -2  --no_dense > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/nopeek/pool4-layer6-gpu.log 2>&1 &


# VGG9 cifar10 nopeek
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 1 --ep -2 > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/nopeek/pool4-layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 2 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/nopeek/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 3 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/nopeek/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/nopeek/pool4-layer4-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 5 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/nopeek/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 6 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/nopeek/pool4-layer6-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 7 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/nopeek/pool4-layer7-gpu.log 2>&1 &



# [1] 2916462
# [2] 2916463
# [3] 2916464
# [4] 2916465
# [5] 2916466
# [6] 2916467
# [7] 2916468


# ResNet18 cifar10 nopeek
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 2 --ep -2 > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/nopeek/pool4-layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 3 --ep -2   > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/nopeek/pool4-layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 5 --ep -2   > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/nopeek/pool4-layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 7 --ep -2   > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/nopeek/pool4-layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 9 --ep -2   > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/nopeek/pool4-layer9-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 11 --ep -2   > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/nopeek/pool4-layer11-gpu.log 2>&1 &

# [9] 2920232
# [10] 2920233
# [11] 2920234
# [12] 2920235
# [13] 2920236
# [14] 2920237



# VGG5 CIFAR10 shredder
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 0 --ep -2 > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/shredder/pool4-layer0-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 1 --ep -2   > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/shredder/pool4-layer1-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2 --ep -2   > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/shredder/pool4-layer2-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 3 --ep -2   > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/shredder/pool4-layer3-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 4 --ep -2   > ../../results/20240702-effectiveInfo/VGG5/effectiveInfo1.12/shredder/pool4-layer4-gpu.log 2>&1 &


# VGG9 cifar10 shredder
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 1 --ep -2 > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/shredder/pool4-layer1-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 2 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/shredder/pool4-layer2-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 3 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/shredder/pool4-layer3-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/shredder/pool4-layer4-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 5 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/shredder/pool4-layer5-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 6 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/shredder/pool4-layer6-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 7 --ep -2   > ../../results/20240702-effectiveInfo/VGG9/effectiveInfo1.12/shredder/pool4-layer7-gpu.log 2>&1 &



# ResNet18 cifar10 shredder
nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 2 --ep -2 > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/shredder/pool4-layer2-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 3 --ep -2   > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/shredder/pool4-layer3-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 5 --ep -2   > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/shredder/pool4-layer5-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 7 --ep -2   > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/shredder/pool4-layer7-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 9 --ep -2   > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/shredder/pool4-layer9-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 11 --ep -2   > ../../results/20240702-effectiveInfo/ResNet18/effectiveInfo1.12/shredder/pool4-layer11-gpu.log 2>&1 &



# ViTb_16 + ImageNet1k no avgpool0
nohup python -u effectInfo1.12.py --device cuda:1 --dataset ImageNet1k --model ViTb_16 --split_layer 0    > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer0-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset ImageNet1k --model ViTb_16 --split_layer 1     > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer1-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset ImageNet1k --model ViTb_16 --split_layer 2     > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer2-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset ImageNet1k --model ViTb_16 --split_layer 3   > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer3-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset ImageNet1k --model ViTb_16 --split_layer 4   > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer4-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset ImageNet1k --model ViTb_16 --split_layer 5  > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer5-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset ImageNet1k --model ViTb_16 --split_layer 6  > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer6-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:1 --dataset ImageNet1k --model ViTb_16 --split_layer 7  > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer7-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset ImageNet1k --model ViTb_16 --split_layer 8  > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer8-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset ImageNet1k --model ViTb_16 --split_layer 9  > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer9-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset ImageNet1k --model ViTb_16 --split_layer 10  > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer10-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset ImageNet1k --model ViTb_16 --split_layer 11  > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer11-gpu.log 2>&1 &
nohup python -u effectInfo1.12.py --device cuda:0 --dataset ImageNet1k --model ViTb_16 --split_layer 12  > ../../results/20240702-effectiveInfo/ViTb_16_ImageNet1k/effectiveInfo1.12/pool4-layer12-gpu.log 2>&1 &
# [1] 3047563
# [2] 3047564
# [3] 3047565
# [4] 3047566
# [5] 3047567
# [6] 3047568
# [7] 3047569
# [8] 3047570
# [9] 3047571
# [10] 3047572
# [11] 3047573
# [12] 3047574
# [13] 3047575


# Credit
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset credit --model credit --split_layer 0  --no_pool  > ../../results/20240702-effectiveInfo/credit_credit/effectiveInfo1.12/layer0-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset credit --model credit --split_layer 1  --no_pool  > ../../results/20240702-effectiveInfo/credit_credit/effectiveInfo1.12/layer1-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset credit --model credit --split_layer 2  --no_pool  > ../../results/20240702-effectiveInfo/credit_credit/effectiveInfo1.12/layer2-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset credit --model credit --split_layer 3  --no_pool  > ../../results/20240702-effectiveInfo/credit_credit/effectiveInfo1.12/layer3-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset credit --model credit --split_layer 4  --no_pool  > ../../results/20240702-effectiveInfo/credit_credit/effectiveInfo1.12/layer4-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset credit --model credit --split_layer 5  --no_pool  > ../../results/20240702-effectiveInfo/credit_credit/effectiveInfo1.12/layer5-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset credit --model credit --split_layer 6  --no_pool  > ../../results/20240702-effectiveInfo/credit_credit/effectiveInfo1.12/layer6-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset credit --model credit --split_layer 7  --no_pool  > ../../results/20240702-effectiveInfo/credit_credit/effectiveInfo1.12/layer7-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset credit --model credit --split_layer 8  --no_pool  > ../../results/20240702-effectiveInfo/credit_credit/effectiveInfo1.12/layer8-gpu.log 2>&1 &
# nohup python -u effectInfo1.12.py --device cuda:0 --dataset credit --model credit --split_layer 9  --no_pool  > ../../results/20240702-effectiveInfo/credit_credit/effectiveInfo1.12/layer9-gpu.log 2>&1 &


