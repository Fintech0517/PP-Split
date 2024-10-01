
###
 # @Author: Ruijun Deng
 # @Date: 2024-09-26 06:27:40
 # @LastEditTime: 2024-09-27 23:41:30
 # @LastEditors: Ruijun Deng
 # @FilePath: /PP-Split/examples/InvMetrics/MI.sh
 # @Description: 
### 
# Resnet18  
# nohup python -u MI.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 2 > ../../results/InvMetric-202403/Resnet18/MI/layer2-gpu.log 2>&1 &
# nohup python -u MI.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 3 > ../../results/InvMetric-202403/Resnet18/MI/layer3-gpu.log 2>&1 &
# nohup python -u MI.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 5 > ../../results/InvMetric-202403/Resnet18/MI/layer5-gpu.log 2>&1 &
# nohup python -u MI.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/InvMetric-202403/Resnet18/MI/layer7-gpu.log 2>&1 &
# nohup python -u MI.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 9 > ../../results/InvMetric-202403/Resnet18/MI/layer9-gpu.log 2>&1 &
# nohup python -u MI.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 11 > ../../results/InvMetric-202403/Resnet18/MI/layer11-gpu.log 2>&1 &


# VGG5
nohup python -u MI.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 0  > ../../results/InvMetric-202403/VGG5/MI/layer0-gpu.log 2>&1 &
nohup python -u MI.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 1  > ../../results/InvMetric-202403/VGG5/MI/layer1-gpu.log 2>&1 &
nohup python -u MI.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  > ../../results/InvMetric-202403/VGG5/MI/layer2-gpu.log 2>&1 &
nohup python -u MI.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 3  > ../../results/InvMetric-202403/VGG5/MI/layer3-gpu.log 2>&1 &
nohup python -u MI.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 4  > ../../results/InvMetric-202403/VGG5/MI/layer4-gpu.log 2>&1 &
nohup python -u MI.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 5  > ../../results/InvMetric-202403/VGG5/MI/layer5-gpu.log 2>&1 &
nohup python -u MI.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 6  > ../../results/InvMetric-202403/VGG5/MI/layer6-gpu.log 2>&1 &

