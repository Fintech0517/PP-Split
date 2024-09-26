
###
 # @Author: Ruijun Deng
 # @Date: 2024-09-26 06:27:40
 # @LastEditTime: 2024-09-26 07:21:34
 # @LastEditors: Ruijun Deng
 # @FilePath: /PP-Split/examples/FIL/fil.sh
 # @Description: 
### 
# Resnet18  
# nohup python -u fil.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 2 > ../../results/inverse-model-results-20240414/Resnet18/drjCodeFIL/inverse_dFIL-layer2-gpu.log 2>&1 &
# nohup python -u fil.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 3 > ../../results/inverse-model-results-20240414/Resnet18/drjCodeFIL/inverse_dFIL-layer3-gpu.log 2>&1 &
# nohup python -u fil.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 5 > ../../results/inverse-model-results-20240414/Resnet18/drjCodeFIL/inverse_dFIL-layer5-gpu.log 2>&1 &
# nohup python -u fil.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/inverse-model-results-20240414/Resnet18/drjCodeFIL/inverse_dFIL-layer7-gpu.log 2>&1 &
# nohup python -u fil.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 9 > ../../results/inverse-model-results-20240414/Resnet18/drjCodeFIL/inverse_dFIL-layer9-gpu.log 2>&1 &
# nohup python -u fil.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 11 > ../../results/inverse-model-results-20240414/Resnet18/drjCodeFIL/inverse_dFIL-layer11-gpu.log 2>&1 &


# VGG5
# nohup python -u fil.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 0  > ../../results/inverse-model-results-20240414/VGG5/drjCodeFIL/inverse_dFIL-layer0-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 1  > ../../results/inverse-model-results-20240414/VGG5/drjCodeFIL/inverse_dFIL-layer1-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  > ../../results/inverse-model-results-20240414/VGG5/drjCodeFIL/inverse_dFIL-layer2-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 3  > ../../results/inverse-model-results-20240414/VGG5/drjCodeFIL/inverse_dFIL-layer3-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 4  > ../../results/inverse-model-results-20240414/VGG5/drjCodeFIL/inverse_dFIL-layer4-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 5  > ../../results/inverse-model-results-20240414/VGG5/drjCodeFIL/inverse_dFIL-layer5-gpu.log 2>&1 &
nohup python -u fil.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 6  > ../../results/inverse-model-results-20240414/VGG5/drjCodeFIL/inverse_dFIL-layer6-gpu.log 2>&1 &

