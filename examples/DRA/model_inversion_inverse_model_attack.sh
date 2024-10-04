
###
 # @Author: Ruijun Deng
 # @Date: 2024-09-26 05:36:51
 # @LastEditTime: 2024-10-02 20:44:03
 # @LastEditors: Ruijun Deng
 # @FilePath: /PP-Split/examples/DRA/model_inversion_inverse_model_attack.sh
 # @Description: 
### 
# Resnet18  cifar10
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 2 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer2-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 3 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer3-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 5 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer5-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer7-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 9 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer9-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR10 --model ResNet18 --split_layer 11 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer11-gpu.log 2>&1 &



# VGG5 CIFAR10
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 0  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer0-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 1  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer1-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer2-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 3  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer3-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 4  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer4-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 5  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer5-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 6  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer6-gpu.log 2>&1 &

# VGG5 MNIST
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 0  > ../../results/inverse-model-results-20240414/VGG5_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer0-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 1  > ../../results/inverse-model-results-20240414/VGG5_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer1-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 2  > ../../results/inverse-model-results-20240414/VGG5_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer2-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 3  > ../../results/inverse-model-results-20240414/VGG5_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer3-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset MNIST --model VGG5 --split_layer 4  > ../../results/inverse-model-results-20240414/VGG5_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer4-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 5  > ../../results/inverse-model-results-20240414/VGG5_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer5-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset MNIST --model VGG5 --split_layer 6  > ../../results/inverse-model-results-20240414/VGG5_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer6-gpu.log 2>&1 &



# VGG9 CIFAR10 [1,4,7,9,10,11,12,13]
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 1  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer1-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 2  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer2-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 3  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer3-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 4  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer4-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 5  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer5-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 6  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer6-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 7  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer7-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 9  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer9-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 10  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer10-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 11  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer11-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 12  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer12-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG9 --split_layer 13  > ../../results/inverse-model-results-20240414/VGG9/InverseModelAttack-defense0.1/InverseNetwork-layer13-gpu.log 2>&1 &



# VGG9 MNIST
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 1  > ../../results/inverse-model-results-20240414/VGG9_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer1-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 2  > ../../results/inverse-model-results-20240414/VGG9_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer2-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 3  > ../../results/inverse-model-results-20240414/VGG9_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer3-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 4  > ../../results/inverse-model-results-20240414/VGG9_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer4-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 5  > ../../results/inverse-model-results-20240414/VGG9_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer5-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 6  > ../../results/inverse-model-results-20240414/VGG9_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer6-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset MNIST --model VGG9 --split_layer 7  > ../../results/inverse-model-results-20240414/VGG9_MNIST/InverseModelAttack-defense0.1/InverseNetwork-layer7-gpu.log 2>&1 &

# Resnet18  cifar100
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 2 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer2-gpu.log 2>&1 &
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 3 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer3-gpu.log 2>&1 &
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 5 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer5-gpu.log 2>&1 &
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 7 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer7-gpu.log 2>&1 &
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 9 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer9-gpu.log 2>&1 &
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 11 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer11-gpu.log 2>&1 &
