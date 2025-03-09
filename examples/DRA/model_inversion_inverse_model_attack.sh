
###
 # @Author: Ruijun Deng
 # @Date: 2024-09-26 05:36:51
 # @LastEditTime: 2024-10-12 22:15:53
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
nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 0  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer0-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 1  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer1-gpu.log 2>&1 &
nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer2-gpu.log 2>&1 &
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
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 2 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer2-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 3 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer3-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 5 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer5-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 7 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer7-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 9 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer9-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:0 --dataset CIFAR100 --model ResNet18 --split_layer 11 > ../../results/inverse-model-results-20240414/Resnet18_CIFAR100/InverseModelAttack-defense0.1/InverseNetwork-layer11-gpu.log 2>&1 &


# Different width ResNet18 CIFAR10
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/inverse-model-results-20240414/Resnet18/Resnet18_20ep_org/InverseModelAttack-defense0.1/pool4-layer7-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/inverse-model-results-20240414/Resnet18/Resnet18_20ep_narrow/InverseModelAttack-defense0.1/pool4-layer7-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/inverse-model-results-20240414/Resnet18/Resnet18_20ep_wide/InverseModelAttack-defense0.1/pool4-layer7-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/inverse-model-results-20240414/Resnet18/Resnet18_20ep_2narrow/InverseModelAttack-defense0.1/pool4-layer7-gpu.log 2>&1 &


# Different deepth ResNet18 CIFAR10
# nohup python -u model_inversion_inverse_model_attack.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/inverse-model-results-20240414/Resnet18/Resnet18_20ep_org/InverseModelAttack-defense0.1/pool4-layer7-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model ResNet34 --split_layer 10 > ../../results/inverse-model-results-20240414/Resnet34/InverseModelAttack-defense0.1/pool4-layer10-gpu.log 2>&1 &


# 不同程度的防御 VGG5 cifar10
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.2 --test_num Gaussian-0.2 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-0.2.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.3 --test_num Gaussian-0.3 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-0.3.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.4 --test_num Gaussian-0.4 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-0.4.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.5 --test_num Gaussian-0.5 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-0.5.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.6 --test_num Gaussian-0.6 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-0.6.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.7 --test_num Gaussian-0.7 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-0.7.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.8 --test_num Gaussian-0.8 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-0.8.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 0.9 --test_num Gaussian-0.9 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-0.9.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 1 --test_num Gaussian-1 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-1.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 5 --test_num Gaussian-5 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-5.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 10 --test_num Gaussian-10 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-10.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 15 --test_num Gaussian-15 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-15.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 20 --test_num Gaussian-20 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-20.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 25 --test_num Gaussian-25 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-25.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 30 --test_num Gaussian-30 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-30.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 35 --test_num Gaussian-35 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-35.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 40 --test_num Gaussian-40 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-40.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 45 --test_num Gaussian-45 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-45.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset CIFAR10 --model VGG5 --split_layer 2  --noise_scale 50 --test_num Gaussian-50 > ../../results/inverse-model-results-20240414/VGG5/Gaussian-0.2/pool4-layer2-gpu-50.log 2>&1 &

# Purchase100
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset purchase --split_layer 0  > ../../results/inverse-model-results-20240414/Purchase/InverseModelAttack-defense0.1/pool4-layer0-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset purchase --split_layer 1  > ../../results/inverse-model-results-20240414/Purchase/InverseModelAttack-defense0.1/pool4-layer1-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset purchase --split_layer 3  > ../../results/inverse-model-results-20240414/Purchase/InverseModelAttack-defense0.1/pool4-layer3-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset purchase --split_layer 5  > ../../results/inverse-model-results-20240414/Purchase/InverseModelAttack-defense0.1/pool4-layer5-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset purchase --split_layer 7  > ../../results/inverse-model-results-20240414/Purchase/InverseModelAttack-defense0.1/pool4-layer7-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset purchase --split_layer 8  > ../../results/inverse-model-results-20240414/Purchase/InverseModelAttack-defense0.1/pool4-layer8-gpu.log 2>&1 &



# Credit
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset credit --model credit --split_layer 0  > ../../results/inverse-model-results-20240414/credit_credit/InverseModelAttack/layer0-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset credit --model credit --split_layer 1  > ../../results/inverse-model-results-20240414/credit_credit/InverseModelAttack/layer1-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset credit --model credit --split_layer 2  > ../../results/inverse-model-results-20240414/credit_credit/InverseModelAttack/layer2-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset credit --model credit --split_layer 3  > ../../results/inverse-model-results-20240414/credit_credit/InverseModelAttack/layer3-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset credit --model credit --split_layer 4  > ../../results/inverse-model-results-20240414/credit_credit/InverseModelAttack/layer4-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset credit --model credit --split_layer 5  > ../../results/inverse-model-results-20240414/credit_credit/InverseModelAttack/layer5-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset credit --model credit --split_layer 6  > ../../results/inverse-model-results-20240414/credit_credit/InverseModelAttack/layer6-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset credit --model credit --split_layer 7  > ../../results/inverse-model-results-20240414/credit_credit/InverseModelAttack/layer7-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset credit --model credit --split_layer 8  > ../../results/inverse-model-results-20240414/credit_credit/InverseModelAttack/layer8-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:1 --dataset credit --model credit --split_layer 9  > ../../results/inverse-model-results-20240414/credit_credit/InverseModelAttack/layer9-gpu.log 2>&1 &

