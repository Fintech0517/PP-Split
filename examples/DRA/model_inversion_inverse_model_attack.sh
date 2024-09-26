
# Resnet18  
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 2 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer2-gpu.log 2>&1 &
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 3 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer3-gpu.log 2>&1 &
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 5 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer5-gpu.log 2>&1 &
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 7 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer7-gpu.log 2>&1 &
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 9 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer9-gpu.log 2>&1 &
nohup python -u model_inversion_inverse_model_attack.py  --device cuda:1 --dataset CIFAR10 --model ResNet18 --split_layer 11 > ../../results/inverse-model-results-20240414/Resnet18/InverseModelAttack-defense0.1/InverseNetwork-layer11-gpu.log 2>&1 &



# VGG5
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 0  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer0-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 1  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer1-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 2  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer2-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 3  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer3-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 4  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer4-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 5  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer5-gpu.log 2>&1 &
# nohup python -u model_inversion_inverse_model_attack.py --device cuda:0 --dataset CIFAR10 --model VGG5 --split_layer 6  > ../../results/inverse-model-results-20240414/VGG5/InverseModelAttack-defense0.1/InverseNetwork-layer6-gpu.log 2>&1 &
