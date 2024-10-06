'''
Author: Ruijun Deng
Date: 2024-08-16 20:50:40
LastEditTime: 2024-10-06 03:53:50
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/training/train-resnet18.py
Description: 
'''
import os
from argparse import ArgumentParser
import wandb

import torch
# torch.use_deterministic_algorithms(True)
import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10,CIFAR100
from tqdm import tqdm
torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler._LRScheduler # debug for torch<=1.13.5


import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# from torchmetrics.functional import accuracy as Accuracy # KIWAN: For compatibility with newer version
from torchmetrics.classification import  Accuracy # KIWAN: For compatibility with newer version



import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from target_model.models.ResNet import resnet18,resnet34,resnet50


class CIFAR10Module(pl.LightningModule):
    def __init__(self, hparams, total_steps):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        

        # self.model = resnet18(pretrained=False, split_layer=13, bottleneck_dim=-1, num_classes=10, activation='relu', pooling='max')
        # self.model = resnet18(pretrained=False, split_layer=13, bottleneck_dim=-1, num_classes=10, activation='gelu', pooling='avg')
        if 'resnet18' in args.classifier:
            self.model = resnet18(pretrained=False, split_layer=-1, bottleneck_dim=-1, num_classes=10, activation='gelu', pooling='avg')
        elif 'resnet34' in args.classifier:
            self.model = resnet34(pretrained=False, split_layer=-1, bottleneck_dim=-1, num_classes=10, activation='gelu', pooling='avg')
        elif 'resnet50' in args.classifier:
            self.model = resnet50(pretrained=False, split_layer=-1, bottleneck_dim=-1, num_classes=10, activation='gelu', pooling='avg')
        else:
            print('nononononono')
            raise ValueError("Invalid classifier")
        
        self.total_steps = total_steps

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    # def configure_optimizers(self):
    #     # optimizer = torch.optim.Adam(self.model.parameters())
    #     optimizer = torch.optim.SGD(
    #         self.model.parameters(),
    #         lr=self.hparams.learning_rate,
    #         weight_decay=self.hparams.weight_decay,
    #         momentum=0.9,
    #         nesterov=True,
    #     )
    #     # total_steps = self.hparams.max_epochs * len(self.train_dataloader())
    #     total_steps = self.total_steps
    #     scheduler = {
    #         "scheduler": WarmupCosineLR(
    #             optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
    #         ),
    #         "interval": "step",
    #         "name": "learning_rate",
    #     }
    #     return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        return [optimizer]

class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def train_dataloader(self):
        transform = T.Compose(
        [
            T.ToTensor(), # 数据中的像素值转换到0～1之间
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
            ) # 接近+-1？ 从[0,1] 不是从[0,255]
            
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class CIFAR100Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def train_dataloader(self):
        transform = T.Compose(
        [
            T.ToTensor(), # 数据中的像素值转换到0～1之间
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
            ) # 接近+-1？ 从[0,1] 不是从[0,255]
            
        dataset = CIFAR100(root=self.hparams.data_dir, train=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR100(root=self.hparams.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

class CIFAR100Module(pl.LightningModule):
    def __init__(self, hparams, total_steps):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=100)
        
        if 'resnet18' in args.classifier:
            self.model = resnet18(pretrained=False, split_layer=13, bottleneck_dim=-1, num_classes=100, activation='gelu', pooling='avg')
        elif 'resnet34' in args.classifier:
            self.model = resnet34(pretrained=False, split_layer=-1, bottleneck_dim=-1, num_classes=100, activation='gelu', pooling='avg')
        elif 'resnet50' in args.classifier:
            self.model = resnet50(pretrained=False, split_layer=-1, bottleneck_dim=-1, num_classes=100, activation='gelu', pooling='avg')
        else:
            raise ValueError("Invalid classifier")
        # self.model = resnet18(pretrained=False, split_layer=13, bottleneck_dim=-1, num_classes=100, activation='relu', pooling='max')

        self.total_steps = total_steps

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.model.parameters())
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        # total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        total_steps = self.total_steps
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.model.parameters())
    #     return [optimizer]


class WarmupCosineLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.
    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.
    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Maximum number of iterations for linear warmup
        max_epochs (int): Maximum number of iterations
        warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        last_epoch: int = -1,
    ) -> None:

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]


def main(args):
    seed_everything(0)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    unit_net_route = f'/home/dengruijun/data/FinTech/PP-Split/results/trained_models/ResNet/{args.classifier}/{args.dataset}/{args.classifier}-drj.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
    unit_net_dir = f'/home/dengruijun/data/FinTech/PP-Split/results/trained_models/ResNet/{args.classifier}/{args.dataset}/' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构

    if args.logger == "wandb":
        # wandb.init(project="cifar10", name=args.classifier)
        logger = WandbLogger(name=f'{args.classifier}-{args.dataset}', project="ResNet")
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger("cifar10", name=args.classifier)

    checkpoint = ModelCheckpoint(monitor="acc/val", 
                                 mode="max", 
                                 dirpath=unit_net_dir,
                                 filename='{epoch}'+ f'{args.classifier}-drj.pth', 
                                 save_top_k=1 #保存每个检查点
                                 ) 

    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        # logger=logger if not bool(args.dev + args.test_phase) else None,
        logger=logger,
        devices=args.device,
        deterministic=True,
        # weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint],
        precision=args.precision,
    )

    if args.dataset == "CIFAR10":
        data = CIFAR10Data(args)
        total_steps = len(data.train_dataloader())*args.max_epochs
        model = CIFAR10Module(args,total_steps=total_steps)
    elif args.dataset == "CIFAR100":
        data = CIFAR100Data(args)
        total_steps = len(data.train_dataloader())*args.max_epochs
        model = CIFAR100Module(args,total_steps=total_steps)


    if bool(args.pretrained): # 训练好了
        print("trained models loading ... ")
        # state_dict = os.path.join(
        #     "cifar10_models", "state_dicts", args.classifier + ".pt"
        # )
        state_dict = unit_net_route
        model.model.load_state_dict(torch.load(state_dict))

    if bool(args.test_phase):
        trainer.test(model, data.test_dataloader())
    else:
        trainer.fit(model, train_dataloaders=data)
        trainer.test(model, data.test_dataloader())

# cifar100+resnet18
# python3 train.py --dataset cifar100 --model resnet18 --activation gelu 
# --bs 128 --lr 0.1 --weight-decay 5e-4 --standardize --nesterov --test-fil 
# --pooling avg --seed 123 --split-layer 7 --bottleneck-dim 2 --jvp-parallelism 
# 100 --save-model --jacloss-alpha 0.1

if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    # parser.add_argument("--data_dir", type=str, default="/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar100/")
    parser.add_argument("--data_dir", type=str, default="/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar10/")
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb"])

    # TRAINER args
    # parser.add_argument("--classifier", type=str, default="resnet50")
    # parser.add_argument("--classifier", type=str, default="resnet34")
    # parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--classifier", type=str, default="resnet18_2narrow")
    # parser.add_argument("--classifier", type=str, default="resnet18_wide")
    # parser.add_argument("--classifier", type=str, default="resnet18_narrow")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1]) # 加载与训练的

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=20) #100
    parser.add_argument("--num_workers", type=int, default=0)
    # parser.add_argument("--gpu_id", type=str, default="1")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # parser.add_argument("--dataset", type=str, default='CIFAR100') # CIFAR10
    parser.add_argument("--dataset", type=str, default='CIFAR10') # CIFAR10
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    main(args)

# nohup python -u train-resnet18.py > resnet18.log 2>&1 & [1] 2928410