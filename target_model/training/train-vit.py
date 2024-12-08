'''
Author: Ruijun Deng
Date: 2024-11-15 04:07:49
LastEditTime: 2024-11-21 19:53:25
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/training/train-vit.py
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
import torch.nn.functional as F

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
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from target_model.models.ImageClassification.ViT import vit_b_16 as ViTb_16
from torchvision.models import ViT_B_16_Weights

from torchvision.datasets import ImageNet
from torchvision.datasets import CIFAR10,CIFAR100


class ViT(pl.LightningModule):
    def __init__(self, model_kwargs,logger):
        super().__init__()
        self.save_hyperparameters()
        model_kwargs = vars(model_kwargs)
        self.model = ViTb_16(**model_kwargs)
        # self.example_input_array = next(iter(train_loader))[0]
        # self.log = logger

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) 
        self.log(f"{mode}_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True) 

        return loss,acc
        

    def training_step(self, batch, batch_idx):
        loss,acc = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss,acc = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        loss,acc = self._calculate_loss(batch, mode="test")


class ImageNetData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

    def train_dataloader(self):
        # transform = T.Compose(
        # [
        #     T.ToTensor(), # 数据中的像素值转换到0～1之间
        #     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #     ]
        #     ) # 接近+-1？ 从[0,1] 不是从[0,255]
        transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        # dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform)
        dataset = ImageNet(root=self.hparams.data_dir, split='train', transform=transform)
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
        # transform = T.Compose(
        #     [
        #         T.ToTensor(),
        #         T.Normalize(self.mean, self.std),
        #     ]
        # )
        transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()

        # dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform)
        dataset = ImageNet(root=self.hparams.data_dir, split='val', transform=transform)

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

class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def train_dataloader(self):
        transform = T.Compose(
        [   
            T.Resize(224),
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


def main(args):
    seed_everything(0)
    # 硬写的
    unit_net_route = f'/home/dengruijun/data/project/data/torch_models/hub/checkpoints/vit_b_16-c867db91.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
    unit_net_dir = f'/home/dengruijun/data/project/data/torch_models/hub/checkpoints/' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构

    if args.logger == "wandb":
        # wandb.init(project="cifar10", name=args.classifier)
        logger = WandbLogger(name=f'{args.classifier}-{args.dataset}', project="ViT")
    elif args.logger == "tensorboard":
        # logger = TensorBoardLogger("cifar10", name=args.classifier)
        pass

    checkpoint = ModelCheckpoint(monitor="acc/val", 
                                mode="max", 
                                dirpath=unit_net_dir,
                                filename='{epoch}'+ f'{args.classifier}-drj.pth', 
                                save_top_k=1  # save top k个检查点
                            #  every_n_epochs=1, # 每n个epoch保存一次
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

    if args.dataset == "ImageNet1k":
        data = ImageNetData(args)
        model = ViT(args,logger)
    elif args.dataset == "CIFAR10":
        data = CIFAR10Data(args)
        model = ViT(args,logger)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 10)


    if bool(args.pretrained): # 训练好了
        print("trained models loading ... ")
        # state_dict = os.path.join(
        #     "target_model.models.ImageClassification.Maeng_FIL_nips23", "state_dicts", args.classifier + ".pt"
        # )
        state_dict = unit_net_route
        model.model.load_state_dict(torch.load(state_dict))

    if bool(args.test_phase):
        trainer.test(model, data.test_dataloader(),verbose=True)
    else:
        trainer.fit(model, train_dataloaders=data,verbose=True)
        trainer.test(model, data.test_dataloader(),verbose=True)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/dengruijun/data/project/data/imageNet1k/ok/")
    # parser.add_argument("--data_dir", type=str, default="/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar10/")

    parser.add_argument("--test_phase", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb"])
    parser.add_argument("--pretrained", type=int, default=1, choices=[0, 1]) # 加载与训练的
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=20) #100
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--classifier", type=str, default="vitb_16")
    parser.add_argument("--dataset", type=str, default='ImageNet1k') 
    # parser.add_argument("--dataset", type=str, default='CIFAR10') 
    parser.add_argument('--device', type=int, default=1)

    # CIFAR10
    # parser.add_argument("--num_classes", type=int, default=10)
    # parser.add_argument("--image_size", type=int, default=32)

    # test only
    args = parser.parse_args()
    main(args)