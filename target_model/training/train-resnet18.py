'''
Author: Ruijun Deng
Date: 2024-08-16 20:50:40
LastEditTime: 2024-08-16 21:21:47
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/training/train-resnet18.py
Description: 
'''
import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from torchmetrics.functional import accuracy as Accuracy # KIWAN: For compatibility with newer version
from schduler import WarmupCosineLR
import pytorch_lightning as pl


import sys
sys.path.append('/home/dengruijun/data/FinTech/PP-Split/')
from target_model.models.PyTorch_CIFAR10.cifar10_models.resnet import resnet18


import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm


class CIFAR10Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.model = resnet18(pretrained=False, split_layer=13, bottleneck_dim=-1, num_classes=10, activation='gelu', pooling='avg')

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
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
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


def main(args):
    seed_everything(0)
    # os.environ["CUDA_VISIBLE_DEVICES"] = 1
    unit_net_route = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/CIFAR10-models/state_dicts/resnet18-drj.pth' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构
    unit_net_dir = '/home/dengruijun/data/FinTech/PP-Split/results/trained_models/CIFAR10-models/' # VGG5-BN+Tanh # 存储的是模型参数，不包括模型结构

    if args.logger == "wandb":
        logger = WandbLogger(name=args.classifier, project="cifar10")
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger("cifar10", name=args.classifier)

    checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", dir_path=unit_net_dir,filename='resnet18-drj.pth')

    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        # logger=logger if not bool(args.dev + args.test_phase) else None,
        logger=logger,
        gpus=-1,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        checkpoint_callback=checkpoint,
        precision=args.precision,
    )

    model = CIFAR10Module(args)
    data = CIFAR10Data(args)

    if bool(args.pretrained): # 训练好了
        # state_dict = os.path.join(
        #     "cifar10_models", "state_dicts", args.classifier + ".pt"
        # )
        state_dict = unit_net_route
        model.model.load_state_dict(torch.load(state_dict))

    if bool(args.test_phase):
        trainer.test(model, data.test_dataloader())
    else:
        trainer.fit(model, data)
        trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/home/dengruijun/data/FinTech/DATASET/image-dataset/cifar10/")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=1) #100
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="3")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()
    main(args)
