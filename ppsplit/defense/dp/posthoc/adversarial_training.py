from collections import OrderedDict
from distutils.command.config import config
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch import optim
from torch.nn.modules.loss import _Loss
# from models.fc import MnistPredModel, FMnistPredModel, UTKPredModel

# from utils import check_and_create_path

# from embedding import setup_vae
# from embedding import get_dataloader

# from models.gen import AdversaryModelGen
# from .relu_nets import ReLUNet


seed = 2
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# 对比学习
class ContrastiveLoss(_Loss):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def get_mask(self, labels): # 区分1类和0类的样本mask
        labels = labels.unsqueeze(-1).to(dtype=torch.float64)
        class_diff = torch.cdist(labels, labels, p=1.0) # p-norm
        return torch.clamp(class_diff, 0, 1) # 类似于clip，上下界裁剪一下
        #  是 PyTorch 中的一个函数，用于将输入张量中的值限制在指定的范围内。具体来说，它会将张量中小于最小值的元素设置为最小值

    def get_pairwise(self, z): # 单个样本的 norm？自己和自己的欧氏距离
        z = z.view(z.shape[0], -1)
        return torch.cdist(z, z, p=2.0) # 具体来说，它计算的是两个输入张量中每对样本之间的距离。

    def forward(self, z, labels): # 损失函数计算
        '''
        使得0类的z：
        最小化它们的距离 pairwise_dist
        对于1类的z：
        如果距离小于margin，则推远它们
        如果距离已经大于margin，则不再施加损失
        '''
        mask = self.get_mask(labels).to(z.device) # label的1-norm
        pairwise_dist = self.get_pairwise(z) # z的2-norm
        loss = (1 - mask) * pairwise_dist +\
               mask * torch.maximum(torch.tensor(0.).to(z.device), self.margin - pairwise_dist)
        return loss.mean()


class ARL(object):
    def __init__(self, config, client_net, server_net, decoder_net,
                  train_loader, test_loader,
                  device, results_dir,
                  ) -> None:
        # Weighing term between privacy and accuracy, higher alpha has
        # higher weight towards privacy
        self.tag = config.get("tag") or None # 
        self.alpha = config["alpha"] # 隐私和准确性权衡参数,越大越偏向隐私
        self.device = device # 设备

        print('device:', self.device)


        self.noise_reg = config.get('noise_reg') or False # 是否加噪声正则化
        self.siamese_reg = config.get('siamese_reg') or False # 是否加对比学习正则化
        self.epoch = config['epoch']

        # 模型：
        self.obfuscator = client_net.to(self.device) # obf模型
        self.pred_model = server_net.to(self.device)
        self.adv_model = decoder_net.to(self.device)

        # 根据数据集，选择不同的预测模型和对抗模型
        # 3个模型损失函数
        self.pred_loss_fn = torch.nn.CrossEntropyLoss()
        self.rec_loss_fn = torch.nn.MSELoss()

        self.obf_optimizer = optim.Adam(self.obfuscator.parameters())
        self.pred_optimizer = optim.Adam(self.pred_model.parameters())
        self.adv_optimizer = optim.Adam(self.adv_model.parameters())

        # obf的损失函数
        if self.noise_reg:
            self.sigma = config["sigma"]
        if self.siamese_reg:
            self._lambda = config["lambda"]
            self.margin = config["margin"]
            self.setup_siamese_reg()

        # 加载预训练的vae和下载好的数据集
        # self.setup_vae()
        # self.setup_data()
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.min_final_loss = np.inf

        # 设置路径
        # self.assign_paths(self.noise_reg, self.siamese_reg)
        # self.base_dir = "../../20241228-defense/Posthoc/experiments/{}_in_{}_out_{}_alpha_{}".format(self.dset,self.obf_in_size,
                                                                        # self.obf_out_size,
                                                                        # self.alpha)
        self.base_dir = results_dir

        if self.noise_reg:
            self.base_dir += "_noisereg_{}".format(self.sigma)
        if self.siamese_reg: # 应该是foalse？
            self.base_dir += "_siamesereg_{}_{}".format(self.margin, self._lambda)
        self.obf_path = self.base_dir + "/obf.pt"
        self.adv_path = self.base_dir + "/adv.pt"
        self.pred_path = self.base_dir + "/pred.pt"

        print("ARL base_dir:", self.base_dir)

    def setup_siamese_reg(self):
        self.contrastive_loss = ContrastiveLoss(self.margin)

    def setup_path(self):
        # Should be only called when it is required to save updated models
        # check_and_create_path(self.base_dir)
        self.imgs_dir = self.base_dir + "/imgs/"
        if not os.path.exists(self.imgs_dir):
            os.makedirs(self.imgs_dir)

    def save_state(self):
        torch.save(self.obfuscator.state_dict(), self.obf_path)
        torch.save(self.adv_model.state_dict(), self.adv_path)
        torch.save(self.pred_model.state_dict(), self.pred_path)

    # def load_state(self):
    #     self.obfuscator.load_state_dict(torch.load(self.obf_path))
    #     self.adv_model.load_state_dict(torch.load(self.adv_path))
    #     self.pred_model.load_state_dict(torch.load(self.pred_path))
    #     print("sucessfully loaded models")

    def on_cpu(self): # 代码挺简洁的，就3个类
        self.obfuscator.cpu()
        self.adv_model.cpu()
        self.pred_model.cpu()

    def gen_lap_noise(self, z): # 提取laplace noise
        noise_dist = torch.distributions.Laplace(torch.zeros_like(z), self.sigma)
        return noise_dist.sample()

    def train(self): # 训练
        self.pred_model.train()
        train_loss = 0

        for epoch in range(1, self.epoch + 1):
            # Start training
            for batch_idx, (data, labels) in enumerate(self.train_loader): # 对每个batch
                # 代码写得如此简单优雅

                data, labels = data.cuda(), labels.cuda()

                z = data

                # pass it through obfuscator
                z_tilde = self.obfuscator(z)

                # Train predictor model
                if self.noise_reg: # 噪声正则化
                    z_server = z_tilde.detach() + self.gen_lap_noise(z_tilde)
                    preds = self.pred_model(z_server)
                else:
                    preds = self.pred_model(z_tilde.detach())
                pred_loss = self.pred_loss_fn(preds, labels)
                self.pred_optimizer.zero_grad()
                pred_loss.backward()
                self.pred_optimizer.step()

                # Train adversary model
                rec = self.adv_model(z_tilde.detach())
                rec_loss = self.rec_loss_fn(rec, data)
                self.adv_optimizer.zero_grad()
                rec_loss.backward()
                self.adv_optimizer.step()

                # Train obfuscator model by maximizing reconstruction loss
                # and minimizing prediction loss
                # 这个单来啊？再来一遍前向推理，不管之前的pred模型和rec模型
                z_tilde = self.obfuscator(z)
                preds = self.pred_model(z_tilde)
                pred_loss = self.pred_loss_fn(preds, labels)
                rec = self.adv_model(z_tilde)
                rec_loss = self.rec_loss_fn(rec, data)
                total_loss = self.alpha*rec_loss + (1-self.alpha)*pred_loss

                # 对比学习正则化，有什么用？z不是vae已经训练好的了嘛
                if self.siamese_reg: 
                    # 这代码写错了吧，应该是z_tilde
                    # siamese_loss = self.contrastive_loss(z, labels)
                    siamese_loss = self.contrastive_loss(z_tilde, labels)

                self.obf_optimizer.zero_grad()
                if self.siamese_reg:
                    total_loss += self._lambda*siamese_loss
                total_loss.backward()
                self.obf_optimizer.step()
                
                train_loss += total_loss.item()
                
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, pred_loss {:.3f}, rec_loss {:.3f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), total_loss.item() / len(data), pred_loss.item(), rec_loss.item()))
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_loader.dataset)))
            self.test(epoch)

    def test(self,epoch):
        self.pred_model.eval()
        test_pred_loss= 0
        test_rec_loss= 0
        pred_correct = 0
        with torch.no_grad():
            for data, labels  in self.test_loader:
                data, labels = data.cuda(), labels.cuda()

                z = data

                # pass it through obfuscator
                z_tilde = self.obfuscator(z)
                # pass obfuscated z through pred_model
                preds = self.pred_model(z_tilde)
                pred_correct += (preds.argmax(dim=1) == labels).sum()

                # train obfuscator and pred_model
                pred_loss = self.pred_loss_fn(preds, labels)

                # pass obfuscated z to adv_model
                rec = self.adv_model(z_tilde)
                rec_loss = self.rec_loss_fn(rec, data)

                test_pred_loss += pred_loss.item()
                test_rec_loss += rec_loss.item()
            
        test_pred_loss /= len(self.test_loader.dataset)
        test_rec_loss /= len(self.test_loader.dataset)
        final_loss = self.alpha*test_rec_loss + (1-self.alpha)*test_pred_loss
        if final_loss < self.min_final_loss:
            # if self.dset in ["mnist", "fmnist"]:
                # rec_imgs = rec.view(-1, 1, 28, 28)
            # elif self.dset == "utkface":
                # rec_imgs = rec.view(-1, 3, 64, 64)
            save_image(rec,'{}/epoch_{}.png'.format(self.base_dir, epoch))
            self.save_state()
            self.min_final_loss = final_loss
        pred_acc = pred_correct.item() / len(self.test_loader.dataset)
        print('====> Test pred loss: {:.4f}, rec loss {:.4f}, acc {:.2f}'.format(test_pred_loss, test_rec_loss, pred_acc))


if __name__ == '__main__':
    # config = {"alpha": 0.99, # 只给了utk的超参数，可以搞懂一下超参数的意思。
    #           "dset": "utkface", "obf_in": 10, "obf_out": 8,
    #           "noise_reg": True, "sigma": 0.01,
    #           "siamese_reg": True, "margin": 25, "lambda": 1.0,
    #           "tag": "gender"
    #           }
    config = {"alpha": 0.0, # 根据privacy estimation来设计 # 隐私和准确性权衡参数
              "dset": "utkface", "obf_in": 10, "obf_out": 8, # obf 输入和输出维度
              "noise_reg": True, "sigma": 0.05, # 噪声正则化； 噪声强度
              "siamese_reg": False, "margin": 25, "lambda": 1.0, # 是否用对比学习正则化
              "tag": "gender" # 主要用于结果目录的命名
              }
    arl = ARL(config)
    arl.setup_path()
    print("starting training", config)
    for epoch in range(1, 51):
        arl.train()
        arl.test()
