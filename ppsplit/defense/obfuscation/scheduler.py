# 防御算法
# from complex_nn import ComplexNN
from .disco import Disco
from .split_inference import SplitInference
from .nopeek import NoPeek
from .uniform_noise import UniformNoise
from .siamese_embedding import SiameseEmbedding
from .pca_embedding import PCAEmbedding
from .deepobfuscator import DeepObfuscator
from .pan import PAN
from .gaussian_blur import GaussianBlur
from .linear_correlation import LinearCorrelation
from .cloak import Cloak
from .shredder import Shredder
from .aioi import AIOI

# 攻击算法
from .supervised_decoder import SupervisedDecoder
from .input_optimization import InputOptimization
from .input_model_optimization import InputModelOptimization
from .discriminator_attacker import DiscriminatorAttack


# 其他
# from data.loaders import DataLoader
# from models.model_zoo import Model
# from utils.utils import Utils
# from utils.config_utils import config_loader
from os import path
import tqdm

import torch, random

import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import wandb

# args = parser.parse_args() 和 准备config
# scheduler = Scheduler(config)
# scheduler.initialize()
# client_model,server_model = scheduler.run_job()


def load_algo(config, client_model=None, dataloader=None, run=None):
    method = config["method"]
    # Defenses
    if method == "split_inference":
        algo = SplitInference(config, client_model=client_model, run=run)
    elif method == "nopeek": # ok
        algo = NoPeek(config, client_model=client_model, run=run)
    elif method == "uniform_noise": # ok
        algo = UniformNoise(config, client_model=client_model, run=run)
    elif method == "siamese_embedding":
        algo = SiameseEmbedding(config, client_model=client_model, run=run)
    elif method == "complex_nn":
        algo = ComplexNN(config, client_model=client_model, run=run)
    elif method == "pca_embedding":
        algo = PCAEmbedding(config, client_model=client_model, run=run)
    elif method == "deep_obfuscator":
        algo = DeepObfuscator(config, client_model=client_model, run=run)
    elif method == "pan":
        algo = PAN(config, client_model=client_model, run=run)
    elif method == "cloak":
        algo = Cloak(config, client_model=client_model, run=run)
    elif method == "shredder": # ok
        algo = Shredder(config, client_model=client_model, run=run)
    elif method == "aioi":
        algo = AIOI(config, client_model=client_model, run=run)
    elif method == "gaussian_blur":
        algo = GaussianBlur(config, client_model=client_model, run=run)
    elif method == "linear_correlation":
        algo = LinearCorrelation(config, client_model=client_model, run=run)
    elif method == "disco":
        algo = Disco(config, client_model=client_model, run=run)
    
    # Attacks
    elif method == "supervised_decoder":
        item = next(iter(dataloader))
        z = item["z"]
        config["adversary"]["channels"] = z.shape[1]
        config["adversary"]["patch_size"] = z.shape[2]
        algo = SupervisedDecoder(config["adversary"])
    elif method == "discriminator":
        item = next(iter(dataloader))
        z = item["z"]
        config["adversary"]["channels"] = z.shape[1]
        config["adversary"]["patch_size"] = z.shape[2]
        algo = DiscriminatorAttack(config["adversary"])
    elif method == "input_optimization":
        config["adversary"]["target_model_path"] = path.join(config["experiments_folder"], config["challenge_experiment"], "saved_models", "client_model.pt")
        config["adversary"]["target_model_config"] = path.join(config["experiments_folder"], config["challenge_experiment"], "configs", f"{config['adversary']['target_model']}.json")
        algo = InputOptimization(config["adversary"])
    elif method == "input_model_optimization":
        config["adversary"]["target_model_path"] = path.join(config["experiments_folder"], config["challenge_experiment"], "saved_models", "client_model.pt")
        config["adversary"]["target_model_config"] = path.join(config["experiments_folder"], config["challenge_experiment"], "configs", f"{config['adversary']['target_model']}.json")
        algo = InputModelOptimization(config["adversary"])
    else:
        raise NotImplementedError("Unknown algorithm {}".format(config["method"]))
        # exit()

    return algo


class Scheduler():
    def __init__(self,config) -> None:
        # log
        wandb_name = f'{config["defense"]["method"]}_{config["general"]["dataset"]}_\
            {config["general"]["model"]}_{config["general"]["split_layer"]}'
        self.wandb = wandb.init(project="defense",
                    name=wandb_name,
                    dir = config["defense"]['results_dir'],
                    config=config)

        config = config['defense']
        self.config = config
        self.config['experiment_type'] = 'defense'
        self.config['seed'] = config.get('seed') or 1
        self.config['logits'] = config.get('logits') or 'softmax'
        self.config['optimizer'] = config.get('optimizer') or 'adam'
        self.config['loss'] = config.get('loss') or 'cross_entropy'
        self.device = torch.device(config["device"])

        # self.wandb = wandb.init(mode="disabled")
        # wandb.define_metric("batch_step")
        # wandb.define_metric("epoch_step")

    def initialize(self, 
                    train_loader=None, 
                    test_loader=None, 
                    client_model = None, 
                    server_model = None,
                    ) -> None:
        assert self.config is not None, "Config should be set when initializing"

        # set seeds
        seed = self.config["seed"]
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

        # dataloader
        self.trainloader = train_loader
        self.testloader = test_loader

        # 单纯 client model 加载
        self.client_model = client_model

        # server side model
        self.server_model = server_model
        if self.server_model:
            print("server model device:",next(self.server_model.parameters()).device)
        if self.client_model:
            print("client model device:",next(self.client_model.parameters()).device)

        # optimizer
        if self.config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(self.server_model.parameters(), lr=self.config["server"]["lr"], weight_decay = 0.01)
            # self.optimizer = torch.optim.Adam(self.server_model.parameters())
        else:
            raise NotImplementedError("Unknown optimizer {}".format(config["optimizer"]))

        # loss function
        if self.config["loss"] == "cross_entropy":
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Unknown loss function {}".format(config["loss"]))

        # 处理得到logits的函数
        if self.config["logits"] == "softmax":
            self.prob_fun = torch.nn.Softmax(dim=1)
        elif config["logits"] == "sigmoid":
            self.prob_fun = torch.nn.Sigmoid(dim=1)
        else:
            raise NotImplementedError("Unknown logits function {}".format(config["logits"]))

        # 单纯算法加载
        algo_config = self.config["client"]
        algo_config["method"] = self.config["method"]
        print("algo_config",algo_config)
        self.algo = load_algo(algo_config, self.client_model, self.trainloader, self.wandb)


    def run_job(self):
        # self.utils.logger.log_console("Starting the job")
        exp_type = self.config["experiment_type"]
        if exp_type == "challenge":
            self.run_challenge_job()
        elif exp_type == "defense": # 应该只会调用这一个。
            return self.run_defense_job()
        elif exp_type == "attack":
            self.run_attack_job()
        else:
            print("unknown experiment type")

    def run_defense_job(self): #防御
        for epoch in range(self.config["total_epochs"]):
            print("Epoch: ",epoch)
            self.defense_train(epoch) # 设计学好防御后的模型
            self.defense_test(epoch) # 测试防御后的模型
            self.epoch_summary(epoch) # 返回client和server的模型
        # self.generate_challenge()
        return self.client_model,self.server_model

    def run_attack_job(self):
        print("running attack job")
        for epoch in range(self.config["total_epochs"]):
            self.attack_train()
            self.attack_test()
            self.epoch_summary()

    def run_challenge_job(self):
        self.utils.load_saved_models()
        self.generate_challenge()

    # def defense_train(self) -> None: # 面向DRA的
    #     self.algo.train()
    #     self.model.train()
    #     for _, sample in enumerate(self.trainloader):
    #         items = self.utils.get_data(sample)
    #         z = self.algo.forward(items)
    #         data = self.model.forward(z)
    #         items["decoder_grads"] = self.algo.infer(data,items["pred_lbls"])
    #         items["server_grads"] = self.model.backward(items["pred_lbls"],items["decoder_grads"])
    #         self.algo.backward(items)

    def defense_train(self, epoch) -> None: # 面向DRA的
        self.algo.train()
        self.server_model.train()

        epoch_loss = 0.0
        NBatch = len(self.trainloader)

        for i, (features, labels) in enumerate(tqdm.tqdm(self.trainloader)):
            features, labels = features.to(self.device), labels.to(self.device)

            # 查看模型&参数的设备：
            # print("features device:", features.device)
            # print("labels device:",labels.device)

            # 前向推理
            self.optimizer.zero_grad()

            # client model推理
            z = self.algo.forward(features) 

            # server 前向推理和反向传播
            z.clone().detach().requires_grad_(True)
            y = self.server_model.forward(z)
            # prob = self.prob_fun(y)
            loss = self.loss(y, labels)
            loss.backward()
            self.optimizer.step()
            
            # 回传梯度
            back_grads = z.grad

            # client端模型反向传播并更新参数
            self.algo.backward(back_grads)

            epoch_loss = epoch_loss + loss.item() / NBatch
            self.wandb.log({"train_loss_batch":loss.item()})

        print("tran loss:",epoch_loss)

        self.wandb.log({"train_loss_epoch":epoch_loss})

    def defense_test(self,epoch) -> None:
        self.algo.eval()
        self.server_model.eval()
        # self.client_model.eval()

        epoch_loss = 0.0
        epoch_acc = 0.0
        NBatch = len(self.testloader)

        for i, (features, labels) in enumerate(tqdm.tqdm(self.testloader)):
            features, labels = features.to(self.device), labels.to(self.device)
            
            z = self.algo.forward(features) 
            y = self.server_model.forward(z)
            loss = self.loss(y, labels)

            epoch_loss = epoch_loss + loss.item()/NBatch
            self.wandb.log({"val_loss_batch":loss.item()})

            # accuracy
            prob = self.prob_fun(y)
            pred = np.argmax(prob.cpu().detach().numpy(), axis = 1)
            groundTruth = labels.cpu().detach().numpy()

            acc = np.mean(pred == groundTruth)
            epoch_acc += acc / NBatch


            # self.algo.infer(y,labels)
            # self.model.compute_loss(data,items["pred_lbls"])

        print("val loss:",epoch_loss)
        print("val acc:",epoch_acc)
        self.wandb.log({"val_loss_epoch":epoch_loss, "val_acc_epoch":epoch_acc})

    def attack_train(self) -> None:
        if self.config.get("no_train"):
            return
        self.algo.train()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            self.algo.backward(items)

    def attack_test(self):
        self.algo.eval()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            self.utils.save_images(z,items["filename"])

    def epoch_summary(self,epoch): # TODO:
        # self.utils.logger.flush_epoch()
        # self.utils.save_models()
        # 打印log

        pass

    def generate_challenge(self) -> None:
        challenge_dir = self.utils.make_challenge_dir(self.config["results_path"])
        self.algo.eval()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            self.utils.save_data(z, items["filename"], challenge_dir)
