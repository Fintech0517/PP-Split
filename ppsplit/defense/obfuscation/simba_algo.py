import torch
import torch.nn as nn
from torchvision import models
from abc import abstractmethod
# from models.Unet import StochasticUNet
# from models.Xception import Xception


class SimbaBase(nn.Module):
    def __init__(self,):
        super(SimbaBase, self).__init__()
        # self.utils = utils
        self.detached = True
        self.logs_enabled = True

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def log_metrics(self):
        pass

    def train(self):
        self.mode = "train"
        self.client_model.train()
        # for _, model in self.utils.model_registry.items():
        #     model.train()

    def eval(self):
        self.mode = "val"
        # for _, model in self.utils.model_registry.items():
        #     model.eval()
        self.client_model.eval()
    
    def set_detached(self, detached):
        self.detached = detached

    def enable_logs(self, enabled):
        self.logs_enabled = enabled

    def init_client_model(self, config):
        pretrained = config["pretrained"]
        if config["model_name"] == "resnet18":
                model = models.resnet18(pretrained=pretrained)                
        elif config["model_name"] == "resnet34":
                model = models.resnet34(pretrained=pretrained)                
        elif config["model_name"] == "resnet50":
                model = models.resnet50(pretrained=pretrained)
        elif config['model_name'] == "vgg16":
                model = models.vgg16(pretrained=pretrained)
        elif config['model_name'] == "stochasticunet":
                model = StochasticUNet()
                return model
        else:
            print("can't find client model")
            exit()  

        model = nn.Sequential(*nn.ModuleList(list(model.children())[:config["split_layer"]]))
            

        return model

    def init_optim(self, config, model):
        if config["optimizer"] == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay = 0.01)
        else:
            print("Unknown optimizer {}".format(config["optimizer"]))

        return optim

    def put_on_gpus(self):
        # self.client_model = self.utils.model_on_gpus(self.client_model)
        self.client_model = self.client_model.to(self.config['device'])

    def infer(self,data,labels):
        pass


class SimbaDefence(SimbaBase):
    def __init__(self):
        super(SimbaDefence, self).__init__()

    def init_client_model(self, client_model):
        self.client_model = client_model

    def put_on_gpus(self):
        # self.client_model = self.utils.model_on_gpus(self.client_model)
        self.client_model = self.client_model.to(self.device)
