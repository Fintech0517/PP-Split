import torch
import torch.nn as nn
from torchvision import models

from .simba_algo import SimbaDefence
# # from models.image_decoder import Decoder
# # from utils.metrics import MetricLoader


class Disco(SimbaDefence):
    def __init__(self, config, utils) -> None:
        super(Disco, self).__init__(utils)
        self.initialize(config)

    def initialize(self, config):
        self.client_model = self.init_client_model(config)
        self.put_on_gpus()
        self.utils.register_model("client_model", self.client_model)
        self.client_optim = self.init_optim(config, self.client_model)

        img_size = config["proxy_adversary"]["img_size"]
        channels, patch_size = self.generate_patch_params(self.client_model, img_size)
        config["img_size"] = img_size
        config["channels"] = channels
        config["patch_size"] = patch_size

        # Pruner parameters
        self.pruner_model = PruningNetwork(config)
        self.utils.model_on_gpus(self.pruner_model)
        self.utils.register_model("pruner_model", self.pruner_model)
        self.pruner_optim = self.init_optim(config, self.pruner_model)

        # Adversary parameters
        self.proxy_adv_model = Decoder(config)
        self.utils.model_on_gpus(self.proxy_adv_model)
        self.utils.register_model("proxy_adv_model", self.proxy_adv_model)
        self.proxy_adv_optim = self.init_optim(config, self.proxy_adv_model)

        self.loss = MetricLoader().l2
        self.alpha = config["alpha"]
        self.adv_tag = "adv"
        self.utils.logger.register_tag("train/" + self.adv_tag)
        self.utils.logger.register_tag("val/" + self.adv_tag)

    def generate_patch_params(self, model, img_size):
        # 3 because of RGB assumption. 1 is batch size
        img = torch.randn(1, 3, img_size, img_size)
        patch = model(img)
        assert patch.shape[2] == patch.shape[3] # only square images
        return patch.shape[1], patch.shape[2]

    def forward(self, items):
        x = items["x"]
        self.unpruned_z_out = self.client_model(x)
        self.unpruned_z_in = self.unpruned_z_out.detach()
        self.unpruned_z_in.requires_grad = True
        self.pruned_z_out = self.pruner_model(self.unpruned_z_in)

        self.pruned_z_in = self.pruned_z_out.detach()
        self.pruned_z_in.requires_grad = True
        
        x_recons = self.proxy_adv_model(self.pruned_z_in)
        self.adv_loss = self.loss(x_recons, x)

        self.utils.logger.add_entry(self.mode + "/" + self.adv_tag,
                                    self.adv_loss.item())
        pruned_z_in = self.pruned_z_out.detach()
        pruned_z_in.requires_grad = True
        return pruned_z_in

    def backward(self, items):
        """DISCO backprop is a little bit tricky so here is the explanation:
        The main idea is to optimize three entities - Client model, Adversary, Pruner
        Adversary minimizes reconstruction loss (Standard idea)
        Pruner maximizes reconstruction loss but minimizes server loss (Standard idea)
        Client model, however, only minimizes the server loss! This is quite different
        To implement this scheme without having to backprop unnecessarily, we first
        backprop on client_model->pruner model. Then we perform gradient descent on
        the client_model parameters. Now we scale the gradients of pruner by 1 - alpha
        and compute adversary loss by scaling it with (-1*alpha). This way pruner gets
        its correct gradient values pertaining to the min-max optimization objective.
        Finally, we undo this for adversary's gradients and optimize it.
        Note that the order in which things are optimized below really matters.
        """
        server_grads = items["server_grads"]
        self.client_optim.zero_grad()
        self.proxy_adv_optim.zero_grad()
        self.pruner_optim.zero_grad()

        # backprop through rec. loss and update proxy adv
        self.adv_loss.backward()
        self.proxy_adv_optim.step()

        # backprop through the pruner model to pass gradients to the client model
        self.pruned_z_out.backward(server_grads, retain_graph=True)
        self.unpruned_z_out.backward(self.unpruned_z_in.grad)
        self.client_optim.step()
        # Scale the task loss gradient with 1 - alpha
        # Higher the alpha, higher the weight for adv loss would be
        for params in self.pruner_model.parameters():
            params.grad *= (1 - self.alpha)

        # backprop through proxy adv loss
        self.pruned_z_out.backward(-1 * self.alpha * self.pruned_z_in.grad)
        self.pruner_optim.step()


class PruningNetwork(nn.Module):
    """ Nothing special about the pruning model,
    it is a standard resnet predictive model. Might update it later
    """
    def __init__(self, config):
        super(PruningNetwork, self).__init__()
        self.pruning_ratio = config["pruning_ratio"]
        self.pruning_style = config["pruning_style"]

        if self.pruning_style == "learnable":
            self.temp = 1/30
            self.logits = config["channels"]
            self.split_layer = config["split_layer"]
            self.model = models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Flatten(),
                                          nn.Linear(num_ftrs, self.logits))

            self.model = nn.ModuleList(list(self.model.children())[self.split_layer:])
            self.model = nn.Sequential(*self.model)
        elif self.pruning_style == "random":
            # decoy layer to allow creation of optimizer
            self.decoy_layer = nn.Linear(10, 10)

    def prune_channels(self, z, indices=None):
        # Indexing is an inplace operation which creates problem during backprop hence z is cloned first
        z = z.clone()
        z[:, indices] = 0.
        return z

    @staticmethod
    def get_random_channels(x, ratio):
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        channels_to_prune = torch.randperm(x.shape[1], device=x.device)[:num_prunable_channels]
        return channels_to_prune

    def custom_sigmoid(self, x, offset):
        exponent = (x - offset) / self.temp
        #print(exponent)
        #answer = (1 / (1 + torch.exp( - exponent / self.temp)))
        answer = nn.Sigmoid()(exponent)
        #print(answer)
        return answer

    def get_channels_from_network(self, x, ratio):
        fmap_score = self.network_forward(x)
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        threshold_score = torch.sort(fmap_score)[0][:, num_prunable_channels].unsqueeze(1)
        fmap_score = self.custom_sigmoid(fmap_score, threshold_score)
        return fmap_score

    def network_forward(self, x):
        return self.model(x)

    def forward(self, x):
        if self.pruning_style == "random":
            indices = self.get_random_channels(x, self.pruning_ratio)
            x = self.prune_channels(x, indices)
        elif self.pruning_style == "learnable":
            # get score for feature maps
            channel_score = self.get_channels_from_network(x, self.pruning_ratio)
            x = x*channel_score.unsqueeze(-1).unsqueeze(-1)
        return x
