'''
Author: Ruijun Deng
Date: 2024-04-22 11:59:24
LastEditTime: 2024-04-27 22:16:44
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/quantification/rep_reading/rep_reader.py
Description: 
'''

from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from itertools import islice
import torch

def project_onto_direction(H, direction): # ？ 
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    # Calculate the magnitude of the direction vector
    # Ensure H and direction are on the same device (CPU or GPU)
    if type(H) != torch.Tensor:
        H = torch.Tensor(H).cuda()
    if type(direction) != torch.Tensor:
        direction = torch.Tensor(direction)
        direction = direction.to(H.device)
    mag = torch.norm(direction)
    assert not torch.isinf(mag).any()
    # Calculate the projection
    projection = H.matmul(direction) / mag
    return projection

def recenter(x, mean=None): # 中心化
    x = torch.Tensor(x,device=x.device)
    if mean is None:
        mean = torch.mean(x,axis=0,keepdims=True)
    else:
        mean = torch.Tensor(mean, device=mean.device)
    return x - mean



class Rep_Reader:
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __init__(self) -> None:
        self.direction_method = None
        self.directions = None # directions accessible via directions[layer][component_index]
        self.direction_signs = None # direction of high concept scores (mapping min/max to high/low)

    @abstractmethod
    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        """Get concept directions for each hidden layer of the model
        
        Args:
            model: Model to get directions for
            tokenizer: Tokenizer to use
            hidden_states: Hidden states of the model on the training data (per layer)
            hidden_layers: Layers to consider

        Returns:
            directions: A dict mapping layers to direction arrays (n_components, hidden_size)
        """
        pass 



class PCA_Reader(Rep_Reader):
    def __init__(self,n_components=1) -> None:
        self.n_components = n_components # 几个主成份
        self.H_means = None
        self.direction = None

    def get_rep_direction(self, hidden_states):
        hidden_states_mean = torch.mean(hidden_states, axis=0,keepdims=True)
        self.H_means = hidden_states_mean # 储存一下均值
        hidden_states_recentered = recenter(hidden_states, mean=hidden_states_mean)

        hidden_states_stack = np.vstack(hidden_states_recentered.cpu().detach()) # 多个batch拼接起来
        pca_model = PCA(n_components=self.n_components, whiten=False).fit(hidden_states_stack)
        direction = pca_model.components_

        self.direction = direction
        return direction
    
    def get_sign(self, hidden_states,train_labels):
        hidden_states_recentered = recenter(hidden_states, mean=self.H_means)
        layer_signs = np.zeros(self.n_components)

        for component_index in range(self.n_components): # 对每个主成份遍历
            # 投影到direction上
            transformed_hidden_states = project_onto_direction(hidden_states_recentered, self.direction[component_index]).cpu()

            pca_outputs_comp = [list(islice(transformed_hidden_states, sum(len(c) for c in train_labels[:i]), sum(len(c) for c in train_labels[:i+1]))) for i in range(len(train_labels))]

            # We do elements instead of argmin/max because sometimes we pad random choices in training
            pca_outputs_min = np.mean([o[train_labels[i].index(1)] == min(o) for i, o in enumerate(pca_outputs_comp)])
            pca_outputs_max = np.mean([o[train_labels[i].index(1)] == max(o) for i, o in enumerate(pca_outputs_comp)])

            layer_signs[component_index] = np.sign(np.mean(pca_outputs_max) - np.mean(pca_outputs_min))
            if layer_signs[component_index] == 0:
                layer_signs[component_index] = 1 # default to positive in case of tie

        return layer_signs