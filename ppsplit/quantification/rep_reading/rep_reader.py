'''
Author: Ruijun Deng
Date: 2024-04-22 11:59:24
LastEditTime: 2024-06-01 22:18:12
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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import tqdm

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
    # x = torch.stack(x).squeeze()
    # x = torch.as_tensor(x,device=x.device)
    if mean is None:
        mean = torch.mean(x,axis=0,keepdims=True)
    else:
        mean = torch.as_tensor(mean, device=mean.device)
    # print('in recenter: ')
    # print(x.shape)
    # print(mean.shape)
    return x - mean

#  Picking the top X probabilities 
def clipDataTopX(dataToClip, top=3):
    sorted_indices = torch.argsort(dataToClip,dim=1,descending=True)[:,:top]
    new_data = torch.gather(dataToClip,1,sorted_indices)
    return new_data


def clipDataFirstX(dataToClip, top=3):
    new_data = dataToClip[:,:top]
    return new_data


# class Rep_Reader:
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



class PCA_Reader(object):
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

        # 改一下格式
        if type(direction) == np.ndarray:
            direction = direction.astype(np.float32)

        self.direction = direction

        return direction
    
    def get_sign(self, hidden_states,train_labels):
        hidden_states_recentered = recenter(hidden_states, mean=self.H_means)
        layer_signs = np.zeros(self.n_components)

        for component_index in range(self.n_components): # 对每个主成份遍历
            # 投影到direction上
            transformed_hidden_states = project_onto_direction(hidden_states_recentered, self.direction[0]).cpu()

            pca_outputs_comp = [list(islice(transformed_hidden_states, sum(len(c) for c in train_labels[:i]), sum(len(c) for c in train_labels[:i+1]))) for i in range(len(train_labels))]

            # We do elements instead of argmin/max because sometimes we pad random choices in training
            # print(train_labels[0][0].shape)
            # print(pca_outputs_comp[0])
            pca_outputs_min = np.mean([o[train_labels[i].index(1)] == min(o) for i, o in enumerate(pca_outputs_comp)])
            pca_outputs_max = np.mean([o[train_labels[i].index(1)] == max(o) for i, o in enumerate(pca_outputs_comp)])

            layer_signs[component_index] = np.sign(np.mean(pca_outputs_max) - np.mean(pca_outputs_min))
            if layer_signs[component_index] == 0:
                layer_signs[component_index] = 1 # default to positive in case of tie

        self.direction_signs = layer_signs
        return layer_signs
    
    

class RepE:
    def __init__(self,n_components=1) -> None:
        self.reader = PCA_Reader(n_components=n_components) # 要的是numpy数据？可以要tensor数据

    def collect_neural_activity(self, train_loader,client_net):

        # 收集所有smashed data
        train_smashed_data_list = []
        device = next(client_net.parameters()).device
        for j, data in enumerate(tqdm.tqdm(train_loader)): # 对trainloader遍历
            # print("data: ", len(data))
            features=data.to(device)
            
            with torch.no_grad():
                pred = client_net(features)
                train_smashed_data_list.append(pred)

        train_smashed_data_list=torch.stack(train_smashed_data_list).squeeze()
        train_smashed_data_list=train_smashed_data_list.reshape(train_smashed_data_list.shape[0],-1)
        # train_smashed_data_list = clipDataTopX(train_smashed_data_list,top=1)
        train_smashed_data_list = clipDataFirstX(train_smashed_data_list,top=20)
        
        # 相对距离
        diff_data = train_smashed_data_list[::2] - train_smashed_data_list[1::2] # np.array
        # print("diff_data.shape: ", diff_data.shape)

        return train_smashed_data_list,diff_data
    
    def construct_linear_model(self,diff_data,train_smashed_data_list,train_labels):
        
        # diff_data = diff_data.reshape(diff_data.shape[0],-1)
        directions = self.reader.get_rep_direction(diff_data)
        signs = self.reader.get_sign(hidden_states=train_smashed_data_list,train_labels=train_labels)

        return directions,signs

    def _quantify_MIA_acc(self, hidden_states, test_labels):
        hidden_states_recentered = recenter(hidden_states, mean=self.reader.H_means)
        transformed_hidden_states = project_onto_direction(hidden_states_recentered, self.reader.direction[0]).cpu()
        unflattened_smashed_data =  [list(islice(transformed_hidden_states, sum(len(c) for c in test_labels[:i]), sum(len(c) for c in test_labels[:i+1]))) for i in range(len(test_labels))]
        print(unflattened_smashed_data[0])
        eval_func = np.argmin if self.reader.direction_signs==-1 else np.argmax
        cors = np.mean([test_labels[i].index(1) == eval_func(H) for i, H in enumerate(unflattened_smashed_data)])

        return cors

    def get_transformed_data(self,test_loader,client_net):
        test_smashed_data_list = []
        device = next(client_net.parameters()).device
        for j, data in enumerate(tqdm.tqdm(test_loader)): # 对trainloader遍历
            features=data[0].to(device)
            with torch.no_grad():
                pred = client_net(features)
                test_smashed_data_list.append(pred)

        test_smashed_data_list=torch.stack(test_smashed_data_list).squeeze()
        test_smashed_data_list=test_smashed_data_list.reshape(test_smashed_data_list.shape[0],-1)
        # test_smashed_data_list = clipDataTopX(test_smashed_data_list,top=1)·
        # 调整smashed data
        test_smashed_data_list = clipDataFirstX(test_smashed_data_list,top=20)
        hidden_states_recentered = recenter(test_smashed_data_list, mean=self.reader.H_means)
        transformed_hidden_states = project_onto_direction(hidden_states_recentered, self.reader.direction[0]).cpu()

        return transformed_hidden_states

    def eval_MIA_acc(self,test_loader,test_labels,client_net):
        test_smashed_data_list = []
        device = next(client_net.parameters()).device
        for j, data in enumerate(tqdm.tqdm(test_loader)): # 对trainloader遍历
            features=data.to(device)
            with torch.no_grad():
                pred = client_net(features)
                test_smashed_data_list.append(pred)

        test_smashed_data_list=torch.stack(test_smashed_data_list).squeeze()
        test_smashed_data_list=test_smashed_data_list.reshape(test_smashed_data_list.shape[0],-1)
        # test_smashed_data_list = clipDataTopX(test_smashed_data_list,top=1)·
        # 调整smashed data
        test_smashed_data_list = clipDataFirstX(test_smashed_data_list,top=20)

        # MIA attack的
        acc = self._quantify_MIA_acc(hidden_states=test_smashed_data_list,test_labels=test_labels)
        
        return acc
    
    def eval_DRA_acc(self, test_loader, client_net):
        pass

