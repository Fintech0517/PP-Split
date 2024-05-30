'''
Author: Ruijun Deng
Date: 2024-05-21 13:41:27
LastEditTime: 2024-05-30 09:55:10
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/attacks/membership_inference/ML_Leaks_attack.py
Description: 
'''
# NDSS'19-ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models
# USENIX Security '24-Quantifying Privacy Risks of Prompts in Visual Prompt Learning
import torch
import tqdm
import random
import torch.nn as nn
import torch.nn.functional as F

class MLP_MIA(nn.Module):
    def __init__(self, dim_in):
        super(MLP_MIA, self).__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(self.dim_in, 32)
        self.fc2 = nn.Linear(32, 2)
        # self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(-1,self.dim_in)
        x = F.relu(self.fc1(x))
        out = F.softmax(input=x,dim=1)
        out = self.fc2(out)
        return out

# #Picking the top X probabilities 
def clipDataTopX(dataToClip, top=3):
    sorted_indices = torch.argsort(dataToClip,dim=1,descending=True)[:,:top]
    new_data = torch.gather(dataTClip,1,sorted_indices)
    
	# res = [sorted(s, reverse=True)[0:top] for s in dataToClip ]
	# return np.array(res)
    # print(new_data[0])
    return new_data

class MLLeaksAttack():
    def __init__(self,smashed_data_size,gpu=True) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
        self.attack_model = MLP_MIA(dim_in = smashed_data_size)
    
    def prepare_model_performance(self, model, member_train_loader, nonmember_train_loader, test_loader):
        
        def _model_predictions(model, dataloader, member=None):
            data_list = []
            label = torch.tensor([0]) if member == False else torch.tensor([1])
            model.to(self.device)
            softmax = nn.Softmax(dim=1)
            for i,(x,y) in enumerate(tqdm.tqdm(dataloader)):
                x = x.to(self.device)
                z = model(x)
                z = z.reshape(z.shape[0],-1)
                z = clipDataTopX(dataToClip=z,top=10) # 前十个最大的
                z = softmax(z)
                data_list.append((z.detach().cpu(),label))
            return data_list
        
        # 得到smashed data
        member_data_list = _model_predictions(model,member_train_loader,True)
        nonmember_data_list = _model_predictions(model,nonmember_train_loader,False)

        # 打乱，构造数据集
        data_list = member_data_list+nonmember_data_list  # 拼接数据集
        random.shuffle(data_list) # 打乱

        return data_list
    
    def train_attack_model(self,attack_train_loader,optimizer=None,epochs=100):
        print("----train MIA attack model----")
        print("MIA_attack_net: ")
        print(self.attack_model)

        self.attack_model.to(self.device)

        if not optimizer:
            optimizer = torch.optim.Adam(self.attack_model.parameters(),lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            # train and update
            epoch_loss = []
            for i, (trn_X, trn_y) in enumerate(tqdm.tqdm(attack_train_loader)):
                trn_X = trn_X.to(self.device)
                trn_y = trn_y.to(self.device)
                batch_loss = []

                optimizer.zero_grad()

                out = self.attack_model(trn_X)
                # print(out.shape)
                # print(trn_y.shape)
                # print(trn_y)
                loss = criterion(out, trn_y)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print("--- epoch: {0}, train_loss: {1}".format(epoch, epoch_loss))


    def MIA_test(self, dataloader):
        pred_labels_list = []
        all_correct_list = []
        self.attack_model.to(self.device)
        for i,(x,y) in enumerate(tqdm.tqdm(dataloader)):
            x,y = x.to(self.device),y.to(self.device)
            z = self.attack_model(x)

            # _,pred_label = z.topk(1,1,True,True)
            pred_label = torch.argmax(z,dim=1)
            # print(pred_label)
            correct = pred_label.eq(y.expand_as(pred_label))

            # 存储结果
            if pred_label.dim() > 1:
                pred_labels_list.extend(pred_label.squeeze().tolist())
                all_correct_list.extend(correct.squeeze().tolist())
            else:  # batch_size = 1
                pred_labels_list.append(pred_label.squeeze().tolist())
                all_correct_list.append(correct.squeeze().tolist())

        print("accuracy: {}".format(sum(all_correct_list)/len(all_correct_list)))

    
    
