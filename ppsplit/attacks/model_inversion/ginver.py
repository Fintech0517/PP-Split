'''
Author: Ruijun Deng
Date: 2023-12-27 12:50:23
LastEditTime: 2023-12-27 13:45:12
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/attacks/model_inversion/ginver.py
Description: 
'''
from .inverse_model import InverseModelAttack
import torch
import tqdm
import torch.nn as nn


class GinverWhiteBoxAttack(InverseModelAttack):
    def __init__(self, gpu=True, decoder_route=None, data_type=0, inverse_dir=None) -> None:
        super().__init__(gpu, decoder_route, data_type, inverse_dir)
        
    def train_decoder(self,client_net,decoder_net,
                    train_loader,test_loader,
                    epochs,optimizer=None):
        
        # 打印相关信息
        print("----train decoder----")
        print("client_net: ")
        print(client_net)
        print("decoder_net: ")
        print(decoder_net)

        # 网络搬到设备上
        client_net.to(self.device)
        decoder_net.to(self.device)
        

        # loss function 统一采用MSELoss？
        if not optimizer:
            optimizer = torch.optim.SGD(decoder_net.parameters(), 1e-3)
        criterion = nn.MSELoss()
        

        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            # train and update
            epoch_loss = []
            for i, (trn_X, trn_y) in enumerate(tqdm.tqdm(train_loader)):
                trn_X = trn_X.to(self.device)
                batch_loss = []

                optimizer.zero_grad()

                s_raw = client_net(trn_X)
                x_reconstr = decoder_net()
                s_reconstr = client_net(x_reconstr)


                loss = criterion(s_reconstr, s_raw)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))


            print("--- epoch: {0}, train_loss: {1}".format(epoch, epoch_loss))

        # 储存decoder模型
        torch.save(decoder_net, self.decoder_route)
        print("model saved")
        return decoder_net
    


    

class GinverBlaxBoxAttack(InverseModelAttack):
    def __init__(self, gpu=True, decoder_route=None, data_type=0, inverse_dir=None) -> None:
        super().__init__(gpu, decoder_route, data_type, inverse_dir)
        self.step = 0.001
        
    def train_decoder(self,client_net,decoder_net,
                    train_loader,test_loader,
                    epochs,optimizer=None):
        
        # 打印相关信息
        print("----train decoder----")
        print("client_net: ")
        print(client_net)
        print("decoder_net: ")
        print(decoder_net)

        # 网络搬到设备上
        client_net.to(self.device)
        decoder_net.to(self.device)
        

        # loss function 统一采用MSELoss？
        if not optimizer:
            optimizer = torch.optim.SGD(decoder_net.parameters(), 1e-3)
        criterion = nn.MSELoss()
        

        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            # train and update
            epoch_grad = []
            for i, (trn_X, trn_y) in enumerate(tqdm.tqdm(train_loader)):
                trn_X = trn_X.to(self.device)
                batch_grad = []

                optimizer.zero_grad()

                s_raw = client_net(trn_X)
                x_reconstr = decoder_net()

                # 选n=100.
                with torch.no_grad(): # step is the search variance
                    grad = torch.zeros_like(x_reconstr)
                    num = 0
                    for j in range(1, 50):
                        random_direction = torch.randn_like(x_reconstr)
                        
                        # gaussian 对称
                        new_pic1 = x_reconstr + self.step * random_direction
                        new_pic2 = x_reconstr - self.step * random_direction
                        
                        # 黑盒访问
                        target1 = client_net(new_pic1, relu=2)
                        target2 = client_net(new_pic2, relu=2)
                   
                        # 计算loss
                        loss1 = criterion(target1, s_raw)
                        loss2 = criterion(target2, s_raw)
                
                        num = num + 2
                        grad = loss1 * random_direction + grad
                        grad = loss2 * -random_direction + grad
                
                    batch_grad = grad / (num * self.step)
                    # grad = grad.squeeze(dim=0)
                #loss_TV = 3*TV(x_reconstr)
                #loss_TV.backward(retain_graph=True)
                x_reconstr.backward(batch_grad)
                optimizer.step()

                batch_grad.append(grad.item())
            epoch_grad.append(sum(batch_grad) / len(batch_grad))


            print("--- epoch: {0}, train_grad: {1}".format(epoch, epoch_grad))

        # 储存decoder模型
        torch.save(decoder_net, self.decoder_route)
        print("model saved")
        return decoder_net