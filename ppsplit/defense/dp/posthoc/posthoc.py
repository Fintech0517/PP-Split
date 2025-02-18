
import time
import numpy as np
import torch
import random
from lipmip.hyperbox import Hyperbox
from lipmip.lipMIP import LipMIP
# from embedding import get_dataloader
from torch.distributions import Laplace
from adversarial_training import ARL

import matplotlib.pyplot as plt



seed = 2
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

class PosthocDefense():
    def __init__(self, config) -> None:
        arl_config = config["arl_config"]
        eval_config = config["eval_config"]

        self.arl_obj = ARL(arl_config, client_net, server_net, decoder_net, train_loader, test_loader)

        self.evaluation = Evaluation(self.arl_obj, eval_config)
        self.evaluation.create_logger(eval_config)

        eval.logger.info(arl_config)
        eval.logger.info(eval_config)

    def defense_train():
        mean_lc, std_lc = eval.test_local_sens()
        eval.proposed_bound = mean_lc + 3*std_lc
        return eval.proposed_bound

    def defense_test():
        eval.test_ptr()
        return


class Evaluation():
    def __init__(self, arl_obj: ARL, config, train_loader, test_loader) -> None:
        # main函数中输入的几个参数
        self.epsilon = config["epsilon"]
        self.delta = config["delta"]
        self.radius = config["radius"]
        self.max_upper_bound_radius = config["max_upper_bound"]
        self.proposed_bound = config["proposed_bound"]
        self.eval_size = config["eval_size"]
        self.arl_obj = arl_obj

        self.max_time = 20 # seconds
        self.margin = 0.001

        # For now we are using center of 0 and l_inf norm of size 1 around the center to define the output space
        self.out_domain = 'l1Ball1'

        # 数据集和模型初步操作 和 路径设置
        self.dset = arl_obj.dset
        self.train_loader, self.test_loader = train_loader, test_loader
        # self.setup_data()
        self.arl_obj.on_cpu() # move the models to cpu
        self.base_dir = "../../20241228-defense/Posthoc/experiments/"


    def create_logger(self, arl_config): # log输出
        import logging
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    # def setup_data(self):
    #     self.train_loader, self.test_loader = get_dataloader(self.dset, batch_size=1)

    def get_noise(self, z_tilde, sens):
        variance = sens / self.epsilon
        return Laplace(torch.zeros_like(z_tilde), variance).sample()

    def test_ptr(self): # PTR机制
        '''
        - 计算Lipschitz常数
        - 检查是否满足proposed_bound
        - 如果满足，添加噪声并进行预测
        - 计算有噪声和无噪声情况下的预测准确率
        '''
        self.arl_obj.vae.eval()
        self.arl_obj.pred_model.eval()
        sample_size, unanswered_samples = 0, 0
        noisy_pred_correct, noiseless_pred_correct = 0, 0

        for batch_idx, (data, labels, _) in enumerate(self.test_loader): # 对每个batch
            data = data.cuda()
            # mu, log_var = self.arl_obj.vae.encode(data.view(-1, 784)) # 28*28
            # 得到原始z
            mu, log_var = self.arl_obj.vae.encode(data) # utkface
            center = self.arl_obj.vae.reparametrize(mu, log_var).cpu()

            lower_bound_s = self.radius
            upper_bound_s = self.max_upper_bound_radius

            # 选第一个样本计算lipschitz常数
            simple_domain = Hyperbox.build_linf_ball(center[0], lower_bound_s) # 邻域
            cross_problem = LipMIP(self.arl_obj.obfuscator.cpu(), simple_domain,
                                   'l1Ball1', num_threads=8, verbose=True)
            cross_problem.compute_max_lipschitz()
            lip_val = cross_problem.result.value
            self.logger.info(lip_val)
             

            if lip_val > self.proposed_bound: # 得到的lipschitz常数大于提议（前一个test函数算的）的上界
                rec = self.arl_obj.vae.decode(center[0].unsqueeze(0).cuda()).cpu()
                plt.imsave("./samples/ptr/{}.png".format(batch_idx), rec[0][0].detach())
                self.logger.info("bot before starting")
                unanswered_samples += 1 # 这个样本 敏感度高于lip_val
            else: # 这个样本的lip_val小于提议的上界
                start_time = time.perf_counter()
                while upper_bound_s >= lower_bound_s + self.margin: # margin是最小的步长
                    # 对于1类的z：
                    # 如果距离小于margin，则推远它们
                    # 如果距离已经大于margin，则不再施加损失
                    radius = (lower_bound_s + upper_bound_s) / 2 # 二分搜索

                    # 计算lipschitz常数
                    simple_domain = Hyperbox.build_linf_ball(center[0], radius)
                    cross_problem = LipMIP(self.arl_obj.obfuscator.cpu(), simple_domain,
                                        'l1Ball1', num_threads=8, verbose=False)
                    cross_problem.compute_max_lipschitz()
                    lip_val = cross_problem.result.value

                    # 找到那个最小的lipschitz常数，使得它小于提议的上界，获取它的半径
                    if self.proposed_bound > lip_val:
                        lower_bound_s = radius
                    else:
                        upper_bound_s = radius
                    time_elapsed = time.perf_counter() - start_time
                    if time_elapsed > self.max_time: # 计时的，超过时间了， 也不用再循环了，是啥就用啥吧
                        self.logger.info("timeout")
                        break

                # Dividing by 2 because that's the actual lower bound as shown in the paper
                real_lower_bound_s = lower_bound_s / 2
                noisy_lower_bound_s = real_lower_bound_s + self.get_noise(torch.zeros(1), self.radius)

                if noisy_lower_bound_s < np.log(1 / self.delta) * (self.radius / self.epsilon): # 这个样本的局部敏感度不满足DP条件
                    self.logger.info("bot {:.4f}".format(noisy_lower_bound_s.item()))
                    unanswered_samples += 1
                else: # 这个样本的局部敏感度满足DP条件
                    # 可以用了，加点噪声，开始预测
                    # pass it through obfuscator
                    z_tilde = self.arl_obj.obfuscator(center)
                    # This is not private yet since the noise added is based on private data
                    # TODO: Switch to PTR version
                    z_hat = z_tilde + self.get_noise(z_tilde, lip_val)
                    # pass obfuscated z through pred_model
                    noisy_preds = self.arl_obj.pred_model(z_hat)
                    noisy_pred_correct += (noisy_preds.argmax(dim=1) == labels).sum()

                    noiseless_preds = self.arl_obj.pred_model(z_tilde)
                    noiseless_pred_correct += (noiseless_preds.argmax(dim=1) == labels).sum()
                    sample_size += 1

            if unanswered_samples + sample_size > self.eval_size:
                break

        assert sample_size + unanswered_samples - 1 == batch_idx
        noisy_pred_acc = noisy_pred_correct.item() / sample_size
        noiseless_pred_acc = noiseless_pred_correct.item() / sample_size
        self.logger.info('====> Unanswered_samples {}/{}, Noisy pred acc {:.2f}, Noiseless pred acc {:.2f}'.format(unanswered_samples, batch_idx, noisy_pred_acc, noiseless_pred_acc))

    def test_local_sens(self): # 计算局部敏感度
        '''
        - 计算每个样本的局部Lipschitz常数
        - 添加基于Lipschitz常数的拉普拉斯噪声
        - 评估模型性能
        - 返回Lipschitz常数的均值和标准差
        '''
        self.arl_obj.vae.eval()
        self.arl_obj.pred_model.eval()
        sample_size, lip_vals = 0, []
        noisy_pred_correct, noiseless_pred_correct = 0, 0

        for batch_idx, (data, labels, _) in enumerate(self.train_loader): # 对每个batch
            data = data.cuda()
            # get sample embedding from the VAE
            mu, log_var = self.arl_obj.vae.encode(data)#data.view(-1, 784))
            center = self.arl_obj.vae.reparametrize(mu, log_var).cpu() # 原始的z

            # We are using the first example for Lip estimation
            # 选第一个样本计算lipschitz常数
            simple_domain = Hyperbox.build_linf_ball(center[0], self.radius) # 邻域
            cross_problem = LipMIP(self.arl_obj.obfuscator.cpu(), simple_domain,
                                   'l1Ball1', num_threads=8, verbose=True)
            cross_problem.compute_max_lipschitz()
            lip_val = cross_problem.result.value
            print(batch_idx, lip_val)
            if lip_val <= 1e-10: # clip
                print("problem")
                lip_val = 1e-10
            lip_vals.append(lip_val)

            # pass it through obfuscator
            z_tilde = self.arl_obj.obfuscator(center)
            # This is not private yet since the noise added is based on private data
            # TODO: Switch to PTR version
            z_hat = z_tilde + self.get_noise(z_tilde, lip_val) # 这个加的噪声还不 DP（因为noise是根据local sensitivity计算的）

            # pass obfuscated z through pred_model
            noisy_preds = self.arl_obj.pred_model(z_hat) 
            noisy_pred_correct += (noisy_preds.argmax(dim=1) == labels).sum()

            # 没加噪声的情况，进行task 任务
            noiseless_preds = self.arl_obj.pred_model(z_tilde)
            noiseless_pred_correct += (noiseless_preds.argmax(dim=1) == labels).sum()

            # 迭代终止条件
            sample_size += 1
            if sample_size > self.eval_size:
                break

        assert sample_size - 1 == batch_idx

        # 整理结果，算平均值
        noisy_pred_acc = noisy_pred_correct.item() / sample_size
        noiseless_pred_acc = noiseless_pred_correct.item() / sample_size
        self.logger.info('====> Noisy pred acc {:.2f}, Noiseless pred acc {:.2f}'.format(noisy_pred_acc, noiseless_pred_acc))
        self.logger.info('====>Lipschitz mean = {:.4f}, std = {:.4f}'.format(np.array(lip_vals).mean(), np.array(lip_vals).std()))

        return np.array(lip_vals).mean(), np.array(lip_vals).std()
