'''
Author: Ruijun Deng
Date: 2024-01-03 15:19:20
LastEditTime: 2024-07-09 09:23:54
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/attacks/membership_inference/Mentr_attack.py
Description: Usenix sec'21-Systematic Evaluation of Privacy Risks of Machine Learning Models
这里面其实包含了4个membership inference attack，集成了
'''
import numpy as np
import math
import torch

class MentrAttack(object):
    # 基于shadow model 的blackbox
    def __init__(self, num_classes,gpu=True):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels. 
        '''

        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    
    def _softmax_by_row(self, logits, T = 1.0): # 用于处理smashed data
        mx = np.max(logits, axis=-1, keepdims=True) # 找到最大的
        exp = np.exp((logits - mx)/T)
        denominator = np.sum(exp, axis=-1, keepdims=True)
        return exp/denominator
    
    def prepare_model_performance(self, shadow_model, shadow_train_loader, shadow_test_loader,
                                target_model, target_train_loader, target_test_loader):
        # 进行模型推理，输出 outputs，和true labels的数组                          
        def _model_predictions(model, dataloader):
            return_outputs, return_labels = [], []
            model.to(self.device)
            for (inputs, labels) in dataloader:
                inputs = inputs.to(self.device)
                return_labels.append(labels.numpy())
                outputs = model.forward(inputs) 
                return_outputs.append( self._softmax_by_row(outputs.data.cpu().detach().numpy()) ) # 归一化后再softmax
            return_outputs = np.concatenate(return_outputs) # 拼接数组
            return_labels = np.concatenate(return_labels) # 拼接数组
            return (return_outputs, return_labels)
        
        # 在训练集、测试集 分别推理一遍 真实模型 和 shadow model
        shadow_train_performance = _model_predictions(shadow_model, shadow_train_loader)
        shadow_test_performance = _model_predictions(shadow_model, shadow_test_loader)
        
        target_train_performance = _model_predictions(target_model, target_train_loader)
        target_test_performance = _model_predictions(target_model, target_test_loader)


        # 设置各类的必要数据
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        
        print(self.t_tr_outputs.shape)
        print(self.t_te_outputs.shape)
        print(self.s_tr_outputs.shape)
        print(self.s_te_outputs.shape)
        # self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        # self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        # self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        # self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)
        
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])
        
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)
        
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)
        # return shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance

    def _log_value(self, probs, small_value=1e-30): # 按元素操作，形状不变
        return -np.log(np.maximum(probs, small_value))
    
    def _entr_comp(self, probs): # 计算entropy
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)
    
    def _m_entr_comp(self, probs, true_labels): # 计算mentropy
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)
    
    def _thre_setting(self, tr_values, te_values): # 设置阈值
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre
    
    def _mem_inf_via_corr(self): # 基于correctness 进行攻击
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
        return
    
    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values): # 基于confidence系列的指标进行攻击
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes): # 对于每一个class
            thre = self._thre_setting(s_tr_values[self.s_tr_labels==num], s_te_values[self.s_te_labels==num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre) # 分类正确的 train 数目
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels==num]<thre) # 分类正确的 test 数目
        mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))
        # mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_outputs)+0.0) + t_te_non_mem/(len(self.t_te_outputs)+0.0))
        print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
        return
    
    def mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]): # 4大类攻击
        if (all_methods) or ('correctness' in benchmark_methods):
            self._mem_inf_via_corr()
        if (all_methods) or ('confidence' in benchmark_methods):
            self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        if (all_methods) or ('entropy' in benchmark_methods):
            self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)

        return

