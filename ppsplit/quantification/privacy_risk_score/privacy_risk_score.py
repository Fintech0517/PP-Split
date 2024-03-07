'''
Author: Ruijun Deng
Date: 2024-01-08 15:43:58
LastEditTime: 2024-03-07 13:37:53
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/quantification/privacy_risk_score/privacy_risk_score.py
Description: 
'''
import numpy as np
import matplotlib.pyplot as plt

def distrs_compute(tr_values, te_values, tr_labels, te_labels, num_bins=5, log_bins=True, plot_name=None):
    
    ### function to compute and plot the normalized histogram for both training and test values class by class.
    ### we recommand using the log scale to plot the distribution to get better-behaved distributions.
    
    num_classes = len(set(tr_labels)) # 类别数目
    print(f'数据共有{num_classes}个类别' )
    sqr_num = int(np.ceil(np.sqrt(num_classes))) # 类数目的开根
    tr_distrs, te_distrs, all_bins = [], [], [] # 待返回的三个列表
    
    # 开始画图了
    plt.figure(figsize = (30,30))
    plt.rc('font', family='serif', size=10)
    plt.rc('axes', linewidth=2)
    # plt.rc('figure.subplot', hspace=0.5)  # 设置垂直间距为0.5
    # plt.rc('figure.subplot', wspace=0.5)  # 设置垂直间距为0.5
    
    
    for i in range(num_classes): # 对每个类别
        tr_list, te_list = tr_values[tr_labels==i], te_values[te_labels==i] # 提取该类别的训练数据和测试数据
        if log_bins:
            # when using log scale, avoid very small number close to 0
            small_delta = 1e-10
            tr_list[tr_list<=small_delta] = small_delta # 进行clip
            te_list[te_list<=small_delta] = small_delta # 进行clip
        n1, n2 = np.sum(tr_labels==i), np.sum(te_labels==i)
        all_list = np.concatenate((tr_list, te_list))
        max_v, min_v = np.amax(all_list), np.amin(all_list)
        
        plt.subplot(sqr_num, sqr_num, i+1)
        plt.subplots_adjust(hspace=0.5)  # 调整垂直间距
        plt.subplots_adjust(wspace=0.5)  # 调整垂直间距

        if log_bins:
            bins = np.logspace(np.log10(min_v), np.log10(max_v),num_bins+1) # 横轴 logbins
            weights = np.ones_like(tr_list)/float(len(tr_list)) # 归一化？ x会乘这个weight的到一个x‘ 也不算是归一化，能说是个取平均
            h1, _,_ = plt.hist(tr_list,bins=bins,facecolor='b',weights=weights,alpha = 0.5) # 画出训练数据集每个类的x的分布
            plt.gca().set_xscale("log")
            weights = np.ones_like(te_list)/float(len(te_list))
            h2, _, _ = plt.hist(te_list,bins=bins,facecolor='r',weights=weights,alpha = 0.5)
            plt.gca().set_xscale("log")
        else:
            bins = np.linspace(min_v, max_v,num_bins+1)
            weights = np.ones_like(tr_list)/float(len(tr_list))
            h1, _,_ = plt.hist(tr_list,bins=bins,facecolor='b',weights=weights,alpha = 0.5)
            weights = np.ones_like(te_list)/float(len(te_list))
            h2, _, _ = plt.hist(te_list,bins=bins,facecolor='r',weights=weights,alpha = 0.5)
        tr_distrs.append(h1)
        te_distrs.append(h2)
        all_bins.append(bins)
    if plot_name == None:
        plot_name='./tmp'
    plt.savefig(plot_name+'.png', bbox_inches='tight')
    tr_distrs, te_distrs, all_bins = np.array(tr_distrs), np.array(te_distrs), np.array(all_bins)
    return tr_distrs, te_distrs, all_bins


def risk_score_compute(tr_distrs, te_distrs, all_bins, data_values, data_labels):
    
    ### Given training and test distributions (obtained from the shadow classifier), 
    ### compute the corresponding privacy risk score for training points (of the target classifier).
    
    def find_index(bins, value):
        # for given n bins (n+1 list) and one value, return which bin includes the value
        if value>=bins[-1]:
            return len(bins)-2 # when value is larger than any bins, we assign the last bin
        if value<=bins[0]:
            return 0  # when value is smaller than any bins, we assign the first bin
        return np.argwhere(bins<=value)[-1][0]
    
    def score_calculate(tr_distr, te_distr, ind): 
        if tr_distr[ind]+te_distr[ind] != 0:
            return tr_distr[ind]/(tr_distr[ind]+te_distr[ind])
        else: # when both distributions have 0 probabilities, we find the nearest bin with non-zero probability
            for t_n in range(1, len(tr_distr)):
                t_ind = ind-t_n
                if t_ind>=0:
                    if tr_distr[t_ind]+te_distr[t_ind] != 0:
                        return tr_distr[t_ind]/(tr_distr[t_ind]+te_distr[t_ind])
                t_ind = ind+t_n
                if t_ind<len(tr_distr):
                    if tr_distr[t_ind]+te_distr[t_ind] != 0:
                        return tr_distr[t_ind]/(tr_distr[t_ind]+te_distr[t_ind])
                    
    risk_score = []   
    for i in range(len(data_values)): # 对每个数据 算risk score
        c_value, c_label = data_values[i], data_labels[i]
        c_tr_distr, c_te_distr, c_bins = tr_distrs[c_label], te_distrs[c_label], all_bins[c_label]
        c_index = find_index(c_bins, c_value)
        c_score = score_calculate(c_tr_distr, c_te_distr, c_index)
        risk_score.append(c_score)
    return np.array(risk_score)

def calculate_risk_score(tr_values, te_values, tr_labels, te_labels, data_values, data_labels, 
                         num_bins=5, log_bins=True):
    
    ########### tr_values, te_values, tr_labels, te_labels are from shadow classifier's training and test data
    ########### data_values, data_labels are from target classifier's training data
    ########### potential choice for the value -- entropy, or modified entropy, or prediction loss (i.e., -np.log(confidence))
    
    tr_distrs, te_distrs, all_bins = distrs_compute(tr_values, te_values, tr_labels, te_labels,  # 这些全是从shadow model的输出算的
                                                    num_bins=num_bins, log_bins=log_bins)
    risk_score = risk_score_compute(tr_distrs, te_distrs, all_bins, data_values, data_labels) # 得到shadow的histogram后，来算target的？
    return risk_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run membership inference attacks')
    parser.add_argument('--dataset', type=str, default='purchase', help='purchase or texas')
    parser.add_argument('--model-dir', type=str, default='/home/data/FinTech/VFL/MIAs/membership_inference_evaluation/adv_reg/training_code/models/purchase_undefended/model_best.pth.tar', help='directory of target model')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size of data loader')
    args = parser.parse_args()
    
    # 加载数据集
    if args.dataset=='purchase': # purchase数据集
        model = PurchaseClassifier(num_classes=100)
        model = torch.nn.DataParallel(model).cuda()
        shadow_train_loader, shadow_test_loader, \
        target_train_loader, target_test_loader = prepare_purchase_data(batch_size=args.batch_size)
    else:
        model = TexasClassifier(num_classes=100)
        model = torch.nn.DataParallel(model).cuda()
        shadow_train_loader, shadow_test_loader, \
        target_train_loader, target_test_loader = prepare_texas_data(batch_size=args.batch_size)
    
    # 加载模型
    checkpoint = torch.load(args.model_dir)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 得到真实模型、shadow model 在训练集、测试集上的表现
    shadow_train_performance,shadow_test_performance,target_train_performance,target_test_performance = \
    prepare_model_performance(model, shadow_train_loader, shadow_test_loader, 
                              model, target_train_loader, target_test_loader)
    
    print('Perform membership inference attacks!!!')
    MIA = black_box_benchmarks(shadow_train_performance,shadow_test_performance,
                         target_train_performance,target_test_performance,num_classes=100)
    # private risk score
    risk_score = calculate_risk_score(MIA.s_tr_m_entr, MIA.s_te_m_entr, MIA.s_tr_labels, MIA.s_te_labels, MIA.t_tr_m_entr, MIA.t_tr_labels)
    # membership inference attack
    MIA._mem_inf_benchmarks() # 执行所有attack
    