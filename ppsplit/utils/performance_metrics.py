import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score,recall_score,roc_curve,precision_recall_curve,classification_report
import torch.nn as nn

# 测试edge model auc
def test_auc(net, test_loader):
    # test数据和网络保持在同样的硬件上
    device = next(net.parameters()).device
    test_epoch_outputs = []
    test_epoch_labels = []
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)

        test_epoch_outputs.append(outputs.cpu().detach())
        test_epoch_labels.append(labels.cpu().detach())

    # print("torch.cat(test_epoch_outputs): ",torch.cat(test_epoch_outputs))
    # print("torch.cat(test_epoch_labels) ",torch.cat(test_epoch_labels))
    test_auc = roc_auc_score(torch.cat(test_epoch_labels), torch.cat(test_epoch_outputs))

    print(f"test_auc: {test_auc}")

    return test_auc


# 从net输出的prob 转为 分类标签
def compute_pred_class(y_preds, y_tagets, threshold=0.5):
    y_pred_labels = []
    for y_pred, y_t in zip(y_preds, y_tagets):
        if y_pred <= threshold:
            y_label = 0
        else:
            y_label = 1
        y_pred_labels.append(y_label)
    return torch.tensor(y_pred_labels)


def test_acc(net, test_loader):
    device = next(net.parameters()).device
    test_epoch_outputs = []
    test_epoch_labels = []
    sigmoid = nn.Sigmoid()  # 分类层 这里sigmoid都是写在外面的吗
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)

        pred_labels = compute_pred_class(sigmoid(outputs), labels)

        test_epoch_outputs.append(pred_labels.cpu().detach())
        test_epoch_labels.append(labels.cpu().detach())

    # print("test_epoch_outputs: ",len(test_epoch_outputs))
    # print("test_epoch_labels: ", len(test_epoch_labels))
    test_acc = accuracy_score(torch.cat(test_epoch_labels), torch.cat(test_epoch_outputs))
    test_precision = precision_score(torch.cat(test_epoch_labels), torch.cat(test_epoch_outputs)) # 增加
    test_recall = recall_score(torch.cat(test_epoch_labels), torch.cat(test_epoch_outputs))
    test_classification_report = classification_report(torch.cat(test_epoch_labels),torch.cat(test_epoch_outputs),labels = [0,1])
    # test_auc = roc_auc_score(torch.cat(test_epoch_labels,torch.cat(test_epoch_outputs)))
    

    print(f"test_acc: {test_acc}")
    print(f"test_precision: {test_precision}")
    print(f"test_recall: {test_recall}")
    print(f"classification_report:")
    print(test_classification_report)
    
    return test_acc,test_precision,test_recall


# pr 曲线：
def test_pr_roc(net, test_loader):
    device = next(net.parameters()).device
    test_epoch_outputs = []
    test_epoch_labels = []
    test_epoch_real_outputs = []
    sigmoid = nn.Sigmoid()  # 分类层 这里sigmoid都是写在外面的吗
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)

        pred_labels = compute_pred_class(sigmoid(outputs), labels)

        test_epoch_outputs.append(pred_labels.cpu().detach())
        test_epoch_labels.append(labels.cpu().detach())
        test_epoch_real_outputs.append(outputs.cpu().detach())

    precision, recall, thresholds = precision_recall_curve(torch.cat(test_epoch_labels), torch.cat(test_epoch_real_outputs))
    fpr, tpr, thresholds = roc_curve(torch.cat(test_epoch_labels), torch.cat(test_epoch_real_outputs))
    
    return [recall,precision],[fpr,tpr]


# 测试f1值
def test_f1(net, test_loader):
    device = next(net.parameters()).device
    test_epoch_outputs = []
    test_epoch_labels = []
    sigmoid = nn.Sigmoid()  # 分类层 # 这里sigmoid都是写在外面的吗

    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        pred_labels = compute_pred_class(sigmoid(outputs), labels)

        test_epoch_outputs.append(pred_labels.cpu().detach())
        test_epoch_labels.append(labels.cpu().detach())

    test_f1 = f1_score(torch.cat(test_epoch_labels), torch.cat(test_epoch_outputs))

    print(f"test_f1: {test_f1}")
    return test_f1

