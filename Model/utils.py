import tensorflow as tf
#from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
import torch

DEVICE = torch.device('cpu')#'cuda' if torch.cuda.is_available() else

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def rv_preprocessing(m):
    m1 = torch.matmul(m,m.T)
    return torch.flatten(m1 - torch.eye(m1.shape[0]) * m1)


def modified_rv(x,y):
    x = rv_preprocessing(x)
    y = rv_preprocessing(y)

    rv = torch.matmul(x,y) / torch.sqrt(torch.matmul(x,x) * torch.matmul(y,y))

    return rv

##################################################################################
# Loss function
##################################################################################


def l1_loss(x, y, graph=True, device=DEVICE):
    if graph:
        loss = torch.mean(torch.abs(x - y))
    else:
        loss = torch.mean(torch.abs(x - y), dim=-1)

    return loss.to(device)


def l2_loss(x, y, graph=True, c=0, device=DEVICE):
    if graph:
        a = torch.mean(torch.pow(x-y, 2))
        loss = torch.clamp(torch.mean(torch.pow(x-y, 2)) - c, min=0., max=100)
    else:
        loss = torch.clamp(torch.mean(torch.pow(x-y, 2), dim=-1) - c, min=0., max=100)

    return loss.to(device)


def cross_entropy(output, lable, graph=True, device=DEVICE):
    if graph:
        loss = torch.mean(-1 * lable * torch.log(output + 1e-15) - (1-lable) * torch.log(1-output + 1e-15))
    else:
        loss = torch.mean(-1 * lable * torch.log(output + 1e-15) - (1-lable) * torch.log(1-output + 1e-15), dim=-1)

    return  loss.to(device)


def predict_result(predict, abnormal, level):
    thers_l = torch.arange(0, 2.51, 0.01)
    label = torch.flatten(abnormal).float()
    predict = torch.flatten(predict).float()
    recall_list = []
    prec_list = []
    f1_list = []
    acc_list = []
    best = [0, 0, 0, 0, 0]
    for thers in thers_l:
        # tp, tn, fp, fn = 0., 0., 0., 0.

        record1 = (predict >= thers).float()

        tp = (label * record1).sum().to(torch.float32)
        tn = ((1 - label) * (1 - record1)).sum().to(torch.float32)
        fp = ((1 - label) * record1).sum().to(torch.float32)
        fn = (label * (1 - record1)).sum().to(torch.float32)

        # for n in range(record1.shape[0]):
        #     if record1[n] == label[n]:
        #         if record1[n] == 0:
        #             tp += 1.
        #         else:
        #             tn += 1.
        #     else:
        #         if record1[n] == 0:
        #             fp += 1.
        #         else:
        #             fn += 1.

        # anomaly_cpu = error.detach().cpu().numpy()
        # anomaly_max = np.max(anomaly_cpu)
        # max_indicate = np.where(anomaly_cpu == anomaly_max)
        # print(max_indicate)
        # if node_exist[max_indicate[0]]:
        #     print('idx={}'.format(idx))
        #     print(anomaly_max)
        #     print(max_indicate)
        #     print('\n')
        # else:
        #     print('Not exists')

        epsilon = 1e-7

        acc = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn + epsilon)
        prec = tp / (tp + fp + epsilon)
        f1 = 2 * (recall * prec) / (recall + prec)

        acc_list.append(acc)
        recall_list.append(recall)
        prec_list.append(prec)
        f1_list.append(f1)

        # print('thers={}, f1={}, acc={}, recall={}, prec={}'.format(thers, f1,acc, recall, prec))

        if f1 > best[1]:
            best = [thers, f1, acc, recall, prec]
        elif f1 == best[1] and acc > best[2]:
            best = [thers, f1, acc, recall, prec]

    print('Best result of ' + level + ' for now: thers={}, f1={}, acc={}, recall={}, prec={}'.format(best[0], best[1],
                                                                                                   best[2], best[3],
                                                                                                   best[4]))

    # plt.figure(1)
    # plt.plot(recall_list,prec_list)
    # plt.title('roc')
    # plt.xlabel('Recall')
    # plt.ylabel('Prec')
    #
    # plt.figure(2)
    # plt.plot(acc_list, thers_l)
    # plt.xlabel('threshold')
    # plt.ylabel('accuracy')
    #
    # plt.figure(3)
    # plt.plot(f1_list, thers_l)
    # plt.xlabel('threshold')
    # plt.ylabel('f1')
    #
    # plt.show()
    #
    # with open('result.txt', 'w') as f:
    #     for i in recall_list:
    #         f.write("%s" % i)
    #     f.write('\n')
    #     for i in prec_list:
    #         f.write("%s" % i)
    #     f.write('\n')
    #     for i in acc_list:
    #         f.write("%s" % i)
    #     f.write('\n')
    #     for i in f1_list:
    #         f.write("%s" % i)

