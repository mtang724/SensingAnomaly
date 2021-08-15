import numpy as np
import csv
import time
import datetime
import torch
import random
from utils import *


def csv_read(path):
    data = []
    with open(path,'r',encoding='utf-8') as f:
        reader = csv.reader(f,dialect='excel')
        next(reader)
        for row in reader:
            data.append(row)
    return np.array(data)#np.transpose(data).astype(np.float)


def Caltime(date1, date2):
    date1 = time.strptime(date1, "%Y/%m/%d %H:%M")
    date2 = time.strptime(date2, "%Y/%m/%d %H:%M")

    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])

    return (date2 - date1).days


def save_graph(feature_path, link_path):
    feature = csv_read(feature_path)
    link = csv_read(link_path)

    start_data = link[0,3]
    new_lable = []
    print('start!')

    node_dict = {}
    for i in range(feature.shape[0]):
        node_dict[feature[i][0]] = i

    for l in link:
        if l[0] not in node_dict.keys() or l[1] not in node_dict.keys():
            continue
        date = Caltime(start_data, l[3])
        if date< 0:
            continue
        if date > 100:
            break

        new_lable.append(node_dict[l[0]])
        new_lable.append(node_dict[l[1]])

    new_lable = sorted(list(set(new_lable)))
    new_feature = []

    for i in new_lable:
        new_feature.append(feature[i])
    new_feature = np.array(new_feature)
    print(new_feature.shape)

    new_node_dict = {}
    for i in range(new_feature.shape[0]):
        new_node_dict[feature[i][0]] = i

    graph = np.zeros((new_feature.shape[0], new_feature.shape[0]), dtype=np.float16)
    node = np.zeros((new_feature.shape[0], new_feature.shape[1] - 1), dtype=np.float16)
    day = 0

    print('start!')
    for l in link:
        if l[0] not in new_node_dict.keys() or l[1] not in new_node_dict.keys():
            continue
        date = Caltime(start_data, l[3])
        if date < 0:
            continue
        if date > 100:
            break

        if date > day:
            day = date

            np.save('reddit_data/node/node' + str(day) + '.npy', node)
            np.save('reddit_data/graph/graph' + str(day) + '.npy', graph)
            np.save('reddit_data/dict/node_dict' + str(day) + '.npy', new_node_dict)

            graph = np.zeros((new_feature.shape[0], new_feature.shape[0]), dtype=np.float16)
            node = np.zeros((new_feature.shape[0], new_feature.shape[1] - 1), dtype=np.float16)

        graph[new_node_dict[l[0]], new_node_dict[l[1]]] = float(l[4])
        graph[new_node_dict[l[0]], new_node_dict[l[0]]] = 1.0
        graph[new_node_dict[l[1]], new_node_dict[l[1]]] = 1.0
        node[new_node_dict[l[0]]] = np.asarray(feature[new_node_dict[l[0]]][1:], dtype=np.float16)
        node[new_node_dict[l[1]]] = np.asarray(feature[new_node_dict[l[1]]][1:], dtype=np.float16)

    np.save('reddit_data/node/node' + str(day) + '.npy', node)
    np.save('reddit_data/graph/graph' + str(day) + '.npy', graph)
    np.save('reddit_data/dict/node_dict' + str(day) + '.npy', new_node_dict)


def load_graph(feature_path, link_path, dict_path=None, abnormal_path=None):
    node = np.load(feature_path, allow_pickle=True)
    graph = np.load(link_path, allow_pickle=True)
    node = torch.tensor(node, dtype=torch.float)
    graph = torch.tensor(graph, dtype=torch.float)
    dict = None
    abnormal = None
    if dict_path is not None:
        d = np.load(dict_path, allow_pickle=True)
        dict = d.item()
    if abnormal_path is not None:
        abnormal = np.load(abnormal_path, allow_pickle=True)
        abnormal = torch.tensor(abnormal, dtype=torch.float)
    return node, graph, dict, abnormal


def load_har(train_list_path, train_label_path, test_list_path, test_label_path, subject_train_path, subject_test_path, abnormal_list_path):
    train_list = np.load(train_list_path, allow_pickle=True)
    train_list = torch.tensor(train_list, dtype=torch.float)

    train_label = np.load(train_label_path, allow_pickle=True)
    train_label = torch.tensor(train_label, dtype=torch.float)

    test_list = np.load(test_list_path, allow_pickle=True)
    test_list = torch.tensor(test_list, dtype=torch.float)

    test_label = np.load(test_label_path, allow_pickle=True)
    test_label = torch.tensor(test_label, dtype=torch.float)

    subject_train = np.load(subject_train_path, allow_pickle=True)
    subject_train = torch.tensor(subject_train, dtype=torch.float)

    subject_test = np.load(subject_test_path, allow_pickle=True)
    subject_test = torch.tensor(subject_test, dtype=torch.float)

    abnormal_list = np.load(abnormal_list_path, allow_pickle=True)
    abnormal_list = torch.tensor(abnormal_list, dtype=torch.float)

    return train_list, train_label, test_list, test_label, subject_train, subject_test, abnormal_list


def har_correlation():
    dataset_name = 'har_clean'
    if 'har' in dataset_name:
        if 'clean' in dataset_name:
            train_list_path = dataset_name + '/train_clean_list.npy'
            train_label_path = dataset_name + '/train_clean_label.npy'
        else:
            train_list_path = dataset_name + '/train_list.npy'
            train_label_path = dataset_name + '/train_label.npy'
        test_list_path = dataset_name + '/test_list.npy'
        test_label_path = dataset_name + '/test_label.npy'
        subject_train_path = dataset_name + '/subject_train.npy'
        subject_test_path = dataset_name + '/subject_test.npy'
        abnormal_list_path = dataset_name + '/ab.npy'

        dataset = load_har(train_list_path, train_label_path, test_list_path, test_label_path, subject_train_path,
                           subject_test_path, abnormal_list_path)

        train_list = dataset[0]
        subject_train = dataset[4]
        del dataset

        num_node = 18
        idx = 0
        edge = torch.eye(num_node) * len(subject_train)

        while idx < len(subject_train):
            start = int(subject_train[idx, 1])
            end = int(subject_train[idx, 2])

            node_feature = train_list[start:end + 1].permute([1,0,2])

            for i in range(num_node):
                for j in range(i+1, num_node):
                    rv = modified_rv(node_feature[i], node_feature[j])
                    edge[i,j] += rv
                    edge[j,i] += rv

            idx += 1

        edge /= len(subject_train)

        np.save('har_clean/edge.npy', edge.numpy())

        return edge


def structure_abnormal(edge, number=500, time=20):
    time = np.random.randint(0, edge.shape[0], size=time)
    node_list = []
    for t in time:
        nodes = np.random.randint(0, edge.shape[1], size=number)
        node_list.append(nodes)
        for n in range(number):
            for m in range(number):
                edge[t,nodes[n],nodes[m]] = 1.
    return edge, time, node_list


def feature_abnormal(node, number=500, time=20):
    times = np.random.randint(0, node.shape[0], size=time)
    node_list = []

    time2 = np.random.randint(0, node.shape[0], size=time)
    nodes2 = np.random.randint(0, node.shape[1], size=int(number/100))

    nozero= torch.sum(node, dim=-1).nonzero()
    indicate = torch.randperm(len(nozero))
    lists = []
    for i in range(int(number / 10 * time)):
        lists.append(nozero[indicate[i]])
    del nozero
    del indicate


    g2 = torch.zeros((int(number/10*time), node.shape[2]), dtype=torch.float)
    for i in range(int(number/10*time)):
        g2[i] = node[lists[i][0], lists[i][1]]

    for t in times:
        nodes = np.random.randint(0, node.shape[1], size=number)
        node_list.append(nodes)
        for n in nodes:
            i = torch.mean(torch.pow(node[t,n]-g2, 2), dim=-1).argmax()
            node[t, n] = g2[i]
    return node, times, node_list


def generate_test_data(dataset_name,test_len,graph_size,channal,train_len):
    all_node = torch.zeros((test_len, graph_size, channal), dtype=torch.float).cpu()
    all_edge = torch.zeros((test_len, graph_size, graph_size), dtype=torch.float).cpu()

    for d in range(test_len):
        node_path = dataset_name + '/node/node' + str(d + train_len + 1) + '.npy'
        edge_path = dataset_name + '/graph/graph' + str(d + train_len + 1) + '.npy'
        dict_path = dataset_name + '/dict/node_dict' + str(
            d + train_len + 1) + '.npy' if dataset_name != 'DBLP5' else None
        all_node[d], all_edge[d], _, _ = load_graph(node_path, edge_path)

    num_node = {'reddit_data': 500, 'DBLP5': 1500}
    num_time = {'reddit_data': 20, 'DBLP5': 2}
    all_node, ft, fnl = feature_abnormal(all_node.detach(), num_node[dataset_name],
                                         num_time[dataset_name])
    all_edge, st, snl = structure_abnormal(all_edge.detach(), num_node[dataset_name],
                                           num_time[dataset_name])

    abnormal = torch.zeros((test_len, graph_size))
    for t in range(len(ft)):
        for n in fnl[t]:
            abnormal[ft[t], n] = 1.

    for t in range(len(st)):
        for n in snl[t]:
            abnormal[st[t], n] = 1.

    all_node = all_node.half()
    all_edge = all_edge.half()
    print(all_edge.shape)
    for i in range(test_len):
        np.save(dataset_name + '/node/testnode' + str(i + 1) + '.npy', all_node[i])
        np.save(dataset_name + '/graph/testgraph' + str(i + 1) + '.npy', all_edge[i])
        np.save(dataset_name + '/abnormal/abnormal' + str(i + 1) + '.npy', abnormal[i])


def structure_abnormal2(edge, number=500, time=20):
    time = np.random.randint(0, edge.shape[0], size=time)
    node_list = []
    for t in time:
        nodes =[]
        for i in range(edge.shape[1]):
            if edge[t,i,i] != 0:
                nodes.append(i)
        random.shuffle(nodes)
        nodes = np.array(nodes[:number])
        node_list.append(nodes)
        for n in range(number):
            for m in range(number):
                edge[t,nodes[n],nodes[m]] = 1.
    return edge, time, node_list


def feature_abnormal2(node, number=500, time=20):
    times = np.random.randint(0, node.shape[0], size=time)
    node_list = []

    nozero= torch.sum(node, dim=-1).nonzero()
    indicate = torch.randperm(len(nozero))
    lists = []
    for i in range(int(number * 100 * time)):
        lists.append(nozero[indicate[i]])
    del nozero
    del indicate


    g2 = torch.zeros((int(number*100*time), node.shape[2]), dtype=torch.float)
    for i in range(int(number*100*time)):
        g2[i] = node[lists[i][0], lists[i][1]]

    for t in times:
        nodes = []
        for i in range(node.shape[1]):
            if torch.sum(node[t, i]) != 0:
                nodes.append(i)
        random.shuffle(nodes)
        nodes = np.array(nodes[:number])
        node_list.append(nodes)
        for n in nodes:
            i = torch.mean(torch.pow(node[t,n]-g2, 2), dim=-1).argmax()
            node[t, n] = g2[i]
    return node, times, node_list


def generate_test_data2(dataset_name,test_len,graph_size,channal,train_len):
    all_node = torch.zeros((test_len, graph_size, channal), dtype=torch.float).cpu()
    all_edge = torch.zeros((test_len, graph_size, graph_size), dtype=torch.float).cpu()

    for d in range(test_len):
        node_path = dataset_name + '/node/node' + str(d + train_len + 1) + '.npy'
        edge_path = dataset_name + '/graph/graph' + str(d + train_len + 1) + '.npy'
        dict_path = dataset_name + '/dict/node_dict' + str(
            d + train_len + 1) + '.npy' if dataset_name != 'DBLP5' else None
        all_node[d], all_edge[d], _, _ = load_graph(node_path, edge_path)

    num_node = {'reddit_data': 5, 'DBLP5': 10}
    num_time = {'reddit_data': 4, 'DBLP5': 2}
    all_node, ft, fnl = feature_abnormal2(all_node.detach(), num_node[dataset_name],
                                         num_time[dataset_name])
    all_edge, st, snl = structure_abnormal2(all_edge.detach(), num_node[dataset_name],
                                           num_time[dataset_name])

    abnormal = torch.zeros((test_len, graph_size))
    for t in range(len(ft)):
        for n in fnl[t]:
            abnormal[ft[t], n] = 1.

    for t in range(len(st)):
        for n in snl[t]:
            abnormal[st[t], n] = 1.

    all_node = all_node.half()
    all_edge = all_edge.half()
    print(all_edge.shape)
    for i in range(test_len):
        np.save(dataset_name + '/node/testnode' + str(i + 1) + '.npy', all_node[i])
        np.save(dataset_name + '/graph/testgraph' + str(i + 1) + '.npy', all_edge[i])
        np.save(dataset_name + '/abnormal/abnormal' + str(i + 1) + '.npy', abnormal[i])


def list2np(feature_map,feature_list, nnode, nfeature):
    feature_np = np.zeros([nnode,nfeature])
    for i, f in enumerate(feature_map[:-7]):
        tem_np = feature_list[f[1]:f[2]+1]
        tem_np = np.array(tem_np)
        tem_np = np.pad(tem_np, (0, nfeature-tem_np.shape[0]))
        feature_np[i] = tem_np

    # angle
    tem_np = feature_list[-7:]
    tem_np = np.array(tem_np)
    tem_np = np.pad(tem_np, (0, nfeature - tem_np.shape[0]))
    feature_np[-1] = tem_np

    return feature_np


def HAR_preprocess(path):
    # feature split
    f = open(path + "/features.txt")
    line = f.readline()
    store_name = line.strip().split('-')[0].split()[1]
    start = 0
    num_line = 0
    result = []
    while line:
        line = f.readline()
        if line:
            current_name = line.strip().split('-')[0].split()[1]
        else:
            current_name = ''
        if current_name != store_name:
            result.append([store_name+'-', start, num_line])
            start = num_line + 1
            store_name = current_name
        num_line += 1
    f.close()
    print(result)

    num_node = len(result) - 7 + 1
    num_feature = max(result, key=lambda x: x[2]-x[1]+1)
    num_feature = num_feature[2] - num_feature[1] + 1
    print(num_node,num_feature)

    # train_list
    train_list = []
    train_clean_list = []
    train_label = []
    train_clean_label = []

    f = open(path + "/train/X_train.txt")
    g = open(path + "/train/y_train.txt")
    h = open(path + "/train/subject_train.txt")

    subline = h.readline()
    store_name = subline.strip()
    start = 0
    num_line = 0
    subject_train = []


    line = f.readline()
    l_feature = list(map(float,line.strip().split()))
    np_feature = list2np(result, l_feature, num_node, num_feature)
    train_list.append(np_feature)

    yline = int(g.readline().strip())
    train_label.append(yline)
    if yline != 3:
        train_clean_list.append(np_feature)
        train_clean_label.append(yline)

    while line:
        line = f.readline()

        subline = h.readline()
        if subline:
            current_name = subline.strip()
        else:
            current_name = ''
        if current_name != store_name:
            subject_train.append([int(store_name), start, num_line, num_line - start + 1])
            start = num_line + 1
            store_name = current_name

        if line:
            l_feature = list(map(float, line.strip().split()))
            np_feature = list2np(result, l_feature, num_node, num_feature)
            train_list.append(np_feature)

            yline = int(g.readline().strip())
            train_label.append(yline)

            if yline != 3:
                train_clean_list.append(np_feature)
                train_clean_label.append(yline)

                num_line += 1

    f.close()
    g.close()
    h.close()
    print(len(train_list), len(train_label))
    print(len(train_clean_list), len(train_clean_label))
    print(subject_train)

    # test_list
    test_list = []
    test_label = []

    f = open(path + "/test/X_test.txt")
    g = open(path + "/test/y_test.txt")

    line = f.readline()
    l_feature = list(map(float, line.strip().split()))
    np_feature = list2np(result, l_feature, num_node, num_feature)
    test_list.append(np_feature)

    yline = int(g.readline().strip())
    test_label.append(yline)

    while line:
        line = f.readline()
        if line:
            l_feature = list(map(float, line.strip().split()))
            np_feature = list2np(result, l_feature, num_node, num_feature)
            test_list.append(np_feature)

            yline = int(g.readline().strip())
            test_label.append(yline)
    f.close()
    g.close()
    print(len(test_list), len(test_label))

    # subject_test
    g = open(path + "/test/subject_test.txt")
    line = g.readline()
    store_name = line.strip()
    start = 0
    num_line = 0
    subject_test = []
    while line:
        line = g.readline()
        if line:
            current_name = line.strip()
        else:
            current_name = ''
        if current_name != store_name:
            subject_test.append([int(store_name), start, num_line, num_line - start + 1])
            start = num_line + 1
            store_name = current_name
        num_line += 1
    g.close()
    print(subject_test)

    # list to numpy
    train_list = np.array(train_list)
    train_clean_list = np.array(train_clean_list)
    train_label = np.array(train_label)
    train_clean_label = np.array(train_clean_label)
    test_list = np.array(test_list)
    test_label = np.array(test_label)
    subject_train = np.array(subject_train)
    subject_test = np.array(subject_test)

    # add sensor abnormal
    test_list, abnormal_list = sensor_abnormal(test_list)

    ab = np.zeros_like(np.sum(test_list, axis=-1))
    print(ab.shape)
    for x in range(abnormal_list.shape[0]):
        p = abnormal_list[x]
        ab[int(p[0]), int(p[1])] = 1

    # save
    check_folder(path + '/numpy')
    np.save(path + '/numpy/train_list.npy', train_list)
    np.save(path + '/numpy/train_clean_list.npy', train_clean_list)
    np.save(path + '/numpy/train_label.npy', train_label)
    np.save(path + '/numpy/train_clean_label.npy', train_clean_label)
    np.save(path + '/numpy/test_list.npy', test_list)
    np.save(path + '/numpy/test_label.npy', test_label)
    np.save(path + '/numpy/subject_train.npy', subject_train)
    np.save(path + '/numpy/subject_test.npy', subject_test)
    np.save(path + '/numpy/abnormal_list.npy', abnormal_list)
    np.save(path + '/numpy/ab.npy', ab)

    return train_list, train_clean_list, train_label, train_clean_label, test_list, test_label, subject_train, subject_test, abnormal_list, ab


def sensor_abnormal(node, number=1, time=60):
    times = np.random.randint(0, node.shape[0], size=time)
    abnormal_list = []

    for t in times:
        node_ind = np.random.randint(0, node.shape[1], size=number)
        for n in node_ind:
            node[t, n] = np.random.normal(0, 2.5) * (node[t, n] != 0)
            abnormal_list.append([t,n])
    abnormal_list = np.array(abnormal_list)
    return node, abnormal_list

if __name__ == '__main__':
    # train_list, train_clean_list, train_label, train_clean_label, test_list, test_label, subject_train, subject_test, abnormal_list, ab = HAR_preprocess(
    #     path='C:/Users/yinan/Desktop/UCI HAR Dataset')

    # dataset_name = 'DBLP5'
    # test_len = 4
    # train_len = 6
    # channal = 100
    # graph_size = 6606
    # generate_test_data2(dataset_name, test_len, graph_size, channal, train_len)

    x = har_correlation()