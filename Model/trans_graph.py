import numpy as np
import csv
import time
import datetime
import torch
import random

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


#def delete(edge, node)

if __name__ == '__main__':
    dataset_name = 'DBLP5'
    test_len = 4
    train_len = 6
    channal = 100
    graph_size = 6606
    generate_test_data2(dataset_name, test_len, graph_size, channal, train_len)
    