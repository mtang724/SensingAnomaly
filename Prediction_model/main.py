import networkx as nx
import os
import torch
import torch.nn as nn
import dgl
import pickle

def read_graphs():
    directory = "../data/45eb179e1d93ae0d"
    date_dir = [dir for dir in os.listdir(directory)]
    graph_feats = []
    graph_lists = []
    for dir in date_dir:
        dir_path = os.path.join(directory, dir)
        files = [file for file in os.listdir(dir_path) if not file.endswith("txt")]
        files.sort(key=lambda x: float(x.split("_")[-1].split(".")[0]))
        for file in files:
            file_path = os.path.join(dir_path, file)
            G = nx.read_gpickle(file_path)
            graph_lists.append(dgl.from_networkx(G))
            outputs = []
            for node in G.nodes:
                if node == "Type":
                    continue
                input = G.nodes[node]['attr']
                input_len = len(input)
                if input_len < 19:
                    input += [0] * (19 - input_len) + [len(input)]
                else:
                    input = input[:19] + [len(input)]
                input_list = []
                for value in input:
                    if value == "Small drink":
                        value = 20
                    input_list.append(value)
                input = torch.FloatTensor(input_list)
                mlp = nn.Linear(20, 10)
                output = mlp(input)
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0)
            graph_feats.append(outputs)
    graph_feats = torch.stack(graph_feats, dim=0)

def read_ys():
    directory = "../data/45eb179e1d93ae0d"
    date_dir = [dir for dir in os.listdir(directory)]
    for dir in date_dir:
        dir_path = os.path.join(directory, dir)
        files = [file for file in os.listdir(dir_path) if file.endswith("txt")]
        file = files[0]
        ys = pickle.load(open(os.path.join(dir_path, file), "rb"))
        print(ys)

read_ys()