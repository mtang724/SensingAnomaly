import networkx as nx
import dgl
import torch
import torch.nn as nn
from Model2.model import GNNAnomalyDetctor

G = nx.read_gpickle("../graphs/user1_0.gpickle")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

outputs = []
for node in G.nodes:
    input = G.nodes[node]['attr']
    input_len = len(input)
    if input_len < 19:
        input += [0] * (19 - input_len) + [len(input)]
    else:
        input = input[:19] + [len(input)]
    # print(len(input))
    input = torch.FloatTensor(input)
    mlp = nn.Linear(20, 10)
    output = mlp(input)
    outputs.append(output)
outputs = torch.stack(outputs, dim=0)
feat = outputs
g = dgl.from_networkx(G)



def preprocess(g):
    in_nodes, out_nodes = g.edges()
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())
    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))
    return neighbor_dict, neighbor_num_list

neighbor_dict, neighbor_num_list = preprocess(g)
in_dim = feat.shape[1]
detector = GNNAnomalyDetctor(in_dim, in_dim, device)
detector(g, feat, neighbor_dict, neighbor_num_list, device, epoch=10)



