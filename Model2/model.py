import networkx as nx
import dgl
import torch.nn as nn
from dgl.nn import GATConv
from Model2.NWR_GAE import GNNStructEncoder
import torch

class GNNAnomalyDetctor(nn.Module):
    def __init__(self, in_dim, hidden_dim, device):
        super(GNNAnomalyDetctor, self).__init__()
        self.in_dim = in_dim
        self.gatconv = GATConv(in_dim, hidden_dim, num_heads=3)
        self.nwr_gae = GNNStructEncoder(in_dim, hidden_dim, layer_num=2, sample_size=5, device=device)
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, g, feat, neighbor_dict, neighbor_num_list, device, epoch=10):
        opt = torch.optim.Adam(self.nwr_gae.parameters(), lr=5e-3, weight_decay=0.00003)
        for i in range(epoch):
            loss, embedding = self.nwr_gae(g, feat, g.in_degrees(), neighbor_dict, neighbor_num_list, 1, device)
            opt.zero_grad()
            loss.backward()
            print(i, loss.item())
            opt.step()
        print(embedding)


