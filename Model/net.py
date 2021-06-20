import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
from torch_geometric.nn import GATConv
from utils import *

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GAE(nn.Module):
    def __init__(self, in_channels, conv_ch, dropout=0.5):
        super(GAE, self).__init__()
        self.conv1 = GATConv(in_channels, conv_ch)
        #self.dropout1 = nn.Dropout(dropout)
        self.conv2 = GATConv(conv_ch, conv_ch*2)
        #self.dropout2 = nn.Dropout(dropout)

        self.edge_decoder_conv = GATConv(conv_ch*2, conv_ch)
        #self.dropout3 = nn.Dropout(dropout)

        self.x_decoder_conv1 = GATConv(conv_ch*2, conv_ch)
        #self.dropout4 = nn.Dropout(dropout)
        self.x_decoder_conv2 = GATConv(conv_ch, in_channels)
        #self.dropout5 = nn.Dropout(dropout)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = torch.nonzero(edge_index).T
        x = F.relu(self.conv1(x, edge_index))
        z = F.relu(self.conv2(x, edge_index))


        recon_edge = F.relu(self.edge_decoder_conv(z,edge_index))
        recon_edge = torch.sigmoid(torch.matmul(recon_edge, recon_edge.T))

        x = F.relu(self.x_decoder_conv1(z, edge_index))
        x = F.relu(self.x_decoder_conv2(x, edge_index))

        return recon_edge, x, z

##################################################################################
# Model (CNN Version)
##################################################################################
class GraphDetector(nn.Module):
    def __init__(self, in_channels, conv_ch, dropout, original_dim, num_head):
        super(GraphDetector, self).__init__()
        self.dropout = dropout
        self.conv_ch = conv_ch

        # Encoder
        self.graph_conv1 = GATConv(in_channels, conv_ch, num_head, concat=False)
        self.conv1 = nn.Conv1d(conv_ch, conv_ch, kernel_size=2)
        self.graph_conv2 = GATConv(conv_ch, conv_ch // 2, num_head, concat=False)
        self.conv2 = nn.Conv1d(conv_ch // 2, conv_ch // 2, kernel_size=2)

        # Reconstruction
        self.reconstruct1 = GATConv(conv_ch // 2, conv_ch, num_head, concat=False)
        self.reconstruct2 = nn.Linear(conv_ch, in_channels)
        self.reconstruct3 = nn.Linear(in_channels, original_dim)

        # Forecasting
        self.forecast1 = GATConv(conv_ch // 2, conv_ch, num_head, concat=False)
        self.forecast2 = nn.Linear(conv_ch, in_channels)
        self.forecast3 = nn.Linear(in_channels, original_dim)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, x, edge):
        """
        :param x: Feature Matrix, Time x Node x Channel
        :param edge: Adjacency matrix Time x Node x Node
        :return: Reconstruction G_t, Forecasting G_t, Embedding E_{t-1}
        """
        out = torch.zeros((x.shape[0], x.shape[1], self.conv_ch), dtype=torch.float).to(x.device)  # TODO: Better Way?
        for i in range(x.shape[0]):
            edge_index = torch.nonzero(edge[i]).T
            out[i] = self.graph_conv1(x[i], edge_index)

        out = F.elu(self.conv1(out.permute([1,2,0]))) # N x C x T
        out = F.dropout(out,self.dropout).permute([2,0,1])

        out2 = torch.zeros((x.shape[0], x.shape[1], self.conv_ch//2), dtype=torch.float).to(x.device)
        for i in range(out.shape[0]):
            edge_index = torch.nonzero(edge[i]).T
            out2[i] = self.graph_conv2(out[i], edge_index)
        E = F.dropout(F.elu(self.conv2(out2.permute([1,2,0]))), self.dropout)  # N x C x T
        E = E.permute([2,0,1])  # T x N x C


        edge_index = torch.nonzero(edge[-1]).T
        recon = F.elu(self.reconstruct1(E[-1], edge_index))
        recon = F.dropout(recon, self.dropout)
        recon = torch.tanh(self.reconstruct2(recon)) # F.elu(self.reconstruct2(recon))
        recon = self.reconstruct3(recon)

        edge_index = torch.nonzero(edge[-2]).T
        forecast = F.elu(self.forecast1(E[-2], edge_index))
        forecast = F.dropout(forecast, self.dropout)
        forecast = torch.tanh(self.reconstruct2(forecast)) # F.elu(self.forecast2(forecast))
        forecast = self.forecast3(forecast)

        # recon = torch.tanh(recon)
        # forecast = torch.tanh(forecast)

        return recon, forecast, E[-2]


class NodeDetector(nn.Module):
    def __init__(self, in_channels, embedding_channels, conv_ch, dropout, original_dim, num_head):
        super(NodeDetector, self).__init__()
        self.dropout = dropout
        self.conv_ch = conv_ch
        self.original_dim = original_dim

        # Projection
        self.node_projection = nn.Parameter(torch.Tensor(in_channels, conv_ch).float())
        self.embedding_projection = nn.Parameter(torch.Tensor(embedding_channels, conv_ch).float())

        # Node aggregation
        self.node_aggr1 = nn.Conv1d(conv_ch, conv_ch, kernel_size=2, stride=2)
        self.node_aggr2 = nn.Linear(conv_ch, conv_ch // 2)

        # Heterogenous Graph Projection
        self.masked_node_projection = nn.Parameter(torch.Tensor(conv_ch//2, conv_ch//2).float())
        self.normal_node_projection = nn.Parameter(torch.Tensor(conv_ch//2, conv_ch//2).float())

        # Graph aggregation
        self.graph_conv1 = GATConv(conv_ch // 2, conv_ch // 2, num_head, concat=False)
        self.graph_conv2 = GATConv(conv_ch // 2, conv_ch // 2, num_head, concat=False)

        # Reconstruction
        self.reconstruct = nn.Linear(conv_ch // 2, original_dim)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_projection, gain=math.sqrt(2.0))
        torch.nn.init.xavier_uniform_(self.embedding_projection, gain=math.sqrt(2.0))
        torch.nn.init.xavier_uniform_(self.masked_node_projection, gain=math.sqrt(2.0))
        torch.nn.init.xavier_uniform_(self.normal_node_projection, gain=math.sqrt(2.0))
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, x, edge, E):
        """
        :param x: Feature Matrix,  Node x Channel
        :param edge: Adjacency matrix Node x Node
        :param E: Embedding E_{t-1} Node x Channel'
        :return: Reconstruction Each Node
        """
        edge_index = torch.nonzero(edge).T

        # Project node feature and node embedding to same space
        x = torch.matmul(x, self.node_projection)  # N x conv_ch
        E = torch.matmul(E, self.embedding_projection)  # N x conv_ch

        recon = torch.ones((x.shape[0], self.original_dim), dtype=torch.float).to(x.device)  # OR change to use a larger tensor to avoid a lot of for loop?
        for i in range(x.shape[0]):
            # Mask each node
            masked = torch.ones_like(x)
            masked[i] *= 0
            masked *= x

            # Aggregation of each node feature and corresponding embeding.
            masked = torch.stack([E, masked]).permute([1,2,0])  # N x C x 2
            masked = self.node_aggr1(masked).squeeze() # N x C
            masked = F.dropout(torch.tanh(masked), self.dropout)
            masked = self.node_aggr2(masked)  # N x C//2

            # Project mask node and other nodes to same place
            projected_masked = torch.matmul(masked, self.normal_node_projection)
            projected_masked[i] = torch.matmul(masked[i], self.masked_node_projection)

            # Graph aggregation
            projected_masked = self.graph_conv1(projected_masked, edge_index)
            projected_masked = F.dropout(F.elu(projected_masked), self.dropout)
            projected_masked = self.graph_conv2(projected_masked, edge_index)
            projected_masked = F.dropout(F.elu(projected_masked), self.dropout) # N x C//2

            # Reconstruction
            recon[i] = self.reconstruct(projected_masked[i].view([1,-1])).squeeze()

        recon = torch.tanh(recon)

        return recon


class Detector(nn.Module):
    def __init__(self, in_channels, conv_ch, num_sensor, embedding_dim, num_head, dropout=0.0, original_dim=1):
        super(Detector, self).__init__()

        self.sensor_embedding = nn.Embedding(num_sensor, embedding_dim)

        self.lin1 = nn.Linear(embedding_dim, embedding_dim)
        self.lin2 = nn.Linear(embedding_dim, embedding_dim)

        self.projection = nn.Parameter(torch.Tensor(1, num_sensor, in_channels, in_channels).float())

        self.graph_detector = GraphDetector(in_channels+embedding_dim, conv_ch, dropout, original_dim, num_head)

        self.node_detector = NodeDetector(in_channels+embedding_dim, conv_ch//2, conv_ch//2, dropout, original_dim, num_head)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.projection, gain=math.sqrt(2.0))
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, x, edge, sensor_indx):
        """
        :param x: Feature Matrix,  Time x Node x Channel
        :param edge: Adjacency matrix Time x Node x Node
        :param sensor_indx: List of sensors
        :return: Reconstruction G_t, Forecasting G_t, Reconstruction Nodes
        """

        # Sensor Embedding
        emb = self.sensor_embedding(sensor_indx) # N x C'
        sen_emb = emb.view([1, len(sensor_indx), -1]).expand([x.shape[0], -1, -1])

        # Projection Graph
        x = torch.matmul(x.view(x.shape[0], x.shape[1], 1, -1), self.projection.expand([x.shape[0], -1, -1, -1]))
        x = torch.cat([x.squeeze(),sen_emb],dim=-1)

        if edge is not None:
            recon, forecast, E_t_1 = self.graph_detector(x, edge)
            node_recon = self.node_detector(x[-1], edge[-1], E_t_1)
        else:
            # edge constructor
            nodevec1 = torch.tanh(self.lin1(emb))
            nodevec2 = torch.tanh(self.lin2(emb))
            a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
            adj = F.relu(torch.tanh(a)).view([1, len(sensor_indx), -1]).expand([x.shape[0], -1, -1])

            recon, forecast, E_t_1 = self.graph_detector(x, adj)
            node_recon = self.node_detector(x[-1], adj[0], E_t_1)

        return recon, forecast, node_recon


if __name__ == '__main__':
    gd = Detector(10, 8, 15, 5, 2, 0.0, 5)
    x = torch.rand([3,15,10])
    e = torch.rand(3,15,15)>0.5
    i = [i for i in range(15)]
    i = torch.tensor(i, dtype=torch.int32)
    E = torch.rand([15,4])

    y1, y2, y3 = gd(x, e, i)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)