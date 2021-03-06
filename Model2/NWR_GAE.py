from Model2.layers import MLP, MLP_generator
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GINConv, GraphConv, GATConv

import scipy
import scipy.optimize
import torch.multiprocessing as mp
import time
import random


# FNN with gumbal_softmax
class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, layer_num):
        super(FNN, self).__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.linear2(F.relu(x))
        x = F.gumbel_softmax(x)
        return x


def chamfer_loss(predictions, targets, mask):
    if mask == 0:
        return 0
    predictions = predictions[:, :mask, :]
    targets = targets[:, :mask, :]
    # predictions and targets shape :: (n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = (predictions - targets).pow(2).mean(1)
    loss = squared_error.min(1)[0] + squared_error.min(2)[0]
    return loss.mean()


def outer(a, b=None):
    if b is None:
        b = a
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b


def per_sample_hungarian_loss(sample_np):
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(sample_np)
    return row_idx, col_idx


def hungarian_loss(predictions, targets, mask, pool):
    # predictions and targets shape :: (n, c, s)
    predictions = predictions[:,:mask,:]
    targets = targets[:,:mask,:]
    # print(predictions.shape)
    predictions = predictions.permute(0, 2, 1)
    targets = targets.permute(0, 2, 1)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = (predictions - targets).pow(2).mean(1)
    squared_error_np = squared_error.detach().cpu().numpy()
    indices = pool.map(per_sample_hungarian_loss, squared_error_np)
    # print(indices)
    losses = [sample[row_idx, col_idx].mean() for sample, (row_idx, col_idx) in zip(squared_error, indices)]
    total_loss = torch.mean(torch.stack(list(losses)))
    return total_loss, indices[0][1]


# GNN encoder to encoder node embeddings, and classifying which Gaussian Distribution the node will fall
def generate_gt_neighbor(neighbor_dict, node_embeddings, neighbor_num_list, in_dim):
    max_neighbor_num = max(neighbor_num_list)
    all_gt_neighbor_embeddings = []
    for i, embedding in enumerate(node_embeddings):
        neighbor_indexes = neighbor_dict[i]
        neighbor_embeddings = []
        for index in neighbor_indexes:
            neighbor_embeddings.append(node_embeddings[index].tolist())
        if len(neighbor_embeddings) < max_neighbor_num:
            for _ in range(max_neighbor_num - len(neighbor_embeddings)):
                neighbor_embeddings.append(torch.zeros(in_dim).tolist())
        all_gt_neighbor_embeddings.append(neighbor_embeddings)
    return all_gt_neighbor_embeddings


class GNNStructEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_num, sample_size, device, GNN_name="GIN"):
        super(GNNStructEncoder, self).__init__()
        self.n_distribution = 7 # How many gaussian distribution should exist
        self.out_dim = hidden_dim
        if GNN_name == "GIN":
            self.linear1 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv1 = GINConv(apply_func=self.linear1, aggregator_type='sum')
            self.linear2 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv2 = GINConv(apply_func=self.linear2, aggregator_type='sum')
        elif GNN_name == "GCN":
            self.graphconv1 = GraphConv(hidden_dim, hidden_dim)
            self.graphconv2 = GraphConv(hidden_dim, hidden_dim)
        else:
            self.graphconv = GATConv(hidden_dim, hidden_dim, num_heads=10)
        # self.neighbor_num_list = neighbor_num_list
        self.linear_classifier = MLP(1, hidden_dim, hidden_dim, self.n_distribution)
        self.neighbor_generator = MLP_generator(hidden_dim, hidden_dim, sample_size).to(device)
        # Gaussian Means, and std
        self.gaussian_mean = nn.Parameter(torch.FloatTensor(sample_size, self.n_distribution, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(sample_size, self.n_distribution, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.m = torch.distributions.Normal(torch.zeros(sample_size, self.n_distribution, hidden_dim), torch.ones(sample_size, self.n_distribution, hidden_dim))

        # Before MLP Gaussian Means, and std
        self.mlp_gaussian_mean = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.mlp_gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.mlp_m = torch.distributions.Normal(torch.zeros(hidden_dim), torch.ones(hidden_dim))

        # Decoders
        self.degree_decoder = FNN(hidden_dim, hidden_dim, 1, 4)
        # self.degree_loss_func = FocalLoss(int(max_degree_num) + 1)
        self.degree_loss_func = nn.MSELoss()
        self.pool = mp.Pool(1)
        self.in_dim = in_dim
        self.sample_size = sample_size

    def forward_encoder(self, g, h):
        # Apply graph convolution and activation
        # l1 = torch.relu(self.graphconv1(g, h))
        # l4 = torch.relu(self.graphconv2(g, l1))
        l5 = self.graphconv1(g, h) # 5 layers
        return l5, h, None

    def neighbor_decoder(self, gij, ground_truth_degree_matrix, temp, g, h, neighbor_dict, device, l4):
        degree_logits = self.degree_decoding(gij)
        ground_truth_degree_matrix = torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix.float())
        _, degree_masks = torch.max(degree_logits.data, dim=1)
        h_loss = 0
        loss_list = []
        total_sample_time = 0
        total_matching_time = 0
        for _ in range(3):
            local_index_loss = 0
            for i1, embedding in enumerate(gij):
                neighbor_embeddings1 = []
                neighbor_indexes = neighbor_dict[i1]
                mask_len = self.sample_size
                if len(neighbor_indexes) < self.sample_size:
                    mask_len = len(neighbor_indexes)
                    sample_indexes = neighbor_indexes
                else:
                    sample_indexes = random.sample(neighbor_indexes, 5)
                for index in sample_indexes:
                    neighbor_embeddings1.append(l4[index].tolist())
                if len(neighbor_embeddings1) < self.sample_size:
                    for _ in range(self.sample_size - len(neighbor_embeddings1)):
                        neighbor_embeddings1.append(torch.zeros(self.out_dim).tolist())
                start_time = time.time()
                zij = F.gumbel_softmax(self.linear_classifier(embedding), tau=temp)
                std_z = self.m.sample().to(device)
                var = self.gaussian_mean + self.gaussian_log_sigma.exp() * std_z
                var = F.dropout(var, 0.2)
                nhij = zij @ var
                generated_neighbors = nhij.tolist()
                sample_time = time.time() - start_time
                total_sample_time += sample_time
                generated_neighbors = torch.unsqueeze(torch.FloatTensor(generated_neighbors), dim=0)
                target_neighbors = torch.unsqueeze(torch.FloatTensor(neighbor_embeddings1), dim=0)
                start_time = time.time()
                new_loss, new_index = hungarian_loss(generated_neighbors, target_neighbors, mask_len, self.pool)
                matching_time = time.time() - start_time
                total_matching_time += matching_time
                local_index_loss += new_loss
            loss_list.append(local_index_loss)
        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)
        loss = h_loss + degree_loss * 10
        return loss, self.forward_encoder(g, h)[0]

    def degree_decoding(self, node_embeddings):
        degree_logits = self.degree_decoder(node_embeddings)
        return degree_logits

    def forward(self, g, h, ground_truth_degree_matrix, neighbor_dict, neighbor_num_list, temp, device):
        gij, l4, l3 = self.forward_encoder(g, h)
        loss, hij = self.neighbor_decoder(gij, ground_truth_degree_matrix, temp, g, h, neighbor_dict, device, l4)
        return loss, hij