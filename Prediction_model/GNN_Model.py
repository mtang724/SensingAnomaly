import torch.nn as nn
from dgl.nn import GraphConv
import torch.functional as F

class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Net, self).__init__()
        self.layer1 = GraphConv(in_dim, hidden_dim)
        self.layer2 = GraphConv(hidden_dim, out_dim)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x
net = Net(12, 6, 2)
print(net)