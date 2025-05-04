import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool, GraphConv, Linear
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, dim):
        super(GCN, self).__init__()

        self.dim = dim
        self.num_features = num_features
        self.num_classes = num_classes

        self.conv1 = GraphConv(self.num_features, dim)
        self.conv2 = GraphConv(dim, dim)
        self.conv3 = GraphConv(dim, dim)
        # self.conv4 = GraphConv(dim, dim)
        # self.conv5 = GraphConv(dim, dim)

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, self.num_classes)

    def forward(self, x, edge_index, batch = None, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        embedding = global_add_pool(x, batch)
        x = self.lin1(embedding).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x, embedding

    
    def predict(self, x, edge_index, batch=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        edge_index = edge_index.to(device)
        if batch is not None:
          batch = batch.to(device)
        out, graph_embedding = self.forward(x, edge_index, batch)
        out = F.softmax(out, dim=1)
        return out, graph_embedding