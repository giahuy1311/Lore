from torch import nn
import torch.nn.functional as F

embedding_dim = 32

class MLPDecoder(nn.Module):
    def __init__(self, num_nodes, hidden_dim, embedding_dim):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(num_nodes, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, num_nodes * embedding_dim)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, embedding_dim)  