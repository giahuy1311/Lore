import torch
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn.models import MLP
from torch_geometric.data import Batch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import pickle
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import to_undirected

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels), 
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True)  
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))  

            in_channels = hidden_channels  

        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels), 
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, batch, return_embeddings=False):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x).relu()  

        node_embeddings = x 
        graph_embedding = global_add_pool(x, batch)
        
        output = self.mlp(graph_embedding)
    
        if return_embeddings:
            return output, node_embeddings, graph_embedding
        
        return output

    
    def predict(self, x, edge_index, batch, get_embeddings=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        edge_index = edge_index.to(device)
        if batch is not None:
          batch = batch.to(device)
        out = self.forward(x, edge_index, batch, return_embeddings=get_embeddings)
        probs = F.softmax(out[0], dim=-1)
        if get_embeddings:
            return probs, out[2]
        else:
            return probs
    
    
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(model,loader, device):
    model.eval()
    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


@torch.no_grad()
def get_node_embeddings(model,loader, device):
    model.eval()
    all_embeddings = []
    for data in loader:
        data = data.to(device)
        _, node_embeddings = model(data.x, data.edge_index, data.batch, data.batch_size, return_embeddings=True)
        all_embeddings.append(node_embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

@torch.no_grad()
def predict_graph(model, device, graph_info):
    model.eval()
    node_features = torch.tensor(graph_info["x"], dtype=torch.float32, device=device)
    edge_index = torch.tensor(graph_info["edge_index"], dtype=torch.long, device=device)

    batch = torch.zeros(node_features.shape[0], dtype=torch.long, device=device)
    
    output = model(node_features, edge_index, batch, batch_size=1)

    predictions = output.argmax(dim=-1) 
    return predictions.numpy().reshape(-1)

def create_adjacency_matrix(edge_index, num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int)
    if edge_index.numel() > 0:  
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_matrix[src, dst] = 1
            adj_matrix[dst, src] = 1  
    return adj_matrix

def ensure_undirected(edge_index):
    edge_set = set()
    unique_edges = []
    
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i].tolist()
        edge = tuple(sorted((u, v)))
        
        if edge not in edge_set:
            edge_set.add(edge)
            unique_edges.append([u, v])

    bidirectional_edges = []
    for u, v in unique_edges:
        bidirectional_edges.append([u, v])
        bidirectional_edges.append([v, u])
    
    return torch.tensor(bidirectional_edges, dtype=torch.long).T

def prepare_dataframe(list_graph, model, device, ground_truth = False, only_edge = False, node_label = False, max_nodes = 0):
    all_embeddings = []
    edge_dicts = []
    labels = []
    node_labels = []
    model.eval() 
    max_nodes = max_nodes if max_nodes > 0 else max(len(graph.x) for graph in list_graph)
    with torch.no_grad():
        for graph in list_graph:
            x = graph.x.to(device)
            edge_index = ensure_undirected(graph.edge_index).to(device)

            if only_edge is False:
            # tinh node embeddings va lay mean
                _, node_embeddings = model(x, edge_index, None, 1, return_embeddings=True)
                mean_node_embedding = node_embeddings.mean(dim=1).cpu().numpy()
                all_embeddings.append(mean_node_embedding.tolist())
            
            num_nodes = graph.x.size(0)
            adj_matrix = create_adjacency_matrix(graph.edge_index.cpu(), num_nodes)

            edge_dict = {}
            for r in range(max_nodes):
                for c in range(r, max_nodes):
                    if r < num_nodes and c < num_nodes:
                        edge_dict[f'n{r}_n{c}'] = adj_matrix[r, c].item()
                    else:
                        edge_dict[f'n{r}_n{c}'] = 0
            edge_dicts.append(edge_dict)
            if ground_truth:
                labels.append(graph.y.item())
            else :
                prediction, embedding = model.predict(x, edge_index, None)
                labels.append(prediction.argmax(dim=-1).item())
                
            atom_types = ["C", "N", "O", "F", "I", "Cl", "Br"]
            if node_label:
                node_label_list = ["None"] * max_nodes
                for i in range(num_nodes):
                    node_label_list[i] = atom_types[x[i].argmax().item()]
                node_labels.append(node_label_list)

    max_embed_dim = max(len(embed) for embed in all_embeddings) if all_embeddings else 0
    embed_columns = [f'nE_{i}' for i in range(max_embed_dim)]
    node_labels_columns = [f'nL_{i}' for i in range(max_nodes)]
    result_df = pd.DataFrame()
    if node_labels:
        df_node_labels = pd.DataFrame(node_labels, columns=node_labels_columns)
        result_df = pd.concat([result_df, df_node_labels], axis=1)
    if all_embeddings:
        df_embeddings = pd.DataFrame(all_embeddings, columns=embed_columns)
        result_df = pd.concat([result_df, df_embeddings], axis=1)
    if only_edge:
        df_edges = pd.DataFrame(edge_dicts).fillna(0).astype(int)
        result_df = pd.concat([result_df, df_edges], axis=1)
         
    result_df['y'] = labels
    
    return result_df