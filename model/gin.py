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

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch, batch_size, return_embeddings=False):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        
        node_embeddings = x 
        graph_embedding = global_add_pool(x, batch, size=batch_size)
        output = self.mlp(graph_embedding)
        
        if return_embeddings:
            return output, node_embeddings
        return output
    
    def predict(self, x, edge_index, batch, batch_size):
        out = self.forward(x, edge_index, batch, batch_size)
        return out.argmax(dim=-1)
    
    # def predict(self, mean_embedding, decoder, batch, batch_size):
    #     mean_embedding = torch.tensor(mean_embedding, dtype=torch.float32)
    #     reconstructed_embeddings = decoder(mean_embedding) 

    #     graph_embedding = global_add_pool(reconstructed_embeddings,batch, batch_size)

    #     graph_label_pred = self.mlp(graph_embedding)

    #     _, predicted = graph_label_pred.max(dim=1)
    #     return predicted.item()
    
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, data.batch_size)
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
        out = model(data.x, data.edge_index, data.batch, data.batch_size)
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
def get_graph_features_df(device, loader):
    all_graphs = []  
    labels = []
    atom_types = ["C", "N", "O", "F", "I", "Cl", "Br"]
    for data in loader:
        data = data.to(device)

        num_graphs = data.batch_size
        for i in range(num_graphs):
            mask = (data.batch == i)  

            # Lấy node labels thay vì data.x
            node_labels = [atom_types.index(atom_types[x.argmax().item()]) + 1 for x in data.x[mask]]

            all_graphs.append(node_labels)
            labels.append(data.y[i].item())

    # Xác định số node tối đa trong batch (để padding)
    max_nodes = max(len(graph) for graph in all_graphs)

    # Padding cho các đồ thị có ít node hơn max_nodes
    padded_graphs = [graph + [0] * (max_nodes - len(graph)) for graph in all_graphs]

    # Tạo tên cột dựa trên số lượng node
    columns = [f'node_{i}' for i in range(max_nodes)]

    # Tạo DataFrame
    df = pd.DataFrame(padded_graphs, columns=columns)
    df['y'] = labels

    return df

@torch.no_grad()
def extract_graph_info(loader):
    graph_info_array = []
    
    for batch in loader:
        edge_index = batch.edge_index
        ptr = batch.ptr  
        num_graphs = len(ptr) - 1 
        
        for i in range(num_graphs):
            start, end = ptr[i].item(), ptr[i + 1].item()  
            mask = (batch.batch == i) 
            
            node_features = batch.x[mask].cpu().numpy()
    
            sub_edge_index = edge_index[:, (edge_index[0] >= start) & (edge_index[0] < end)] - start
            sub_edge_index = sub_edge_index.cpu().numpy()
            label = batch.y[i].item()
            
            graph_info = {
                "x": node_features,
                "edge_index": sub_edge_index,
                "y": label
            }
            graph_info_array.append(graph_info)
    
    return graph_info_array

@torch.no_grad()
def predict_graph(model, device, graph_info):
    model.eval()
    node_features = torch.tensor(graph_info["x"], dtype=torch.float32, device=device)
    edge_index = torch.tensor(graph_info["edge_index"], dtype=torch.long, device=device)

    batch = torch.zeros(node_features.shape[0], dtype=torch.long, device=device)
    
    output = model(node_features, edge_index, batch, batch_size=1)

    predictions = output.argmax(dim=-1) 
    return predictions.numpy().reshape(-1)

@torch.no_grad()
def predict_single_graph(model, graph, device):
    model.eval()
    
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)  
    batch = Batch.from_data_list([graph]).to(device)

    out = model(batch.x, batch.edge_index, batch.batch, batch_size=1)
    
    predicted_label = out.argmax(dim=-1).item()
    
    return predicted_label

def generate_modified_graphs(graph, model, device, num_graphs=50):
    modified_graphs = []
    num_nodes = graph.x.size(0)

    for _ in range(num_graphs):
        edge_index = graph.edge_index.clone()
        edges = edge_index.t().tolist()

        mode = "add" if torch.rand(1).item() > 0.5 else "remove"

        if mode == "remove" and len(edges) > 0:
            remove_idx = torch.randint(0, len(edges), (1,)).item()
            edges.pop(remove_idx)

        elif mode == "add":
            while True:
                u, v = torch.randint(0, num_nodes, (2,)).tolist()
                if u != v and [u, v] not in edges and [v, u] not in edges:
                    edges.append([u, v])
                    break

        new_edge_index = torch.tensor(edges, dtype=torch.long).t()

        new_graph = Data(x=graph.x, edge_index=new_edge_index)
        new_graph.y = torch.tensor([predict_single_graph(model, new_graph, device)], dtype=torch.long)

        modified_graphs.append(new_graph)

    return modified_graphs

# @torch.no_grad()
# def get_mean_node_embeddings_df(model,device,loader):
#     model.eval()
#     all_graphs = []  
#     labels = []
#     for data in loader:
#         data = data.to(device)
#         _, node_embeddings = model(data.x, data.edge_index, data.batch, data.batch_size, return_embeddings=True)

#         num_graphs = data.batch_size
#         for i in range(num_graphs):
#             mask = (data.batch == i)  
#             mean_node_embedding = node_embeddings[mask].mean(dim=1).cpu().numpy() 
#             labels.append(data.y[i].item())
#             all_graphs.append(mean_node_embedding.flatten().tolist())

#     max_nodes = max(len(graph) for graph in all_graphs)
#     padded_graphs = [list(graph) + [0] * (max_nodes - len(graph)) for graph in all_graphs]
#     columns = [f'n_{i}' for i in range(max_nodes)]

#     df = pd.DataFrame(padded_graphs, columns=columns)
#     df['y'] = labels
#     return df


# @torch.no_grad()
# def create_adjacency_matrix(edge_index, num_nodes):
#     adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int) 
#     edges = edge_index.cpu().numpy().T 
    
#     for src, dst in edges:
#         adj_matrix[src, dst] = 1  
#         adj_matrix[dst, src] = 1  
    
#     return adj_matrix

def create_adjacency_matrix(edge_index, num_nodes):
    """Create an adjacency matrix from edge indices."""
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int)
    if edge_index.numel() > 0:  # Check if there are any edges
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_matrix[src, dst] = 1
            adj_matrix[dst, src] = 1  # Ensure undirected graph
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

def prepare_dataframe(list_graph, model, device, predict = False):
    all_embeddings = []
    edge_dicts = []
    labels = []
    
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for graph in list_graph:
            x = graph.x.to(device)
            edge_index = ensure_undirected(graph.edge_index).to(device)

            _, node_embeddings = model(x, edge_index, None, 1, return_embeddings=True)

            mean_node_embedding = node_embeddings.mean(dim=1).cpu().numpy()
            all_embeddings.append(mean_node_embedding.tolist())
            
            num_nodes = graph.num_nodes
            adj_matrix = create_adjacency_matrix(graph.edge_index.cpu(), num_nodes)
            
            # Convert adjacency matrix to dictionary
            edge_dict = {f'n{r}_n{c}': adj_matrix[r, c].item() 
                         for r in range(num_nodes) for c in range(r, num_nodes)}
            edge_dicts.append(edge_dict)
            
            if predict:
                labels.append(graph.y.item())
            else :
                prediction = model.predict(x, edge_index, None, 1)
                labels.append(prediction.item())
                
    
    # Create DataFrame
    max_embed_dim = max(len(embed) for embed in all_embeddings)
    embed_columns = [f'nE_{i+1}' for i in range(max_embed_dim)]
    df_embeddings = pd.DataFrame(all_embeddings, columns=embed_columns)
    
    # Create DataFrame for edge information
    df_edges = pd.DataFrame(edge_dicts).fillna(0).astype(int)
    
    # Combine DataFrames
    result_df = pd.concat([df_embeddings, df_edges], axis=1)
    result_df['y'] = labels
    
    return result_df

# @torch.no_grad()
# def get_mean_node_embeddings_df(model, device, loader):
#     model.eval()
#     all_graphs = []  
#     labels = []
#     edge_dicts = []

#     for data in loader:
#         data = data.to(device)
#         _, node_embeddings = model(data.x, data.edge_index, data.batch, data.batch_size, return_embeddings=True)

#         num_graphs = data.batch_size
#         for i in range(num_graphs):
#             mask = (data.batch == i)  
#             node_indices = torch.where(mask)[0].cpu().numpy()
#             mean_node_embedding = node_embeddings[mask].mean(dim=1).cpu().numpy()

#             labels.append(data.y[i].item())
#             all_graphs.append(mean_node_embedding.flatten().tolist())

#             node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
#             edges = data.edge_index[:, mask[data.edge_index[0]]].cpu().numpy().T
#             edges_mapped = [(node_mapping[src], node_mapping[dst]) for src, dst in edges]

#             # ma trận kề
#             num_nodes = len(node_indices)
#             adj_matrix = create_adjacency_matrix(torch.tensor(edges_mapped).T, num_nodes)

#             # convert ma trận kề thành dict để concat vào DataFrame
#             edge_dict = {f'n{r}_n{c}': adj_matrix[r, c] for r in range(num_nodes) for c in range(r, num_nodes)}
#             edge_dicts.append(edge_dict)

#     max_nodes = max(len(graph) for graph in all_graphs)
#     embed_columns = [f'nE_{i+1}' for i in range(max_nodes)]
#     df = pd.DataFrame(all_graphs, columns=embed_columns)

#     df_edges = pd.DataFrame(edge_dicts).fillna(0).astype(int)  

#     df = pd.concat([df, df_edges], axis=1)
#     df['y'] = labels  

#     return df


def convert_dict_to_graph(graph_info):
    x = torch.tensor(graph_info["x"], dtype=torch.float)
    edge_index = torch.tensor(graph_info["edge_index"], dtype=torch.long)
    y = torch.tensor([graph_info["y"]], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y)
def save_graphs_to_pickle(graphs, filename="graphs.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(graphs, f)



def generate_samples(graph2X, blackbox, device, num_samples=100):
    graph_x_info = convert_dict_to_graph(graph2X)
    print("Graph info: ",graph_x_info)
    # Lưu graph đã convert
    save_graphs_to_pickle(graph_x_info, "graph2X.pkl")
    samples = generate_modified_graphs(graph_x_info,blackbox, device, num_samples)
    data_loader = DataLoader(samples, batch_size=32, shuffle=False)
    dfZ = get_mean_node_embeddings_df(blackbox, device, data_loader)
    Z = dfZ.drop(columns=['y']).values
    
    return dfZ, Z

