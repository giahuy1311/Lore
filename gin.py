import torch
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn.models import MLP
from torch_geometric.data import Batch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import pickle

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
    
    def predict(self, data_loader, device):
        model = self.to(device)
        predictions = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                output = model(data.x, data.edge_index, data.batch, batch_size=data.num_graphs)
                predictions.append(output.cpu())
        
        predictions = torch.cat(predictions, dim=0)
        return predictions.argmax(dim=-1).numpy()
    
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
def get_graph_info_array(model, device, loader):
    model.eval()
    
    graph_info_list = []
    
    for data in loader:
        data = data.to(device)
        _, node_embeddings = model(data.x, data.edge_index, data.batch, data.batch_size, return_embeddings=True)
        
        num_graphs = data.batch_size
        for i in range(num_graphs):
            mask = (data.batch == i)  
            # mean_node_embedding = node_embeddings[mask].mean(dim=0).cpu().numpy()  # Mean pooling
            # sum_node_embedding = node_embeddings[mask].sum(dim=0).cpu().numpy()   # Sum pooling
            # mean_feature_vector = data.x[mask].mean(dim=0).cpu().numpy()
            
            # Lưu thông tin của từng graph dưới dạng dictionary
            graph_info = {
                # "mean_node_embedding": mean_node_embedding,
                # "mean_feature_vector": mean_feature_vector,
                # "sum_pooling": sum_node_embedding,
                "x": data.x[mask].cpu().numpy(),  # Node features
                "edge_index": data.edge_index.cpu().numpy(),  # Edge list
                "y": data.y[i].item()  # Label
            }
            
            graph_info_list.append(graph_info)
    
    # Chuyển thành numpy array
    graph_info_array = np.array(graph_info_list, dtype=object)  
    return graph_info_array

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
    
    # Forward qua model
    output = model(node_features, edge_index, batch, batch_size=1)
    
    # Lấy nhãn dự đoán
    predictions = output.argmax(dim=-1)  # Trả về tensor có nhiều giá trị
    return predictions.numpy().reshape(-1)

@torch.no_grad()
def predict_single_graph(model, graph, device):
    model.eval()
    
    # Tạo batch giả (dù chỉ có 1 graph)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)  # Tất cả node thuộc batch 0
    batch = Batch.from_data_list([graph]).to(device)

    # Forward qua mô hình
    out = model(batch.x, batch.edge_index, batch.batch, batch_size=1)
    
    # Lấy nhãn dự đoán
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

        # Dự đoán nhãn mới cho đồ thị đã sửa đổi
        new_graph = Data(x=graph.x, edge_index=new_edge_index)
        new_graph.y = torch.tensor([predict_single_graph(model, new_graph, device)], dtype=torch.long)

        modified_graphs.append(new_graph)

    return modified_graphs

@torch.no_grad()
def get_mean_node_embeddings_df(model,device,loader):
    model.eval()
    all_graphs = []  
    labels = []
    for data in loader:
        data = data.to(device)
        _, node_embeddings = model(data.x, data.edge_index, data.batch, data.batch_size, return_embeddings=True)

        num_graphs = data.batch_size
        for i in range(num_graphs):
            mask = (data.batch == i)  
            mean_node_embedding = node_embeddings[mask].mean(dim=1).cpu().numpy() 
            labels.append(data.y[i].item())
            all_graphs.append(mean_node_embedding.flatten().tolist())

    max_nodes = max(len(graph) for graph in all_graphs)
    padded_graphs = [list(graph) + [None] * (max_nodes - len(graph)) for graph in all_graphs]
    columns = [f'node_{i}' for i in range(max_nodes)]

    df = pd.DataFrame(padded_graphs, columns=columns)
    df['y'] = labels
    return df

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
    samples = generate_modified_graphs(graph_x_info,blackbox, device, 200)
    data_loader = DataLoader(samples, batch_size=32, shuffle=False)
    dfZ = get_mean_node_embeddings_df(blackbox, device, data_loader)
    Z = dfZ.drop(columns=['y']).values
    
    return dfZ, Z