import numpy as np
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from util import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lore
from prepare_dataset import *
from neighbor_generator import *
import torch
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.datasets.motif_generator import CycleMotif
import random
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.nn import GINConv, GIN 
import pickle
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GIN
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn.models import MLP

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
def predict(model,graph, device):
    model.eval()
    graph = graph.to(device)

    output, _ = model(graph.x, graph.edge_index)
    pred = output.argmax(dim=-1).item()

    return pred


@torch.no_grad()
def get_node_embeddings(model,loader, device):
    model.eval()
    all_embeddings = []
    for data in loader:
        data = data.to(device)
        _, node_embeddings = model(data.x, data.edge_index, data.batch, data.batch_size, return_embeddings=True)
        all_embeddings.append(node_embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

class GINBlackBox:
    def __init__(self, model_path, dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GIN(in_channels=10, hidden_channels=64, out_channels=2, num_layers=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
    def predict(self, X):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in X:  
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1).cpu().numpy()
                predictions.extend(pred)
        return np.array(predictions)
    
def extract_graph_features(data):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    avg_degree = 2 * num_edges / num_nodes
 
    return [num_nodes, num_edges, avg_degree]

def generate_dataset():
    dataset = BA2MotifDataset(root='data/BA2Motif')
    return dataset.shuffle()

def build_df(dataset):
    features = []
    labels = []
    for data in dataset:
        features.append(extract_graph_features(data))
        labels.append(data.y.item())
    columns = ['num_nodes', 'num_edges', 'avg_degree'] 
    df = pd.DataFrame(features, columns=columns)
    df['y'] = labels
    
    return df
    

def prepare_dataset(df):
    columns = df.columns.tolist()
    class_name = 'y'

    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)
    print(type_features)
    discrete = ['y']
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete,
                                                   continuous=None)
    print(discrete, continuous)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values


    dataset = {
        'df': df, 
        'columns': columns, 
        'class_name': class_name,  
        'possible_outcomes': possible_outcomes, 
        'type_features': type_features, 
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous, 
        'label_encoder': label_encoder,   
        'idx_features': idx_features,  
        'X': X,
        'y': y 
    }

    return dataset

def eval(x, y):
    return 1 if x == y else 0

def extract_features(dataset):
    X = []
    for data in dataset:
        node_features = data.x.mean(dim=0).cpu().numpy()  # Tính trung bình node features
        X.append(node_features)
    return np.array(X) 

def main():
    model_path = 'model/gin_model.pt'
    
    
    dataset = generate_dataset()
    train_dataset, test_dataset, val_dataset = dataset[:800], dataset[800:900], dataset[900:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    model = torch.load(model_path)

    # blackbox
    df = build_df(dataset)
    dataset = prepare_dataset(df)
    
    
    path_data = 'datasets/'
    idx_record2explain = 111
    X2E = get_node_embeddings(model, test_loader, device)
    print('X2E', X2E)

    #y2E = np.asarray([dataset['possible_outcomes'][i] for i in y2E])

    explanation, infos = lore.explain(idx_record2explain, X2E, dataset, model,
                                        ng_function=genetic_neighborhood,
                                        discrete_use_probabilities=True,
                                        continuous_function_estimation=False,
                                        returns_infos=True,
                                        path=path_data, sep=';', log=False)
    
    dfX2E = build_df2explain(model, X2E, dataset).to_dict('records')
    dfx = dfX2E[idx_record2explain]
    # x = build_df2explain(blackbox, X2E[idx_record2explain].reshape(1, -1), dataset).to_dict('records')[0]
    covered = lore.get_covered(explanation[0][1], dfX2E, dataset)
    # for sample in covered:
    #     print(dataset['df'].iloc[sample])
    print('x = %s' % dfx)
    
    print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
    for delta in explanation[1]:
        print('delta', delta)
        
    precision = [1-eval(v, explanation[0][0][dataset['class_name']]) for v in y2E[covered]]
    print(precision)
    print(np.mean(precision), np.std(precision))
    
    
if __name__ == "__main__":
    main()

    