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

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True)

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
    
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

def main():
    
    dataset = generate_dataset()
    df = build_df(dataset)
    dataset = prepare_dataset(df) # dùng lại từ lore
    
    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # blackbox
    model = RandomForestClassifier(random_state=42) # thay blackbox model
    
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print('Accuracy: ', accuracy)
    
    path_data = 'datasets/'
    idx_record2explain = 111
    X2E = X_test
    y2E = model(X2E)
    y2E = np.asarray([dataset['possible_outcomes'][i] for i in y2E])

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

    