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
from model.gin import GIN
from model.gin import *
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
from genetic import *

def permute_graph(data):
    num_nodes = data.num_nodes
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    mapping = {node: i for i, node in enumerate(nodes)}
    print(mapping)
    
    edge_index = data.edge_index.clone()
    edge_index[0] = torch.tensor([mapping[node] for node in edge_index[0].tolist()])
    edge_index[1] = torch.tensor([mapping[node] for node in edge_index[1].tolist()])
    
    x = data.x.clone()
    x = x[nodes]

    permuted_data = data.clone()
    permuted_data.edge_index = edge_index
    permuted_data.x = x
    return permuted_data
    
def split_loader(dataset):
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_idx, temp_idx = train_test_split(indices, train_size=train_size, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_size, random_state=42)

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

def generate_dataset(dataset_name):
    if dataset_name == 'BA2Motif':
        dataset = BA2MotifDataset(root='data/BA2Motif')
    elif dataset_name == 'MUTAG':
        dataset = TUDataset(root="data/TUDataset", name="MUTAG")
    return dataset

def prepare_dataset(df):
    columns = df.columns.tolist()
    class_name = 'y'

    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)
    discrete = ['y']
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete,
                                                   continuous=None)
    print('discrete: ', discrete)
    print('continuous: ', continuous)
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
    ba_name = 'BA2Motif'
    mutag_name = 'MUTAG'
    
    dataset = generate_dataset(ba_name)
    test_dataset = dataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GIN(
        in_channels=dataset.num_features,
        hidden_channels=32,
        out_channels=dataset.num_classes,
        num_layers=5,
    ).to(device)
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()
    model.to(device)

    # blackbox
    df = prepare_dataframe(test_dataset, model, device, ground_truth=True, only_edge=True)
    print('df: ', df)
    dataset = prepare_dataset(df)
    path_data = 'datasets/'
    idx_record2explain = 724 #(72 & 724)
    
    #generate sample
    graphX = test_dataset[idx_record2explain]
    
    graphX = Data(x=graphX.x, edge_index=filter_undirected_edges(graphX.edge_index))
    final_population = genetic_algorithm(graphX = graphX, populationSize=400, generations=20, blackbox=model, 
                                     distance_function=my_distance2, 
                                     alpha1=0.5, alpha2=0.5)
    
    dfZ = prepare_dataframe(final_population, model, device, ground_truth=False, only_edge=True)
    print('df_0: ', dfZ[dfZ['y'] == 0].shape[0])
    print('df_1: ', dfZ[dfZ['y'] == 1].shape[0])
    graphX.edge_index = ensure_undirected(graphX.edge_index)

    y_pred_list = df['y'].tolist()
    y2E = np.asarray([dataset['possible_outcomes'][i] for i in y_pred_list])

    explanation, infos = lore.explain_graph(idx_record2explain, dfZ, graphX, dataset, model,
                                        discrete_use_probabilities=True,
                                        continuous_function_estimation=False,
                                        returns_infos=True,
                                        path=path_data, sep=';', log=False)
    dfX2E = df.to_dict('records')
    # dfx = dfX2E[idx_record2explain]
    dfx = prepare_dataframe([graphX], model, device, ground_truth=False, only_edge=True)
    # x = build_df2explain(blackbox, X2E[idx_record2explain].reshape(1, -1), dataset).to_dict('records')[0]
    covered = lore.get_covered(explanation[0][1], dfX2E, dataset)
    # for sample in covered:
    #     print(dataset['df'].iloc[sample])
        
    print("covered len:", len(covered))
    print('x = %s' % dfx)
    
    print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
    for delta in explanation[1]:
        print('delta', delta)
    def eval(x, y):
        return 1 if x == y else 0
        
    precision = [1-eval(v, explanation[0][0][dataset['class_name']]) for v in y2E[covered]]
    print(precision)
    print(np.mean(precision), np.std(precision))
    
    
if __name__ == "__main__":
    main()

    