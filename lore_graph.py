import numpy as np
from util import *
from sklearn.model_selection import train_test_split
import lore
from prepare_dataset import *
from neighbor_generator import *
import torch
import random
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.nn import GIN 
import pickle
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GIN
from torch_geometric.nn.models import MLP
from model.gin import GIN
from model.gin import *
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
from genetic import *
from model.gcn import GCN

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

def load_test_dataset(dataset_name, path_data):
    if dataset_name == 'ba2motif':
        dataset = BA2MotifDataset(root=path_data + 'BA2Motif')
        with open('datasets/Ba2motif_splits.pkl', 'rb') as f:
            split_indices = pickle.load(f)
        test_idx = split_indices['test_idx']
        test_dataset = [dataset[i] for i in test_idx]
        test_dataset = dataset
    elif dataset_name == 'mutag':
        dataset = TUDataset(root=path_data + "TUDataset", name="MUTAG")
        with open('datasets/MUTAG_splits.pkl', 'rb') as f:
            split_indices = pickle.load(f)
        test_idx = split_indices['test_idx']
        test_dataset = [dataset[i] for i in test_idx]
    
    return dataset, test_dataset

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
    ba_name = 'ba2motif'
    mutag_name = 'mutag'
    
    dataset, test_dataset = load_test_dataset(mutag_name, '')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GIN(
        in_channels=dataset.num_features,
        hidden_channels=32,
        out_channels=dataset.num_classes,
        num_layers=5,
    )
    model.load_state_dict(torch.load('model/GIN_mutag.pth'))
    model.eval()

    graph2X = dataset[0]
    # blackbox
    df = prepare_dataframe(test_dataset, model, device, ground_truth=True, only_edge=True, node_label=True)
    print('df: ', df)
    dataset = prepare_dataset(df)
    path_data = 'datasets/'
    idx_record2explain = 0 #(72 & 724)
    
    #generate sample
    
    
    with open('mutagX.pkl', 'wb') as f:
        pickle.dump(graph2X, f)
    
    final_population = genetic_algorithm(graph2X, population_size=2000, generations=2, blackbox=model, 
                                      dataset=test_dataset)
    
    
    list_graph = []
    for individual in final_population:
        graph = from_networkx(individual.graph)
        list_graph.append(graph)
        
    max_nodes = max(len(graph.x) for graph in list_graph)
    dfZ = prepare_dataframe(list_graph, model, device, ground_truth=False, only_edge=True, node_label=True, max_nodes=max_nodes)

    print('df_0: ', dfZ[dfZ['y'] == 0].shape[0])
    print('df_1: ', dfZ[dfZ['y'] == 1].shape[0])
    # graphX = dataset[idx_record2explain]

    y_pred_list = df['y'].tolist()
    y2E = np.asarray([dataset['possible_outcomes'][i] for i in y_pred_list])
    formatted_neighbor = prepare_dataset(dfZ)
    explanation, infos = lore.explain_graph(idx_record2explain, dfZ, graph2X, formatted_neighbor, model,
                                        discrete_use_probabilities=True,
                                        continuous_function_estimation=False,
                                        returns_infos=True,
                                        path=path_data, sep=';', log=True, max_nodes=max_nodes)
    dfX2E = df.to_dict('records')
    # dfx = dfX2E[idx_record2explain]
    dfx = prepare_dataframe([graph2X], model, device, ground_truth=False, only_edge=True)
    # x = build_df2explain(blackbox, X2E[idx_record2explain].reshape(1, -1), dataset).to_dict('records')[0]
    covered = lore.get_covered(explanation[0][1], dfX2E, formatted_neighbor)
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

    