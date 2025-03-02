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
from gin import GIN
from gin import *


    
def extract_graph_features(data):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    avg_degree = 2 * num_edges / num_nodes
 
    return [num_nodes, num_edges, avg_degree]

def generate_dataset():
    dataset = BA2MotifDataset(root='data/BA2Motif')
    return dataset.shuffle()

def prepare_dataset(df):
    columns = df.columns.tolist()
    class_name = 'y'

    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)
    discrete = ['y']
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete,
                                                   continuous=None)

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
    dataset = generate_dataset()
    train_dataset, test_dataset, val_dataset = dataset[:800], dataset[800:900], dataset[900:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GIN(
        in_channels=dataset.num_features,
        hidden_channels=32,
        out_channels=dataset.num_classes,
        num_layers=5,
    ).to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # blackbox
    df = get_mean_node_embeddings_df(model,  device, test_loader)
    print('df: ', df)
    dataset = prepare_dataset(df)
    
    
    path_data = 'datasets/'
    idx_record2explain = 13
    X2E = df.drop(columns=['y']).values
    # print('X2E', X2E)
    # print('dataset', dataset)
    dataset_info = extract_graph_info(test_loader)
    #y2E = np.asarray([dataset['possible_outcomes'][i] for i in y2E])

    explanation, infos = lore.explain_graph(idx_record2explain, X2E, dataset, model,
                                        discrete_use_probabilities=True,
                                        continuous_function_estimation=False,
                                        returns_infos=True,
                                        path=path_data, sep=';', log=False, testloader = test_loader, dataset_info = dataset_info)
    
    dfX2E = build_df2explain(model, X2E, dataset, graphlist=test_loader).to_dict('records')
    dfx = dfX2E[idx_record2explain]
    # x = build_df2explain(blackbox, X2E[idx_record2explain].reshape(1, -1), dataset).to_dict('records')[0]
    #covered = lore.get_covered(explanation[0][1], dfX2E, dataset)
    # for sample in covered:
    #     print(dataset['df'].iloc[sample])
    print("explaination: ", explanation)
    print('x = %s' % dfx)
    
    print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
    for delta in explanation[1]:
        print('delta', delta)
        
    # precision = [1-eval(v, explanation[0][0][dataset['class_name']]) for v in y2E[covered]]
    # print(precision)
    # print(np.mean(precision), np.std(precision))
    
    
if __name__ == "__main__":
    main()

    