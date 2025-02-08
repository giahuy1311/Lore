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

def add_node_features(dataset):
    new_data_list = []
    for data in dataset:
        data.x = torch.ones((data.num_nodes, 10))
        new_data_list.append(data)
    return new_data_list

def extract_graph_features(data):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    avg_degree = 2 * num_edges / num_nodes
    
    if data.x is not None:
        node_feature_mean = data.x.mean(dim=0).numpy()
    else:
        node_feature_mean = np.zeros(10) 
    return [num_nodes, num_edges, avg_degree] + node_feature_mean.tolist()

def generate_dataset():
    dataset1 = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=25, num_edges=1),
        motif_generator=HouseMotif(),
        num_motifs=1,
        num_graphs=500,
    )

    dataset2 = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=25, num_edges=1),
        motif_generator=CycleMotif(5),
        num_motifs=1,
        num_graphs=500,
    )

    new_dataset1 = []
    for data in dataset1:
        data = data.clone()  # Clone để có thể sửa đổi
        data.graph_label = torch.tensor([0])
        new_dataset1.append(data)

    new_dataset2 = []
    for data in dataset2:
        data = data.clone()
        data.graph_label = torch.tensor([1])
        new_dataset2.append(data)

    # Thêm node features (với clone bên trong hàm add_node_features nếu cần)
    new_dataset1 = add_node_features(new_dataset1)
    new_dataset2 = add_node_features(new_dataset2)

    # Kết hợp dataset
    dataset = new_dataset1 + new_dataset2
    random.shuffle(dataset)
    return dataset

def build_df(dataset):
    features = []
    labels = []
    for data in dataset:
        features.append(extract_graph_features(data))
        labels.append(data.graph_label.item())
    columns = ['num_nodes', 'num_edges', 'avg_degree'] + [f'feature_{i}' for i in range(10)]
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
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    path_data = 'datasets/'
    idx_record2explain = 123
    X2E = X_test
    y2E = model.predict(X2E)
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
    for sample in covered:
        print(dataset['df'].iloc[sample])
    print('x = %s' % dfx)
    
    print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
    for delta in explanation[1]:
        print('delta', delta)
        
    # precision = [1-eval(v, explanation[0][0][dataset['class_name']]) for v in y2E[covered]]
    # print(precision)
    # print(np.mean(precision), np.std(precision))
    
    
if __name__ == "__main__":
    main()

    