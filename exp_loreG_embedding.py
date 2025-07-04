import numpy as np
from util import *
from sklearn.model_selection import train_test_split
import lore
from prepare_dataset import *
from neighbor_generator import *
import torch
import random
from torch_geometric.datasets import BA2MotifDataset
import pickle
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from genetic import *
from hydra import compose, initialize
from model.gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader
import os
from model.gin import prepare_dataframe

initialize(config_path="config")
config = compose(config_name="config", overrides=["datasets=ba_2motifs", "models=gcn_old", "explainers=tux"])

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
    pwd = os.getcwd()
    config.datasets.dataset_root = os.path.join(pwd, "dataset")
    config.models.gnn_saving_path = os.path.join(pwd, "checkpoints")
    config.models.param = config.models.param[config.datasets.dataset_name]
    dataset = get_dataset(
        dataset_root=config.datasets.dataset_root,
        dataset_name=config.datasets.dataset_name,
    )
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {
            "batch_size": config.models.param.batch_size,
            "random_split_flag": config.datasets.random_split_flag,
            "data_split_ratio": config.datasets.data_split_ratio,
            "seed": config.datasets.seed,
        }
        loader = get_dataloader(dataset, **dataloader_params)
        test_indices = loader["test"].dataset.indices
        if config.datasets.data_explain_cutoff > 0:
            test_indices = test_indices[: config.datasets.data_explain_cutoff]

    else:
        node_indices_mask = (dataset.data.y != 0) * dataset.data.test_mask
        node_indices = torch.where(node_indices_mask)[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    test_dataset = [dataset[i] for i in test_indices]
    model = get_gnnNets(
        input_dim=dataset.num_node_features,
        output_dim=dataset.num_classes,
        model_config=config.models,
    )
    state_dict = torch.load(
        os.path.join(
            config.models.gnn_saving_path,
            config.datasets.dataset_name,
            f"{config.models.gnn_name}_"
            f"{len(config.models.param.gnn_latent_dim)}l_latest.pth",
        )
    )["net"]
    model.load_state_dict(state_dict)
    model.eval()

    graph2X = test_dataset[3]
    # blackbox
    df = prepare_dataframe(test_dataset, model, device, ground_truth=True, only_edge=True, node_label=False)
    print('df: ', df)
    dataset = prepare_dataset(df)
    path_data = 'datasets/'
    idx_record2explain = 3 #(72 & 724)
    
    #generate sample
    
    
    with open('mutagX.pkl', 'wb') as f:
        pickle.dump(graph2X, f)
    
    final_population = genetic_algorithm(graph2X, population_size=1000, generations=2, blackbox=model, 
                                      dataset=test_dataset)
    
    
    list_graph = []
    for individual in final_population:
        graph = from_networkx(individual.graph)
        list_graph.append(graph)
        
    max_nodes = max(len(graph.x) for graph in list_graph)
    dfZ = prepare_dataframe(list_graph, model, device, ground_truth=False, only_edge=True, node_label=False, max_nodes=max_nodes)

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
    
    cc_outcome_lore = explanation[0][0][dataset['class_name']]
    lrule = explanation[0][1]
    covered_lore = lore.get_covered(lrule, dfX2E, dataset)
    coverage_lore = len(covered_lore) / len(dfX2E)
    def eval(x, y):
        return 1 if x == y else 0
    precision_lore = [1 - eval(v, cc_outcome_lore) for v in y2E[covered_lore]]
    covered_Z_lore = lore.get_covered(lrule, infos['dfZ'].to_dict('records'), dataset)
    coverage_Z_lore = len(covered_Z_lore) / len(infos['dfZ'])
    print('coverage_lore: ', coverage_lore)
    print('coverage_Z_lore: ', coverage_Z_lore)
    print('cc_outcome_lore: ', cc_outcome_lore)
    
    
    print("covered len:", len(covered))
    print('x = %s' % dfx)
    
    print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
    for delta in explanation[1]:
        print('delta', delta)
    
        
    precision = [1-eval(v, explanation[0][0][dataset['class_name']]) for v in y2E[covered]]
    print(precision)
    print(np.mean(precision), np.std(precision))
    
    
if __name__ == "__main__":
    main()

    