import torch
import numpy as np
import datetime
import lore
from model.gin import *
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import from_networkx
from torch_geometric.datasets import BA2MotifDataset, TUDataset
from genetic import *
from util import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import random
import os
import hydra
from dataset import get_dataset, get_dataloader
from model.gnnNets import get_gnnNets



def mask(G, rule):
    feature_dim = len(G.nodes[0].get('x')) if G.nodes else 0
    G_maskeout = G.copy()
    G_mask = nx.Graph()
    for key, cond in rule.items():
        if '_' in key and not key.startswith('nL_'): # edge
            u, v = key.split('_')
            u, v = int(u[1:]), int(v[1:])
            if cond == '>0':
                if not G_mask.has_edge(u, v):
                    if not G_mask.has_node(u) and G.has_node(u):
                        attrs = G.nodes[u]
                        G_mask.add_node(u, **attrs)
                    if not G_mask.has_node(v) and G.has_node(v):
                        attrs = G.nodes[v]
                        G_mask.add_node(v, **attrs)
                    G_mask.add_edge(u, v)
                
                if G_maskeout.has_edge(u, v):
                    G_maskeout.remove_edge(u, v)
            elif cond == '<=0':
                if not G_maskeout.has_edge(u, v):
                    if not G_maskeout.has_node(u):
                        if G.has_node(u):
                            attrs = G.nodes[u]
                            G_maskeout.add_node(u, **attrs)
                        else:
                            new_feature = torch.zeros(feature_dim, dtype=torch.float)
                            node_label = random.choice(range(feature_dim))
                            new_feature[node_label] = 1
                            G_maskeout.add_node(u, x=new_feature)
                    if not G_maskeout.has_node(v):
                        if G.has_node(v):
                            attrs = G.nodes[v]
                            G_maskeout.add_node(v, **attrs)
                        else:
                            new_feature = torch.zeros(feature_dim, dtype=torch.float)
                            node_label = random.choice(range(feature_dim))
                            new_feature[node_label] = 1
                            G_maskeout.add_node(v, x=new_feature)
                    G_maskeout.add_edge(u, v)

        elif key.startswith('nL_'): # node label
            node_id = int(key.split('_')[1])
            if cond != 'None':
                if G_maskeout.has_node(node_id):
                    G_maskeout.remove_node(node_id)
                if not G_mask.has_node(node_id) and G.has_node(node_id):
                    attrs = G.nodes[node_id]
                    G_mask.add_node(node_id, **attrs)
            else: # None
                if not G_maskeout.has_node(node_id):
                    new_feature = torch.zeros(feature_dim, dtype=torch.float)
                    node_label = random.choice(range(feature_dim))
                    new_feature[node_label] = 1
                    G_maskeout.add_node(node_id, x=new_feature)

    return G_maskeout, G_mask

def hit_outcome(bb_outcome, cc_outcome):
    return 1 if bb_outcome == cc_outcome else 0

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

def evaluate(model, graph, rule, device='cpu'):
    G = to_networkx(graph, to_undirected=True, node_attrs="x")
    l, _ = model(graph.x, graph.edge_index, None)
    p = l.squeeze(0).softmax(dim=-1)
    target_class = torch.argmax(p).item()  
    conf_target = p[target_class].item()
    
    G_maskout, G_mask = mask(G, rule)
    
    if G_maskout.number_of_nodes() < 1:
        fidelity = 0
    else:
        g_maskout = from_networkx(G_maskout)
        l_mo, _ = model(g_maskout.x, g_maskout.edge_index, None)
        p_mo = l_mo.squeeze(0).softmax(dim=-1)
        conf_maskout = p_mo[target_class].item()
        
        fidelity = conf_target - conf_maskout
    
    if G_mask.number_of_nodes() < 1:
        inv_fidelity = 0
    else:
        g_mask = from_networkx(G_mask)
        l_m, _ = model(g_mask.x, g_mask.edge_index, None)
        p_m = l_m.squeeze(0).softmax(dim=-1)
        conf_mask = p_m[target_class].item()
        
        inv_fidelity = conf_target - conf_mask

    total_nodes = G.number_of_nodes()
    explained_nodes = G_mask.number_of_nodes()
    sparsity = 1 - (explained_nodes / total_nodes)

    return fidelity, inv_fidelity, sparsity


def run_experiment(blackbox, graph2X, dfX2E, X2E, idx_record2explain, path_data,device,  verbose=False, popSize = 1000, gens=5):
    #graph2X format: pyg Data
    
    # Prepare test dataset
    formatted_dataset = prepare_dataset(dfX2E)
    y2E = np.asarray(dfX2E['y'].tolist())
    y2E = np.asarray([formatted_dataset['label_encoder'][formatted_dataset['class_name']].classes_[i] for i in y2E])
    
    bb_outcome = y2E[idx_record2explain]
    
    # Gen neighbors of graph with idx_record2explain
    final_population = genetic_algorithm(graph2X = graph2X, population_size=popSize, generations=gens, blackbox=blackbox, 
                                      dataset=X2E)
    Z = []
    max_node = 0
    for individual in final_population:
        graph = from_networkx(individual.graph)
        Z.append(graph)
        max_node = max(max_node, len(graph.x))
    
    # Prepare dataframe for neighbors
    dfZ = prepare_dataframe(Z, blackbox, device, ground_truth=False, only_edge=True, node_label=False, max_nodes=max_node)
    formatted_neighborhood = prepare_dataset(dfZ)

    individual_hit_lore = 0
    fidelity_acc_lore = fidelity_f1_lore = coverage_lore = coverage_Z_lore = 0
    precision_lore = [0]
    fi = inv_fi = sparsity = 0

    def eval(x, y):
        return 1 if x == y else 0

    print(datetime.datetime.now(), '\tLORE')
    attempt = 0
    lore_explanation, infos = lore.explain_graph(idx_record2explain, dfZ, graph2X, formatted_neighborhood, blackbox,
                                        discrete_use_probabilities=True,
                                        continuous_function_estimation=False,
                                        returns_infos=True,
                                        path=path_data, sep=';', log=False, max_nodes=max_node)
    cc_outcome_lore = lore_explanation[0][0][formatted_neighborhood['class_name']]
    print(cc_outcome_lore, type(cc_outcome_lore), bb_outcome, type(bb_outcome))
    individual_hit_lore = hit_outcome(bb_outcome, cc_outcome_lore)

    y_pred_bb_lore = infos['y_pred_bb']
    y_pred_cc_lore = infos['y_pred_cc']
    fidelity_acc_lore = accuracy_score(y_pred_bb_lore, y_pred_cc_lore)
    fidelity_f1_lore = f1_score(y_pred_bb_lore, y_pred_cc_lore)

    lrule = lore_explanation[0][1]
    print(lrule)
    dfX2E = dfX2E.to_dict('records')
    covered_lore = lore.get_covered(lrule, dfX2E, formatted_neighborhood)
    print('covered_lore: ', covered_lore)
    coverage_lore = len(covered_lore) / len(dfX2E)
    precision_lore = [1 - eval(v, bb_outcome) for v in y2E[covered_lore]]
    covered_Z_lore = lore.get_covered(lrule, infos['dfZ'].to_dict('records'), formatted_neighborhood)
    coverage_Z_lore = len(covered_Z_lore) / len(infos['dfZ'])
    
    fi, inv_fi, sparsity = evaluate(blackbox, graph2X, lrule, device=device)

    res = '%d,%.6f,%.6f,%.6f,%.6f,%.6f, %s' % (
        individual_hit_lore, fidelity_acc_lore, fidelity_f1_lore,
        coverage_lore, np.mean(precision_lore), coverage_Z_lore, lrule
    )
    return res, fi, inv_fi, sparsity

@hydra.main(config_path="config", config_name="config")
def main(config):
    cwd = os.path.dirname(os.path.abspath(__file__))
    pwd = cwd
    
    config.datasets.dataset_root = os.path.join(pwd, "dataset")
    config.models.gnn_saving_path = os.path.join(pwd, "checkpoints")

    config.models.param = config.models.param[config.datasets.dataset_name]
    
    dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
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
        
    model = get_gnnNets(
        input_dim=dataset.num_node_features,
        output_dim=dataset.num_classes,
        model_config=config.models,
    )

    state_dict = torch.load(
        os.path.join(
            config.models.gnn_saving_path,
            f"{config.datasets.dataset_name}",
            f"{config.models.gnn_name}_"
            f"{len(config.models.param.gnn_latent_dim)}l_latest.pth",
        )
    )["net"]
    model.load_state_dict(state_dict)

    start_index = -1

    path = './'
    path_exp = path + 'experiments/'
    path_data = path + 'datasets/'
    popSize = config.pop_size
    gens = config.gens

    experiments_results = open(path_exp + 'lore_graph_%s_%s_%d_%d_logits.csv' % (config.datasets.dataset_name, config.models.gnn_name, popSize, gens), 'w')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X2E = [dataset[i] for i in test_indices]

    dfX2E = prepare_dataframe(X2E, model, device, ground_truth=True, only_edge=True, node_label=False)
    fi_list = []
    inv_fi_list = []
    sp_list = []
    for idx_record2explain in range(len(X2E)):
        if idx_record2explain <= start_index:
            continue
        graph2X = X2E[idx_record2explain]
        print(datetime.datetime.now(), '%d - %.2f' % (idx_record2explain, idx_record2explain / len(X2E)))
        res, fi, inv_fi, sp = run_experiment(model, graph2X, dfX2E, X2E, idx_record2explain,
                                path_data, device, verbose=False, popSize=popSize, gens=gens)
        res = '%d,%s,%s,%s\n' % (idx_record2explain, config.datasets.dataset_name, config.models.gnn_name, res)
        print(res)
        print("Fidelity: %.6f, Inv Fidelity: %.6f, Sparsity: %.6f" % (fi, inv_fi, sp))
        fi_list.append(fi)
        inv_fi_list.append(inv_fi)
        sp_list.append(sp)
        experiments_results.write(res)
        experiments_results.write('Fidelity: %.6f, Inv Fidelity: %.6f, Sparsity: %.6f\n' % (fi, inv_fi, sp))
        experiments_results.flush()
        # break

    avg_fidelity = np.mean(fi_list)
    std_fidelity = np.std(fi_list)
    avg_inv_fidelity = np.mean(inv_fi_list)
    std_inv_fidelity = np.std(inv_fi_list)
    avg_sparsity = np.mean(sp_list)
    std_sparsity = np.std(sp_list)
    experiments_results.write(
        'Mean Fidelity: %.6f, Std: %.6f\n' % (avg_fidelity, std_fidelity))
    experiments_results.write(
        'Mean Inv Fidelity: %.6f, Std: %.6f\n' % (avg_inv_fidelity, std_inv_fidelity))
    experiments_results.write(
        'Mean Sparsity: %.6f, Std: %.6f\n' % (avg_sparsity, std_sparsity))
    experiments_results.close()
    
if __name__ == '__main__':
    main()