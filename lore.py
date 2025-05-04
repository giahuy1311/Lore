import pyyadt
import random

from neighbor_generator import *
from gpdatagenerator import calculate_feature_values

from model.gin import *
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def explain(idx_record2explain, X2E, dataset, blackbox,
            ng_function=genetic_neighborhood, #generate_random_data, #genetic_neighborhood, random_neighborhood
            discrete_use_probabilities=False,
            continuous_function_estimation=False,
            returns_infos=False, path='./', sep=';', log=False):

    random.seed(0)
    class_name = dataset['class_name']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']
    label_encoder = dataset['label_encoder']
    possible_outcomes = dataset['possible_outcomes']
    
    # Dataset Preprocessing
    dataset['feature_values'] = calculate_feature_values(X2E, columns, class_name, discrete, continuous, 1000,
                                                         discrete_use_probabilities, continuous_function_estimation)
    
    dfZ, x = dataframe2explain(X2E, dataset, idx_record2explain, blackbox)
    print('dfZ',dfZ)
    
    # Generate Neighborhood
    dfZ, Z = ng_function(dfZ, x, blackbox, dataset)

    # Build Decision Tree
    dt, dt_dot = pyyadt.fit(dfZ, class_name, columns, features_type, discrete, continuous, path=path, sep=sep, log=log)

    # Apply Black Box and Decision Tree on instance to explain
    bb_outcome = blackbox.predict(x.reshape(1, -1))[0]

    # predict the outcome of the instance to explain
    dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0] 
    cc_outcome, rule, tree_path = pyyadt.predict_rule(dt, dfx, class_name, features_type, discrete, continuous)

    # Apply Black Box and Decision Tree on neighborhood
    #y_pred_bb = blackbox.predict(Z) # nhãn dự đoán của blackbox cho các điểm lân cận
    
    # y_pred_cc: nhãn dự đoán của decision tree cho các điểm lân cận
    y_pred_cc, leaf_nodes = pyyadt.predict(dt, dfZ.to_dict('records'), class_name, features_type,
                                           discrete, continuous)

    def predict(X):
        y, ln, = pyyadt.predict(dt, X, class_name, features_type, discrete, continuous)
        return y, ln

    # Update labels if necessary
    if class_name in label_encoder:
        cc_outcome = label_encoder[class_name].transform(np.array([cc_outcome]))[0]

    if class_name in label_encoder:
        y_pred_cc = label_encoder[class_name].transform(y_pred_cc)

    # Extract Coutnerfactuals
    diff_outcome = get_diff_outcome(bb_outcome, possible_outcomes)
    counterfactuals = pyyadt.get_counterfactuals(dt, tree_path, rule, diff_outcome,
                                                class_name, continuous, features_type)
    
    explanation = (rule, counterfactuals)

    infos = {
        #'bb_outcome': bb_outcome,
        'cc_outcome': cc_outcome,
        #'y_pred_bb': y_pred_bb,
        'y_pred_cc': y_pred_cc,
        'dfZ': dfZ,
        'Z': Z,
        'dt': dt,
        'tree_path': tree_path,
        'leaf_nodes': leaf_nodes,
        #'diff_outcome': diff_outcome,
        'predict': predict,
    }

    if returns_infos:
        return explanation, infos

    return explanation


def is_satisfied(x, rule, discrete, features_type):
    for col, val in rule.items():
        if col in discrete:
            if str(x[col]).strip() != val:
                return False
        else:
            if '<=' in val and '<' in val and val.find('<=') < val.find('<'):
                val = val.split(col)
                thr1 = pyyadt.yadt_value2type(val[0].replace('<=', ''), col, features_type)
                thr2 = pyyadt.yadt_value2type(val[1].replace('<', ''), col, features_type)
                # if thr2 < x[col] <= thr1: ok
                if x[col] > thr1 or x[col] <= thr2:
                    return False
            elif '<' in val and '<=' in val and val.find('<') < val.find('<='):
                val = val.split(col)
                thr1 = pyyadt.yadt_value2type(val[0].replace('<', ''), col, features_type)
                thr2 = pyyadt.yadt_value2type(val[1].replace('<=', ''), col, features_type)
                # if thr2 < x[col] <= thr1: ok
                if x[col] >= thr1 or x[col] < thr2:
                    return False
            elif '<=' in val:
                thr = pyyadt.yadt_value2type(val.replace('<=', ''), col, features_type)
                if x[col] > thr:
                    return False
            elif '>' in val:
                thr = pyyadt.yadt_value2type(val.replace('>', ''), col, features_type)
                if x[col] <= thr:
                    return False
    return True


def explain_graph(idx_record2explain, dfZ, graphX, dataset, blackbox, 
            ng_function=genetic_neighborhood,
            #generate_random_data, #genetic_neighborhood, random_neighborhood
            discrete_use_probabilities=False,
            continuous_function_estimation=False,
            returns_infos=False, path='./', sep=';', log=False, max_nodes=0):

    random.seed(0)
    class_name = dataset['class_name']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']
    label_encoder = dataset['label_encoder']
    possible_outcomes = dataset['possible_outcomes']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('dfZ of modified: ', dfZ)
    # Apply Black Box and Decision Tree on instance to explain
    pred, embedding = blackbox.predict(graphX.x, graphX.edge_index, None)
    bb_outcome = pred.argmax(dim=-1).item()
    print('bb_outcome: ', bb_outcome)

    # Build Decision Tree
    dt, dt_dot = pyyadt.fit(dfZ, class_name, columns, features_type, discrete, continuous, path=path, sep=sep, log=log)

    # predict the outcome of the instance to explain
    #dfx = build_df2explain(blackbox, x, dataset).to_dict('records')[0]
    dfx = prepare_dataframe([graphX], blackbox, device, ground_truth=False, only_edge=True, node_label=True, max_nodes=max_nodes)
    
    print('dfx: ', dfx.to_dict('records')[0])
    cc_outcome, rule, tree_path = pyyadt.predict_rule(dt, dfx, class_name, features_type, discrete, continuous)

    # Apply Black Box and Decision Tree on neighborhood
    #y_pred_bb = blackbox.predict(Z) # nhãn dự đoán của blackbox cho các điểm lân cận
    
    # y_pred_cc: nhãn dự đoán của decision tree cho các điểm lân cận
    y_pred_cc, leaf_nodes = pyyadt.predict(dt, dfZ.to_dict('records'), class_name, features_type,
                                           discrete, continuous)

    def predict(X):
        y, ln, = pyyadt.predict(dt, X, class_name, features_type, discrete, continuous)
        return y, ln

    # Update labels if necessary
    if class_name in label_encoder:
        cc_outcome = label_encoder[class_name].transform(np.array([cc_outcome]))[0]

    if class_name in label_encoder:
        y_pred_cc = label_encoder[class_name].transform(y_pred_cc)

    # # Extract Coutnerfactuals
    diff_outcome = get_diff_outcome(bb_outcome, possible_outcomes)
    counterfactuals = pyyadt.get_counterfactuals(dt, tree_path, rule, diff_outcome,
                                                class_name, continuous, features_type)
    
    explanation = (rule, counterfactuals)

    infos = {
        #'bb_outcome': bb_outcome,
        'cc_outcome': cc_outcome,
        #'y_pred_bb': y_pred_bb,
        'y_pred_cc': y_pred_cc,
        'dfZ': dfZ,
        # 'Z': Z,
        'dt': dt,
        'tree_path': tree_path,
        'leaf_nodes': leaf_nodes,
        #'diff_outcome': diff_outcome,
        'predict': predict,
    }

    if returns_infos:
        return explanation, infos

    return explanation

def get_covered(rule, X, dataset):
    covered_indexes = list()
    for i, x in enumerate(X):
        if is_satisfied(x, rule, dataset['discrete'], dataset['features_type']):
            covered_indexes.append(i)
    return covered_indexes

