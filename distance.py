import torch
import numpy as np
import ot
from torch_geometric.utils import to_dense_adj


def filter_undirected_edges(edge_index):
    edge_set = set()
    unique_edges = []
    
    for i in range(edge_index.size(1)):
        u, v = sorted(edge_index[:, i].tolist()) 
        edge = (u, v)
        if edge not in edge_set:
            edge_set.add(edge)
            unique_edges.append([u, v])
    
    return torch.tensor(unique_edges, dtype=torch.long).T 

def ensure_undirected(edge_index):
    edge_set = set()
    unique_edges = []
    
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i].tolist()
        edge = tuple(sorted((u, v)))
        
        if edge not in edge_set:
            edge_set.add(edge)
            unique_edges.append([u, v])

    bidirectional_edges = []
    for u, v in unique_edges:
        bidirectional_edges.append([u, v])
        bidirectional_edges.append([v, u])
    
    return torch.tensor(bidirectional_edges, dtype=torch.long).T

def reformat(graph):
    graph.edge_index = ensure_undirected(graph.edge_index)
    adj = to_dense_adj(graph.edge_index)[0].numpy()
    features = graph.x.numpy()
    p = np.ones(graph.num_nodes) / graph.num_nodes
    
    return adj, features, p

def fgw_distance(graph1, graph2, alpha=0.5):
    graph1 = graph1.clone()
    graph2 = graph2.clone()
    C1, features1, p = reformat(graph1)
    C2, features2, q = reformat(graph2)
    max_nodes = max(features1.shape[0], features2.shape[0])

    # padding thêm
    features1 = np.pad(features1, ((0, max_nodes - features1.shape[0]), (0, 0)), mode='constant')
    features2 = np.pad(features2, ((0, max_nodes - features2.shape[0]), (0, 0)), mode='constant')

    C1 = np.pad(C1, ((0, max_nodes - C1.shape[0]), (0, max_nodes - C1.shape[1])), mode='constant')
    C2 = np.pad(C2, ((0, max_nodes - C2.shape[0]), (0, max_nodes - C2.shape[1])), mode='constant')
    # print("diff:", np.sum(C1 - C2))
    features1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-10)
    features2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-10)
    p = p / np.sum(p)
    q = q / np.sum(q)
    M = ot.dist(features1, features2)  #diff feature matrix
    # print("M matrix sum:", np.sum(M))
    fgw_dist = ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, p,q,
        loss_fun='square_loss', alpha=alpha
    )
    
    return fgw_dist

def pad_matrix(matrix, target_size):
    pad_size = target_size - matrix.shape[0]
    if pad_size > 0:
        pad_matrix = torch.zeros((pad_size, matrix.shape[1]), dtype=matrix.dtype, device=matrix.device)
        matrix = torch.cat([matrix, pad_matrix], dim=0)
        
        pad_matrix = torch.zeros((matrix.shape[0], pad_size), dtype=matrix.dtype, device=matrix.device)
        matrix = torch.cat([matrix, pad_matrix], dim=1)
    return matrix

def pad_features(features, target_size):
    pad_size = target_size - features.shape[0]
    if pad_size > 0:
        pad_matrix = torch.zeros((pad_size, features.shape[1]), dtype=features.dtype, device=features.device)
        features = torch.cat([features, pad_matrix], dim=0)
    return features

def my_distance(data1, data2, adj_weight=0.5, feature_weight=0.5, normalize=True):

    max_nodes = max(data1.x.shape[0], data2.x.shape[0])

    data1.edge_index = ensure_undirected(data1.edge_index)
    data2.edge_index = ensure_undirected(data2.edge_index)
    
    adj1 = torch.zeros((data1.num_nodes, data1.num_nodes), dtype=torch.float32)
    adj1[data1.edge_index[0], data1.edge_index[1]] = 1
    adj1 = pad_matrix(adj1, max_nodes)

    adj2 = torch.zeros((data2.num_nodes, data2.num_nodes), dtype=torch.float32)
    adj2[data2.edge_index[0], data2.edge_index[1]] = 1
    adj2 = pad_matrix(adj2, max_nodes)

    adj_distance = torch.norm(adj1 - adj2, p='fro')
    
    if data1.x is not None and data2.x is not None:
        feature1 = pad_features(data1.x, max_nodes)
        feature2 = pad_features(data2.x, max_nodes)
        feature_distance = torch.norm(feature1 - feature2, p='fro')
    else:
        feature_distance = torch.tensor(0.0)

    max_adj_dist = torch.norm(torch.ones_like(adj1), p='fro')  
    max_feat_dist = torch.norm(torch.ones_like(feature1), p='fro') if data1.x is not None else 1.0

    adj_distance = adj_distance / max_adj_dist
    feature_distance = feature_distance / max_feat_dist

        
    total_distance = adj_weight * adj_distance + feature_weight * feature_distance
    #print("distance: ", adj_distance.item())
    return total_distance.item()

def my_distance2(data1, data2, adj_weight=0.5, feature_weight=0.5):
    
    edges_1 = set(tuple(edge.tolist()) for edge in data1.edge_index.T)
    edges_2 = set(tuple(edge.tolist()) for edge in data2.edge_index.T)
    
    num_common_edges = len(edges_1.intersection(edges_2))
    num_total_edges = len(edges_1.union(edges_2))
    adj_distance = 1.0 - num_common_edges / num_total_edges
    
    max_nodes = max(data1.x.shape[0], data2.x.shape[0])
    feature_distance = torch.tensor(0.0)
    if data1.x is not None and data2.x is not None:
        feature1 = pad_features(data1.x, max_nodes)
        feature2 = pad_features(data2.x, max_nodes)
        feature_distance = torch.norm(feature1 - feature2, p='fro')
    else:
        feature_distance = torch.tensor(0.0)
    
    feature_distance = feature_distance / torch.norm(torch.ones_like(feature1), p='fro')
    
    total_distance = adj_weight * adj_distance + feature_weight * feature_distance
    return total_distance.item()
    
        
    
