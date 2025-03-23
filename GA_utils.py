import random
from model.gin import GIN
from torch_geometric.data import DataLoader
import torch
import random
from torch_geometric.data import Data
from torch_geometric.datasets import BA2MotifDataset
import ot
import networkx as nx
from torch_geometric.utils import to_dense_adj
import numpy as np
import ot
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix, coalesce
import numpy as np
import torch
from torch_geometric.data import Data
import random

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

def pyg_to_fgw_format(graph):
    graph.edge_index = ensure_undirected(graph.edge_index)
    adj = to_dense_adj(graph.edge_index)[0].numpy()
    features = graph.x.numpy()
    p = np.ones(graph.num_nodes) / graph.num_nodes
    
    return adj, features, p

def compute_fgw_distance(graph1, graph2, alpha=0.5):
    # clone graphs
    graph1 = graph1.clone()
    graph2 = graph2.clone()
    C1, features1, p = pyg_to_fgw_format(graph1)
    C2, features2, q = pyg_to_fgw_format(graph2)
    max_nodes = max(features1.shape[0], features2.shape[0])

    # Padding f
    features1 = np.pad(features1, ((0, max_nodes - features1.shape[0]), (0, 0)), mode='constant')
    features2 = np.pad(features2, ((0, max_nodes - features2.shape[0]), (0, 0)), mode='constant')

    C1 = np.pad(C1, ((0, max_nodes - C1.shape[0]), (0, max_nodes - C1.shape[1])), mode='constant')
    C2 = np.pad(C2, ((0, max_nodes - C2.shape[0]), (0, max_nodes - C2.shape[1])), mode='constant')
    # print("diff:", np.sum(C1 - C2))
    features1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-10)
    features2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-10)
    p = p / np.sum(p)
    q = q / np.sum(q)
    M = ot.dist(features1, features2)  
    # print("M matrix sum:", np.sum(M))
    fgw_dist = ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, p,q,
        loss_fun='square_loss', alpha=alpha
    )
    
    return fgw_dist

import random
import torch
from torch_geometric.data import Data

class GraphGenome:
    def __init__(self, data):
        self.num_nodes = data.num_nodes
        self.x = data.x.clone()
        self.edge_index = data.edge_index.clone()
        self.innovation_numbers = {tuple(edge.tolist()): i for i, edge in enumerate(self.edge_index.T)}
        self.fitness = 0
    
    def clone(self):
        return GraphGenome(Data(x=self.x.clone(), edge_index=self.edge_index.clone()))
    
    def eval_fitness_sso(self, graphX, blackbox, distance_function, alpha1, alpha2):
        similarity = 1.0 - distance_function(graphX, self, alpha=0.5)
        similarity = 0.0 if similarity >= 0.999 else similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1)
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1)
        target_similarity = 1.0 if y_X == y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def eval_fitness_sdo(self, graphX, blackbox, distance_function, alpha1, alpha2):
        similarity = 1.0 - distance_function(graphX, self, alpha=0.5)
        similarity = 0.0 if similarity >= 0.999 else similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1)
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1)
        target_similarity = 1.0 if y_X != y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def eval_fitness_dso(self, graphX, blackbox, distance_function, alpha1, alpha2):
        similarity = 1.0 - distance_function(graphX, self, alpha=0.5)
        similarity = 0.0 if similarity <= 0.9688 else 1 - similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1)
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1)
        target_similarity = 1.0 if y_X == y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def eval_fitness_ddo(self, graphX, blackbox, distance_function, alpha1, alpha2):
        similarity = 1.0 - distance_function(graphX, self, alpha=0.5)
        similarity = 0.0 if similarity <= 0.9688 else 1 - similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1)
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1)
        target_similarity = 1.0 if y_X != y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def mutate(self, graphX, blackbox, distance_function, alpha1, alpha2, case_type):
        edge_list = set(tuple(edge.tolist()) for edge in self.edge_index.T)
        num_nodes = self.x.size(0)
        
        if random.random() <= 0.5 or len(edge_list) < 15:  # Add edge
            u, v = random.sample(range(num_nodes), 2)
            edge = (u, v) if u < v else (v, u)
            if edge not in edge_list:
                edge_list.add(edge)
                self.innovation_numbers[edge] = len(self.innovation_numbers) + 1
        else:  # Remove edge
            if edge_list:
                edge = random.choice(list(edge_list))
                edge_list.remove(edge)
                self.innovation_numbers.pop(edge, None)
            
                
        new_edge_index = torch.tensor(list(edge_list), dtype=torch.long).T
        new_x = self.x.clone()
        mutation_rate = 0.1
        for i in range(num_nodes):
            if random.random() < mutation_rate:
                noise = torch.randn_like(new_x[i]) * 0.05
                new_x[i] = torch.clamp(new_x[i] + noise, 0, 1)
        
        mutated_graph = GraphGenome(Data(x=new_x, edge_index=new_edge_index))
        
        # Evaluate fitness based on case type
        if case_type == 'sso':
            mutated_graph.fitness = mutated_graph.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'sdo':
            mutated_graph.fitness = mutated_graph.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'dso':
            mutated_graph.fitness = mutated_graph.eval_fitness_dso(graphX, blackbox, distance_function, alpha1, alpha2)
        else:  # 'ddo'
            mutated_graph.fitness = mutated_graph.eval_fitness_ddo(graphX, blackbox, distance_function, alpha1, alpha2)
            
        return mutated_graph
    
    def crossover(self, other, graphX, blackbox, distance_function, alpha1, alpha2, case_type):
        parent1_edges = list(set(tuple(edge.tolist()) for edge in self.edge_index.T))
        parent2_edges = list(set(tuple(edge.tolist()) for edge in other.edge_index.T))

        if len(parent1_edges) < 2 or len(parent2_edges) < 2:
            return self  
        cut1, cut2 = sorted(random.sample(range(len(parent1_edges)), 2))

        child_edges = parent1_edges[:cut1] + parent2_edges[cut1:cut2] + parent1_edges[cut2:]
        child_edges = list(set(child_edges)) 
        
        new_edge_index = torch.tensor(child_edges, dtype=torch.long).T

        new_x = self.x.clone()
        for i in range(self.x.size(0)):
            new_x[i] = (self.x[i] + other.x[i]) / 2 if random.random() < 0.5 else other.x[i]

        child = GraphGenome(Data(x=new_x, edge_index=new_edge_index))
        child.innovation_numbers = {edge: i for i, edge in enumerate(child_edges)}
        
        # Evaluate fitness based on case type
        if case_type == 'sso':
            child.fitness = child.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'sdo':
            child.fitness = child.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'dso':
            child.fitness = child.eval_fitness_dso(graphX, blackbox, distance_function, alpha1, alpha2)
        else:  # 'ddo'
            child.fitness = child.eval_fitness_ddo(graphX, blackbox, distance_function, alpha1, alpha2)

        return child


def initialize_population(size, graph, blackbox, distance_function, alpha1, alpha2, case_type):
    population = [GraphGenome(graph) for _ in range(size)]
    for individual in population:
        if case_type == 'sso':
            individual.fitness = individual.eval_fitness_sso(graph, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'sdo':
            individual.fitness = individual.eval_fitness_sdo(graph, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'dso':
            individual.fitness = individual.eval_fitness_dso(graph, blackbox, distance_function, alpha1, alpha2)
        else:  # 'ddo'
            individual.fitness = individual.eval_fitness_ddo(graph, blackbox, distance_function, alpha1, alpha2)
            
    return population

def select_parents(population, ratio):
    population.sort(key=lambda individual: individual.fitness, reverse=True)
    return population[:int(ratio * len(population))]

def genetic_algorithm(graphX, populationSize, generations, blackbox, distance_function, alpha1, alpha2):
    case_population_size = populationSize // 4
    case_type = ['sso', 'sdo', 'dso', 'ddo']
    populations = {}
        
    
    # Initialize populations for both cases
    for type in case_type:
        populations[type] = initialize_population(
                    case_population_size, graphX, blackbox, distance_function, alpha1, alpha2, case_type
                )
        for gen in range(generations):
            print(f"\n===== {type} : Generation {gen+1}/{generations} =====")
            new_population = []
            best_individuals = select_parents(populations[type], 0.4)
            new_population.extend(best_individuals[:len(best_individuals) // 2])
            while len(new_population) < case_population_size:
                parent1, parent2 = random.sample(best_individuals, 2)
                child = parent1.crossover(parent2, graphX, blackbox, distance_function, alpha1, alpha2, type)
                if random.random() < 0.2:
                    child = child.mutate(graphX, blackbox, distance_function, alpha1, alpha2, type)
                new_population.append(child)
                    
            populations[type] = new_population
            
    final_population = []
    for population in populations.values():
        final_population.extend(population)
        
    return final_population
        
