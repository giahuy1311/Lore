import random
from torch_geometric.data import DataLoader
import torch
import random
from torch_geometric.data import Data
from distance import *
import networkx as nx
import random
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def permute_graph(data):
    num_nodes = data.num_nodes
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    mapping = {node: i for i, node in enumerate(nodes)}
    
    edge_index = data.edge_index.clone()
    edge_index[0] = torch.tensor([mapping[node] for node in edge_index[0].tolist()])
    edge_index[1] = torch.tensor([mapping[node] for node in edge_index[1].tolist()])
    
    x = data.x.clone()
    x = x[nodes]

    permuted_data = data.clone()
    permuted_data.edge_index = edge_index
    permuted_data.x = x
    return permuted_data

# class GraphGenome:
#     def __init__(self, data):
#         self.num_nodes = data.x.size(0)
#         self.x = data.x.clone()
#         self.edge_index = data.edge_index.clone().to(device)
#         self.fitness = 0
    
#     def clone(self):
#         return GraphGenome(Data(x=self.x.clone().to(device), edge_index=self.edge_index.clone().to(device)))
    
#     def eval_fitness_sso(self, graphX, blackbox, distance_function, alpha1, alpha2):
#         '''
#         u = softmax(graph_embedding)
#         '''
#         prob_X, embedding_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None)
#         prob_G, embedding_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None)
        
#         uX = F.normalize(embedding_X, p=2, dim=1)
#         uG = F.normalize(embedding_G, p=2, dim=1)
#         #uX = F.softmax(embedding_X, dim=1)
#         #uG = F.softmax(embedding_G, dim=1)
        
#         distance = torch.norm(uX - uG, p=2).item()
#         #distance = 0 if prob_X.argmax().item() == prob_G.argmax().item() else 1
#         if distance <= 1e-8:
#             distance = 1
#         #print('distance sdo: ', distance, ' ---- label: ', prob_G.argmax().item(), ' ---- prob: ', prob_G)
#         return distance
    
#     def eval_fitness_sdo(self, graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs_embedding):
#         '''
#         DO_graphs_embedding: list of embedding of graphs that has other label to graphX
#         '''
#         prob_G, embedding_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None)
#         uG = F.normalize(embedding_G, p=2, dim=1)
#         #uG = F.softmax(embedding_G, dim=1)
#         distances = []
#         for embedding in DO_graphs_embedding:
#             uX = embedding
            
#             distance = torch.norm(uX - uG, p=2).item()
#             #distance = 0 if uX.argmax().item() == prob_G.argmax().item() else 1
#             distances.append(distance)
        
#         distance = min(distances)
#         if distance <= 1e-8:
#             distance = 1
#         #print('distance sdo: ', distance, ' ---- label: ', prob_G.argmax().item(), ' ---- prob: ', prob_G)
#         return distance
    
#     def mutate(self, graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs):
#         edge_list = set(map(tuple, self.edge_index.T.tolist()))
#         num_nodes = self.num_nodes
#         feature_dim = self.x.size(1)
#         mutation_choices = ['add_edge', 'remove_edge', 'change_node', 'add_node', 'remove_node']
#         mutation_probs = [0.2, 0.2, 0.2, 0.2, 0.2]  

#         mutation_type = random.choices(mutation_choices, weights=mutation_probs, k=1)[0]
#         new_x = self.x.clone()
        
#         if mutation_type == 'add_edge':
#             for _ in range(10):
#                 u, v = random.sample(range(num_nodes), 2)
#                 edge = (u, v) if u < v else (v, u)
#                 if edge not in edge_list:
#                     edge_list.add(edge)
#                     break

#         elif mutation_type == 'remove_edge':
#             if edge_list:
#                 edge = random.choice(list(edge_list))
#                 edge_list.remove(edge)
                
#         elif mutation_type == 'add_node':
#             new_node = torch.zeros_like(self.x[0])
#             new_node[random.randint(0, feature_dim - 1)] = 1
#             new_x = torch.cat([new_x, new_node.unsqueeze(0)], dim=0)
            
#             if edge_list:
#                 u = random.randint(0, num_nodes - 2)
#                 edge = (u, num_nodes - 1) if u < num_nodes - 1 else (num_nodes - 1, u)
#                 edge_list.add(edge)
                
#         elif mutation_type == 'remove_node':
#             index = random.randint(0, num_nodes - 1)
#             new_x = torch.cat([new_x[:index], new_x[index + 1:]], dim=0)
#             edge_list = {edge for edge in edge_list if index not in edge}
#             edge_list = {(u - 1 if u > index else u, v - 1 if v > index else v) for u, v in edge_list}
#             edge_list = set(tuple(edge) for edge in edge_list)
                
#         else:
#             index = random.randint(0, num_nodes - 1)
#             current_label = torch.argmax(new_x[index]).item()
            
#             new_label = random.choice([j for j in range(new_x.size(1)) if j != current_label])
#             new_x[index] = torch.zeros_like(new_x[index])
#             new_x[index][new_label] = 1
        
#         edge_list = list(edge_list)
#         if len(edge_list) < 2 or new_x.size(0) < 2:
#             return self
        
#         new_edge_index = torch.tensor(edge_list, dtype=torch.long).T
#         mutated_graph = GraphGenome(Data(x=new_x, edge_index=new_edge_index))
        
#         if case_type == 'sso':
#             mutated_graph.fitness = mutated_graph.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
#         else:
#             mutated_graph.fitness = mutated_graph.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs)
        
#         return mutated_graph
    
#     def crossover(self, other, graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs):
#         parent1_edges = [tuple(edge.tolist()) for edge in self.edge_index.T]
#         parent2_edges = [tuple(edge.tolist()) for edge in other.edge_index.T]
#         parent1_edges = list(set(parent1_edges))
#         parent2_edges = list(set(parent2_edges))
        
#         if len(parent1_edges) < 2 or len(parent2_edges) < 2:
#             return self  

#         parent_size = min(len(parent1_edges), len(parent2_edges))
#         cut1, cut2 = sorted(random.sample(range(parent_size), 2))
#         if cut1 == cut2:
#             cut2 = min(cut1 + 1, parent_size - 1)
#         child_edges = parent1_edges[:cut1] + parent2_edges[cut1:cut2] + parent1_edges[cut2:]
#         child_edges = list(set(child_edges)) 
#         all_node_ids = sorted(set([u for u, v in child_edges] + [v for u, v in child_edges] + list(range(self.num_nodes))))
#         node_id_map = {old_id: new_id for new_id, old_id in enumerate(all_node_ids)}

#         remapped_edges = [(node_id_map[u], node_id_map[v]) for u, v in child_edges]
#         new_edge_index = torch.tensor(remapped_edges, dtype=torch.long).T
#         feature_dim = self.x.size(1)
#         new_x = torch.zeros((len(all_node_ids), feature_dim))

#         for old_id, new_id in node_id_map.items():
#             if old_id < self.num_nodes:
#                 new_x[new_id] = self.x[old_id]
#             elif old_id < other.num_nodes:
#                 new_x[new_id] = other.x[old_id]
#             else:
#                 print(f"Warning: Node ID {old_id} not found in either parent.")
#                 new_x[new_id] = torch.randn(feature_dim)
                
#         child = GraphGenome(Data(x=new_x, edge_index=new_edge_index))
        
#         if case_type == 'sso':
#             child.fitness = child.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
#         else:
#             child.fitness = child.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs)

#         return child


    
#     # def mutate(self, graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs):
#     #     edge_list = set(tuple(edge.tolist()) for edge in self.edge_index.T)
#     #     num_nodes = self.num_nodes
        
#     #     mutation_type = random.choice(['add_edge', 'remove_edge', 'change_node'])
#     #     new_x = self.x.clone()
        
#     #     if mutation_type == 'add_edge':
#     #         u, v = random.sample(range(num_nodes), 2)
#     #         edge = (u, v) if u < v else (v, u)
#     #         if edge not in edge_list:
#     #             edge_list.add(edge)
#     #     elif mutation_type == 'remove_edge':
#     #         if edge_list:
#     #             edge = random.choice(list(edge_list))
#     #             edge_list.remove(edge)
#     #     else:
#     #         index = random.randint(0, num_nodes - 1)
#     #         current_label = torch.argmax(new_x[index]).item()
            
#     #         new_label = random.choice([j for j in range(new_x.size(1)) if j != current_label])
#     #         new_x[index] = torch.zeros_like(new_x[index])
#     #         new_x[index][new_label] = 1
                
#     #     new_edge_index = torch.tensor(list(edge_list), dtype=torch.long).T
#     #     mutated_graph = GraphGenome(Data(x=new_x, edge_index=new_edge_index))
        
#     #     if case_type == 'sso':
#     #         mutated_graph.fitness = mutated_graph.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
#     #     elif case_type == 'sdo':
#     #         mutated_graph.fitness = mutated_graph.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs)
#     #     elif case_type == 'dso':
#     #         mutated_graph.fitness = mutated_graph.eval_fitness_dso(graphX, blackbox, distance_function, alpha1, alpha2)
#     #     else:
#     #         mutated_graph.fitness = mutated_graph.eval_fitness_ddo(graphX, blackbox, distance_function, alpha1, alpha2)
            
#     #     return mutated_graph
    
#     # def crossover(self, other, graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs):
#     #     parent1_edges = [tuple(edge.tolist()) for edge in self.edge_index.T]
#     #     parent2_edges = [tuple(edge.tolist()) for edge in other.edge_index.T]

#     #     if len(parent1_edges) < 2 or len(parent2_edges) < 2:
#     #         return self  

#     #     parent_size = len(parent1_edges)
#     #     cut1, cut2 = sorted(random.sample(range(parent_size), 2))
#     #     child_edges = parent1_edges[:cut1] + parent2_edges[cut1:cut2] + parent1_edges[cut2:]
#     #     child_edges = list(set(child_edges))  
#     #     new_edge_index = torch.tensor(child_edges, dtype=torch.long).T
        
#     #     new_x = self.x.clone()
#     #     # if node comes from parent2, change label
#     #     for i in range(cut1, cut2):
#     #         index = self.edge_index[0][i].item() # nút source (source, target) trong edge
#     #         current_label = torch.argmax(new_x[index]).item()
#     #         new_label = torch.argmax(other.x[index]).item()
#     #         if current_label != new_label:
#     #             new_x[index] = torch.zeros_like(new_x[index])
#     #             new_x[index][new_label] = 1
                
#     #     child = GraphGenome(Data(x=new_x, edge_index=new_edge_index))
        
#     #     if case_type == 'sso':
#     #         child.fitness = child.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
#     #     elif case_type == 'sdo':
#     #         child.fitness = child.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs)
#     #     elif case_type == 'dso':
#     #         child.fitness = child.eval_fitness_dso(graphX, blackbox, distance_function, alpha1, alpha2)
#     #     else:
#     #         child.fitness = child.eval_fitness_ddo(graphX, blackbox, distance_function, alpha1, alpha2)

#     #     return child

# def initialize_population(size, graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs):
#     population = []
#     for _ in range(size):
#         individual = GraphGenome(graphX).mutate(graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs)
#         population.append(individual)
    
#     for individual in population:
#         if case_type == 'sso':
#             individual.fitness = individual.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
#         else:
#             individual.fitness = individual.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs)
        
#     return population

# def select_parents(population, ratio):
#     unique_fitness = {}
#     for individual in population:
#         if individual.fitness not in unique_fitness:
#             unique_fitness[individual.fitness] = individual
    
#     distinct_population = list(unique_fitness.values())
#     distinct_population.sort(key=lambda individual: individual.fitness)
    
#     return distinct_population[:int(ratio * len(distinct_population))]

# def get_graphs_DO_embedding(label, dataset, blackbox):
#     DO_graphs_embedding = []
#     for i in range(len(dataset)):
#         if dataset[i].y != label:
#             prob_X, embedding_X = blackbox.predict(dataset[i].x, dataset[i].edge_index, None)
        
#             # uX = F.softmax(embedding_X, dim=1)
#             uX = F.normalize(embedding_X, p=2, dim=1)
#             DO_graphs_embedding.append(uX)
#     return DO_graphs_embedding

# def genetic_algorithm(graphX, populationSize, generations, blackbox, distance_function, alpha1, alpha2, dataset):
#     case_population_size = populationSize // 2
#     case_type = ['sso', 'sdo']
#     populations = {}
#     DO_graphs_embedding = get_graphs_DO_embedding(graphX.y, dataset, blackbox)
        
#     for type in case_type:
#         populations[type] = initialize_population(
#                     case_population_size, graphX, blackbox, distance_function, alpha1, alpha2, type, DO_graphs_embedding
#                 )
        
        
#         for gen in range(generations):
#             print(f"{type} : Generation {gen+1}/{generations} =====")
#             new_population = []
#             best_individuals = select_parents(populations[type], 0.1)
#             new_population.extend(best_individuals[:len(best_individuals) // 2]) # di truyen qua the he sau
#             fitness_set = set(ind.fitness for ind in new_population)
#             while len(new_population) < case_population_size: #200
#                 parent1, parent2 = random.sample(best_individuals, 2)
#                 child = parent1.crossover(parent2, graphX, blackbox, distance_function, alpha1, alpha2, type, DO_graphs_embedding)
#                 child = child.mutate(graphX, blackbox, distance_function, alpha1, alpha2, type, DO_graphs_embedding)
                
#                 #child = child.mutate(graphX, blackbox, distance_function, alpha1, alpha2, type, DO_graphs_embedding)
                
#                 phi = 1e-9  
#                 isUnique = True  
#                 for f in fitness_set:
#                     if abs(child.fitness - f) < phi:
#                         isUnique = False
#                         break 

#                 if isUnique:
#                     fitness_set.add(child.fitness)
#                     new_population.append(child)
                    
#             populations[type] = new_population
            
#     final_population = []
#     for population in populations.values():
#         final_population.extend(population)
        
#     return final_population

import random
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import from_networkx
import copy
import torch
import networkx as nx
import torch.nn.functional as F
# from ged4py import GraphEditDistance

def prepare_networkx_for_pyg(graph_nx):
    for _, data in graph_nx.nodes(data=True):
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = torch.tensor(v, dtype=torch.float)
    for _, _, data in graph_nx.edges(data=True):
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = torch.tensor(v, dtype=torch.float)

def add_bidirectional_edges(edge_index):
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

# def eval_fitness_so(child_nx, graphX_nx, blackbox):
#     child_string = graph_to_string(child_nx.graph)
#     graphX_string = graph_to_string(graphX_nx)
#     distance = levenshtein_ed(child_string, graphX_string)
#     norm_distance = distance / max(len(child_string), len(graphX_string))
#     if norm_distance == 0:
#         norm_distance = 1
    
#     child = from_networkx(child_nx.graph)
#     graphX = from_networkx(graphX_nx)
#     prob_child, _ = blackbox.predict(child.x, child.edge_index, None)
#     prob_G, _ = blackbox.predict(graphX.x, graphX.edge_index, None)
#     target_similarity = torch.norm(prob_child - prob_G, p=2).item()

#     return 0.5 * norm_distance + 0.5 * target_similarity

# def eval_fitness_do(child_nx, blackbox, other_graphs):
#     distances = []
#     fitness = []
#     child = from_networkx(child_nx.graph)
#     prob_child, _ = blackbox.predict(child.x, child.edge_index, None)
#     child_string = graph_to_string(child_nx.graph)
#     for graph, prob in other_graphs:
#         graph_string = graph_to_string(graph)
#         distance = levenshtein_ed(child_string, graph_string)
#         norm_distance = distance / max(len(child_string), len(graph_string))
#         if norm_distance == 0:
#             norm_distance = 1
            
#         target_similarity = torch.norm(prob_child - prob, p=2).item()
#         fitness.append(0.5 * norm_distance + 0.5 * target_similarity)
        
            
#     if not fitness:
#         return 1

#     return min(fitness)

def eval_fitness_so(child_nx, graphX_nx, blackbox):    
    child = from_networkx(child_nx.graph)
    graphX = from_networkx(graphX_nx)
    
    logits_X, embedding_X = blackbox(graphX.x, add_bidirectional_edges(graphX.edge_index), None)
    logits_G, embedding_G = blackbox(child.x, add_bidirectional_edges(child.edge_index), None)
    # prob_X = F.softmax(logits_X, dim=1)
    # prob_G = F.softmax(logits_G, dim=1)
    
    #distance = torch.norm(F.normalize(embedding_X, p=2, dim=1) - F.normalize(embedding_G, p=2, dim=1), p=2).item()
    distance = torch.norm(logits_X - logits_G, p=2).item()
    
    if distance <= 0:
        distance = 1
    #distance = 0 if prob_X.argmax().item() == prob_G.argmax().item() else 1
    return distance
def eval_fitness_do(child_nx, blackbox, DO_graphs_embedding):
    
    child = from_networkx(child_nx.graph)
    logit_G, embedding_G = blackbox(child.x, add_bidirectional_edges(child.edge_index), None)
    prob_G = F.softmax(logit_G, dim=1)
    uG = F.normalize(embedding_G, p=2, dim=1)
    distances = []
    for uX in DO_graphs_embedding:
        distance = torch.norm(uX - logit_G, p=2).item()
        if distance <= 0:
            distance = 1
        #distance = 0 if uX.argmax().item() == prob_G.argmax().item() else 1
        distances.append(distance)
        
    if not distances:
        return 1
    
    distance = min(distances)
    return distance

class Individual:
    def __init__(self, graph):
        self.graph = graph
        self.fitness = 0

    def clone(self):
        return Individual(copy.deepcopy(self.graph))

    def mutate(self, graphX, blackbox, case_type, DO_graphs):
        mutation_choices = ['add_edge', 'remove_edge']
        mutation_probs = [0.5, 0.5]  # Equal probability for each mutation type
        mutation_type = random.choices(mutation_choices, weights=mutation_probs, k=1)[0]

        mutated_graph = copy.deepcopy(self.graph)
        if mutated_graph.nodes is None or mutated_graph.edges is None:
            return None
        
        any_node = next(iter(mutated_graph.nodes))
        feature_dim = len(mutated_graph.nodes[any_node].get('x'))
        nodes = list(mutated_graph.nodes)
        edges = list(mutated_graph.edges)

        if mutation_type == 'add_edge':
            for _ in range(10):
                u, v = random.sample(nodes, 2)
                if not mutated_graph.has_edge(u, v):
                    mutated_graph.add_edge(u, v)
                    break

        elif mutation_type == 'remove_edge' and edges:
            u, v = random.choice(edges)
            mutated_graph.remove_edge(u, v)

        elif mutation_type == 'add_node':
            new_node_id = max(mutated_graph.nodes) + 1
            new_feature = torch.zeros(feature_dim, dtype=torch.float)
            node_label = random.choice(range(feature_dim))
            new_feature[node_label] = 1
            
            mutated_graph.add_node(new_node_id, x=new_feature)
            
            existing_node = random.choice(nodes)
            if existing_node != new_node_id:
                mutated_graph.add_edge(new_node_id, existing_node)

        elif mutation_type == 'remove_node':
            node = random.choice(nodes)
            mutated_graph.remove_node(node)

        else: #change node label
            node_id = random.choice(nodes)
            old_feature = torch.tensor(mutated_graph.nodes[node_id]['x'], dtype=torch.float)
            current_index = torch.argmax(old_feature).item()
            
            other_indexes = [i for i in range(feature_dim) if i != current_index]
            new_index = random.choice(other_indexes)

            new_feature = torch.zeros(feature_dim, dtype=torch.float)
            new_feature[new_index] = 1.0
            mutated_graph.nodes[node_id]['x'] = new_feature

        if len(mutated_graph.nodes) < 2 or len(mutated_graph.edges) < 1:
            return None
        # if not nx.is_connected(mutated_graph):
        #     return self
        
        prepare_networkx_for_pyg(mutated_graph)
        child = Individual(mutated_graph)
        if case_type == 'sso':
            child.fitness = eval_fitness_so(child, graphX, blackbox)
        else:
            child.fitness = eval_fitness_do(child, blackbox, DO_graphs)
        return child

    def crossover(self, other, graphX, blackbox, case_type, DO_graphs):
        G1 = copy.deepcopy(self.graph)
        G2 = copy.deepcopy(other.graph)

        nodes1 = list(G1.nodes)
        nodes2 = list(G2.nodes)

        cut = random.choice(range(1, len(nodes1)))
        part1 = nodes1[:cut]
        part2 = nodes2[cut:]
        
        child = nx.Graph()

        child.add_nodes_from((n, G1.nodes[n]) for n in part1 if n not in child)
        child.add_nodes_from((n, G2.nodes[n]) for n in part2 if n not in child)

        child.add_edges_from((u, v) for u, v in G1.edges if u in part1 and v in part1)
        child.add_edges_from((u, v) for u, v in G2.edges if u in part2 and v in part2)
        
        child.add_edges_from((u, v) for u, v in G1.edges if (u in part1 and v in part2) or (v in part1 and u in part2))
        child.add_edges_from((u, v) for u, v in G2.edges if (u in part1 and v in part2) or (v in part1 and u in part2))

        # if not nx.is_connected(child):
        #     return None
        
        prepare_networkx_for_pyg(child)
        child = Individual(child)
        if case_type == 'sso':
            child.fitness = eval_fitness_so(child, graphX, blackbox)
        else:
            child.fitness = eval_fitness_do(child, blackbox, DO_graphs)
        return child

def initialize_population(size, graphX, blackbox, case_type, DO_graphs):
    population = []
    for _ in range(size):
        ind_graph = copy.deepcopy(graphX)
        individual = Individual(ind_graph).mutate(graphX, blackbox, case_type, DO_graphs)
        if individual is not None:
            population.append(individual)
    return population

def select_parents(population, ratio):
    unique = {}
    for ind in population:
        if ind.fitness not in unique:
            unique[ind.fitness] = ind
    distinct = list(unique.values())
    distinct.sort(key=lambda x: x.fitness)
    return distinct[:int(ratio * len(distinct))]

def get_graphs_DO_embedding(label, dataset, blackbox):
    DO_graphs_embedding = []
    for i in range(len(dataset)):
        if dataset[i].y.item() != label:
            logits, embedding_X = blackbox(dataset[i].x, dataset[i].edge_index, None)
            uX = F.normalize(embedding_X, p=2, dim=1)
            prob_X = F.softmax(logits, dim=1)
            DO_graphs_embedding.append(logits)
            
            # G = to_networkx(dataset[i], to_undirected=True, node_attrs="x")
            # prob, _ = blackbox.predict(dataset[i].x, dataset[i].edge_index, None)
            # DO_graphs_embedding.append((G, prob))
    print("Number of graphs with different labels: ", len(DO_graphs_embedding))
    return DO_graphs_embedding

def genetic_algorithm(graph2X, population_size, generations, blackbox, dataset):
    graphX = to_networkx(graph2X, to_undirected=True, node_attrs="x")
    graphX.graph['y'] = graph2X.y.item() 
    DO_graphs_embedding = get_graphs_DO_embedding(graphX.graph['y'], dataset, blackbox)
    case_types = ['sso', 'sdo']
    populations = {}

    for case_type in case_types:
        population = initialize_population(population_size // 2, graphX, blackbox, case_type, DO_graphs_embedding)
        
        for gen in range(generations):
            print(f"{case_type} : Generation {gen+1}/{generations}")
            new_population = []
            best = select_parents(population, ratio = 0.1) 
            new_population.extend(best[:len(best) // 2])
            fitness_set = set(ind.fitness for ind in new_population)

            while len(new_population) < (population_size // 2):
                p1, p2 = random.sample(best, 2)
                
                child = p1.crossover(p2, graphX, blackbox, case_type, DO_graphs_embedding)
                if child is None:
                    continue
                child = child.mutate(graphX, blackbox, case_type, DO_graphs_embedding)
                if child is None:
                    continue
                
                if all(abs(child.fitness - f) > 1e-12 for f in fitness_set):
                    fitness_set.add(child.fitness)
                    new_population.append(child)
                    
            population = new_population
            
        populations[case_type] = population
        
    final_population = []
    for population in populations.values():
        final_population.extend(population)
        
    return final_population
