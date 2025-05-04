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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def has_house(list_edge):
    edges = set(tuple(sorted(edge)) for edge in list_edge)
    ring = {
        (20, 21), (21, 22), (22, 23), (23, 24), (20, 24)
    }
    subsets_house = {(21, 24), (21, 23), (20, 23), (20, 22), (22, 24)}
    house_motif = []
    for subset in subsets_house:
        house = ring.copy()
        house.add(tuple(sorted(subset)))
        house_motif.append(house)
    
    
    for house in house_motif:
        if house.issubset(edges):
            return True
    return False

def has_cycle(list_edge):
    edges = set(tuple(sorted(edge)) for edge in list_edge)
    ring = {
        (20, 23), (21, 24), (20, 24), (22, 23), (21, 22), 
    }
    subsets_house = {(20, 21), (20, 22), (23, 24), (21, 23), (22, 24)}
    cycle_motif = []
    for subset in subsets_house:
        cycle_c = ring.copy()
        cycle_c.add(tuple(sorted(subset)))
        cycle_motif.append(cycle_c)
    
    for ring in cycle_motif:
        if ring.issubset(edges):
            return False
    return True

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

class GraphGenome:
    def __init__(self, data):
        self.num_nodes = data.x.size(0)
        self.x = data.x.clone()
        self.edge_index = data.edge_index.clone().to(device)
        self.fitness = 0
    
    def clone(self):
        return GraphGenome(Data(x=self.x.clone().to(device), edge_index=self.edge_index.clone().to(device)))
    
    def eval_fitness_sso(self, graphX, blackbox, distance_function, alpha1, alpha2):
        '''
        u = softmax(graph_embedding)
        '''
        prob_X, embedding_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None)
        prob_G, embedding_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None)
        
        uX = F.normalize(embedding_X, p=2, dim=1)
        uG = F.normalize(embedding_G, p=2, dim=1)
        #uX = F.softmax(embedding_X, dim=1)
        #uG = F.softmax(embedding_G, dim=1)
        
        distance = torch.norm(uX - uG, p=2).item()
        #distance = 0 if prob_X.argmax().item() == prob_G.argmax().item() else 1
        if distance <= 1e-8:
            distance = 1
        #print('distance sdo: ', distance, ' ---- label: ', prob_G.argmax().item(), ' ---- prob: ', prob_G)
        return distance
    
    def eval_fitness_sdo(self, graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs_embedding):
        '''
        DO_graphs_embedding: list of embedding of graphs that has other label to graphX
        '''
        prob_G, embedding_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None)
        uG = F.normalize(embedding_G, p=2, dim=1)
        #uG = F.softmax(embedding_G, dim=1)
        distances = []
        for embedding in DO_graphs_embedding:
            uX = embedding
            
            distance = torch.norm(uX - uG, p=2).item()
            #distance = 0 if uX.argmax().item() == prob_G.argmax().item() else 1
            distances.append(distance)
        
        distance = min(distances)
        if distance <= 1e-8:
            distance = 1
        #print('distance sdo: ', distance, ' ---- label: ', prob_G.argmax().item(), ' ---- prob: ', prob_G)
        return distance
    
    def mutate(self, graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs):
        edge_list = set(map(tuple, self.edge_index.T.tolist()))
        num_nodes = self.num_nodes
        feature_dim = self.x.size(1)
        mutation_choices = ['add_edge', 'remove_edge', 'change_node', 'add_node', 'remove_node']
        mutation_probs = [0.2, 0.2, 0.2, 0.2, 0.2]  

        mutation_type = random.choices(mutation_choices, weights=mutation_probs, k=1)[0]
        new_x = self.x.clone()
        
        if mutation_type == 'add_edge':
            for _ in range(10):
                u, v = random.sample(range(num_nodes), 2)
                edge = (u, v) if u < v else (v, u)
                if edge not in edge_list:
                    edge_list.add(edge)
                    break

        elif mutation_type == 'remove_edge':
            if edge_list:
                edge = random.choice(list(edge_list))
                edge_list.remove(edge)
                
        elif mutation_type == 'add_node':
            new_node = torch.zeros_like(self.x[0])
            new_node[random.randint(0, feature_dim - 1)] = 1
            new_x = torch.cat([new_x, new_node.unsqueeze(0)], dim=0)
            
            if edge_list:
                u = random.randint(0, num_nodes - 2)
                edge = (u, num_nodes - 1) if u < num_nodes - 1 else (num_nodes - 1, u)
                edge_list.add(edge)
                
        elif mutation_type == 'remove_node':
            index = random.randint(0, num_nodes - 1)
            new_x = torch.cat([new_x[:index], new_x[index + 1:]], dim=0)
            edge_list = {edge for edge in edge_list if index not in edge}
            edge_list = {(u - 1 if u > index else u, v - 1 if v > index else v) for u, v in edge_list}
            edge_list = set(tuple(edge) for edge in edge_list)
                
        else:
            index = random.randint(0, num_nodes - 1)
            current_label = torch.argmax(new_x[index]).item()
            
            new_label = random.choice([j for j in range(new_x.size(1)) if j != current_label])
            new_x[index] = torch.zeros_like(new_x[index])
            new_x[index][new_label] = 1
        
        edge_list = list(edge_list)
        if len(edge_list) < 2 or new_x.size(0) < 2:
            return self
        
        new_edge_index = torch.tensor(edge_list, dtype=torch.long).T
        mutated_graph = GraphGenome(Data(x=new_x, edge_index=new_edge_index))
        
        if case_type == 'sso':
            mutated_graph.fitness = mutated_graph.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
        else:
            mutated_graph.fitness = mutated_graph.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs)
        
        return mutated_graph
    
    def crossover(self, other, graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs):
        parent1_edges = [tuple(edge.tolist()) for edge in self.edge_index.T]
        parent2_edges = [tuple(edge.tolist()) for edge in other.edge_index.T]
        parent1_edges = list(set(parent1_edges))
        parent2_edges = list(set(parent2_edges))
        
        if len(parent1_edges) < 2 or len(parent2_edges) < 2:
            return self  

        parent_size = min(len(parent1_edges), len(parent2_edges))
        cut1, cut2 = sorted(random.sample(range(parent_size), 2))
        if cut1 == cut2:
            cut2 = min(cut1 + 1, parent_size - 1)
        child_edges = parent1_edges[:cut1] + parent2_edges[cut1:cut2] + parent1_edges[cut2:]
        child_edges = list(set(child_edges)) 
        all_node_ids = sorted(set([u for u, v in child_edges] + [v for u, v in child_edges] + list(range(self.num_nodes))))
        node_id_map = {old_id: new_id for new_id, old_id in enumerate(all_node_ids)}

        remapped_edges = [(node_id_map[u], node_id_map[v]) for u, v in child_edges]
        new_edge_index = torch.tensor(remapped_edges, dtype=torch.long).T
        feature_dim = self.x.size(1)
        new_x = torch.zeros((len(all_node_ids), feature_dim))

        for old_id, new_id in node_id_map.items():
            if old_id < self.num_nodes:
                new_x[new_id] = self.x[old_id]
            elif old_id < other.num_nodes:
                new_x[new_id] = other.x[old_id]
            else:
                print(f"Warning: Node ID {old_id} not found in either parent.")
                new_x[new_id] = torch.randn(feature_dim)
                
        child = GraphGenome(Data(x=new_x, edge_index=new_edge_index))
        
        if case_type == 'sso':
            child.fitness = child.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
        else:
            child.fitness = child.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs)

        return child


    
    # def mutate(self, graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs):
    #     edge_list = set(tuple(edge.tolist()) for edge in self.edge_index.T)
    #     num_nodes = self.num_nodes
        
    #     mutation_type = random.choice(['add_edge', 'remove_edge', 'change_node'])
    #     new_x = self.x.clone()
        
    #     if mutation_type == 'add_edge':
    #         u, v = random.sample(range(num_nodes), 2)
    #         edge = (u, v) if u < v else (v, u)
    #         if edge not in edge_list:
    #             edge_list.add(edge)
    #     elif mutation_type == 'remove_edge':
    #         if edge_list:
    #             edge = random.choice(list(edge_list))
    #             edge_list.remove(edge)
    #     else:
    #         index = random.randint(0, num_nodes - 1)
    #         current_label = torch.argmax(new_x[index]).item()
            
    #         new_label = random.choice([j for j in range(new_x.size(1)) if j != current_label])
    #         new_x[index] = torch.zeros_like(new_x[index])
    #         new_x[index][new_label] = 1
                
    #     new_edge_index = torch.tensor(list(edge_list), dtype=torch.long).T
    #     mutated_graph = GraphGenome(Data(x=new_x, edge_index=new_edge_index))
        
    #     if case_type == 'sso':
    #         mutated_graph.fitness = mutated_graph.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
    #     elif case_type == 'sdo':
    #         mutated_graph.fitness = mutated_graph.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs)
    #     elif case_type == 'dso':
    #         mutated_graph.fitness = mutated_graph.eval_fitness_dso(graphX, blackbox, distance_function, alpha1, alpha2)
    #     else:
    #         mutated_graph.fitness = mutated_graph.eval_fitness_ddo(graphX, blackbox, distance_function, alpha1, alpha2)
            
    #     return mutated_graph
    
    # def crossover(self, other, graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs):
    #     parent1_edges = [tuple(edge.tolist()) for edge in self.edge_index.T]
    #     parent2_edges = [tuple(edge.tolist()) for edge in other.edge_index.T]

    #     if len(parent1_edges) < 2 or len(parent2_edges) < 2:
    #         return self  

    #     parent_size = len(parent1_edges)
    #     cut1, cut2 = sorted(random.sample(range(parent_size), 2))
    #     child_edges = parent1_edges[:cut1] + parent2_edges[cut1:cut2] + parent1_edges[cut2:]
    #     child_edges = list(set(child_edges))  
    #     new_edge_index = torch.tensor(child_edges, dtype=torch.long).T
        
    #     new_x = self.x.clone()
    #     # if node comes from parent2, change label
    #     for i in range(cut1, cut2):
    #         index = self.edge_index[0][i].item() # nút source (source, target) trong edge
    #         current_label = torch.argmax(new_x[index]).item()
    #         new_label = torch.argmax(other.x[index]).item()
    #         if current_label != new_label:
    #             new_x[index] = torch.zeros_like(new_x[index])
    #             new_x[index][new_label] = 1
                
    #     child = GraphGenome(Data(x=new_x, edge_index=new_edge_index))
        
    #     if case_type == 'sso':
    #         child.fitness = child.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
    #     elif case_type == 'sdo':
    #         child.fitness = child.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs)
    #     elif case_type == 'dso':
    #         child.fitness = child.eval_fitness_dso(graphX, blackbox, distance_function, alpha1, alpha2)
    #     else:
    #         child.fitness = child.eval_fitness_ddo(graphX, blackbox, distance_function, alpha1, alpha2)

    #     return child



def initialize_population(size, graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs):
    population = []
    for _ in range(size):
        individual = GraphGenome(graphX).mutate(graphX, blackbox, distance_function, alpha1, alpha2, case_type, DO_graphs)
        population.append(individual)
    
    for individual in population:
        if case_type == 'sso':
            individual.fitness = individual.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
        else:
            individual.fitness = individual.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2, DO_graphs)
        
    return population

def select_parents(population, ratio):
    unique_fitness = {}
    for individual in population:
        if individual.fitness not in unique_fitness:
            unique_fitness[individual.fitness] = individual
    
    distinct_population = list(unique_fitness.values())
    distinct_population.sort(key=lambda individual: individual.fitness)
    
    return distinct_population[:int(ratio * len(distinct_population))]

def get_graphs_DO_embedding(label, dataset, blackbox):
    DO_graphs_embedding = []
    for i in range(len(dataset)):
        if dataset[i].y != label:
            prob_X, embedding_X = blackbox.predict(dataset[i].x, dataset[i].edge_index, None)
        
            # uX = F.softmax(embedding_X, dim=1)
            uX = F.normalize(embedding_X, p=2, dim=1)
            DO_graphs_embedding.append(uX)
    return DO_graphs_embedding

def genetic_algorithm(graphX, populationSize, generations, blackbox, distance_function, alpha1, alpha2, dataset):
    case_population_size = populationSize // 2
    case_type = ['sso', 'sdo']
    populations = {}
    DO_graphs_embedding = get_graphs_DO_embedding(graphX.y, dataset, blackbox)
        
    for type in case_type:
        populations[type] = initialize_population(
                    case_population_size, graphX, blackbox, distance_function, alpha1, alpha2, type, DO_graphs_embedding
                )
        
        
        for gen in range(generations):
            print(f"{type} : Generation {gen+1}/{generations} =====")
            new_population = []
            best_individuals = select_parents(populations[type], 0.1)
            new_population.extend(best_individuals[:len(best_individuals) // 2]) # di truyen qua the he sau
            fitness_set = set(ind.fitness for ind in new_population)
            while len(new_population) < case_population_size: #200
                parent1, parent2 = random.sample(best_individuals, 2)
                child = parent1.crossover(parent2, graphX, blackbox, distance_function, alpha1, alpha2, type, DO_graphs_embedding)
                child = child.mutate(graphX, blackbox, distance_function, alpha1, alpha2, type, DO_graphs_embedding)
                
                #child = child.mutate(graphX, blackbox, distance_function, alpha1, alpha2, type, DO_graphs_embedding)
                
                phi = 1e-8  
                if not any(abs(child.fitness - f) < phi for f in fitness_set):
                    fitness_set.add(child.fitness)
                    new_population.append(child)
                    
            populations[type] = new_population
            
    final_population = []
    for population in populations.values():
        final_population.extend(population)
        
    return final_population

