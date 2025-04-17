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
        similarity = 1.0 - distance_function(graphX, self)
        similarity = 0.0 if similarity >= 1 else similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1).argmax(dim=-1).item()
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1).argmax(dim=-1).item()
        target_similarity = 1.0 if y_X == y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def eval_fitness_sdo(self, graphX, blackbox, distance_function, alpha1, alpha2):
        similarity = 1.0 - distance_function(graphX, self)
        similarity = 0.0 if similarity >= 1 else similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1).argmax(dim=-1).item()
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1).argmax(dim=-1).item()
        target_similarity = 1.0 if y_X != y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def eval_fitness_dso(self, graphX, blackbox, distance_function, alpha1, alpha2):
        similarity = 1.0 - distance_function(graphX, self)
        similarity = 0.0 if similarity <= 0 else 1.0 - similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1)
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1)
        target_similarity = 1.0 if y_X == y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def eval_fitness_ddo(self, graphX, blackbox, distance_function, alpha1, alpha2):
        similarity = 1.0 - distance_function(graphX, self)
        similarity = 0.0 if similarity <= 0 else 1.0 - similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1)
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1)
        target_similarity = 1.0 if y_X != y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def mutate(self, graphX, blackbox, distance_function, alpha1, alpha2, case_type):
        edge_list = set(tuple(edge.tolist()) for edge in self.edge_index.T)
        num_nodes = self.num_nodes
        
        if random.random() <= 0.5:
            u, v = random.sample(range(num_nodes), 2)
            edge = (u, v) if u < v else (v, u)
            if edge not in edge_list:
                edge_list.add(edge)
        else:  
            if edge_list:
                edge = random.choice(list(edge_list))
                edge_list.remove(edge)
        
        new_edge_index = torch.tensor(list(edge_list), dtype=torch.long).T
    
        new_x = self.x.clone()
        mutation_rate = 0.1
        for i in range(self.num_nodes):
            if random.random() < mutation_rate:
                if torch.all((new_x[i] == 0) | (new_x[i] == 1)):  
                    current_category = torch.argmax(new_x[i]).item()
                    new_category = random.choice([j for j in range(new_x.size(1)) if j != current_category])
                    new_x[i] = torch.zeros_like(new_x[i])
                    new_x[i][new_category] = 1
                else:
                    noise = torch.randn_like(new_x[i]) * 0.05
                    new_x[i] = torch.nn.functional.softmax(new_x[i] + noise, dim=0)
                
        mutated_graph = GraphGenome(Data(x=new_x, edge_index=new_edge_index))
        
        if case_type == 'sso':
            mutated_graph.fitness = mutated_graph.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'sdo':
            mutated_graph.fitness = mutated_graph.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'dso':
            mutated_graph.fitness = mutated_graph.eval_fitness_dso(graphX, blackbox, distance_function, alpha1, alpha2)
        else:
            mutated_graph.fitness = mutated_graph.eval_fitness_ddo(graphX, blackbox, distance_function, alpha1, alpha2)
            
        return mutated_graph
    
    def crossover(self, other, graphX, blackbox, distance_function, alpha1, alpha2, case_type):
        # cat doan edge tu cha va me
        parent1_edges = [tuple(edge.tolist()) for edge in self.edge_index.T]
        parent2_edges = [tuple(edge.tolist()) for edge in other.edge_index.T]

        if len(parent1_edges) < 2 or len(parent2_edges) < 2:
            return self  

        cut1, cut2 = sorted(random.sample(range(len(parent1_edges)), 2))
        child_edges = parent1_edges[:cut1] + parent2_edges[cut1:cut2] + parent1_edges[cut2:]

        child_edges = list(set(child_edges))  
        
        new_edge_index = torch.tensor(child_edges, dtype=torch.long).T
        
        child = GraphGenome(Data(x=self.x.clone(), edge_index=new_edge_index))
        
        if case_type == 'sso':
            child.fitness = child.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'sdo':
            child.fitness = child.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'dso':
            child.fitness = child.eval_fitness_dso(graphX, blackbox, distance_function, alpha1, alpha2)
        else:
            child.fitness = child.eval_fitness_ddo(graphX, blackbox, distance_function, alpha1, alpha2)

        return child



def initialize_population(size, graphX, blackbox, distance_function, alpha1, alpha2, case_type):
    population = []
    for _ in range(size):
        individual = GraphGenome(graphX).mutate(graphX, blackbox, distance_function, alpha1, alpha2, case_type)
        population.append(individual)
        
    
    for individual in population:
        if case_type == 'sso':
            individual.fitness = individual.eval_fitness_sso(graphX, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'sdo':
            individual.fitness = individual.eval_fitness_sdo(graphX, blackbox, distance_function, alpha1, alpha2)
        elif case_type == 'dso':
            individual.fitness = individual.eval_fitness_dso(graphX, blackbox, distance_function, alpha1, alpha2)
        else:
            individual.fitness = individual.eval_fitness_ddo(graphX, blackbox, distance_function, alpha1, alpha2)
            
    return population

def select_parents(population, ratio):
    population.sort(key=lambda individual: individual.fitness, reverse=True)
    return population[:int(ratio * len(population))]

def tournament_selection(population, tournament_size=3):
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda ind: ind.fitness, reverse=True)
    return selected[0] 

def genetic_algorithm(graphX, populationSize, generations, blackbox, distance_function, alpha1, alpha2):
    case_population_size = populationSize // 2
    case_type = ['sso', 'sdo']
    populations = {}
        
    for type in case_type:
        populations[type] = initialize_population(
                    case_population_size, graphX, blackbox, distance_function, alpha1, alpha2, type
                )
        for gen in range(generations):
            print(f"{type} : Generation {gen+1}/{generations} =====")
            new_population = []
            best_individuals = select_parents(populations[type], 0.4)
            new_population.extend(best_individuals[:len(best_individuals) // 2])
            while len(new_population) < case_population_size:
                parent1 = tournament_selection(populations[type])
                parent2 = tournament_selection(populations[type])
                child = parent1.crossover(parent2, graphX, blackbox, distance_function, alpha1, alpha2, type)
                if random.random() < 0.5:
                    child = child.mutate(graphX, blackbox, distance_function, alpha1, alpha2, type)
                
                new_population.append(child)
                    
            populations[type] = new_population
            
    final_population = []
    for population in populations.values():
        final_population.extend(population)
        
    return final_population

