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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_connected(edge_index, num_nodes):
        G = nx.Graph()
        G.add_edges_from(edge_index.T.tolist())
        return nx.is_connected(G)
class GraphGenome:
    def __init__(self, data):
        self.num_nodes = data.num_nodes
        self.x = data.x.clone()
        self.edge_index = data.edge_index.clone().to(device)
        self.fitness = 0
    
    def clone(self):
        return GraphGenome(Data(x=self.x.clone().to(device), edge_index=self.edge_index.clone().to(device)))
    
    def eval_fitness_sso(self, graphX, blackbox, distance_function, alpha1, alpha2):
        similarity = 1.0 - distance_function(graphX, self)
        similarity = 0.0 if similarity >= 1 else similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1)
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1)
        target_similarity = 1.0 if y_X == y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def eval_fitness_sdo(self, graphX, blackbox, distance_function, alpha1, alpha2):
        similarity = 1.0 - distance_function(graphX, self)
        similarity = 0.0 if similarity >= 1 else similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1)
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1)
        target_similarity = 1.0 if y_X != y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def eval_fitness_dso(self, graphX, blackbox, distance_function, alpha1, alpha2):
        similarity = 1.0 - distance_function(graphX, self)
        similarity = 0.0 if similarity <= 0.9688 else 1 - similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1)
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1)
        target_similarity = 1.0 if y_X == y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def eval_fitness_ddo(self, graphX, blackbox, distance_function, alpha1, alpha2):
        similarity = 1.0 - distance_function(graphX, self)
        similarity = 0.0 if similarity <= 0.9688 else 1 - similarity
        
        y_X = blackbox.predict(graphX.x, ensure_undirected(graphX.edge_index), None, 1)
        y_G = blackbox.predict(self.x, ensure_undirected(self.edge_index), None, 1)
        target_similarity = 1.0 if y_X != y_G else 0.0
        
        #print('record_similarity: ', similarity, '-- evaluation: ', target_similarity)
        return alpha1 * similarity + alpha2 * target_similarity
    
    def mutate(self, graphX, blackbox, distance_function, alpha1, alpha2, case_type, dataset):
        edge_list = set(tuple(edge.tolist()) for edge in self.edge_index.T)
        num_nodes = self.x.size(0)
        
        # them hoac xoa 1 canh lay tu 1 do thi khac trong dataset
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
        new_edge_index = torch.tensor(list(edge_list), dtype=torch.long, device=device).T

        
        # thay doi feature vector cua 1 node
        new_x = self.x.clone().to(device)
        mutation_rate = 0.1
        for i in range(num_nodes):
            if random.random() < mutation_rate:
                # other_graph = random.choice(dataset)
                # new_x[i] = other_graph.x[i].clone()
                if torch.all((new_x[i] == 0) | (new_x[i] == 1)):  
                    # Nếu x[i] chỉ có 0 và 1 (như MUTAG), dùng bit flipping
                    current_category = torch.argmax(new_x[i]).item()
                    new_category = random.choice([j for j in range(new_x.size(1)) if j != current_category])
                    new_x[i] = torch.zeros_like(new_x[i])
                    new_x[i][new_category] = 1
                else:
                    # Nếu x[i] là giá trị liên tục, thêm nhiễu Gaussian
                    noise = torch.randn_like(new_x[i]) * 0.05
                    new_x[i] = torch.clamp(new_x[i] + noise, 0, 1)
                
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

        child_edges = list(set(tuple(edge) for edge in child_edges))  
        new_edge_index = torch.tensor(child_edges, dtype=torch.long).T
        child = GraphGenome(Data(x=self.x, edge_index=new_edge_index))
        
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
    population = [GraphGenome(graph.to(device)) for _ in range(size)]
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

def genetic_algorithm(graphX, populationSize, generations, blackbox, distance_function, alpha1, alpha2, dataset):
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
                parent1, parent2 = random.sample(best_individuals, 2)
                child = parent1.crossover(parent2, graphX, blackbox, distance_function, alpha1, alpha2, type)
                if random.random() < 0.2:
                    child = child.mutate(graphX, blackbox, distance_function, alpha1, alpha2, type, dataset)
                new_population.append(child)
                    
            populations[type] = new_population
            
    final_population = []
    for population in populations.values():
        final_population.extend(population)
        
    return final_population