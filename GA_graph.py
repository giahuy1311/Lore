import random
import torch
from torch_geometric.data import Data
from genetic import *
from torch_geometric.datasets import BA2MotifDataset
from model.gin import *

def create_graph(data):
    return {
        'num_nodes': data.num_nodes,
        'x': data.x.clone(),
        'edge_index': data.edge_index.clone(),
        'innovation_numbers': {tuple(edge.tolist()): i for i, edge in enumerate(data.edge_index.T)},
        'fitness': 0
    }

def clone_graph(graph):
    return create_graph(Data(x=graph['x'].clone(), edge_index=graph['edge_index'].clone()))

def calculate_similarity(graph_x, graph, distance_function):
    return 1.0 - distance_function(graph_x, graph, alpha=0.5)

def predict_outcomes(graph_x, graph, blackbox):
    y_X = blackbox.predict(graph_x['x'], ensure_undirected(graph_x['edge_index']), None, 1)
    y_G = blackbox.predict(graph['x'], ensure_undirected(graph['edge_index']), None, 1)
    return y_X, y_G

def eval_fitness(graph, graph_x, blackbox, distance_function, alpha1, alpha2, case_type):
    similarity = calculate_similarity(graph_x, graph, distance_function)
    y_X, y_G = predict_outcomes(graph_x, graph, blackbox)
    
    if case_type.startswith('s'):  # Similar
        similarity = 0.0 if similarity >= 0.987 else similarity
    else:  # Different
        similarity = 0.0 if similarity <= 0.9688 else 1 - similarity
        
    # Determine target similarity based on prediction outcomes
    if case_type.endswith('o'):  # Opposite
        target_similarity = 1.0 if y_X != y_G else 0.0
    else:  # Same
        target_similarity = 1.0 if y_X == y_G else 0.0
    
    print(f'case: {case_type} - similarity: {similarity} - evaluation: {target_similarity}')
    return alpha1 * similarity + alpha2 * target_similarity

def mutate(graph, graph_x, blackbox, distance_function, alpha1, alpha2, case_type):
    # Convert edge index to a set for easier manipulation
    edge_list = set(tuple(edge.tolist()) for edge in graph['edge_index'].T)
    num_nodes = graph['x'].size(0)
    innovation_numbers = graph['innovation_numbers'].copy()
    
    # Edge mutation: add or remove an edge
    if random.random() <= 0.5 or len(edge_list) < 15:  # Add edge
        u, v = random.sample(range(num_nodes), 2)
        edge = (u, v) if u < v else (v, u)
        if edge not in edge_list:
            edge_list.add(edge)
            innovation_numbers[edge] = len(innovation_numbers) + 1
    else:  # Remove edge
        if edge_list:
            edge = random.choice(list(edge_list))
            edge_list.remove(edge)
            if edge in innovation_numbers:
                del innovation_numbers[edge]
    
    # Create new edge index tensor
    new_edge_index = torch.tensor(list(edge_list), dtype=torch.long).T
    
    # Node feature mutation
    new_x = graph['x'].clone()
    mutation_rate = 0.1
    for i in range(num_nodes):
        if random.random() < mutation_rate:
            noise = torch.randn_like(new_x[i]) * 0.05
            new_x[i] = torch.clamp(new_x[i] + noise, 0, 1)
    
    # Create and evaluate the mutated graph
    mutated_graph = {
        'num_nodes': num_nodes,
        'x': new_x,
        'edge_index': new_edge_index,
        'innovation_numbers': innovation_numbers,
        'fitness': 0
    }
    
    # Evaluate fitness
    mutated_graph['fitness'] = eval_fitness(
        mutated_graph, graph_x, blackbox, distance_function, alpha1, alpha2, case_type
    )
    
    return mutated_graph

def crossover(parent1, parent2, graph_x, blackbox, distance_function, alpha1, alpha2, case_type):
    parent1_edges = list(set(tuple(edge.tolist()) for edge in parent1['edge_index'].T))
    parent2_edges = list(set(tuple(edge.tolist()) for edge in parent2['edge_index'].T))

    # Return parent1 if not enough edges for crossover
    if len(parent1_edges) < 2 or len(parent2_edges) < 2:
        return parent1
        
    # Two-point crossover for edges
    cut1, cut2 = sorted(random.sample(range(len(parent1_edges)), 2))
    child_edges = parent1_edges[:cut1] + parent2_edges[cut1:cut2] + parent1_edges[cut2:]
    child_edges = list(set(child_edges))  # Remove duplicates
    new_edge_index = torch.tensor(child_edges, dtype=torch.long).T

    # Feature crossover
    new_x = parent1['x'].clone()
    for i in range(parent1['x'].size(0)):
        new_x[i] = (parent1['x'][i] + parent2['x'][i]) / 2 if random.random() < 0.5 else parent2['x'][i]

    # Create child graph
    child = {
        'num_nodes': parent1['num_nodes'],
        'x': new_x,
        'edge_index': new_edge_index,
        'innovation_numbers': {edge: i for i, edge in enumerate(child_edges)},
        'fitness': 0
    }
    
    # Evaluate fitness
    child['fitness'] = eval_fitness(
        child, graph_x, blackbox, distance_function, alpha1, alpha2, case_type
    )

    return child

def initialize_population(size, graph_x, blackbox, distance_function, alpha1, alpha2, case_type):
    population = [create_graph(graph_x) for _ in range(size)]
    
    for individual in population:
        individual['fitness'] = eval_fitness(
            individual, graph_x, blackbox, distance_function, alpha1, alpha2, case_type
        )
            
    return population

def select_parents(population, selection_ratio=0.4):
    """Select top performers from population"""
    sorted_population = sorted(population, key=lambda x: x['fitness'], reverse=True)
    return sorted_population[:int(selection_ratio * len(population))]

def evolve_population(population, graph_x, blackbox, distance_function, alpha1, alpha2, case_type, generation):
    print(f"{case_type.upper()} Generation {generation+1}")
    
    new_population = []
    best_individuals = select_parents(population)
    
    # Elitism: Keep top performers
    new_population.extend(best_individuals[:len(best_individuals) // 2])
    
    # Generate the rest of the population
    while len(new_population) < len(population):
        parent1, parent2 = random.sample(best_individuals, 2)
        child = crossover(
            parent1, parent2, graph_x, blackbox, 
            distance_function, alpha1, alpha2, case_type
        )
        
        # Apply mutation with some probability
        if random.random() < 0.2:
            child = mutate(
                child, graph_x, blackbox, distance_function,
                alpha1, alpha2, case_type
            )
            
        new_population.append(child)
        
    return new_population

def run_genetic_algorithm(graph_x, population_size=100, generations=50, blackbox=None, 
                          distance_function=None, alpha1=0.5, alpha2=0.5):
    case_types = ['sso', 'sdo', 'dso', 'ddo']
    size_per_case = population_size // len(case_types)
    
    # Initialize populations for all case types
    populations = {}
    for case_type in case_types:
        populations[case_type] = initialize_population(
            size_per_case, graph_x, blackbox, distance_function, alpha1, alpha2, case_type
        )
    
    # Run evolution for each case type
    for gen in range(generations):
        print(f"\n===== Generation {gen+1}/{generations} =====")
        
        for case_type in case_types:
            populations[case_type] = evolve_population(
                populations[case_type], graph_x, blackbox, 
                distance_function, alpha1, alpha2, case_type, gen
            )
            
    final_population = []
    for population in populations.values():
        final_population.extend(population)
        
    return final_population

# Example usage
def main():
    dataset = BA2MotifDataset(root='data/BA2Motif')
    graph_x = dataset[0]
    model = GIN(
        in_channels=dataset.num_features,
        hidden_channels=32,
        out_channels=dataset.num_classes,
        num_layers=5,
    )
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()
    blackbox = model
    result = run_genetic_algorithm(
        graph_x=graph_x,
        population_size=100,
        generations=10,
        blackbox=blackbox,
        distance_function=compute_fgw_distance,
        alpha1=0.5,
        alpha2=0.5
    )
    # Get best individual across all cases
    best_individual = max(result, key=lambda x: x['fitness'])
    return best_individual

if __name__ == '__main__':
    main()