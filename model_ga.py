import torch, random, os, json
import torch.nn as nn
from torch.optim import AdamW
from copy import deepcopy

from model_cnn import CNN

# Define the search space for CNN architecture
class CNNSearchSpace:
    def __init__(self):
        self.conv_layers = [1, 2, 3, 4]
        self.filters = [16, 32, 64, 128]
        self.kernel_sizes = [3, 5, 7]
        self.pool_types = ['max', 'avg']
        self.activations = ['relu', 'leaky_relu']
        self.fc_units = [64, 128, 256, 512]

# Encode architecture as a chromosome (gene representation)
class Architecture:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = self.random_genes()
        else:
            self.genes = genes
        self.fitness = 0
        self.accuracy = 0
        self.best_epoch = 0
        # for Q1B - separate conv/fc tracking
        self.conv_params = 0
        self.fc_params = 0
        self.conv_penalty = 0
        self.fc_penalty = 0
    
    def random_genes(self):
        space = CNNSearchSpace()
        num_conv = random.choice(space.conv_layers)
        
        genes = {
            'num_conv': num_conv,
            'conv_configs': [],
            'pool_type': random.choice(space.pool_types),
            'activation': random.choice(space.activations),
            'fc_units': random.choice(space.fc_units)
        }
        
        for _ in range(num_conv):
            genes['conv_configs'].append({
                'filters': random.choice(space.filters),
                'kernel_size': random.choice(space.kernel_sizes)
            })
        
        return genes
    
    def __repr__(self):
        return f"Arch(conv={self.genes['num_conv']}, acc={self.accuracy:.4f})"

# Genetic Algorithm Operations
class GeneticAlgorithm:
    def __init__(self, population_size=20, generations=10, mutation_rate=0.2, crossover_rate=0.7):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_architecture = None
        self.search_space = CNNSearchSpace()
    
    def initialize_population(self):
        self.population = [Architecture() for _ in range(self.population_size)]
    
    def count_conv_fc_params(self, model):
        """separate count for conv and fc params"""
        conv_count = 0
        fc_count = 0
        
        for name, param in model.named_parameters():
            if 'features' in name:
                conv_count = conv_count + param.numel()
            elif 'classifier' in name:
                fc_count = fc_count + param.numel()
        
        return conv_count, fc_count
    
    def evaluate_fitness(self, architecture, train_loader, val_loader, device, epochs=100):
        """Train and evaluate a single architecture"""
        try:
            model = CNN(architecture.genes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=0.001)
            
            # Quick training
            best_acc = 0
            patience = 10
            step = 1
            best_epoch = 1
            for epoch in range(1, epochs+1):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
                # Evaluation
                model.eval()
                correct = 0
                # total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        # total += labels.size(0)
                        correct += (predicted == labels).sum().item()
            
                accuracy = correct / len(val_loader.dataset)
                if accuracy > best_acc:
                    step = 0
                    best_acc = accuracy
                    best_epoch = epoch
                else:
                    step += 1
                if step >= patience:
                    break
            
            # separate conv and fc params for different penalties
            conv_p, fc_p = self.count_conv_fc_params(model)
            
            # different weights: conv is more expensive computationally
            w_conv = 0.0008
            w_fc = 0.0002
            
            penalty_conv = w_conv * conv_p
            penalty_fc = w_fc * fc_p
            total_pen = penalty_conv + penalty_fc
            
            # store for logging
            architecture.conv_params = conv_p
            architecture.fc_params = fc_p
            architecture.conv_penalty = penalty_conv
            architecture.fc_penalty = penalty_fc

            del model, inputs, outputs, labels
            torch.cuda.empty_cache()
            
            # fitness = accuracy - penalties
            architecture.accuracy = best_acc
            architecture.best_epoch = best_epoch
            architecture.fitness = best_acc - total_pen
            
            return architecture.fitness
            
        except Exception as e:
            print(f"Error evaluating architecture: {e}", flush=True)
            architecture.fitness = 0
            architecture.accuracy = 0
            architecture.conv_params = 0
            architecture.fc_params = 0
            architecture.conv_penalty = 0
            architecture.fc_penalty = 0
            return 0
    
    def selection(self):
        """Roulette wheel selection - fitness proportionate"""
        # get all fitness scores first
        fitness_vals = []
        for arch in self.population:
            fitness_vals.append(arch.fitness)
        
        # make sure no negative or zero values (for probability calc)
        adjusted = []
        for f in fitness_vals:
            if f < 1e-6:
                adjusted.append(1e-6)
            else:
                adjusted.append(f)
        
        total = sum(adjusted)
        
        # edge case: if total is too small, just return copies
        if total < 1e-10:
            result = []
            for a in self.population:
                result.append(deepcopy(a))
            return result
        
        # calculate probabilities (relative fitness)
        probs = []
        for f in adjusted:
            probs.append(f / total)
        
        # build cumulative distribution
        cumsum = []
        s = 0.0
        for p in probs:
            s = s + p
            cumsum.append(s)
        
        # log for assignment
        print(f"\n{'='*60}", flush=True)
        print("Roulette-Wheel Selection - Fitness and Probabilities", flush=True)
        print(f"{'='*60}", flush=True)
        for i in range(len(self.population)):
            print(f"Arch {i+1}: Fitness={self.population[i].fitness:.6f}, "
                  f"Relative={probs[i]:.6f}, Cumulative={cumsum[i]:.6f}", flush=True)
        
        # roulette wheel selection
        selected = []
        for _ in range(self.population_size):
            r = random.uniform(0.0, 1.0)
            chosen = False
            for j in range(len(cumsum)):
                if r <= cumsum[j]:
                    selected.append(deepcopy(self.population[j]))
                    chosen = True
                    break
            if not chosen:
                # fallback to last one
                selected.append(deepcopy(self.population[-1]))
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Single-point crossover for architectures"""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        child1_genes = deepcopy(parent1.genes)
        child2_genes = deepcopy(parent2.genes)
        
        # Crossover number of conv layers and pool type
        if random.random() < 0.5:
            child1_genes['num_conv'], child2_genes['num_conv'] = child2_genes['num_conv'], child1_genes['num_conv']
        
        # Crossover pool type and activation
        if random.random() < 0.5:
            child1_genes['pool_type'], child2_genes['pool_type'] = child2_genes['pool_type'], child1_genes['pool_type']
            child1_genes['activation'], child2_genes['activation'] = child2_genes['activation'], child1_genes['activation']
        
        # Adjust conv_configs to match num_conv
        min_len = min(child1_genes['num_conv'], len(child1_genes['conv_configs']))
        child1_genes['conv_configs'] = child1_genes['conv_configs'][:min_len]
        while len(child1_genes['conv_configs']) < child1_genes['num_conv']:
            child1_genes['conv_configs'].append({
                'filters': random.choice(self.search_space.filters),
                'kernel_size': random.choice(self.search_space.kernel_sizes)
            })
        
        min_len = min(child2_genes['num_conv'], len(child2_genes['conv_configs']))
        child2_genes['conv_configs'] = child2_genes['conv_configs'][:min_len]
        while len(child2_genes['conv_configs']) < child2_genes['num_conv']:
            child2_genes['conv_configs'].append({
                'filters': random.choice(self.search_space.filters),
                'kernel_size': random.choice(self.search_space.kernel_sizes)
            })
        
        return Architecture(child1_genes), Architecture(child2_genes)
    
    def mutation(self, architecture):
        """Mutate architecture genes"""
        if random.random() > self.mutation_rate:
            return architecture
        
        genes = deepcopy(architecture.genes)
        mutation_type = random.choice(['conv_param', 'num_layers', 'pool_activation', 'fc_units'])
        
        if mutation_type == 'conv_param' and genes['conv_configs']:
            # Mutate a random conv layer
            idx = random.randint(0, len(genes['conv_configs']) - 1)
            genes['conv_configs'][idx]['filters'] = random.choice(self.search_space.filters)
            genes['conv_configs'][idx]['kernel_size'] = random.choice(self.search_space.kernel_sizes)
        
        elif mutation_type == 'num_layers':
            # Change number of conv layers
            genes['num_conv'] = random.choice(self.search_space.conv_layers)
            # Adjust conv_configs
            if genes['num_conv'] > len(genes['conv_configs']):
                for _ in range(genes['num_conv'] - len(genes['conv_configs'])):
                    genes['conv_configs'].append({
                        'filters': random.choice(self.search_space.filters),
                        'kernel_size': random.choice(self.search_space.kernel_sizes)
                    })
            else:
                genes['conv_configs'] = genes['conv_configs'][:genes['num_conv']]
        
        elif mutation_type == 'pool_activation':
            genes['pool_type'] = random.choice(self.search_space.pool_types)
            genes['activation'] = random.choice(self.search_space.activations)
        
        elif mutation_type == 'fc_units':
            genes['fc_units'] = random.choice(self.search_space.fc_units)
        
        return Architecture(genes)
    
    def evolve(self, train_loader, val_loader, device, run=1):
        parent = os.path.abspath('')
        """Main evolutionary loop"""
        self.initialize_population()
        print(f"Starting with {self.population_size} Population:\n{self.population}\n", flush=True)
        
        for generation in range(self.generations):
            print(f"\n{'='*60}", flush=True)
            print(f"Generation {generation + 1}/{self.generations}", flush=True)
            print(f"{'='*60}", flush=True)
            
            # Evaluate fitness
            for i, arch in enumerate(self.population):
                print(f"Evaluating architecture {i+1}/{self.population_size}...", end=' ', flush=True)
                fitness = self.evaluate_fitness(arch, train_loader, val_loader, device)
                # log params and penalties for Q1B
                print(f"Fitness: {fitness:.6f}, Accuracy: {arch.accuracy:.6f}, "
                      f"Conv Params: {arch.conv_params}, FC Params: {arch.fc_params}, "
                      f"Conv Penalty: {arch.conv_penalty:.6f}, FC Penalty: {arch.fc_penalty:.6f}, "
                      f"Total Penalty: {arch.conv_penalty + arch.fc_penalty:.6f}", flush=True)
            
            # Sort by fitness score
            print(f"\nSorting population in terms of fitness score (high -> low) ...", flush=True)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best
            if self.best_architecture is None or self.population[0].fitness > self.best_architecture.fitness:
                self.best_architecture = deepcopy(self.population[0])
            
            print(f"Best in generation: {self.population[0]}\n", flush=True)
            print(f"Best overall: {self.best_architecture}", flush=True)
            
            # Selection using roulette wheel
            print(f"\nPerforming roulette-wheel selection of total population: {self.population_size} ...", flush=True)
            selected = self.selection()
            
            # Crossover and Mutation
            print(f"Performing Crossover & Mutation ...", flush=True)
            next_generation = []
            
            # Elitism: keep top 2 architectures
            print(f"Elitism: Keeping top 2 architectures in next generation.", flush=True)
            next_generation.extend([deepcopy(self.population[0]), deepcopy(self.population[1])])
            
            while len(next_generation) < self.population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                next_generation.append(child1)
                if len(next_generation) < self.population_size:
                    next_generation.append(child2)
            
            self.population = next_generation
            print(f"Next Generation: {self.population}", flush=True)
            with open(os.path.join(parent, 'outputs', f'run_{run}', f"generation_{generation}.jsonl"), 'w') as f:
                for obj in self.population:
                    f.write(json.dumps(obj.genes))
        
        return self.best_architecture