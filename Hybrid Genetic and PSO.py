import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Defining problem-specific constants
D = np.array([125000, 318000, 50000])  # Customer demands

# Defining the fitness function for multi-objective optimization
def fitness_function(individual):
    # Extracting decision variables
    xh1i = individual[0]
    xh2i = individual[1]
    xie  = individual[2]
    xej1 = individual[3]
    xej2 = individual[4]
    xej3 = individual[5]

    # Defining the three objective functions
    cost_minimization = (36000 + 12000 + 5.59e-4 * xh1i + 3.706e-3 * xh2i + 7.12 * xie + 4.94e-3 * xej1 + 2.67e-3 * xej2 + 0.0119 * xej3)
    fill_rate_maximization = (xej1 + xej2 + xej3) / np.sum(D)
    co2_minimization = (3.33e-3 * 0.673 * 2.3e-5 * xie) + (45.9 * xh1i + 25.4 * xh1i + 0 * xie + 17 * xej1 + 23 * xej2 + 16.4 * xej3) * 0.7 * 2.3e-5

    # Checking constraints
    constraint1 = xh1i + xh2i - 550000
    constraint2 = xh1i + xh2i - xie
    constraint3 = xie - (xej1 + xej2 + xej3)
    constraint4 = (xej1 + xej2 + xej3) - 1000000
    constraint5 = 0.8 <= (xej1 + xej2 + xej3) / 493000 <= 1
    constraint6 = (3.33e-3 * 0.673 * 2.3e-5 * xie) + (45.9 * xh1i + 25.4 * xh1i + 0 * xie + 17 * xej1 + 23 * xej2 + 16.4 * xej3) * 0.7 * 2.3e-5 <= 2250000
    constraint7 = all(x >= 0 for x in [xej1, xej2, xej3, xie, xh1i, xh2i])

    if constraint1 >= 0 or constraint2 < 0 or constraint3 < 0 or constraint4 > 0 or not constraint5 or not constraint6 or not constraint7:
        # Penalize the objectives if constraints are not satisfied
        return float("inf"), float("-inf"), float("inf")

    return cost_minimization, fill_rate_maximization, co2_minimization

# Defining the problem as a multi-objective optimization problem
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Creating a Toolbox for evolution
toolbox = base.Toolbox()

# Defining the decision variables with specified ranges
toolbox.register("attr_xh1i", random.uniform, 0, 300000)
toolbox.register("attr_xh2i", random.uniform, 0, 250000)
toolbox.register("attr_xie", random.uniform, 0, 500000)
toolbox.register("attr_xej1", random.uniform, 0, 125000)
toolbox.register("attr_xej2", random.uniform, 0, 318000)
toolbox.register("attr_xej3", random.uniform, 0, 50000)

# Decision variable bounds
num_decision_variables = 6

# Function to create an individual with decision variables within specified ranges
def create_individual():
    return [
        toolbox.attr_xh1i(),
        toolbox.attr_xh2i(),
        toolbox.attr_xie(),
        toolbox.attr_xej1(),
        toolbox.attr_xej2(),
        toolbox.attr_xej3()
    ]

# Registering the individual creation function
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
# Creating a population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Defining the evaluation function
toolbox.register("evaluate", fitness_function)

# Defining the selection operator
toolbox.register("select", tools.selNSGA2)

# Defining the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# Defining the mutation operator
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)

# Create an initial population
population = toolbox.population(n=100)

# Evaluating the entire population
fitnesses = map(toolbox.evaluate, population)
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# Evolutionary algorithm parameters
num_generations = 100
cxpb = 0.7  # Crossover probability
mutpb = 0.2  # Mutation probability

# Defining a PSO-based function to update a subset of individuals in the population
def pso_update(population, toolbox):
    # Number of particles (individuals) to update using PSO
    num_particles = 20
    
    # Extracting decision variables of the population
    pop_xh1i = [ind[0] for ind in population]
    pop_xh2i = [ind[1] for ind in population]
    pop_xie = [ind[2] for ind in population]
    pop_xej1 = [ind[3] for ind in population]
    pop_xej2 = [ind[4] for ind in population]
    pop_xej3 = [ind[5] for ind in population]
    
    # PSO parameters
    num_dimensions = len(population[0])  # Number of decision variables
    num_iterations = 10  # Number of PSO iterations
    c1 = 1.49618
    c2 = 1.49618
    w = 0.729844
    
    # Initializing particle positions and velocities
    particle_positions = np.array([pop_xh1i, pop_xh2i, pop_xie, pop_xej1, pop_xej2, pop_xej3]).T
    particle_velocities = np.zeros_like(particle_positions)
    
    # Initializing personal best positions and fitness values
    personal_best_positions = particle_positions.copy()
    personal_best_fitness = [toolbox.evaluate(ind)[0] for ind in population]
    
    # Initializing global best position and fitness value
    global_best_position = np.copy(personal_best_positions[np.argmin(personal_best_fitness)])
    global_best_fitness = np.min(personal_best_fitness)
    
    for iteration in range(num_iterations):
        for i in range(num_particles):
            r1, r2 = random.random(), random.random()
            inertia = w * particle_velocities[i]
            cognitive = c1 * r1 * (personal_best_positions[i] - particle_positions[i])
            social = c2 * r2 * (global_best_position - particle_positions[i])
            particle_velocities[i] = inertia + cognitive + social
            particle_positions[i] = particle_positions[i] + particle_velocities[i]
            
            # Applying boundary constraints to particle positions
            particle_positions[i] = np.maximum(0, particle_positions[i])
            
            # Evaluating the particle's fitness
            fitness = toolbox.evaluate(list(particle_positions[i]))
            cost = fitness[0]
            
            # Updating personal best if needed
            if cost < personal_best_fitness[i]:
                personal_best_positions[i] = np.copy(particle_positions[i])
                personal_best_fitness[i] = cost
                
                # Updating global best if needed
                if cost < global_best_fitness:
                    global_best_position = np.copy(particle_positions[i])
                    global_best_fitness = cost
                    
        # Updating the population with the best particles from PSO
        for i, ind in enumerate(population):
            # Check if the PSO particle has a better fitness than the individual in the population
            if personal_best_fitness[i] < toolbox.evaluate(ind)[0]:
                population[i] = creator.Individual(list(personal_best_positions[i]))

# Running the evolutionary algorithm (NSGA-II) with PSO updates
for generation in range(num_generations):
    algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=cxpb, mutpb=mutpb, ngen=1,
                              stats=None, halloffame=None, verbose=True)
    
    # Applying PSO updates every 10 generations
    if generation % 10 == 0:
        pso_update(population, toolbox)

# Extracting Pareto front solutions and objectives
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
costs = []
fill_rates = []
co2_values = []

# Calculating crowding distance for each solution in the Pareto front
def crowding_distance(ind, front):
    distance = 0
    objectives = list(zip(*[fit.values for fit in front]))
    for obj in objectives:
        sorted_indices = np.argsort(obj)
        distance += obj[sorted_indices[-1]] - obj[sorted_indices[0]]
        for i in range(1, len(obj) - 1):
            distance += obj[sorted_indices[i + 1]] - obj[sorted_indices[i - 1]]
    return distance

for ind in pareto_front:
    ind.crowding_distance = 0

num_objectives = len(pareto_front[0].fitness.values)
for i in range(num_objectives):
    pareto_front.sort(key=lambda ind: ind.fitness.values[i])
    pareto_front[0].crowding_distance = pareto_front[-1].crowding_distance = float("inf")
    min_obj = pareto_front[0].fitness.values[i]
    max_obj = pareto_front[-1].fitness.values[i]
    if min_obj == max_obj:
        continue
    for j in range(1, len(pareto_front) - 1):
        pareto_front[j].crowding_distance += (pareto_front[j + 1].fitness.values[i] - pareto_front[j - 1].fitness.values[i]) / (max_obj - min_obj)

# Extracting crowding distances for each chromosome in the Pareto front
crowding_distances = [ind.crowding_distance for ind in pareto_front]

# Printing solution variables and objectives
for idx, ind in enumerate(pareto_front):
    objectives = fitness_function(ind)
    costs.append(objectives[0])
    fill_rates.append(objectives[1])
    co2_values.append(objectives[2])
    
    print(f"Solution {idx + 1}:")
    print(f"Decision Variables (xh1i, xh2i, xie, xej1, xej2, xej3): {ind}")
    print(f"Objective Values (Cost, Fill Rate, CO2): {objectives}\n")

# Creating subplots for each pair of objectives
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(hspace=0.5)

# Plotting Cost vs. Fill Rate
axs[0, 0].scatter(costs, fill_rates, c='b', marker='o', label='Pareto Front')
axs[0, 0].set_xlabel('Cost Minimization')
axs[0, 0].set_ylabel('Fill Rate Maximization')
axs[0, 0].set_title('Cost vs. Fill Rate')

# Plotting Cost vs. CO2
axs[0, 1].scatter(costs, co2_values, c='g', marker='o', label='Pareto Front')
axs[0, 1].set_xlabel('Cost Minimization')
axs[0, 1].set_ylabel('CO2 Minimization')
axs[0, 1].set_title('Cost vs. CO2')

# Plotting Fill Rate vs. CO2
axs[1, 0].scatter(fill_rates, co2_values, c='r', marker='o', label='Pareto Front')
axs[1, 0].set_xlabel('Fill Rate Maximization')
axs[1, 0].set_ylabel('CO2 Minimization')
axs[1, 0].set_title('Fill Rate vs. CO2')

# Removing the empty subplot
fig.delaxes(axs[1, 1])

# Showing the plot
plt.tight_layout()
plt.legend()
plt.suptitle('Pareto Front for Multi-Objective Optimization')

# Creating a 3D scatter plot for all three objectives
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Plotting the Pareto front in 3D
ax_3d.scatter(costs, fill_rates, co2_values, c='b', marker='o', label='Pareto Front')

# Setting axis labels
ax_3d.set_xlabel('Cost Minimization')
ax_3d.set_ylabel('Fill Rate Maximization')
ax_3d.set_zlabel('CO2 Minimization')

# Showing the 3D plot
plt.legend()
plt.title('Pareto Front (3D) for Multi-Objective Optimization')

# Creating a parallel coordinates plot without pandas
fig_parallel = plt.figure(figsize=(8, 6))
ax_parallel = fig_parallel.add_subplot(111)

# Convering lists to NumPy arrays for normalization
costs = np.array(costs)
fill_rates = np.array(fill_rates)
co2_values = np.array(co2_values)

# Normalizing objectives for better visualization
normalized_costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))
normalized_fill_rates = (fill_rates - np.min(fill_rates)) / (np.max(fill_rates) - np.min(fill_rates))
normalized_co2_values = (co2_values - np.min(co2_values)) / (np.max(co2_values) - np.min(co2_values))

# Plotting the parallel coordinates plot
for i in range(len(costs)):
    ax_parallel.plot([0, 1, 2], [normalized_costs[i], normalized_fill_rates[i], normalized_co2_values[i]])

# Setting axis labels and ticks
ax_parallel.set_xticks([0, 1, 2])
ax_parallel.set_xticklabels(['Cost', 'Fill Rate', 'CO2'])
ax_parallel.set_xlabel('Objectives')
ax_parallel.set_ylabel('Normalized Values')
ax_parallel.set_title('Parallel Coordinates Plot for Pareto Front')

# Adding legend
ax_parallel.legend(loc='upper right')

# Showing the parallel coordinates plot
plt.tight_layout()

plt.show()
