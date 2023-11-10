import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define problem-specific constants
D = np.array([125000, 318000, 50000])  # Customer demands

# Define the fitness function for multi-objective optimization
def fitness_function(individual):
    # Extract decision variables
    xh1i = individual[0]
    xh2i = individual[1]
    xie  = individual[2]
    xej1 = individual[3]
    xej2 = individual[4]
    xej3 = individual[5]

    # Define the three objective functions
    cost_minimization = (36000 + 12000 + 5.59e-4 * xh1i + 3.706e-3 * xh2i + 7.12 * xie + 4.94e-3 * xej1 + 2.67e-3 * xej2 + 0.0119 * xej3)
    fill_rate_maximization = (xej1 + xej2 + xej3) / np.sum(D)
    co2_minimization = (3.33e-3 * 0.673 * 2.3e-5 * xie) + (45.9 * xh1i + 25.4 * xh1i + 0 * xie + 17 * xej1 + 23 * xej2 + 16.4 * xej3) * 0.7 * 2.3e-5

    # Check constraints
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

# Define the problem as a multi-objective optimization problem
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Create a Toolbox for evolution
toolbox = base.Toolbox()

# Define the decision variables
toolbox.register("attr_float", random.uniform, 0, 500000)

# Decision variable bounds
num_decision_variables = 6
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_decision_variables)

# Create a population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the evaluation function
toolbox.register("evaluate", fitness_function)

# Define the selection operator
toolbox.register("select", tools.selNSGA2)

# Define the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# Define the mutation operator
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)

# Create an initial population
population = toolbox.population(n=100)

# Evaluate the entire population
fitnesses = map(toolbox.evaluate, population)
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# Evolutionary algorithm parameters
num_generations = 100
cxpb = 0.7  # Crossover probability
mutpb = 0.2  # Mutation probability

# Run the evolutionary algorithm (NSGA-II)
algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=cxpb, mutpb=mutpb, ngen=num_generations,
                          stats=None, halloffame=None, verbose=True)

# Extract Pareto front solutions and objectives
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
costs = []
fill_rates = []
co2_values = []

# Print solution variables and objectives
for idx, ind in enumerate(pareto_front):
    objectives = fitness_function(ind)
    costs.append(objectives[0])
    fill_rates.append(objectives[1])
    co2_values.append(objectives[2])
    
    print(f"Solution {idx + 1}:")
    print(f"Decision Variables (xh1i, xh2i, xie, xej1, xej2, xej3): {ind}")
    print(f"Objective Values (Cost, Fill Rate, CO2): {objectives}\n")

# Create subplots for each pair of objectives
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(hspace=0.5)

# Plot Cost vs. Fill Rate
axs[0, 0].scatter(costs, fill_rates, c='b', marker='o', label='Pareto Front')
axs[0, 0].set_xlabel('Cost Minimization')
axs[0, 0].set_ylabel('Fill Rate Maximization')
axs[0, 0].set_title('Cost vs. Fill Rate')

# Plot Cost vs. CO2
axs[0, 1].scatter(costs, co2_values, c='g', marker='o', label='Pareto Front')
axs[0, 1].set_xlabel('Cost Minimization')
axs[0, 1].set_ylabel('CO2 Minimization')
axs[0, 1].set_title('Cost vs. CO2')

# Plot Fill Rate vs. CO2
axs[1, 0].scatter(fill_rates, co2_values, c='r', marker='o', label='Pareto Front')
axs[1, 0].set_xlabel('Fill Rate Maximization')
axs[1, 0].set_ylabel('CO2 Minimization')
axs[1, 0].set_title('Fill Rate vs. CO2')

# Remove the empty subplot
fig.delaxes(axs[1, 1])

# Show the plot
plt.tight_layout()
plt.legend()
plt.suptitle('Pareto Front for Multi-Objective Optimization')

# Create a 3D scatter plot for all three objectives
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Plot the Pareto front in 3D
ax_3d.scatter(costs, fill_rates, co2_values, c='b', marker='o', label='Pareto Front')

# Set axis labels
ax_3d.set_xlabel('Cost Minimization')
ax_3d.set_ylabel('Fill Rate Maximization')
ax_3d.set_zlabel('CO2 Minimization')

# Show the 3D plot
plt.legend()
plt.title('Pareto Front (3D) for Multi-Objective Optimization')

# Create a parallel coordinates plot without pandas
fig_parallel = plt.figure(figsize=(8, 6))
ax_parallel = fig_parallel.add_subplot(111)

# Convert lists to NumPy arrays for normalization
costs = np.array(costs)
fill_rates = np.array(fill_rates)
co2_values = np.array(co2_values)

# Normalize objectives for better visualization
normalized_costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))
normalized_fill_rates = (fill_rates - np.min(fill_rates)) / (np.max(fill_rates) - np.min(fill_rates))
normalized_co2_values = (co2_values - np.min(co2_values)) / (np.max(co2_values) - np.min(co2_values))

# Plot the parallel coordinates plot
for i in range(len(costs)):
    ax_parallel.plot([0, 1, 2], [normalized_costs[i], normalized_fill_rates[i], normalized_co2_values[i]], label=f'Solution {i + 1}')

# Set axis labels and ticks
ax_parallel.set_xticks([0, 1, 2])
ax_parallel.set_xticklabels(['Cost', 'Fill Rate', 'CO2'])
ax_parallel.set_xlabel('Objectives')
ax_parallel.set_ylabel('Normalized Values')
ax_parallel.set_title('Parallel Coordinates Plot for Pareto Front')

# Add legend
ax_parallel.legend(loc='upper right')

# Show the parallel coordinates plot
plt.tight_layout()


# Show all plots
plt.show()
end
