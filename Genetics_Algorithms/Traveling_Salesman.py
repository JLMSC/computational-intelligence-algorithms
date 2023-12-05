import numpy as np
import matplotlib.pyplot as plt


def fitness_function(individual, points) -> np.ndarray:

    def calculate_distance(point_1, point_2) -> float:
        return np.sqrt(np.sum(a=(point_1 - point_2) ** 2))

    indices = individual.astype(int)
    distances = np.array(object=[
        calculate_distance(point_1=points[indices[i]], point_2=points[indices[i + 1]])
        for i in range(len(indices) - 1)
    ])
    distances = np.append(arr=distances, values=calculate_distance(point_1=points[indices[-1]], point_2=points[indices[0]]))
    return np.sum(a=distances)


def generate_points(N: int = 40) -> np.ndarray:
    x_partition = np.random.uniform(low=-10, high=10, size=(N, 3))
    y_partition = np.random.uniform(low=0, high=20, size=(N, 3))
    z_partition = np.random.uniform(low=-20, high=0, size=(N, 3))
    w_partition = np.random.uniform(low=0, high=20, size=(N, 3))

    x_partition += np.tile(A=np.array(object=[[20, -20, -20]]), reps=(N, 1))
    y_partition += np.tile(A=np.array(object=[[-20, 20, 20]]), reps=(N, 1))
    z_partition += np.tile(A=np.array(object=[[-20, 20, -20]]), reps=(N, 1))
    w_partition += np.tile(A=np.array(object=[[20, 20, -20]]), reps=(N, 1))

    return np.concatenate((x_partition, y_partition, z_partition, w_partition), axis=0)


def generate_population(points: np.ndarray, population_size: int = 100) -> np.ndarray:  
    population = np.empty(shape=(0, points.shape[0]))
    for _ in range(population_size):
        individual = np.random.permutation(x=points.shape[0]).reshape(1, points.shape[0])
        population = np.concatenate((population, individual))
    return population


def tournament_selection(population: np.ndarray, fitness: np.ndarray, tournament_size: int) -> np.ndarray:
    selected_indices = []
    population_size = len(population)
    for _ in range(population_size):
        # Randomly choose some inviduals for the tournament.
        tournament_indices = np.random.choice(a=population_size, size=tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]

        # Select the best individual in the tournament.
        winner_index = tournament_indices[np.argmin(a=tournament_fitness)]
        selected_indices.append(winner_index)
    return population[selected_indices]


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> list[np.ndarray]:
    # Recombination probability
    pc = np.random.uniform(low=0.85, high=0.95)

    if np.random.rand() < pc:
        # Recombine both parent's chromossomes without repetition.
        start, end = sorted(np.random.choice(a=len(parent1), size=2, replace=False))

        # First child.
        child1 = [gene for gene in parent1[start:end+1]]
        child1 += [gene for gene in parent2 if gene not in child1]
        child1 = np.array(object=child1)
        
        # Second child.
        child2 = [gene for gene in parent2[start:end+1]]
        child2 += [gene for gene in parent1 if gene not in child2]
        child2 = np.array(object=child2)

        # Try to mutate the childrens.
        mutate(individual=child1)
        mutate(individual=child2)

        return [child1, child2]
    
    # No recombinations.
    return [parent1, parent2]


def mutate(individual: np.ndarray, pm: float = 0.1) -> None:
    if np.random.rand() < pm:
        # ? Is this doing something?
        mutation_point1, mutation_point2 = np.random.choice(a=len(individual), size=2, replace=False)
        individual[mutation_point1], individual[mutation_point2] = individual[mutation_point1], individual[mutation_point2]


def main() -> None:
    population_size = 100
    max_generations = 100
    amount_of_points = 40
    tournament_size = 10

    # Generate points.
    points = generate_points(N=amount_of_points)
    _I = np.random.permutation(x=amount_of_points * 4)
    points = np.delete(arr=points, obj=_I[0], axis=0)

    # Generate population.
    population = generate_population(points=points, population_size=population_size)

    # Origin vector.
    origin_vector = np.tile(A=np.array(object=[[int(_I[0])]]), reps=(population_size, 1))

    # Paths.
    paths = np.concatenate((origin_vector, population, origin_vector), axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='#248DD2', marker='o')

    # Fitness for initial population.
    fitness = np.apply_along_axis(func1d=fitness_function, axis=1, arr=paths, points=points)

    for generation in range(max_generations):
        # Selection.
        selected_inviduals = tournament_selection(population=population, fitness=fitness, tournament_size=tournament_size)

        # Crossover and mutation.
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_inviduals[i]
            parent2 = selected_inviduals[i + 1]
            children = crossover(parent1=parent1, parent2=parent2)
            new_population.extend(children)
        
        # Substitute new population.
        population = np.array(object=new_population)

        # Update path.
        paths = np.concatenate((origin_vector, population), axis=1)

        # Fitness of the new population.
        fitness = np.apply_along_axis(func1d=fitness_function, axis=1, arr=paths, points=points)
        
        # Mean fitness in current population.
        avg_fitness = np.mean(a=fitness)
        print(f'Generation {generation + 1}: Average Fitness = {avg_fitness}')
    
    # Best individual, same as best path.
    best_path = paths[np.argmin(a=fitness)]
    print(f'Best path: {best_path}\nCost: {np.min(a=fitness)}')

    # Change starting and ending point marker.
    origin_point = points[int(best_path[0]), :].reshape(1, 3)
    ax.scatter(origin_point[0:, 0], origin_point[0:, 1], origin_point[0:, 2], c='green', marker='x', linewidth=3, s=60)

    # Plot best path.
    p_i, p_j = 0, 0
    for i in range(0, len(best_path) - 1):
        p_i = points[int(best_path[i]), :].reshape(1, 3)
        p_j = points[int(best_path[i + 1]), :].reshape(1, 3)
        ax.plot([p_i[0, 0], p_j[0, 0]], [p_i[0, 1], p_j[0, 1]], [p_i[0, 2], p_j[0, 2]],color='k')

    # Plot to return to initial point.
    ax.plot([p_j[0, 0], origin_point[0, 0]], [p_j[0, 1], origin_point[0, 1]], [p_j[0, 2], origin_point[0, 2]], color='k')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()