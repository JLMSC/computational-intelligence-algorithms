import numpy as np


class Board:
    def __init__(self, chromossomes: list, fitness: int) -> None:
        self.chromossomes = chromossomes
        self.fitness = fitness


def fitness_function(chromossomes: list) -> int:
    # Start by counting attacks in rows.
    attacks = abs(len(chromossomes) - len(np.unique(ar=chromossomes)))
    # Count and sum up diagonal attacks.
    for i in range(len(chromossomes)):
        for j in range(len(chromossomes)):
            if i != j:
                if abs(i - j) == abs(chromossomes[i] - chromossomes[j]):
                    attacks += 1
    # 28 defines the numbers of arrangemens of non attacking pairs.
    return 28 - attacks


def generate_chromosomes(N: int = 8) -> list:
    distribution = np.arange(N)
    np.random.shuffle(x=distribution)
    return list(distribution)


def generate_population(population_size: int = 100) -> list[Board]:
    population = []
    for _ in range(population_size):
        chromossomes = generate_chromosomes()
        fitness = fitness_function(chromossomes=chromossomes)
        population.append(Board(chromossomes=chromossomes, fitness=fitness))
    return population


def wheel_selection(population: list[Board]) -> Board:
    # Calculates every probabilitie for each individual in the population,
    # it sums up to 100%, or 1.0.
    probabilities = [
        individual.fitness / sum(individual.fitness for individual in population)
        for individual in population
    ]

    # Select individual by the wheel method.
    i, probs_sum = 0, probabilities[0]
    r = np.random.uniform(low=0, high=1)
    while probs_sum < r:
        i += 1
        probs_sum += probabilities[i]
    return population[i]


# TODO: Try crossover between two points.
def crossover(parent1: Board, parent2: Board, nd: int = 8) -> list[Board]:
    # Recombination probability
    pc = np.random.uniform(low=0.85, high=0.95)

    if np.random.rand() < pc:
        # Select recombination point between 1 and nd - 2.
        # Assuring that there'll be chromossomes from both parents.
        xi = np.random.randint(low=1, high=nd - 2)

        # Recombine both parent's chromossomes to both childrens.
        child1_chromossomes = parent1.chromossomes[:xi] + parent2.chromossomes[xi:]
        child2_chromossomes = parent2.chromossomes[:xi] + parent1.chromossomes[xi:]
        child1 = Board(chromossomes=child1_chromossomes, fitness=fitness_function(chromossomes=(child1_chromossomes)))
        child2 = Board(chromossomes=child2_chromossomes, fitness=fitness_function(chromossomes=(child2_chromossomes)))

        # Try to mutate the childrens.
        mutate(individual=child1)
        mutate(individual=child2)

        return [child1, child2]

    # No recombinations.
    return [parent1, parent2]


# TODO: Try unordered mutation and gaussian mutation.
def mutate(individual: Board, nd: int = 8, pm: float = 0.1) -> None:
    for j in range(nd):
        if np.random.rand() < pm:
            individual.chromossomes[j] = np.random.randint(low=0, high=nd)


def main() -> None:
    # TODO: Test with random values, get optimal solution.
    population_size = 100
    max_generations = 10000
    population = generate_population(population_size=population_size)

    for generation in range(max_generations):
        # Selection.
        selected_parents = [wheel_selection(population=population) for _ in range(population_size)]

        # Crossover and mutation.
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            children = crossover(parent1=parent1, parent2=parent2)
            new_population.extend(children)
        
        # Substitute new population.
        population = new_population

        # Fitness of the new population.
        for individual in population:
            individual.fitness = fitness_function(chromossomes=individual.chromossomes)
        
        # Mean fitness in current population.
        avg_fitness = np.mean(a=[individual.fitness for individual in population])
        print(f'Generation {generation + 1}: Mean Fitness = {avg_fitness}')
    
    # Best individual.
    best_individual = max(population, key=lambda individual: individual.fitness)
    print(f'Best individual: Chromossomes = {best_individual.chromossomes}, Fitness = {best_individual.fitness}')


if __name__ == '__main__':
    main()
