# Genetic Algorithms for Combinatorial Optimization

This work presents computational implementations of Genetic Algorithms (GA) applied to two classical combinatorial optimization problems: The Traveling Salesman Problem (TSP) and the Eight Queens Problem. The system models candidate solutions as chromosomes, evolves populations through selection, crossover, and mutation, and iteratively improves solution quality based on fitness evaluation.

## Introduction

This project implements evolutionary strategies using Python, with the following objectives:
- Represent combinatorial problems as chromosome-based populations
- Define problem-specific fitness functions
- Apply Genetic Algorithm operators (selection, crossover, mutation)
- Iteratively evolve populations toward optimal or near-optimal solutions
- Provide visualization for spatial problems (TSP)

## Problems Overview

### A. Traveling Salesman Problem (TSP)
Given a set of points in 3D space, the objective is to determine the shortest possible route that:
- Visits each point exactly once
- Returns to the starting point

### B. Eight Queens Problem
The objective is to place eight queens on a chessboard such that:
- No two queens attack each other
- Constraints include rows, columns, and diagonals

## Chromosome Representation

### A. TSP
- Each individual is a permutation of point indices
- The chromosome defines the visiting order of points

### B. Eight Queens
- Each chromosome is a list of integers
- Index represents the column, value represents the row of a queen

## Genetic Algorithm Components

### A. Population Initialization
- **TSP:** Random permutations of points are generated
- **Eight Queens:** Random distributions of queen positions are created

*Implemented in:* `generate_population(...)`, `generate_chromosomes(...)`

### B. Fitness Function
- **TSP:** Computes total Euclidean distance of the closed path
- **Eight Queens:** Computes number of non-attacking queen pairs

*Implemented in:* `fitness_function(...)`

### C. Selection
- **TSP (Tournament Selection):** Random subsets of individuals compete, best is selected
- **Eight Queens (Roulette Wheel Selection):** Probability proportional to fitness

*Implemented in:* `tournament_selection(...)`, `wheel_selection(...)`

### D. Crossover (Recombination)
- **TSP:** Ordered crossover preserving permutation validity. Subsequences inherited from one parent, remainder filled from the other.
- **Eight Queens:** Single-point crossover. Chromosomes split and recombined.

*Implemented in:* `crossover(...)`

### E. Mutation
- **TSP:** Swap mutation between two positions
- **Eight Queens:** Random reassignment of gene values

*Implemented in:* `mutate(...)`

## Algorithm Workflow

For both problems, the algorithm follows these steps:
1. Initialize population
2. Evaluate fitness of all individuals
3. Repeat for a fixed number of generations:
    - Select parents
    - Apply crossover
    - Apply mutation
    - Form new population
    - Recompute fitness
4. Extract best individual

## Traveling Salesman Implementation

**Additional features:**
- 3D point generation using clustered spatial partitions
- Visualization of:
    - Points in 3D space
    - Best path found by the algorithm

**Key steps:**
- Points are generated and slightly perturbed
- A random origin is selected
- Paths are constructed as closed loops
- Final solution is plotted

**Output:**
- Best path (sequence of indices)
- Total path cost (distance)
- 3D visualization of the route

## Eight Queens Implementation

**Key characteristics:**
- Fitness is maximized (ideal value = 28 non-attacking pairs)
- Population evolves toward valid board configurations

**Output:**
- Best chromosome (queen positions)
- Corresponding fitness score

## Results

The system successfully demonstrates the application of Genetic Algorithms in solving discrete optimization problems.
- **TSP:**
    - Produces near-optimal route over generations
    - Convergence observed through decreasing average fitness (distance)
- **Eight Queens:**
    - Evolves toward valid configurations with minimal or zero conflicts
    - Fitness increases toward the optimal value

Console output includes:
- Generation-wise average fitness
- Final best solution

## Limitations

- No guarantee of global optimality due to stochastic nature
- Performance sensitive to hyperparameters:
    - Population size
    - Mutation rate
    - Number of generations
- TSP implementation does not include advanced optimizations (e.g., elitism, adaptive mutation)
- Eight Queens mutation may introduce invalid states without repair mechanisms
- Computational cost increases with population size and problem complexity
