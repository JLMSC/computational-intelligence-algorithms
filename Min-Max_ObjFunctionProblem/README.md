# Stochastic Optimization Algorithms

This work presents computational implementations of four stochastic optimization algorithms for continuous search spaces: Global Random Search, Local Random Search, Hill Climbing, and Simulated Annealing. Each method explores the search space using different strategies to approximate optimal solutions for a given objective function.

## Introduction

This project implements optimization techniques with the following objectives:
- Optimize arbitrary 2D functions (maximization or minimization)
- Explore different stochastic search strategies
- Visualize optimization processes in 3D space
- Analyze convergence behavior of each algorithm
- Compare global vs local search approaches

## Problem Representation

The optimization problem is defined as:

$$ f(x_1, x_2) $$

Where:
- $x_1, x_2 \in \mathbb{R}$: decision variables
- $f$: objective function to be minimized or maximized

The search space is constrained by:

$$ x_1 \in [a_1, b_1] $$
$$ x_2 \in [a_2, b_2] $$

Each algorithm attempts to find:

$$ (x_{1}^{}, x_{2}^{}) \text{ such that } f(x_{1}^{}, x_{2}^{}) \text{ is optimal} $$

More problems can be found under `Utils/Problems.py`.

## Visualization

All methods include 3D visualization of the objective function:
- Surface plot of $f(x_1, x_2)$
- Iterative plotting of candidate solutions
- Highlighting of improved solutions over time

## Algorithms

### A. Global Random Search
Explores the search space by sampling candidates uniformly across the entire domain.

**Key characteristics:**
- No locality assumption
- High exploration capability
- Slow convergence

**Steps:**
- Randomly sample candidate solutions within bounds
- Evaluate objective function
- Update best solution if improvement is found
- Terminate after max iterations or stagnation

### B. Local Random Search
Performs exploration around the current best solution using Gaussian perturbations.

**Key characteristics:**
- Local exploration
- Faster convergence than global search
- Sensitive to initial conditions

**Steps:**
- Generate candidates using normal distribution centered at current best
- Accept candidate if it improves the objective
- Ensure candidates remain within bounds

### C. Hill Climbing
A greedy local search method that iteratively improves the solution by exploring nearby candidates.

**Key characteristics:**
- Exploits local gradients
- Fast convergence
- Prone to local optima

**Steps:**
- Start from an initial solution
- Generate multiple local candidates
- Move to the first improving candidate
- Repeat until no improvement or dropout condition

### D. Simulated Annealing
A probabilistic method that allows temporary acceptance of worse solutions to escape local optima.

**Key characteristics:**
- Balances exploration and exploitation
- Uses temperature parameter to control randomness
- Capable of escaping local minima

**Acceptance criterion:**
Accept if:
- Improvement, or  
- With probability $P = \exp\left(-\frac{\Delta f}{T}\right)$

Where:
- $\Delta f$: change in objective value
- $T$: temperature

**Steps:**
- Generate local candidate
- Evaluate objective
- Accept based on probability criterion
- Gradually decrease temperature

## Algorithm Workflow

All methods follow a general structure:
1. Initialize candidate solution
2. Evaluate objective function
3. Iteratively:
    - Generate new candidates
    - Evaluate objective
    - Update best solution based on strategy
4. Terminate based on:
    - Maximum iterations
    - Stagnation (dropout condition)

## Results

The system demonstrates distinct behaviors across optimization strategies:

- **Global Random Search:**
    - Wide exploration of search space
    - Slow improvement rate
- **Local Random Search:**
    - Efficient local refinement
    - Limited global awareness
- **Hill Climbing:**
    - Rapid convergence
    - Sensitive to local optima
- **Simulated Annealing:**
    - Robust against local minima
    - Gradual convergence controlled by temperature

Console output includes:
- Iteration progress
- Dropout activation (stagnation detection)

Visualization shows:
- Evolution of candidate solutions
- Convergence toward optimal regions

## Limitations

- Limited to 2D functions ($x_1, x_2$)
- No adaptive parameter tuning
- Fixed dropout strategy for stagnation detection
- Performance dependent on hyperparameters:
    - Step size ($\epsilon, \sigma$)
    - Temperature schedule
    - Iteration limits
- No comparison metrics between algorithms
- Visualization overhead may impact performance
