import numpy as np

# Problems (Objective functions & optimization methods).
from utils.SummarizeProblems import problems

# Models.
from HC.HillClimbing import HillClimbing


def main() -> None:
    # The amount of data to be generated for each problem.
    data_amount = 1000
    for problem_label, problem in problems.items():
        print(f'Running "{problem_label}"')
        # fixme: third problem, fourth problem, sixth problem, eiggth problem
        # thus HC is not guaranteed to find the optimal solution.
        HillClimbing.search(
            f=problem.objective_function,
            f_x1_bounds=problem.x1_bounds,
            f_x2_bounds=problem.x2_bounds,
            f_data_amount=data_amount,
            optimization_method=problem.optimization_method,
            dropout_max_iterations=10,
            max_iterations=1000,
            max_candidates=1000,
            epsilon=0.2,
        )
    pass


if __name__ == "__main__":
    main()
