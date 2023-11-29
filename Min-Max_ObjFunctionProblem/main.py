import numpy as np

# Problems (Objective functions).
from utils.GenerateData import problem_one

# Models.
from HC.HillClimbing import HillClimbing


def main() -> None:
    epsilon_value = 0.1
    hc = HillClimbing.fit(problem_one, epsilon=epsilon_value)
    pass


if __name__ == '__main__':
    main()

