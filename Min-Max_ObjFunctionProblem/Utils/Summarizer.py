from Utils.Problems import *
from dataclasses import dataclass
from typing import Callable, Tuple, Any


@dataclass
class Problem:
    f: Callable[[Any, Any], Any]
    x1_bounds: Tuple[float, float]
    x2_bounds: Tuple[float, float]


ProblemContainer: dict[str, Problem] = {
    'Problem_1': Problem(f=problem_1, x1_bounds=(-100, 100), x2_bounds=(-100, 100)),
    'Problem_2': Problem(f=problem_2, x1_bounds=(-2, 4), x2_bounds=(-2, 5)),
    'Problem_3': Problem(f=problem_3, x1_bounds=(-8, 8), x2_bounds=(-8, 8)),
    'Problem_4': Problem(f=problem_4, x1_bounds=(-5.12, 5.12), x2_bounds=(-5.12, 5.12)),
    'Problem_5': Problem(f=problem_5, x1_bounds=(-2, 2), x2_bounds=(-1, 3)),
    'Problem_6': Problem(f=problem_6, x1_bounds=(-1, 3), x2_bounds=(-1, 3)),
    'Problem_7': Problem(f=problem_7, x1_bounds=(0, np.pi), x2_bounds=(0, np.pi)),
    'Problem_8': Problem(f=problem_8, x1_bounds=(-200, 20), x2_bounds=(-200, 20)),
}
