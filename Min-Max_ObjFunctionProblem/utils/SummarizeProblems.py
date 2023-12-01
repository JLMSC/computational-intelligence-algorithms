import numpy as np
from typing import NamedTuple, Callable, Any, Tuple
from utils.Problems import *


class Problem(NamedTuple):
    """Represents a problem."""

    objective_function: Callable[..., Any]
    optimization_method: Callable[..., Any]
    x1_bounds: Tuple[float, float]
    x2_bounds: Tuple[float, float]


problems = {
    # Problem Label: Objective function, Optimization method.
    "first_problem": Problem(first_problem, np.min, (-100, 100), (-100, 100)),
    "second_problem": Problem(second_problem, np.max, (-2, 4), (-2, 5)),
    "third_problem": Problem(third_problem, np.min, (-8, 8), (-8, 8)),
    "fourth_problem": Problem(fourth_problem, np.min, (-5.12, 5.12), (-5.12, 5.12)),
    "fifth_problem": Problem(fifth_problem, np.min, (-2, 2), (-1, 3)),
    "sixth_problem": Problem(sixth_problem, np.max, (-1, 3), (-1, 3)),
    "seventh_problem": Problem(seventh_problem, np.min, (0, np.pi), (0, np.pi)),
    "eighth_problem": Problem(eighth_problem, np.min, (-200, 20), (-200, 20)),
}
