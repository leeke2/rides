import random
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
from rides_env.solution import LSSDPSolution


def random_neighbours(solution: LSSDPSolution, k: int):
    inst_nstops = solution._inst.nstops
    inst_nbuses = solution._inst.nbuses

    actions = (
        list(range(inst_nstops))
        + ([-1] if solution._lss.nbuses < inst_nbuses - 1 else [])  # Add bus
        + ([-2] if solution._lss.nbuses > 1 else [])  # Remove bus
    )

    for _ in range(k):
        if len(actions) == 0:
            break

        action = actions.pop(random.randrange(len(actions)))
        candidate = deepcopy(solution)

        if action == -1:
            candidate.add_bus()
        elif action == -2:
            candidate.remove_bus()
        else:
            candidate.toggle(action)

        yield (action, candidate)


def tabu_search(
    initial_solution: LSSDPSolution,
    tabu_tenure: int = 10,
    max_epochs: int = 100,
    patience: int = 50,
    nneighbours: int = 10,
):
    is_better = lambda before, after: before > after
    is_better_eq = lambda before, after: before >= after

    stagnant_steps = 0
    tabu_list: deque = deque([], tabu_tenure)
    best_obj = initial_solution._obj
    logs = [(0, best_obj)]

    best_solution: LSSDPSolution = initial_solution
    best_candidate: LSSDPSolution = best_solution

    t = time.time()

    for epoch in range(max_epochs):
        current: LSSDPSolution = best_candidate
        best_candidate = None

        for _, candidate in random_neighbours(current, nneighbours):
            if best_candidate is None or (
                candidate._lss not in tabu_list
                and is_better(best_candidate._obj, candidate._obj)
            ):
                best_candidate = candidate

        tabu_list.append(best_candidate._lss)

        if is_better_eq(best_solution._obj, best_candidate._obj):
            stagnant_steps = -1
            best_solution = best_candidate

            if best_solution._obj != best_obj:
                logs.append((time.time() - t, best_solution._obj))
                best_obj = best_solution._obj

        stagnant_steps += 1

        if stagnant_steps > patience:
            break

    return best_solution, logs
