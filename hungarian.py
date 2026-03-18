import random
import numpy as np
from numpy import typing as npt
from scipy.optimize import linear_sum_assignment
from typing import List, Dict


def min_num_lines(M: npt.NDArray):
    """
    Given a matrix, returns the minimum number of lines required to cover all 0s.

    >>> import numpy as np
    >>> min_num_lines(np.array([[0, 1], [1, 0]]))
    2

    >>> min_num_lines(np.array([[0, 0], [0, 0]]))
    2

    >>> min_num_lines(np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]]))
    2

    >>> min_num_lines(np.array([[1, 1], [1, 1]]))
    0

    >>> min_num_lines(np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]]))
    1

    >>> min_num_lines(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    3

    >>> min_num_lines(np.array([[1]]))
    0

    >>> min_num_lines(np.array([[0]]))
    1
    """
    row_ids, col_ids = np.where(M == 0)  #
    row_lines, col_lines = [], []
    mask = np.ones_like(M).astype(bool)
    crosses = 0
    while True:
        if len(row_ids) == 0 or len(col_ids) == 0:
            return crosses, mask, row_lines, col_lines

        row_idx_val, row_idx_freq = np.unique(row_ids, return_counts=True)
        col_idx_val, col_idx_freq = np.unique(col_ids, return_counts=True)
        most_freq_row_elem_count,  most_freq_row_elem_idx = np.max(row_idx_freq), np.argmax(row_idx_freq)
        most_freq_row_elem  = row_idx_val[most_freq_row_elem_idx]

        most_freq_col_elem_count, most_freq_col_elem_idx = np.max(col_idx_freq), np.argmax(col_idx_freq)
        most_freq_col_elem = col_idx_val[most_freq_col_elem_idx]

        if (most_freq_row_elem_count > most_freq_col_elem_count):
            v, f = most_freq_row_elem, most_freq_row_elem_count
            crosses += 1
            idx = 0

            while idx < len(row_ids):
                if row_ids[idx] == v:
                    mask[row_ids[idx], :] = 0
                    row_lines.append(row_ids[idx])
                    row_ids = np.delete(row_ids, idx)
                    col_ids = np.delete(col_ids, idx)
                    f -= 1
                    if f == 0:
                        break
                else:
                    idx += 1

            if len(row_ids) == 0 or len(col_ids) == 0:
                return crosses, mask, row_lines, col_lines
        else:
            v, f = most_freq_col_elem, most_freq_col_elem_count
            crosses += 1
            idx = 0
            while idx < len(col_ids):
                if col_ids[idx] == v:
                    col_lines.append(col_ids[idx])
                    mask[:, col_ids[idx]] = 0
                    col_ids = np.delete(col_ids, idx)
                    row_ids = np.delete(row_ids, idx)
                    f -= 1
                    if f == 0:
                        break
                else:
                    idx += 1


def broken_hungarian_algorithm(agent_goal_costs: dict):
    """
    :param agent_goal_costs: A dictionary mapping agent ids to a list of costs for each goal.
                            The order of goals is the same for all agents.
    :return: A dictionary mapping agent id to goal id.

    >>> hungarian_algorithm({0: [9, 2, 7], 1: [3, 6, 3], 2: [5, 8, 1]})
    {0: 1, 1: 0, 2: 2}  # cost: 2 + 3 + 1 = 6

    >>> hungarian_algorithm({0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]})
    {0: 0, 1: 1, 2: 2}  # cost: 1 + 5 + 9 = 15

    >>> hungarian_algorithm({0: [4, 1, 3], 1: [2, 0, 5], 2: [3, 2, 2]})
    {0: 1, 1: 0, 2: 2}  # cost: 1 + 2 + 2 = 5

    >>> hungarian_algorithm({0: [3, 5], 1: [10, 1]})
    {0: 0, 1: 1}  # cost: 3 + 1 = 4

    >>> hungarian_algorithm({0: [1, 3], 1: [3, 1]})
    {0: 0, 1: 1}  # cost: 1 + 1 = 2 (or {0:1, 1:0}
    """
    ##############################

    mapping = dict()

    n = len(agent_goal_costs)
    M = np.zeros((n, n))

    for row_idx, agent_ids in enumerate(agent_goal_costs):
        for col_idx, task_cost in enumerate(agent_goal_costs[agent_ids]):
            M[row_idx][col_idx] = task_cost

    min_val_in_cols = np.min(M, axis=0)  # (1, c) this gives the minimum over the rows, giving 1 val for each c
    M = (M - np.tile(min_val_in_cols[np.newaxis, :], (n, 1)))

    min_val_in_rows = np.min(M, axis=1)  # (r, 1) this gives the minimum over the rows, giving c
    M = (M - np.tile(min_val_in_rows[:, np.newaxis], (1, n)))

    while True:
        crosses, true_for_uncovered_mask, row_lines, col_lines = min_num_lines(M)

        if crosses == n:
            break

        intersections = np.zeros_like(M).astype(bool)

        min_val = np.min(M[true_for_uncovered_mask])
        M[true_for_uncovered_mask] = M[true_for_uncovered_mask] - min_val

        for row_idx in row_lines:
            for col_idx in col_lines:
                intersections[row_idx][col_idx] = 1

        M[intersections] = M[intersections] + min_val

    assigned_rows = set()
    assigned_cols = set()

    while len(mapping) < n:
        best_row, best_col = -1, -1
        best_count = float('inf')

        for row_idx in range(n):
            if row_idx in assigned_rows:
                continue
            zero_cols = [c for c in range(n) if M[row_idx][c] == 0 and c not in assigned_cols]
            if len(zero_cols) == 0:
                continue
            if len(zero_cols) < best_count:
                best_count = len(zero_cols)
                best_row = row_idx
                best_col = zero_cols[0]

        assigned_rows.add(best_row)
        assigned_cols.add(best_col)
        agent_ids = list(agent_goal_costs.keys())
        mapping[agent_ids[best_row]] = best_col

    return mapping


def hungarian_algorithm(agent_goal_costs: List[List[int]]) -> List[int]:
    """
    Solves the assignment problem using scipy's linear_sum_assignment :/
    """
    rows_ids, col_ids = linear_sum_assignment(agent_goal_costs)
    return list(col_ids)


def dict2list(some_dict: Dict[int, List[int]]) -> List[List[int]]:
    """
    Solves the assignment problem using scipy's linear_sum_assignment :/
    """
    return [some_dict[k] for k in some_dict]
