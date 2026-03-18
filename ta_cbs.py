# Generic imports.
import copy
import time as timer
import numpy as np
import heapq
import random
# Project imports.
from hungarian import hungarian_algorithm, dict2list
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_path_lengths
from kr_cbs import detect_first_collision_for_path_pair, detect_collisions_among_all_paths, KRCBSSolver, \
    standard_splitting
from copy import deepcopy

DEBUG = True


class TACBSSolver(KRCBSSolver):
    """The high-level search of TA-CBS."""

    def __init__(self, my_map, starts, goals, k=0):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        k           - the parameter for K-Robust CBS
        """
        super().__init__(my_map, starts, goals, k)
        self.start_time = None
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

        # The parameter for K-Robust CBS.
        self.k = k

    def find_solution(self):
        """
        Finds shortest paths and an optimal target assignment for all agents.
        """
        self.start_time = timer.time()
        # Generate the root node
        # constraints - list of constraints.
        # paths       - list of paths, one for each agent
        #             [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions  - list of collisions in paths.
        # Mc          - Mc[i][j] is the cost of the shortest path (under constraints) for agent i to target j.
        root = {'cost': 0,
                'constraints': [],  # Like in CBS, a list of dictionaries, each dictionary is a constraint.
                'collisions': [],  # Like in CBS.
                'paths': [],  # The paths, one for each agent, that are planned for the optimal assignment under Mc.
                'Mc': {i: [float('inf') for g in range(len(self.goals))] for i in range(self.num_of_agents)}
                # Dict[Int: List[Int]]
                # Mc[i][j] is the cost of the shortest path (under constraints) for agent i to target j.
                }
        ##############################
        # Find initial paths for each agent to all targets.
        # Populate root['paths'] and root['Mc'] with the paths and costs.

        for goal_idx in range(self.num_of_agents):
            for agent_idx, agent_start in enumerate(self.starts):
                root["Mc"][agent_idx][goal_idx] = self.heuristics[goal_idx][agent_start]

        opt_task_alloc = hungarian_algorithm(dict2list(root["Mc"]))
        curr_goal_alloc = [self.goals[idx] for idx in opt_task_alloc]

        for agent_idx in range(self.num_of_agents):
            curr_goal_loc = curr_goal_alloc[agent_idx]
            root["paths"].append(a_star(self.my_map, self.starts[agent_idx], curr_goal_loc,
                                        self.heuristics[opt_task_alloc[agent_idx]], agent_idx, root["constraints"]))

        root['cost'] = get_sum_of_path_lengths(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'], self.k)
        self.push_node(root)

        ##############################
        # High-Level Search
        #  Repeat the following as long as the open list is not empty:
        while len(self.open_list) > 0:
            # 1. Get the next node from the open list (you can use self.pop_node()
            curr_node = self.pop_node()

            if self.a_star_calls > 50000:
                return None

            #    2. If this node has no collisions, return the solution stored in its 'paths' field.
            if len(curr_node["collisions"]) == 0:
                self.print_results(curr_node)
                return curr_node["paths"]

            #    3. Otherwise, choose the first collision and convert to a list of constraints (using your
            #       standard_splitting function).
            curr_collision = curr_node["collisions"][0]
            constraints = standard_splitting(curr_collision)

            self.a_star_calls += 1
            #       For each constraint created:
            for constraint in constraints:
                # 3a. Create a new child CT node.
                neighbor = deepcopy(curr_node)
                neighbor["constraints"].append(constraint)
                agent_idx = constraint["agent"]
                # 3b. Replan the affected agent paths to all goals and update Mc with the costs.
                for orig_goal_idx, orig_goal_loc in enumerate(self.goals):
                    path = a_star(self.my_map, self.starts[agent_idx], orig_goal_loc, self.heuristics[orig_goal_idx],
                                  agent_idx, neighbor["constraints"])
                    if path is not None:
                        neighbor["Mc"][agent_idx][orig_goal_idx] = len(path) - 1

                    # 3c. Find the new optimal assignment and paths.
                opt_task_alloc = hungarian_algorithm(dict2list(neighbor["Mc"]))
                curr_goal_alloc = [self.goals[idx] for idx in opt_task_alloc]

                paths_possible = True
                for agent_idx in range(self.num_of_agents):
                    path_i = a_star(self.my_map, self.starts[agent_idx], curr_goal_alloc[agent_idx],
                                    self.heuristics[opt_task_alloc[agent_idx]], agent_idx, neighbor["constraints"])
                    if path_i is not None:
                        neighbor["paths"][agent_idx] = path_i
                    else:
                        neighbor["paths"][agent_idx] = float("inf")
                        paths_possible = False
                        break

                if paths_possible:
                    neighbor["collisions"] = detect_collisions_among_all_paths(neighbor["paths"], self.k)
                    neighbor["cost"] = get_sum_of_path_lengths(neighbor["paths"])

                    # 3d. Add the new child CT node to the open list.
                    self.push_node(neighbor)

        print("NO SOLUTION FOUND")
        return None
