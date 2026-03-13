import copy
import time as timer
import heapq
import random
from copy import deepcopy
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost, get_k_deferred_location

DEBUG = False

def detect_first_collision_for_path_pair(path1, path2, k):
    ##############################
    # Return the first collision that occurs between two robot paths (or None if there is no collision)
    # There are two types of collisions: vertex collision and edge collision.
    # A vertex collision occurs if both robots occupy the same location at the same timestep
    # An edge collision occurs if the robots swap their location at the same timestep.
    # You should use "get_location(path, t)" to get the location of a robot at time t.
    max_len = max(len(path1), len(path2))
    for t in range(max_len):
        loc1 = get_location(path1, t)
        loc2 = get_location(path2, t)

        ## VERTEX COLLISION DETECTION from 1s perspective for 2's future
        for delta in range(k + 1):
            if loc1 == get_location(path2, t + delta):
                return {'loc': [loc1], 'a1_t': t, 'a2_t': t + delta}

        ## VERTEXT COLLISION DETECTION from 2s perspective for 1's future k steps
        for delta in range(1, k + 1):
            if loc2 == get_location(path1, t + delta):
                return {'loc': [loc2], 'a1_t': t + delta, 'a2_t': t}

        ## EDGE COLLISION DETECTION (only needed for k=0)
        if k == 0 and t > 0:
            prev1 = get_location(path1, t - 1)
            prev2 = get_location(path2, t - 1)
            if loc1 == prev2 and loc2 == prev1:
                return {'loc': [prev1, loc1], 'a1_t': t, 'a2_t': t}

    return None


def detect_collisions_among_all_paths(paths, k):
    ##############################
    # Return a list of first collisions between all robot pairs.
    # A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    # causing the collision, and the timestep at which the collision occurred.
    # You should use your detect_collision function to find a collision between two robots.
    retval = []

    for a1_idx in range(len(paths)):
        for a2_idx in range(a1_idx + 1, len(paths)):
            coll = detect_first_collision_for_path_pair(paths[a1_idx], paths[a2_idx], k)
            if coll is not None:
                retval.append({"a1": a1_idx,
                               "a2": a2_idx,
                               "loc": coll['loc'],
                               "timestep": (coll['a1_t'], coll['a2_t'])})

    return retval


def standard_splitting(collision):
    ##############################
    # Return a list of (two) constraints to resolve the given collision
    # Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                  specified timestep, and the second constraint prevents the second agent to be at the
    #                  specified location at the specified timestep.
    # Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                specified timestep, and the second constraint prevents the second agent to traverse the
    #                specified edge at the specified timestep
    if len(collision["loc"]) == 1:
        a1_constraint = {
            "agent": collision["a1"],
            "loc": collision["loc"],
            "timestep": collision["timestep"][0]
        }

        a2_constraint = {
            "agent": collision["a2"],
            "loc": collision["loc"],
            "timestep": collision["timestep"][1]
        }
    elif len(collision["loc"]) == 2:
        a1_constraint = {
            "agent": collision["a1"],
            "loc": collision["loc"],
            "timestep": collision["timestep"][0]
        }

        a2_constraint = {
            "agent": collision["a2"],
            "loc": [collision["loc"][1], collision["loc"][0]],
            "timestep": collision["timestep"][1]
        }
    else:
        raise ValueError("Impossible State in Standard Splitting")

    return a1_constraint, a2_constraint


class KRCBSSolver(object):
    """The high-level search of K-Robust CBS."""

    def __init__(self, my_map, starts, goals, k=0):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        k           - the parameter for K-Robust CBS
        """

        self.start_time = None
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.a_star_calls = 0

        self.open_list = []

        # Compute heuristics for the low-level search.
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

        # The parameter for K-Robust CBS.
        self.k = k

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        """
        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}

        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'], self.k)
        self.push_node(root)

        while len(self.open_list) > 0:
            curr_node = self.pop_node()

            if len(curr_node["collisions"]) == 0:
                self.print_results(curr_node)
                return curr_node["paths"]

            if DEBUG:
                print(self.a_star_calls)

            if self.a_star_calls > 5000:
                raise BaseException("NO VALID PATH NOT FOUND")

            collision = curr_node["collisions"][0]
            constraints = standard_splitting(collision)

            self.a_star_calls += 1
            for constraint in constraints:
                neighbor = deepcopy(curr_node)
                neighbor["constraints"].append(constraint)
                agent_idx = constraint["agent"]
                path = a_star(self.my_map, self.starts[agent_idx], self.goals[agent_idx], self.heuristics[agent_idx],
                              agent_idx, neighbor["constraints"])
                if path is not None:
                    neighbor["paths"][agent_idx] = path
                    neighbor["collisions"] = detect_collisions_among_all_paths(neighbor["paths"], self.k)
                    neighbor["cost"] = get_sum_of_cost(neighbor["paths"])
                    if DEBUG:
                        print(f"Neighbor with path {path}, constraint {constraint}, and {neighbor['collisions']} collisions.")
                    self.push_node(neighbor)

        ##############################
        # High-Level Search
        # Repeat the following as long as the open list is not empty:
        #   1. Get the next node from the open list (you can use self.pop_node()
        #   2. If this node has no collision, return solution
        #   3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #      standard_splitting function). Add a new child node to your open list for each constraint
        # Ensure to create a copy of any objects that your child nodes might inherit

        # These are just to print debug output - can be modified once you implement the high-level search
        # self.print_results(root)
        print("!!!! PATH NOT FOUND !!!!")

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
