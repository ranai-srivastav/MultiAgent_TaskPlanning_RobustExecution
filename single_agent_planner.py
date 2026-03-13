import heapq
from typing import List, Tuple

directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]  # Move Left, Down, Right, Up, Wait

def move(loc, dir):
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def is_valid_motion(old_loc, new_loc):
    ##############################
    # Check if a move from old_loc to new_loc is valid
    # Check if two agents are in the same location (vertex collision)
    new_set = {loc: idx for idx, loc in enumerate(new_loc)}
    if len(new_set) != len(new_loc):
        return False

    # Check edge collision
    for old_idx, old in enumerate(old_loc):
        if (old in new_set):
            new_idx = new_set[old]
            prev_loc_a = old
            curr_loc_a = new_loc[old_idx]
            prev_loc_b = old_loc[new_idx]
            curr_loc_b = new_loc[new_idx]

            if (old_idx != new_idx and
                    prev_loc_a == curr_loc_b and
                    prev_loc_b == curr_loc_a):
                return False

    return True


def get_sum_of_cost(paths):
    rst = 0
    if paths is None:
        return -1
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
                    or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # Build the heuristics table.
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    retval = dict()

    for constraint in constraints:
        if constraint["agent"] == agent:
            if constraint["timestep"] in retval:
                retval[constraint["timestep"]].append(constraint)
            else:
                retval[constraint["timestep"]] = [constraint]

    return retval


def get_k_deferred_location(path, time, k):
    """
    Given a time as an integer index and k as a delta from time t to (time - k), inclusive, give all relevant paths

    >>> get_k_deferred_location([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 2, 0)
    [(0, 2)]

    >>> get_k_deferred_location([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 2, 1)
    [(0, 1), (0, 2)]

    >>> get_k_deferred_location([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 2, 2)
    [(0, 0), (0, 1), (0, 2)]

    >>> get_k_deferred_location([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 2, 3)
    [(0, 0), (0, 0), (0, 1), (0, 2)]

    >>> get_k_deferred_location([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 8, 3)
    [(0, 4), (0, 4), (0, 4), (0, 4)]

    >>> get_k_deferred_location([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 8, 0)
    [(0, 4)]
    """
    retval = [get_location(path, time)]
    while k > 0:
        k -= 1
        time -= 1
        retval.append(get_location(path, time))
    retval.reverse()
    return retval


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]

def get_location_2(path: List[Tuple[int, int]], time: int, k: int = 0) -> List[Tuple[int, int]]:
    """
    Given a time as an integer index and k as a delta from time t to (time - k), inclusive, give all relevant paths

    >>> get_location_2([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 2, 0)
    [(0, 2)]

    >>> get_location_2([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 2, 1)
    [(0, 1), (0, 2)]

    >>> get_location_2([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 2, 2)
    [(0, 0), (0, 1), (0, 2)]

    >>> get_location_2([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 2, 3)
    [(0, 0), (0, 0), (0, 1), (0, 2)]

    >>> get_location_2([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 8, 3)
    [(0, 4), (0, 4), (0, 4), (0, 4)]

    >>> get_location_2([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 8, 0)
    [(0, 4)]
    """
    if k > len(path):
        raise ValueError("k cannot be greater than path")

    if time >= len(path):
        path_idx = max(0, time - k)
        num_repeat = max(0, time - len(path) + 1)
        # assert (len(path) - path_idx) + num_repeat == k
        return path[path_idx:] + [path[-1]] * num_repeat
    elif time - k < 0:
        num_path = abs(time - k)
        num_repeat = k - num_path
        assert num_path + num_repeat == k
        return [path[0]] * num_repeat + path[1:num_path+1]
    else:
        return path[time-k:time+1]




def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Check if a move from curr_loc to next_loc at time step next_time violates
    # any given constraint. For efficiency the constraints are indexed in a constraint_table
    # by time step, see build_constraint_table.
    if next_time in constraint_table:
        for constraint in constraint_table[next_time]:
            assert isinstance(constraint, dict)
            assert isinstance(constraint["loc"], list)
            assert isinstance(constraint["loc"][0], tuple)
            nogo = constraint["loc"]
            if len(nogo) == 1 and nogo[0] == next_loc:
                return True
            elif len(nogo) == 2 and curr_loc == nogo[0] and next_loc == nogo[1]:
                return True
    return False


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def in_map(map, loc):
    if loc[0] >= len(map) or loc[1] >= len(map[0]) or min(loc) < 0:
        return False
    else:
        return True


def all_in_map(map, locs):
    for loc in locs:
        if not in_map(map, loc):
            return False
    return True


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints, max_path_length=-1):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        h_values    - precomputed heuristic values for each location on the map
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    #############################
    # Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    prev_goal_timestep = 0
    h_value = h_values[start_loc]
    agent_table = build_constraint_table(constraints, agent)

    # If the start position is forbidden at t=0, no path exists
    if is_constrained(start_loc, start_loc, 0, agent_table):
        return None
    root = {'loc': start_loc,
            'g_val': 0,
            'h_val': h_value,
            'parent': None,
            "timestep": 0}

    push_node(open_list, root)
    closed_list[(root['loc'], root["timestep"])] = root
    while len(open_list) > 0:
        curr = pop_node(open_list)

        if max_path_length != -1 and curr["timestep"] > max_path_length:
            return None

        #############################
        # Task 2.2: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc:
            prev_goal_timestep = curr["timestep"]   #### <-----------------------------------
            can_stop = True                         #
            for k in agent_table.keys():            # To solve the case when a lower priority agent
                if (k > prev_goal_timestep):        #  blocks a higher priority agent because A*
                    can_stop = False                #  is myopic, I introduce a prev_goal_timestep
                    break                           #  and check if there is a contraint in the
                                                    # dict at a later timestep
            if can_stop:                            #### <--------------------------------------
                return get_path(curr)

        for dir in range(len(directions)):
            child_loc = move(curr['loc'], dir)
            if (not in_map(my_map, child_loc) or
                    my_map[child_loc[0]][child_loc[1]] or
                    is_constrained(curr["loc"], child_loc, curr["timestep"]+1, agent_table)): # (curr["loc"] == (5, 4) or child_loc==(5, 4)) and agent in [5, 6]
                continue

            child = {'loc': child_loc,
                     'g_val': curr['g_val'] + 1,
                     'h_val': h_values[child_loc],
                     'parent': curr,
                     "timestep": curr["timestep"] + 1}

            if (child['loc'], child["timestep"]) in closed_list:
                existing_node = closed_list[(child['loc'], child["timestep"])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child["timestep"])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child["timestep"])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions
