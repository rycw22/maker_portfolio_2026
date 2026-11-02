# Ryan Chang-Wu
# 12/10/24

# Getting necessary libraries

import random
import math
import time
import copy

import networkx as nx
import matplotlib.pyplot as plt

from heapq import heappush, heappop

import numpy as np
import tsplib95

from networkx.algorithms.approximation import traveling_salesman_problem

starting_pos = [(-600, -1200)]
rings = [(-1500, 1200), (-1500, -1200), (-1200, 1500), (-1200, 1200), 
        (-1200, -1200), (-1200, -1500), (-600, 1200), (-600, 600), (-600, -600), 
        (-600, -1200), (0, 1500), (0, 0), (0, 1500), (600, 1200), (600, 600), (600, -600), 
        (600, -1200), (1800, 1800), (1800, -1800), (1200, 1500), (1200, 1200), (1200, -1200), 
        (1200, -1500), (1500, 1200), (1500, -1200)]
doubled_rings = []
goals = [(-1200, 600), (-1200, -600), (1200, 0)]
goals_w_ring = [(1500, 600), (1500, -600)]
alliance_stakes = [(-1800, 0), (1800, 0)]
wall_stakes = [(0, 1800), (0, -1800)]
corners = [(-1800, 1800), (-1800, -1800), (1800, 1800), (1800, -1800)]
goal_and_ring = goals + rings
scoring = goals + wall_stakes + goals_w_ring

def construct_distance_matrix(*nodes):
    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            distance = np.linalg.norm(np.array(nodes[i])-np.array(nodes[j]))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

# gets fitness of the tour/path
def get_fitness(tour):
    distance_matrix = construct_distance_matrix(tour)
    distance = 0
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = None
        if i+1 < len(tour):
            to_city = tour[i+1]
        else:
            to_city = None
    # might be wrong ... check later
        distance += distance_matrix[tour[i-1]][tour[i]]
    fitness = 1/distance
    return fitness

def get_cost(tour):
    distance = 0
    for i in range(len(tour) - 1):
        distance += get_distance(tour[i], tour[i + 1])
    distance += get_distance(tour[-1], tour[0])  # Closing the loop
    return distance
    # might be wrong ... check later

# changes tour to meet requirements

def get_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def apply_constraints(tour):

    # Sets the starting point to a certain node
    start_coord = (-1200, -600)

    # Ensure the start node is the first node
    if tour[0] != start_coord:
        # Find where the desired start node is in the current tour
        start_index = tour.index(start_coord)
        # Move it to the front
        tour[0], tour[start_index] = tour[start_index], tour[0] 
        # tour.insert(0, tour.pop(start_index))

    '''if tour[0] != start_coord:
        start_index = tour.index(start_coord)
        tour[0], tour[start_index] = tour[start_index], tour[0]'''

    # Initialize counters
    ring_count = 0
    goal_count = 0
    n = len(tour)

    # Create distance matrix for fast lookups
    coords = np.array(tour)
    dist_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)

    # Helper to find nearest point from a given point
    def find_nearest(current_index, candidates):
        heap = []
        for candidate in candidates:
            candidate_index = point_to_index.get(candidate)
            if candidate_index is not None and candidate_index > current_index:
                heappush(heap, (dist_matrix[current_index, candidate_index], candidate_index))
        return heappop(heap)[1] if heap else None

    # Map points to their indices for faster lookups
    point_to_index = {point: i for i, point in enumerate(tour)}

    # Precompute point categories
    goals_set = set(goals)
    rings_set = set(rings)
    doubled_rings_set = set(doubled_rings)
    wall_stakes_set = set(wall_stakes)
    corners_set = set(corners)

    for i in range(1, n):
        current_point = tour[i]

        # Update counts
        if current_point in rings_set:
            ring_count += 1
        elif current_point in doubled_rings_set:
            ring_count += 1
        elif current_point in goals_set:
            goal_count += 1
        else:
            goal_count += 0

        # Rule: Must visit a goal when ring count is 0-2
        if 0 <= ring_count <= 2 and goal_count == 0:
            nearest_goal_index = find_nearest(i, goals_set)
            if nearest_goal_index:
                # Swap to ensure goal is visited
                tour[i+1], tour[nearest_goal_index] = tour[nearest_goal_index], tour[i+1]

        # Rule: Visit a corner and reset counts when ring count is 6
        elif ring_count == 6:
            nearest_corner_index = find_nearest(i, corners_set)
            if nearest_corner_index:
                tour[i+1], tour[nearest_corner_index] = tour[nearest_corner_index], tour[i+1]
            ring_count = 0
            goal_count = 0

        # Rule: If carrying rings and a goal, visit a wall stake to drop a ring
        elif ring_count > 0 and goal_count > 0:
            nearest_stake_index = find_nearest(i, wall_stakes_set)
            if nearest_stake_index:
                tour[i+1], tour[nearest_stake_index] = tour[nearest_stake_index], tour[i+1]
                ring_count -= 1
    return tour



    # After that start a counter - when the counter has reached 6 - go to corner - go to another goal, or go to a ring then to a goal

    # If it has a ring, then it can also go to a wall stake or alliance stake


# changes tour
def inverse_changer(tour):
    excerpt = tour[1:]
    node_one = random.choice(excerpt)
    new_list = list(filter(lambda city: city != node_one, excerpt))
    if not new_list:
        return tour
    node_two = random.choice(new_list)
    tour[min(tour.index(node_one),tour.index(node_two)):max(tour.index(node_one),tour.index(node_two))] = tour[min(tour.index(node_one),tour.index(node_two)):max(tour.index(node_one),tour.index(node_two))][::-1]
    return tour

# also changes tour
def swap_changer(tour):
    excerpt = tour[1:]
    pos_one = random.choice(range(len(excerpt)))
    pos_two = random.choice(range(len(excerpt)))
    tour[pos_one], tour[pos_two] = tour[pos_two], tour[pos_one]
    return tour

def insert_changer(tour):
    excerpt = tour[1:]
    node_j = random.choice(excerpt)
    tour.remove(node_j)
    excerpt = tour[1:]
    node_i = random.choice(excerpt)
    index = tour.index(node_i)
    tour.insert(index, node_j)
    return tour
    
def swap_routes_changer(tour):
    excerpt = tour[1:]
    subroute_a = random.choice(range(len(excerpt)))
    subroute_b = random.choice(range(len(excerpt)))
    subroute = tour[min(subroute_a,subroute_b):max(subroute_a,subroute_b)]
    del tour[min(subroute_a,subroute_b):max(subroute_a,subroute_b)]
    insert_pos = random.choice(range(len(tour)))
    for i in subroute:
        tour.insert(insert_pos, i)
    return tour

def get_init_tour(*nodes):
    shuffle = list(nodes)
    random.shuffle(shuffle)
    return shuffle

def get_neighbors(tour):
    neighbor = copy.deepcopy(tour)

    rand = random.choice([0,1,2,3])
    if rand == 0:
        inverse_changer(neighbor)
    elif rand == 1:
        insert_changer(neighbor)
    elif rand == 2:
        swap_changer(neighbor)
    else:
        swap_routes_changer(neighbor)
    
    return neighbor

def simulated_annealing_heuristic(nodes):
    init_tour = nodes
    #random.shuffle(init_tour)


    # tune these variables later
    init_temp = 5000
    alpha = 0.99

    current_temp = init_temp

    solution = init_tour
    same_solution = 0
    same_cost_diff = 0

    # tune these variables later
    while same_solution < 15000 and same_cost_diff < 60000:
        
        '''if same_solution % 50 == 0:
            apply_constraints(solution)'''

        neighbor = get_neighbors(solution)

        cost_diff = get_cost(solution) - get_cost(neighbor)
        if cost_diff > 0:
            solution = neighbor
            same_solution -= 100
            same_cost_diff -= 100

        elif cost_diff == 0:
            solution = neighbor
            same_solution += 0
            same_cost_diff += 10
        
        else:
            if random.uniform(0, 1) <= math.exp(float(cost_diff) / float(current_temp)):
                solution = neighbor
                same_solution =  0
                same_cost_diff = 0
            else:
                same_solution += 10
                same_cost_diff += 10
       
        current_temp = current_temp*alpha
        #print(1/get_cost(solution), same_solution)
    #print(1/get_cost(solution))
    return solution, get_cost(solution)
    #1/get_cost(solution)

  

if __name__ == '__main__':
    # test for nearest neighbor construction heuristic
    #G = generate_full(5)

    #approx_tour = traveling_salesman_problem(G, cycle=True)
    #approx_tour_cost = calculate_tour_cost(G, approx_tour)

    #print(approx_tour)
    #print(approx_tour_cost)

    #nearest_neighbor_construction_heuristic(G, start_node = 0)

    # test for distance matrix code that might not be needed

    #nodes = [(0, 0), (3, 4), (6, 8), (3, 0)]  
    #distance_matrix = construct_distance_matrix(nodes)
    #print("Distance Matrix:")
    #print(distance_matrix)
    all_coords = [(-1200, -600), (-600, -1200), (-1500, 1200), (-1500, -1200), (-1200, 1500), 
                  (-1200, 1200), (-1200, -1200), (-1200, -1500),(-600, 1200), (-600, 600), 
                  (-600, -600), (0, 1500), (0, 0), (0, 1500),(600, 1200), (600, 600), (600, -600),
                    (600, -1200), (1800, 1800), (1800, -1800),  (1200, 1500), (1200, 1200),
                    (1200, -1200), (1200, -1500), (1500, 1200), (1500, -1200),(-1200, 600),
                    (1200, 0), (1500, 600), (1500, -600), (-1800, 0), (1800, 0), (0, 1800), 
                    (0, -1800), (-1800, 1800), (-1800, -1800), (1800, 1800), (1800, -1800)]
    some_coords =   [(1, 2), (5, -3), (0, 0), (-2, 4), (3, 0), (0, -5), (-4, -1)]
    best_route_distance = []
    best_route = []
    #convergence_time = []
    for i in range(1):  
        print(simulated_annealing_heuristic(all_coords))  
        #print(simulated_annealing_heuristic(all_coords))
        route, route_distance = simulated_annealing_heuristic(all_coords)
        G = nx.Graph()

        # Add edges between consecutive coordinates
        for i in range(len(route) - 1):
            G.add_edge(route[i], route[i + 1])

        # Plot the graph
        pos = {node: node for node in G.nodes()}  # Use coordinates as positions
        nx.draw(G, pos, with_labels=True, node_size=300, font_size=10, node_color="skyblue")
        plt.xlabel('X Coordinates')
        plt.ylabel('Y Coordinates')
        plt.show()
