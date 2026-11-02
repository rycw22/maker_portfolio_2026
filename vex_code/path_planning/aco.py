import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
# import matplotlib.pyplot as plt

# set up the stuff the alrogithm is going to use
@dataclass(frozen=True)
class Node:
    id: int
    name: str
    kind: str 
    x: float
    y: float
    initial_blocks: int = 0
    edge_stack_bonus: bool = False  


@dataclass
class Env:
    nodes: List[Node]
    start_id: int
    pickup_ids: List[int]
    deposit_ids: List[int]
    capacity: int = 8

    def dist(self, a: int, b: int) -> float:
        na, nb = self.nodes[a], self.nodes[b]
        return math.hypot(na.x - nb.x, na.y - nb.y)



def simulate_route(env: Env, route: List[int]) -> Tuple[float, float]:
    # we want the path to hit many of the nodes while keeping track of the blocks at each node
    remaining = {nid: env.nodes[nid].initial_blocks for nid in env.pickup_ids}
    cleared_edge_bonus_given = {nid: False for nid in env.pickup_ids}

    carried = 0
    points = 0.0
    distance = 0.0

    def is_edge_bonus_pickup(nid: int) -> bool:
        return env.nodes[nid].kind == "pickup" and env.nodes[nid].edge_stack_bonus

    for i in range(len(route) - 1):
        cur, nxt = route[i], route[i + 1]
        distance += env.dist(cur, nxt)

        node = env.nodes[nxt]

        if node.kind == "pickup":
            if remaining[nxt] <= 0:
                continue
            space = env.capacity - carried
            if space <= 0:
                continue
            taken = min(space, remaining[nxt])
            remaining[nxt] -= taken
            carried += taken

            # we want to account for the clear bonus
            if is_edge_bonus_pickup(nxt) and remaining[nxt] == 0 and not cleared_edge_bonus_given[nxt]:
                points += 5
                cleared_edge_bonus_given[nxt] = True

        elif node.kind == "deposit":
            # Deposit everything carried: 1 point per block
            points += carried
            carried = 0

        # start node: nothing special

    return points, distance


# ----------------------------
# Ant Colony Optimization
# ----------------------------

class ACO:
    def __init__(
        self,
        env: Env,
        n_ants: int = 60,
        n_iters: int = 250,
        alpha: float = 1.2,   # pheromone 
        beta: float = 3.0,    # heuristic influence
        rho: float = 0.15,    # evaporation rate
        q: float = 1.0,       # pheromone deposit scaling
        lam: float = 0.35,    # distance penalty in fitness
        max_steps: int = 60,
        seed: int = 7
    ):
        self.env = env
        self.n_ants = n_ants
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.lam = lam
        self.max_steps = max_steps
        random.seed(seed)

        n = len(env.nodes)
        # pheromone on directed edges
        self.tau = [[1.0 for _ in range(n)] for _ in range(n)]

        # Precompute distances for speed
        self.d = [[env.dist(i, j) for j in range(n)] for i in range(n)]

    def fitness(self, points: float, dist: float) -> float:
        return points - self.lam * dist

    def build_ant_route(self) -> List[int]:
        env = self.env
        start = env.start_id

        # State for da ant
        remaining = {nid: env.nodes[nid].initial_blocks for nid in env.pickup_ids}
        carried = 0
        route = [start]
        cur = start

        def best_deposit_distance(from_id: int) -> float:
            return min(self.d[from_id][dep] for dep in env.deposit_ids)

        def heuristic(cur_id: int, nxt_id: int) -> float:
            # we want it to be higher
            node = env.nodes[nxt_id]
            dist = self.d[cur_id][nxt_id] + 1e-9

            if node.kind == "pickup":
                if remaining[nxt_id] <= 0:
                    return 1e-12
                space = env.capacity - carried
                if space <= 0:
                    return 1e-12
                taken = min(space, remaining[nxt_id])

                bonus = 0.0
                if node.edge_stack_bonus and remaining[nxt_id] - taken == 0:
                    bonus = 5.0

                # future cost estimate
                future = best_deposit_distance(nxt_id)
                # Effective value (this move and the next predicted move)
                return (taken + bonus) / (dist + 0.7 * future + 1e-9)

            if node.kind == "deposit":
                if carried <= 0:
                    return 1e-12
                return carried / dist

            return 1e-12

        def allowed_moves(cur_id: int) -> List[int]:
            moves = []

            # if full then you should go to depot
            if carried > 0:
                moves.extend(env.deposit_ids)

            # If you have space, pickups are possible
            if carried < env.capacity:
                for pid in env.pickup_ids:
                    if remaining[pid] > 0:
                        moves.append(pid)

            # return to start if nothing else is good
            moves.append(env.start_id)
            seen = set()
            uniq = []
            for m in moves:
                if m not in seen:
                    uniq.append(m)
                    seen.add(m)
            return uniq

        for step in range(self.max_steps):
            candidates = allowed_moves(cur)

            # if we can only go to start, end route
            if len(candidates) == 1 and candidates[0] == start:
                break

            # calculate the probabilities for each choice
            weights = []
            for nxt in candidates:
                tau = self.tau[cur][nxt] ** self.alpha
                eta = heuristic(cur, nxt) ** self.beta
                weights.append(tau * eta)

            # we should stop if the weights are close to 0
            s = sum(weights)
            if s < 1e-20:
                break

            r = random.random() * s
            acc = 0.0
            chosen = candidates[-1]
            for nxt, w in zip(candidates, weights):
                acc += w
                if acc >= r:
                    chosen = nxt
                    break

            # ant go to chosen state
            route.append(chosen)

            node = env.nodes[chosen]
            if node.kind == "pickup":
                space = env.capacity - carried
                if space > 0 and remaining[chosen] > 0:
                    taken = min(space, remaining[chosen])
                    remaining[chosen] -= taken
                    carried += taken

            elif node.kind == "deposit":
                carried = 0

            # if the ant goes home, end route
            cur = chosen
            if cur == start and step > 3:
                break

        # end at the start! (for that sweet park bonus)
        if route[-1] != start:
            route.append(start)

        return route

    def run(self) -> Tuple[List[int], float, float, float]:
        best_route = None
        best_points = -1e9
        best_dist = 1e9
        best_fit = -1e18

        for it in range(self.n_iters):
            ant_solutions = []

            # Build routes
            for _ in range(self.n_ants):
                route = self.build_ant_route()
                points, dist = simulate_route(self.env, route)
                fit = self.fitness(points, dist)
                ant_solutions.append((route, points, dist, fit))

            # Find best this iteration
            ant_solutions.sort(key=lambda x: x[3], reverse=True)
            if ant_solutions[0][3] > best_fit:
                best_route, best_points, best_dist, best_fit = ant_solutions[0]

            # Evaporate pheromone
            n = len(self.env.nodes)
            for i in range(n):
                for j in range(n):
                    self.tau[i][j] *= (1.0 - self.rho)
                    if self.tau[i][j] < 1e-6:
                        self.tau[i][j] = 1e-6

            # Deposit pheromone from top-k ants (stabilizes learning)
            top_k = max(3, self.n_ants // 8)
            for route, points, dist, fit in ant_solutions[:top_k]:
                # stronger pheromone for better fitness, but keep it positive
                deposit = self.q * max(0.0, fit)
                for a, b in zip(route[:-1], route[1:]):
                    self.tau[a][b] += deposit

        return best_route, best_points, best_dist, best_fit


# ----------------------------
# Example environment
# ----------------------------


def make_example_env() -> Env:
    """
    Field coordinate system:
      - 144 x 144 inches
      - (0,0) bottom-left
      - Uses VEXcode VR GPS convention for element locations (mm, origin at field center),
        converted to inches with:
            inches_per_mm = 24/600 = 0.04
            x_in = 72 + 0.04*x_mm
            y_in = 72 + 0.04*y_mm

    Start node:
      - Center of the RED parking zone on the left side of the field: (12, 72) inches.
    """
    nodes: List[Node] = []
    nid = 0

    INCH_PER_MM = 24.0 / 600.0  # 0.04 in/mm

    def mm_to_in(x_mm: float, y_mm: float) -> tuple[float, float]:
        return (72.0 + INCH_PER_MM * x_mm, 72.0 + INCH_PER_MM * y_mm)

    # -----------------------
    # START = Red Parking Zone (center)
    # -----------------------
    nodes.append(Node(nid, "RED_PARKING_ZONE", "start", x=12.0, y=72.0))
    start_id = nid
    nid += 1

    # -----------------------
    # DEPOSITS (Goals) — official GPS points (mm)
    # -----------------------
    deposit_ids: List[int] = []

    goal_points_mm = [
        # Long goal endpoints
        ("TOP_GOAL_LEFT",     -620,  1200),
        ("TOP_GOAL_RIGHT",     620,  1200),
        ("BOTTOM_GOAL_LEFT",  -620, -1200),
        ("BOTTOM_GOAL_RIGHT",  620, -1200),

        # Center goal points
        ("CENTER_UL",         -205,   200),
        ("CENTER_UR",          205,   200),
        ("CENTER_LL",         -205,  -200),
        ("CENTER_LR",          205,  -200),
    ]

    for name, x_mm, y_mm in goal_points_mm:
        x, y = mm_to_in(x_mm, y_mm)
        nodes.append(Node(nid, name, "deposit", x=x, y=y))
        deposit_ids.append(nid)
        nid += 1

    # -----------------------
    # PICKUPS — field block stacks (mm)
    # -----------------------
    pickup_ids: List[int] = []

    field_stack_mm = [
        # Big edge stacks (6)
        ("STACK_TOP",       0,   1200, 6, True),
        ("STACK_BOTTOM",    0,  -1200, 6, True),
        ("STACK_LEFT",  -1700,     0, 6, True),
        ("STACK_RIGHT",  1700,     0, 6, True),

        # Inner clusters (4)
        ("CLUSTER_UL",   -600,   600, 4, False),
        ("CLUSTER_UR",    600,   600, 4, False),
        ("CLUSTER_LL",   -600,  -600, 4, False),
        ("CLUSTER_LR",    600,  -600, 4, False),
    ]

    for name, x_mm, y_mm, blocks, bonus in field_stack_mm:
        x, y = mm_to_in(x_mm, y_mm)
        nodes.append(
            Node(
                nid, name, "pickup",
                x=x, y=y,
                initial_blocks=blocks,
                edge_stack_bonus=bonus
            )
        )
        pickup_ids.append(nid)
        nid += 1

    # -----------------------
    # MATCH LOADERS — official GPS points (mm)
    # -----------------------
    loader_mm = [
        ("LOADER_TL", -1725,  1200),
        ("LOADER_TR",  1725,  1200),
        ("LOADER_BL", -1725, -1200),
        ("LOADER_BR",  1725, -1200),
    ]

    for name, x_mm, y_mm in loader_mm:
        x, y = mm_to_in(x_mm, y_mm)
        nodes.append(
            Node(
                nid, name, "pickup",
                x=x, y=y,
                initial_blocks=6,
                edge_stack_bonus=False
            )
        )
        pickup_ids.append(nid)
        nid += 1

    return Env(
        nodes=nodes,
        start_id=start_id,
        pickup_ids=pickup_ids,
        deposit_ids=deposit_ids,
        capacity=8
    )
# ----------------------------
# Run it
# ----------------------------

def main():
    env = make_example_env()
    aco = ACO(
        env,
        n_ants=70,
        n_iters=300,
        alpha=1.2,
        beta=3.2,
        rho=0.18,
        q=0.004,   # keep small because fitness can be big
        lam=0.35,
        max_steps=70,
        seed=4
    )

    best_route, best_points, best_dist, best_fit = aco.run()
    print("Best fitness:", best_fit)
    print("Points:", best_points)
    print("Distance:", best_dist)
    print("Route:")
    plot_route_on_field(env, best_route)
    for nid in best_route:
        print("  -", env.nodes[nid].name)
def plot_route_on_field(env: Env, route: List[int], image_path="field.png", field_size=144):
    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg

    # Ensure route returns to start
    if route and route[-1] != env.start_id:
        route = route + [env.start_id]

    # fig, ax = plt.subplots(figsize=(7, 7))

    # Load and display background field image
    # img = mpimg.imread(image_path)

    # Important: set extent so image coordinates match 0–144 field
    #ax.imshow(img, extent=[0, field_size, 0, field_size])

    #ax.set_xlim(0, field_size)
    #ax.set_ylim(0, field_size)
    #ax.set_aspect("equal")

    # Draw route
    xs = [env.nodes[i].x for i in route]
    ys = [env.nodes[i].y for i in route]

    #ax.plot(xs, ys, linewidth=3)
    #ax.scatter(xs, ys, s=40)

    # Label visit order
    '''for k, nid in enumerate(route):
        ax.text(env.nodes[nid].x + 1,
                env.nodes[nid].y + 1,
                str(k),
                fontsize=8)'''

    #ax.set_title("ACO Route Overlay")
    #plt.show()

if __name__ == "__main__":
    main()