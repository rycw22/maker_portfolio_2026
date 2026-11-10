import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


# -- Data model --

@dataclass(frozen=True)
class Node:
    id: int
    name: str
    kind: str  # "start" | "pickup" | "deposit"
    x: float
    y: float
    initial_blocks: int = 0
    edge_stack_bonus: bool = False  # True for the 6-block edge stacks


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


# -- Simulator --

def simulate_route(env: Env, route: List[int]) -> Tuple[float, float]:
    remaining = {nid: env.nodes[nid].initial_blocks for nid in env.pickup_ids}
    cleared_edge_bonus = {nid: False for nid in env.pickup_ids}

    carried = 0
    points = 0.0
    dist_total = 0.0

    def is_edge(nid: int) -> bool:
        return env.nodes[nid].kind == "pickup" and env.nodes[nid].edge_stack_bonus

    for a, b in zip(route[:-1], route[1:]):
        dist_total += env.dist(a, b)
        node = env.nodes[b]

        if node.kind == "pickup":
            if remaining[b] <= 0:
                continue
            space = env.capacity - carried
            if space <= 0:
                continue
            take = min(space, remaining[b])
            remaining[b] -= take
            carried += take

            if is_edge(b) and remaining[b] == 0 and not cleared_edge_bonus[b]:
                points += 5.0
                cleared_edge_bonus[b] = True

        elif node.kind == "deposit":
            points += carried
            carried = 0

    return points, dist_total


# -- Discrete decoding from a QPSO particle --

def decode_particle_to_route(
    env: Env,
    priorities: List[float],
    max_steps: int = 70,
    stop_if_low_value: bool = False
) -> List[int]:
    start = env.start_id
    route = [start]
    cur = start

    remaining = {nid: env.nodes[nid].initial_blocks for nid in env.pickup_ids}
    carried = 0

    def best_deposit_distance(from_id: int) -> float:
        return min(env.dist(from_id, dep) for dep in env.deposit_ids)

    def estimated_gain_if_go(nid: int) -> float:
        node = env.nodes[nid]
        d = env.dist(cur, nid) + 1e-9

        if node.kind == "pickup":
            if remaining[nid] <= 0:
                return -1e9
            space = env.capacity - carried
            if space <= 0:
                return -1e9
            take = min(space, remaining[nid])
            bonus = 5.0 if (node.edge_stack_bonus and remaining[nid] - take == 0) else 0.0
            future = best_deposit_distance(nid)
            # points per "effective distance"
            return (take + bonus) / (d + 0.7 * future)

        if node.kind == "deposit":
            if carried <= 0:
                return -1e9
            return (carried * (carried / env.capacity)) / d

        return -1e9

    for step in range(max_steps):
        candidates = []

        # if the robot has blocks it can visit a deposit
        have_pickups_left = any(remaining[pid] > 0 for pid in env.pickup_ids)
        if carried >= env.capacity * 0.5 or (carried > 0 and not have_pickups_left):
            candidates.extend(env.deposit_ids)

        # if the robot has space it can get blocks
        if carried < env.capacity:
            for pid in env.pickup_ids:
                if remaining[pid] > 0:
                    candidates.append(pid)

        # can always go home
        candidates.append(start)

        # try not to visit the same place too many times
        candidates = list(dict.fromkeys(candidates))

        # go home if its the only option left
        if len(candidates) == 1 and candidates[0] == start:
            break

        # how we choose the best next node
        best = None
        best_score = -1e18
        for nid in candidates:
            # priority dominates global "style", gain adds local intelligence
            score = 1.0 * priorities[nid] + 2.0 * estimated_gain_if_go(nid)
            if score > best_score:
                best_score = score
                best = nid

        # we can go home
        if best == start and step > 3:
            route.append(start)
            break

        # move
        route.append(best)
        node = env.nodes[best]

        if node.kind == "pickup":
            space = env.capacity - carried
            if space > 0 and remaining[best] > 0:
                take = min(space, remaining[best])
                remaining[best] -= take
                carried += take

        elif node.kind == "deposit":
            carried = 0

        cur = best

    if route[-1] != start:
        route.append(start)

    return route


# -- Quantum Particle Swarm Optimization (QPSO) --

class QPSO:

    def __init__(
        self,
        env: Env,
        swarm_size: int = 50,
        iters: int = 250,
        beta_start: float = 1.2,
        beta_end: float = 0.5,
        lam: float = 0.35,      # distance penalty
        max_steps: int = 70,
        seed: int = 9
    ):
        self.env = env
        self.swarm_size = swarm_size
        self.iters = iters
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.lam = lam
        self.max_steps = max_steps
        random.seed(seed)

        self.dim = len(env.nodes)

        # positions (priority vectors)
        self.x = [[random.uniform(-1.0, 1.0) for _ in range(self.dim)]
                  for _ in range(swarm_size)]

        # personal bests
        self.pbest = [row[:] for row in self.x]
        self.pbest_fit = [-1e18 for _ in range(swarm_size)]

        # global best
        self.gbest = self.x[0][:]
        self.gbest_fit = -1e18

    def fitness_from_priorities(self, priorities: List[float]) -> Tuple[float, float, float, List[int]]:
        route = decode_particle_to_route(self.env, priorities, max_steps=self.max_steps)
        pts, dist = simulate_route(self.env, route)
        fit = pts - self.lam * dist
        return fit, pts, dist, route

    def run(self) -> Tuple[List[int], float, float, float]:
        # initialize pbest/gbest
        best_route = None
        best_pts = 0.0
        best_dist = 0.0

        for i in range(self.swarm_size):
            fit, pts, dist, route = self.fitness_from_priorities(self.x[i])
            self.pbest_fit[i] = fit
            self.pbest[i] = self.x[i][:]
            if fit > self.gbest_fit:
                self.gbest_fit = fit
                self.gbest = self.x[i][:]
                best_route, best_pts, best_dist = route, pts, dist

        # main loop
        for t in range(self.iters):
            # anneal beta (explore -> exploit)
            beta = self.beta_start + (self.beta_end - self.beta_start) * (t / max(1, self.iters - 1))

            # mbest = mean of all pbest
            mbest = [0.0] * self.dim
            for d in range(self.dim):
                mbest[d] = sum(self.pbest[i][d] for i in range(self.swarm_size)) / self.swarm_size

            for i in range(self.swarm_size):
                # Update each dimension
                for d in range(self.dim):
                    u = random.random()
                    u = max(u, 1e-12)

                    r1 = random.random()
                    r2 = random.random()
                    c1 = 1.0
                    c2 = 1.0
                    denom = c1 * r1 + c2 * r2 + 1e-12

                    p = (c1 * r1 * self.pbest[i][d] + c2 * r2 * self.gbest[d]) / denom

                    sign = 1.0 if random.random() < 0.5 else -1.0
                    self.x[i][d] = p + sign * beta * abs(mbest[d] - self.x[i][d]) * math.log(1.0 / u)

                    # make sure the numbers don't explode
                    if self.x[i][d] > 10.0:
                        self.x[i][d] = 10.0
                    elif self.x[i][d] < -10.0:
                        self.x[i][d] = -10.0

                # evaluate the fitness of the particle
                fit, pts, dist, route = self.fitness_from_priorities(self.x[i])

                # save the best
                if fit > self.pbest_fit[i]:
                    self.pbest_fit[i] = fit
                    self.pbest[i] = self.x[i][:]

                # globally save the best
                if fit > self.gbest_fit:
                    self.gbest_fit = fit
                    self.gbest = self.x[i][:]
                    best_route, best_pts, best_dist = route, pts, dist

        return best_route, best_pts, best_dist, self.gbest_fit

# -- Example environment --

def make_example_env() -> Env:
    """
    Coordinates match the Push Back field.png:
      - Field is 144 x 144 inches
      - (0,0) is bottom-left
      - VEX VR GPS coords are in mm, origin at field center (0,0)
      - Convert via: inches_per_mm = 24/600 = 0.04
          x_in = 72 + 0.04*x_mm
          y_in = 72 + 0.04*y_mm

    START = Red parking zone center (left side): (12, 72)
    """
    nodes: List[Node] = []
    nid = 0

    INCH_PER_MM = 24.0 / 600.0  # 0.04

    def mm_to_in(x_mm: float, y_mm: float) -> Tuple[float, float]:
        return (72.0 + INCH_PER_MM * x_mm, 72.0 + INCH_PER_MM * y_mm)

    # START (Red parking zone)
    nodes.append(Node(nid, "RED_PARKING_ZONE", "start", x=12.0, y=72.0))
    start_id = nid
    nid += 1

    # Deposits (goals)
    deposit_ids: List[int] = []

    goal_points_mm = [
        # Long goal endpoints
        ("TOP_GOAL_LEFT",     -620,  1200),
        ("TOP_GOAL_RIGHT",     620,  1200),
        ("BOTTOM_GOAL_LEFT",  -620, -1200),
        ("BOTTOM_GOAL_RIGHT",  620, -1200),

        # Center goal points around the X
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

   # Matchloaders
    pickup_ids: List[int] = []

    # Field stacks (official GPS set)
    field_pickups = [
        # Edge stacks (6 blocks, edge bonus when cleared)
        ("STACK_TOP",       0,   1200, 6, True),
        ("STACK_BOTTOM",    0,  -1200, 6, True),
        ("STACK_LEFT",  -1700,     0, 6, True),
        ("STACK_RIGHT",  1700,     0, 6, True),

        # Inner 4-block clusters
        ("CLUSTER_UL",   -600,   600, 4, False),
        ("CLUSTER_UR",    600,   600, 4, False),
        ("CLUSTER_LL",   -600,  -600, 4, False),
        ("CLUSTER_LR",    600,  -600, 4, False),
    ]

    for name, x_mm, y_mm, blocks, bonus in field_pickups:
        x, y = mm_to_in(x_mm, y_mm)
        nodes.append(Node(nid, name, "pickup", x=x, y=y, initial_blocks=blocks, edge_stack_bonus=bonus))
        pickup_ids.append(nid)
        nid += 1

    # 4 corner loaders; model them as pickup nodes with 6 blocks initially.
    loader_mm = [
        ("LOADER_TL", -1725,  1200),
        ("LOADER_TR",  1725,  1200),
        ("LOADER_BL", -1725, -1200),
        ("LOADER_BR",  1725, -1200),
    ]

    for name, x_mm, y_mm in loader_mm:
        x, y = mm_to_in(x_mm, y_mm)
        nodes.append(Node(nid, name, "pickup", x=x, y=y, initial_blocks=6, edge_stack_bonus=False))
        pickup_ids.append(nid)
        nid += 1

    return Env(nodes=nodes, start_id=start_id, pickup_ids=pickup_ids, deposit_ids=deposit_ids, capacity=8)

def plot_route_on_field(env: Env, route: List[int], image_path="field.png"):
    """
    Draws the route over the 144x144 Push Back field image.
    """

    # Ensure route ends at start
    if route and route[-1] != env.start_id:
        route = route + [env.start_id]

    # fig, ax = plt.subplots(figsize=(7, 7))

    # img = mpimg.imread(image_path)

    # Align field coordinates (0–144 inches)
    # ax.imshow(img, extent=[0, 144, 0, 144])

    # ax.set_xlim(0, 144)
    # ax.set_ylim(0, 144)
    # ax.set_aspect("equal")

    # Extract coordinates
    xs = [env.nodes[nid].x for nid in route]
    ys = [env.nodes[nid].y for nid in route]

    # Draw route line
    # ax.plot(xs, ys, linewidth=3)
    # ax.scatter(xs, ys, s=40)

    # Label order
    '''for i, nid in enumerate(route):
        ax.text(env.nodes[nid].x + 1,
                env.nodes[nid].y + 1,
                str(i),
                fontsize=8)'''

    # ax.set_title("QPSO Route Overlay")

    # plt.show()
# -- Run --

def main():
    env = make_example_env()

    qpso = QPSO(
        env,
        swarm_size=120,
        iters=800,
        beta_start=1.25,
        beta_end=0.55,
        lam=0.05,
        max_steps=110,
        seed=3
    )

    best_route, best_pts, best_dist, best_fit = qpso.run()

    print("Best fitness:", best_fit)
    print("Points:", best_pts)
    print("Distance:", best_dist)
    print("Route:")
    for nid in best_route:
        print("  -", env.nodes[nid].name)

    plot_route_on_field(env, best_route)

if __name__ == "__main__":
    main()