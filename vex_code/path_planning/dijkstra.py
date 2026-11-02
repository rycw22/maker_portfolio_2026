import math
import heapq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class Node:
    id: int
    name: str
    kind: str  # "start" | "pickup" | "deposit"
    x: float
    y: float
    blocks: int = 0             # for pickup nodes (4 or 6)
    edge_bonus: bool = False    # True for the 6-block edge stacks (gives +5 when cleared)


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


# ----------------------------
# Build an example environment (EDIT COORDS!)
# ----------------------------

def make_example_env() -> Env:
    """
    Matches aco.py / quantum_swarm.py:
      - 144 x 144 inches
      - (0,0) bottom-left
      - VEX VR GPS coords in mm, origin at field center -> convert:
            inches_per_mm = 24/600 = 0.04
            x_in = 72 + 0.04*x_mm
            y_in = 72 + 0.04*y_mm

    START = red parking zone center: (12, 72)
    """
    nodes: List[Node] = []
    nid = 0

    INCH_PER_MM = 24.0 / 600.0  # 0.04

    def mm_to_in(x_mm: float, y_mm: float) -> Tuple[float, float]:
        return (72.0 + INCH_PER_MM * x_mm, 72.0 + INCH_PER_MM * y_mm)

    # START (red parking zone)
    nodes.append(Node(nid, "RED_PARKING_ZONE", "start", x=12.0, y=72.0))
    start_id = nid
    nid += 1

    # Deposits (goals)
    deposit_ids: List[int] = []
    goal_points_mm = [
        ("TOP_GOAL_LEFT",     -620,  1200),
        ("TOP_GOAL_RIGHT",     620,  1200),
        ("BOTTOM_GOAL_LEFT",  -620, -1200),
        ("BOTTOM_GOAL_RIGHT",  620, -1200),
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

    # Pickups (stacks on field + match loaders)
    pickup_ids: List[int] = []

    # Field stacks (6-block edge stacks give +5 when cleared)
    field_pickups = [
        ("STACK_TOP",       0,   1200, 6, True),
        ("STACK_BOTTOM",    0,  -1200, 6, True),
        ("STACK_LEFT",  -1700,     0,  6, True),
        ("STACK_RIGHT",  1700,     0,  6, True),
        ("CLUSTER_UL",   -600,   600,  4, False),
        ("CLUSTER_UR",    600,   600,  4, False),
        ("CLUSTER_LL",   -600,  -600,  4, False),
        ("CLUSTER_LR",    600,  -600,  4, False),
    ]
    for name, x_mm, y_mm, blocks, bonus in field_pickups:
        x, y = mm_to_in(x_mm, y_mm)
        nodes.append(Node(nid, name, "pickup", x=x, y=y, blocks=blocks, edge_bonus=bonus))
        pickup_ids.append(nid)
        nid += 1

    # Match loaders (corners)
    loaders_mm = [
        ("LOADER_TL", -1725,  1200),
        ("LOADER_TR",  1725,  1200),
        ("LOADER_BL", -1725, -1200),
        ("LOADER_BR",  1725, -1200),
    ]
    for name, x_mm, y_mm in loaders_mm:
        x, y = mm_to_in(x_mm, y_mm)
        nodes.append(Node(nid, name, "pickup", x=x, y=y, blocks=6, edge_bonus=False))
        pickup_ids.append(nid)
        nid += 1

    return Env(nodes=nodes, start_id=start_id, pickup_ids=pickup_ids, deposit_ids=deposit_ids, capacity=8)


def optimize_with_dijkstra(env: Env) -> Tuple[List[int], int, float]:
    pickup_ids = env.pickup_ids
    nP = len(pickup_ids)

    radices = [env.nodes[pid].blocks + 1 for pid in pickup_ids]
    multipliers = [1] * nP
    for i in range(1, nP):
        multipliers[i] = multipliers[i - 1] * radices[i - 1]

    def encode_remaining(rem_list: List[int]) -> int:
        code = 0
        for i, r in enumerate(rem_list):
            code += r * multipliers[i]
        return code

    def get_remaining(code: int, i: int) -> int:
        return (code // multipliers[i]) % radices[i]

    def set_remaining(code: int, i: int, new_r: int) -> int:
        old_r = get_remaining(code, i)
        return code + (new_r - old_r) * multipliers[i]

    pindex = {pid: i for i, pid in enumerate(pickup_ids)}

    init_rem = [env.nodes[pid].blocks for pid in pickup_ids]
    init_code = encode_remaining(init_rem)

    start = (env.start_id, 0, init_code, 0) 

    # what are the best distances and data for each point achieved so far
    best: Dict[Tuple[int, int, int, int], Tuple[int, float]] = {start: (0, 0.0)}

    parent: Dict[Tuple[int, int, int, int], Tuple[Tuple[int, int, int, int] | None, int]] = {
        start: (None, env.start_id)
    }

    # get more points with less distance
    pq: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
    heapq.heappush(pq, (0, 0.0, start))  # store (-points, distance, state) but points=0 so 0

    def push_if_better(cur_state, nxt_state, add_dist, add_points, via_node):
        cur_points, cur_dist = best[cur_state]
        nxt_points = cur_points + add_points
        nxt_dist = cur_dist + add_dist

        prev = best.get(nxt_state)
        if prev is None or (nxt_points > prev[0]) or (nxt_points == prev[0] and nxt_dist + 1e-12 < prev[1]):
            best[nxt_state] = (nxt_points, nxt_dist)
            parent[nxt_state] = (cur_state, via_node)
            heapq.heappush(pq, (-nxt_points, nxt_dist, nxt_state))

    # HARD SAFETY STOP so it can't run forever (tune this)
    EXPANSION_LIMIT = 500_000
    expansions = 0

    while pq:
        neg_pts, cur_dist, state = heapq.heappop(pq)
        pts = -neg_pts

        # stale
        if best.get(state) != (pts, cur_dist):
            continue

        expansions += 1
        if expansions > EXPANSION_LIMIT:
            break

        pos, carried, rem_code, edge_mask = state

        # pickups
        space = env.capacity - carried
        if space > 0:
            for pid in pickup_ids:
                i = pindex[pid]
                r = get_remaining(rem_code, i)
                if r <= 0:
                    continue

                take = min(space, r)  # takes the best
                nxt_rem = set_remaining(rem_code, i, r - take)

                add_points = 0
                nxt_edge_mask = edge_mask
                if env.nodes[pid].edge_bonus and (r - take) == 0:
                    bit = 1 << i
                    if (edge_mask & bit) == 0:
                        add_points += 5
                        nxt_edge_mask |= bit

                nxt_state = (pid, carried + take, nxt_rem, nxt_edge_mask)
                push_if_better(state, nxt_state, env.dist(pos, pid), add_points, via_node=pid)

        # deposit
        if carried > 0:
            for dep in env.deposit_ids:
                nxt_state = (dep, 0, rem_code, edge_mask)
                push_if_better(state, nxt_state, env.dist(pos, dep), carried, via_node=dep)

        # return to start
        if pos != env.start_id:
            nxt_state = (env.start_id, carried, rem_code, edge_mask)
            push_if_better(state, nxt_state, env.dist(pos, env.start_id), 0, via_node=env.start_id)

    # choose best terminal: at start and carried==0
    best_state = None
    best_points = -1
    best_distance = float("inf")

    for st, (pts, d) in best.items():
        pos, carried, rem_code, edge_mask = st
        if pos == env.start_id and carried == 0:
            if pts > best_points or (pts == best_points and d < best_distance):
                best_points = pts
                best_distance = d
                best_state = st

    if best_state is None:
        return [env.start_id], 0, 0.0

    # reconstruct route
    route: List[int] = []
    cur = best_state
    while cur is not None:
        prev, via = parent[cur]
        route.append(via)
        cur = prev
    route.reverse()

    return route, best_points, best_distance

def plot_route_on_field(env: Env, route: List[int], image_path: str = "field.png", title: str = "Dijkstra Route Overlay"):
    """
    Overlays a route on top of the Push Back field image.
    Assumes env node coordinates are in inches with (0,0) bottom-left and field is 144x144.
    """
    # Ensure route ends at start (nice closed loop)
    if route and route[-1] != env.start_id:
        route = route + [env.start_id]

    # fig, ax = plt.subplots(figsize=(7, 7))
    # img = mpimg.imread(image_path)

    # Align image to coordinate space
    # ax.imshow(img, extent=[0, 144, 0, 144])

    # ax.set_xlim(0, 144)
    # ax.set_ylim(0, 144)
    # ax.set_aspect("equal")

    # Route polyline
    xs = [env.nodes[nid].x for nid in route]
    ys = [env.nodes[nid].y for nid in route]
    # ax.plot(xs, ys, linewidth=3)
    # ax.scatter(xs, ys, s=40)

    # Label visit order
    '''for i, nid in enumerate(route):
        ax.text(env.nodes[nid].x + 1,
                env.nodes[nid].y + 1,
                str(i),
                fontsize=8)'''

    #ax.set_title(title)
    #ax.set_xlabel("x (in)")
    #ax.set_ylabel("y (in)")
    #plt.show()

# ----------------------------
# Run
# ----------------------------

def main():
    env = make_example_env()
    route, points, distance = optimize_with_dijkstra(env)

    print("Max points:", points)
    print("Min distance (among max-point routes):", distance)
    print("Route:")
    for nid in route:
        print("  -", env.nodes[nid].name)

    # 👇 Plot overlay
    plot_route_on_field(env, route, image_path="field.png", title="Dijkstra Route Overlay")

if __name__ == "__main__":
    main()