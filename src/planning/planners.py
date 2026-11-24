"""Path planners on a cost grid: A*, Dijkstra, straight-line baseline, plus metrics.

Grid cells are (row, col). Moves are 8-connected; moving between two cells costs
step_length * average(cost of the two cells). inf cost = blocked.
"""
import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2)
MOVES = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def nearest_free(cost, cell):
    """If `cell` is blocked, return the closest passable cell (else the cell itself)."""
    if np.isfinite(cost[cell]):
        return cell
    free = np.argwhere(np.isfinite(cost))
    if len(free) == 0:
        return None
    d = (free[:, 0] - cell[0]) ** 2 + (free[:, 1] - cell[1]) ** 2
    return tuple(free[np.argmin(d)])


def _search(cost, start, goal, use_heuristic):
    h_cost = cost.shape
    finite = cost[np.isfinite(cost)]
    if len(finite) == 0:
        return None
    min_c = float(finite.min())

    def h(c):   # admissible: cheapest possible cost per step * octile distance
        if not use_heuristic:
            return 0.0
        dr, dc = abs(c[0] - goal[0]), abs(c[1] - goal[1])
        return min_c * (max(dr, dc) + (SQRT2 - 1) * min(dr, dc))

    best = {start: 0.0}
    parent = {}
    pq = [(h(start), 0.0, start)]
    while pq:
        _, g, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        if g > best.get(cur, np.inf):
            continue
        for dr, dc, step in MOVES:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < h_cost[0] and 0 <= nc < h_cost[1]) or not np.isfinite(cost[nr, nc]):
                continue
            if dr and dc and not (np.isfinite(cost[cur[0] + dr, cur[1]]) and np.isfinite(cost[cur[0], cur[1] + dc])):
                continue   # no cutting corners through blocked cells
            ng = g + step * (cost[cur] + cost[nr, nc]) / 2
            if ng < best.get((nr, nc), np.inf):
                best[(nr, nc)] = ng
                parent[(nr, nc)] = cur
                heapq.heappush(pq, (ng + h((nr, nc)), ng, (nr, nc)))
    return None


def astar(cost, start, goal):
    return _search(cost, start, goal, use_heuristic=True)


def dijkstra(cost, start, goal):
    return _search(cost, start, goal, use_heuristic=False)


def straight_line(start, goal):
    """Baseline: cells along the straight segment, ignoring terrain."""
    n = max(abs(goal[0] - start[0]), abs(goal[1] - start[1])) + 1
    rr = np.rint(np.linspace(start[0], goal[0], n)).astype(int)
    cc = np.rint(np.linspace(start[1], goal[1], n)).astype(int)
    return list(dict.fromkeys(zip(rr.tolist(), cc.tolist())))


def path_metrics(path, cost, class_grid, blocked_penalty=100.0):
    """Proxy metrics (no ground truth exists for 'best route')."""
    if not path:
        return None
    c = np.where(np.isfinite(cost), cost, blocked_penalty)
    total, length = 0.0, 0.0
    for a, b in zip(path[:-1], path[1:]):
        step = SQRT2 if (a[0] != b[0] and a[1] != b[1]) else 1.0
        total += step * (c[a] + c[b]) / 2
        length += step
    cls = np.array([class_grid[p] for p in path])
    return {
        "total_cost": round(float(total), 2),
        "length": round(float(length), 2),
        "pct_safe": round(float(np.isin(cls, (0, 1)).mean() * 100), 1),
        "hazard_cells": int(np.isin(cls, (2, 3)).sum()),
        "blocked_cells": int(sum(not np.isfinite(cost[p]) for p in path)),
    }
