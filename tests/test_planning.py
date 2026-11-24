import numpy as np

from src.planning.cost_map import class_grid_to_cost, mask_to_grid
from src.planning.planners import astar, dijkstra, path_metrics, straight_line

COSTS = {"bedrock": 1.0, "soil": 2.0, "sand": 5.0, "big_rock": None, "unknown": 8.0}


def test_astar_matches_dijkstra_cost():
    rng = np.random.default_rng(0)
    cls = rng.integers(0, 3, (20, 20))
    cost = class_grid_to_cost(cls, COSTS, inflate=0)
    a, d = astar(cost, (19, 10), (0, 10)), dijkstra(cost, (19, 10), (0, 10))
    ma, md = path_metrics(a, cost, cls), path_metrics(d, cost, cls)
    assert abs(ma["total_cost"] - md["total_cost"]) < 1e-6   # both optimal


def test_avoids_wall_and_straight_line_does_not():
    cls = np.zeros((15, 15), int)
    cls[7, 2:13] = 3                                   # wall of big rocks, gap at the edges
    cost = class_grid_to_cost(cls, COSTS, inflate=0)
    path = astar(cost, (14, 7), (0, 7))
    assert path is not None and all(np.isfinite(cost[p]) for p in path)
    assert path_metrics(straight_line((14, 7), (0, 7)), cost, cls)["blocked_cells"] > 0


def test_no_path_returns_none():
    cls = np.zeros((10, 10), int)
    cls[5, :] = 3
    cost = class_grid_to_cost(cls, COSTS, inflate=0)
    assert astar(cost, (9, 5), (0, 5)) is None


def test_mask_to_grid_shape_and_unknown():
    mask = np.full((256, 256), 255, np.uint8)
    g = mask_to_grid(mask, 32)
    assert g.shape == (32, 32) and (g == 4).all()
