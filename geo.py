"""Geospatial helpers: distances, local projection, PIN adjacency graph,
beat-tour length estimation."""
from __future__ import annotations
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

EARTH_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km (broadcasts like numpy)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def project_km(lat, lon, lat0=None, lon0=None):
    """Equirectangular projection to local km (fine at city scale)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lat0 = np.mean(lat) if lat0 is None else lat0
    lon0 = np.mean(lon) if lon0 is None else lon0
    x = (lon - lon0) * np.cos(np.radians(lat0)) * 111.320
    y = (lat - lat0) * 110.574
    return np.column_stack([x, y])


def pairwise_km(xy_a: np.ndarray, xy_b: np.ndarray) -> np.ndarray:
    d = xy_a[:, None, :] - xy_b[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def delaunay_adjacency(xy: np.ndarray, prune_factor: float = 3.0) -> list[set]:
    """Neighbour graph over PIN centroids (Delaunay edges, long hull edges
    pruned).  Used for contiguity checks and boundary-move candidates."""
    n = len(xy)
    adj = [set() for _ in range(n)]
    if n < 2:
        return adj
    if n < 4:  # Delaunay needs >=3 non-collinear points; fall back to complete graph
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i].add(j)
        return adj
    try:
        tri = Delaunay(xy)
        edges = set()
        for s in tri.simplices:
            for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
                edges.add((min(a, b), max(a, b)))
    except Exception:  # degenerate geometry -> k-NN fallback
        d = pairwise_km(xy, xy)
        edges = set()
        for i in range(n):
            for j in np.argsort(d[i])[1:5]:
                edges.add((min(i, int(j)), max(i, int(j))))
    lens = np.array([np.linalg.norm(xy[a] - xy[b]) for a, b in edges])
    cut = prune_factor * np.median(lens) if len(lens) else np.inf
    for (a, b), L in zip(edges, lens):
        if L <= cut:
            adj[a].add(b)
            adj[b].add(a)
    # make sure nobody is isolated after pruning
    d = None
    for i in range(n):
        if not adj[i]:
            if d is None:
                d = pairwise_km(xy, xy)
            j = int(np.argsort(d[i])[1])
            adj[i].add(j)
            adj[j].add(i)
    return adj


def n_components(nodes: list[int], adj: list[set]) -> int:
    """Number of connected components of the sub-graph induced by `nodes`."""
    if not nodes:
        return 0
    idx = {v: k for k, v in enumerate(nodes)}
    rows, cols = [], []
    for v in nodes:
        for w in adj[v]:
            if w in idx:
                rows.append(idx[v]); cols.append(idx[w])
    m = len(nodes)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, m))
    return int(connected_components(g, directed=False)[0])


def tour_length_km(points_xy: np.ndarray, start_xy: np.ndarray | None = None) -> float:
    """Closed-tour length estimate: nearest-neighbour construction + 2-opt.
    Straight-line km (caller applies road circuity).  Good to ~5-8 % of
    optimal for the <40 stops we see per territory."""
    pts = np.asarray(points_xy, float)
    if start_xy is not None:
        pts = np.vstack([np.asarray(start_xy, float)[None, :], pts])
    n = len(pts)
    if n <= 1:
        return 0.0
    if n == 2:
        return 2 * float(np.linalg.norm(pts[0] - pts[1]))
    D = pairwise_km(pts, pts)
    # nearest neighbour from node 0
    order, seen = [0], np.zeros(n, bool)
    seen[0] = True
    for _ in range(n - 1):
        last = order[-1]
        cand = np.where(~seen)[0]
        nxt = cand[np.argmin(D[last, cand])]
        order.append(int(nxt)); seen[nxt] = True
    # 2-opt
    improved, it = True, 0
    while improved and it < 50:
        improved, it = False, it + 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = order[i - 1], order[i]
                c, d_ = order[j], order[(j + 1) % n]
                if D[a, c] + D[b, d_] < D[a, b] + D[c, d_] - 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    return float(sum(D[order[k], order[(k + 1) % n]] for k in range(n)))
