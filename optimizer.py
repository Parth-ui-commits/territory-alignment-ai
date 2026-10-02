"""Territory optimiser.

Problem (per Area-Manager region)
---------------------------------
Atomic unit : PIN code  (visit-hours W_p, potential P_p, stop frequency F_p)
Decision    : owner[p] in {employees}   -> every PIN has exactly ONE owner, so
              territories never overlap and there are no coverage gaps.

Objective   : J = w_wl  * mean_i (util_i - U*)^2            workload balance
                + w_pot * mean_i (potential_i/target_i - 1)^2 opportunity balance
                + w_cap * mean_i max(0, util_i - 1)^2        hard-ish capacity
                + w_geo * sum_i sum_p W_p |x_p - c_i| / (W_tot * R)   compactness
                + w_cont* relationship-weight lost / total   continuity
Constraints : each rep keeps >= 1 PIN; moves may not fragment a territory.

Algorithm   : (1) capacity-aware, continuity-biased Lloyd initialisation
              (2) steepest-descent local search over boundary-PIN moves and
                  boundary swaps (exact incremental cost on the 2 touched
                  territories)
              (3) fragment repair
This is a classical OR heuristic (a.k.a. 'balanced districting'); it is fast
(~1 s / region), explainable, and warm-startable for what-if re-alignment.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

import config as C
from geo import project_km, delaunay_adjacency, n_components
from workload import add_customer_fields, pin_table

TRAVEL_K = 1.28         # calibrated: true beat-tour hours / proxy hours (median 1.28, see README)
BHH = 1.9               # tour ~ BHH * sqrt(m) * mean-dist-to-centroid (random-uniform heuristic)


@dataclass
class SolveInfo:
    region: str
    sweeps: int
    moves: int
    swaps: int
    j_init: float
    j_final: float
    history: list


class RegionProblem:
    def __init__(self, pins_r: pd.DataFrame, cust_r: pd.DataFrame, emps_r: pd.DataFrame):
        self.region = str(pins_r.region.iloc[0])
        self.pin_ids = pins_r.pin.tolist()
        self.emp_ids = emps_r.emp_id.tolist()
        self.n_p, self.n_e = len(self.pin_ids), len(self.emp_ids)
        lat0, lon0 = pins_r.lat.mean(), pins_r.lon.mean()
        self.lat0, self.lon0 = lat0, lon0
        self.xy = project_km(pins_r.lat, pins_r.lon, lat0, lon0)
        self.W = pins_r.visit_hours.values.astype(float)
        self.P = pins_r.potential_lakh.values.astype(float)
        self.F = pins_r.stop_freq.values.astype(float)
        self.stop = self.F > 0
        self.cap = emps_r.capacity_hours.values.astype(float)
        self.speed = emps_r.speed_kmph.values.astype(float)
        self.potw = emps_r.potential_weight.values.astype(float)
        self.home_xy = project_km(emps_r.home_lat, emps_r.home_lon, lat0, lon0)
        self.adj = delaunay_adjacency(self.xy)
        pos = {p: i for i, p in enumerate(self.pin_ids)}
        epos = {e: i for i, e in enumerate(self.emp_ids)}
        # relationship matrix: rel-weight of customers in PIN p legacy-owned by emp e
        R = np.zeros((self.n_p, self.n_e))
        cw = cust_r if "rel_weight" in cust_r else add_customer_fields(cust_r)
        for p, e, w in zip(cw.pin, cw.legacy_emp, cw.rel_weight):
            if e in epos and p in pos:
                R[pos[p], epos[e]] += w
        self.R = R
        self.relw_live = R.sum(1)
        self.rel_total = max(R.sum(), 1e-9)
        self.cont_cost = (self.relw_live[:, None] - R) / self.rel_total   # (pin, emp)
        self.Wtot = max(self.W.sum(), 1e-9)
        self.Pstar = max(self.P.sum(), 1e-9) * self.potw / self.potw.sum()
        # typical territory radius -> scale for the compactness term
        from scipy.spatial import ConvexHull
        try:
            area = ConvexHull(self.xy).volume
        except Exception:
            area = 100.0
        self.R_len = max(np.sqrt(area / max(self.n_e, 1)), 1.0)
        # anchors: blend of legacy-customer centroid and home
        self.anchor0 = self.home_xy.copy()
        for e in range(self.n_e):
            w = R[:, e]
            if w.sum() > 0 and (w > 0).sum() >= 3:
                leg = np.average(self.xy, axis=0, weights=w)
                self.anchor0[e] = 0.5 * leg + 0.5 * self.home_xy[e]
        self.U_star = 0.9

    # ------------------------------------------------------------ costs
    def stats(self, e: int, members: np.ndarray):
        """(utilisation, potential_ratio, geo_cost, travel_hours) for territory e."""
        if len(members) == 0:
            return 0.0, 0.0, 0.0, 0.0
        vis = self.W[members].sum()
        pot = self.P[members].sum()
        st = members[self.stop[members]]
        if len(st):
            sxy = self.xy[st]
            c = sxy.mean(0)
            d = np.linalg.norm(sxy - c, axis=1)
            L = BHH * np.sqrt(len(st)) * d.mean() + 2 * np.linalg.norm(self.home_xy[e] - c)
            trav = TRAVEL_K * C.ROAD_CIRCUITY * self.F[st].mean() * L / self.speed[e]
            geo = float((self.W[st] * d).sum()) / (self.Wtot * self.R_len)
        else:
            trav, geo = 0.0, 0.0
        return (vis + trav) / self.cap[e], pot / self.Pstar[e], geo, trav

    def terr_cost(self, e: int, members: np.ndarray, w: C.Weights) -> float:
        util, pr, geo, _ = self.stats(e, members)
        n = self.n_e
        return (w.w_wl * (util - self.U_star) ** 2 / n
                + w.w_pot * (pr - 1.0) ** 2 / n
                + w.w_cap * max(0.0, util - 1.0) ** 2 / n
                + w.w_geo * geo)

    def total_cost(self, assign: np.ndarray, w: C.Weights) -> float:
        tot = 0.0
        for e in range(self.n_e):
            tot += self.terr_cost(e, np.where(assign == e)[0], w)
        ok = assign >= 0
        tot += w.w_cont * self.cont_cost[np.where(ok)[0], assign[ok]].sum()
        return float(tot)

    # --------------------------------------------------------- initialise
    def init_assign(self, w: C.Weights, rng: np.random.Generator, jitter: float = 0.0) -> np.ndarray:
        anchors = self.anchor0 + rng.normal(0, jitter, self.anchor0.shape)
        target = self.cap / self.cap.sum() * self.W.sum()          # capacity-proportional visit hours
        assign = -np.ones(self.n_p, int)
        for it in range(8):
            load = np.zeros(self.n_e)
            assign = -np.ones(self.n_p, int)
            order = np.argsort(-self.W + rng.normal(0, 1e-3, self.n_p))
            for p in order:
                d = np.linalg.norm(self.xy[p] - anchors, axis=1) / self.R_len
                over = np.maximum(0.0, (load + self.W[p]) / np.maximum(target, 1e-9) - 0.95)
                cost = d + 2.5 * over - 0.6 * self.cont_cost_bonus(p)
                e = int(np.argmin(cost))
                assign[p] = e
                load[e] += self.W[p]
            for e in range(self.n_e):                              # Lloyd update
                m = np.where(assign == e)[0]
                if len(m):
                    ww = self.W[m] + 1e-3
                    anchors[e] = np.average(self.xy[m], axis=0, weights=ww)
        return self._ensure_nonempty(assign)

    def cont_cost_bonus(self, p: int) -> np.ndarray:
        """1 for the employee holding all (live) relationships in PIN p, 0 for others."""
        if self.relw_live[p] <= 0:
            return np.zeros(self.n_e)
        return self.R[p] / self.relw_live[p]

    def _ensure_nonempty(self, assign: np.ndarray) -> np.ndarray:
        assign = assign.copy()
        for e in range(self.n_e):
            if not (assign == e).any():
                # steal the PIN nearest to this rep's home from the biggest territory
                sizes = np.bincount(assign[assign >= 0], minlength=self.n_e)
                cand = np.where(sizes[assign] > 1)[0]
                d = np.linalg.norm(self.xy[cand] - self.home_xy[e], axis=1)
                assign[cand[np.argmin(d)]] = e
        return assign

    # ------------------------------------------------------- local search
    def local_search(self, assign: np.ndarray, w: C.Weights, rng: np.random.Generator,
                     max_sweeps: int | None = None, allow_moves_from=None):
        max_sweeps = max_sweeps or w.max_sweeps
        assign = assign.copy()
        members = [set(np.where(assign == e)[0].tolist()) for e in range(self.n_e)]
        arr = lambda s: np.fromiter(s, int, len(s))
        history, n_moves, n_swaps = [], 0, 0
        eps = 1e-9

        for sweep in range(max_sweeps):
            # refresh the utilisation target from current totals
            tot_h = sum(self.stats(e, arr(members[e]))[0] * self.cap[e] for e in range(self.n_e))
            self.U_star = tot_h / self.cap.sum()
            cur = [self.terr_cost(e, arr(members[e]), w) for e in range(self.n_e)]
            changed = False

            # ---- single-PIN boundary moves
            order = rng.permutation(self.n_p)
            for p in order:
                a = assign[p]
                if a < 0 or len(members[a]) <= 1:
                    continue
                nbr_t = {assign[q] for q in self.adj[p] if assign[q] >= 0 and assign[q] != a}
                if not nbr_t:
                    continue
                Ma = members[a] - {p}
                new_a = self.terr_cost(a, arr(Ma), w)
                best, best_b, best_nb = -eps, None, None
                for b in nbr_t:
                    Mb = members[b] | {p}
                    new_b = self.terr_cost(b, arr(Mb), w)
                    dlt = (new_a + new_b - cur[a] - cur[b]
                           + w.w_cont * (self.cont_cost[p, b] - self.cont_cost[p, a]))
                    if dlt < best:
                        best, best_b, best_nb = dlt, b, new_b
                if best_b is not None:
                    if n_components(list(Ma), self.adj) <= n_components(list(members[a]), self.adj):
                        members[a], members[best_b] = Ma, members[best_b] | {p}
                        assign[p] = best_b
                        cur[a], cur[best_b] = new_a, best_nb
                        n_moves += 1; changed = True

            # ---- boundary swaps (help when both territories sit at capacity)
            edges = [(p, q) for p in range(self.n_p) for q in self.adj[p]
                     if p < q and assign[p] != assign[q] and assign[p] >= 0 and assign[q] >= 0]
            rng.shuffle(edges)
            for p, q in edges:
                a, b = assign[p], assign[q]
                if a == b:
                    continue
                Ma, Mb = (members[a] - {p}) | {q}, (members[b] - {q}) | {p}
                dlt = (self.terr_cost(a, arr(Ma), w) + self.terr_cost(b, arr(Mb), w) - cur[a] - cur[b]
                       + w.w_cont * (self.cont_cost[p, b] + self.cont_cost[q, a]
                                     - self.cont_cost[p, a] - self.cont_cost[q, b]))
                if dlt < -eps:
                    if (n_components(list(Ma), self.adj) <= n_components(list(members[a]), self.adj) and
                            n_components(list(Mb), self.adj) <= n_components(list(members[b]), self.adj)):
                        members[a], members[b] = Ma, Mb
                        assign[p], assign[q] = b, a
                        cur[a], cur[b] = self.terr_cost(a, arr(Ma), w), self.terr_cost(b, arr(Mb), w)
                        n_swaps += 1; changed = True

            history.append(self.total_cost(assign, w))
            if not changed:
                break
        return assign, history, n_moves, n_swaps, sweep + 1

    # ------------------------------------------------------ fragment repair
    def fix_fragments(self, assign: np.ndarray, w: C.Weights) -> np.ndarray:
        assign = assign.copy()
        for _ in range(5):
            fixed_any = False
            for e in range(self.n_e):
                mem = np.where(assign == e)[0].tolist()
                if len(mem) < 2 or n_components(mem, self.adj) <= 1:
                    continue
                # connected components via simple BFS
                comps, seen = [], set()
                ms = set(mem)
                for s in mem:
                    if s in seen:
                        continue
                    stack, comp = [s], set()
                    while stack:
                        v = stack.pop()
                        if v in comp:
                            continue
                        comp.add(v); seen.add(v)
                        stack.extend(x for x in self.adj[v] if x in ms and x not in comp)
                    comps.append(comp)
                comps.sort(key=lambda c: -sum(self.W[list(c)]))
                for comp in comps[1:]:                       # keep heaviest, re-home the rest
                    for p in comp:
                        nb = {assign[q] for q in self.adj[p] if assign[q] != e and assign[q] >= 0}
                        if nb:
                            costs = {b: self.terr_cost(b, np.append(np.where(assign == b)[0], p), w)
                                        - self.terr_cost(b, np.where(assign == b)[0], w)
                                        + w.w_cont * self.cont_cost[p, b] for b in nb}
                            assign[p] = min(costs, key=costs.get)
                            fixed_any = True
            if not fixed_any:
                break
        return self._ensure_nonempty(assign)

    # --------------------------------------------------------------- solve
    def solve(self, w: C.Weights, seed: int = 0, init: np.ndarray | None = None,
              max_sweeps: int | None = None) -> tuple[np.ndarray, SolveInfo]:
        best, best_j, best_info = None, np.inf, None
        n_try = 1 if init is not None else max(1, w.n_restarts)
        for r in range(n_try):
            rng = np.random.default_rng(seed + 1000 * r)
            a0 = self._complete(init, w) if init is not None else \
                self.init_assign(w, rng, jitter=0.0 if r == 0 else 0.6 * self.R_len)
            j0 = self.total_cost(a0, w)
            a, hist, mv, sw, sweeps = self.local_search(a0, w, rng, max_sweeps)
            a = self.fix_fragments(a, w)
            a, hist2, mv2, sw2, sweeps2 = self.local_search(a, w, rng, max_sweeps=10)
            a = self.fix_fragments(a, w)
            j = self.total_cost(a, w)
            if j < best_j:
                best, best_j = a, j
                best_info = SolveInfo(self.region, sweeps + sweeps2, mv + mv2, sw + sw2, j0, j, hist + hist2)
        return best, best_info

    def _complete(self, init: np.ndarray, w: C.Weights) -> np.ndarray:
        """Fill unassigned (-1) PINs by greedy cheapest-neighbour growth."""
        a = init.copy()
        todo = set(np.where(a < 0)[0].tolist())
        guard = 0
        while todo and guard < 10 * self.n_p:
            guard += 1
            progress = False
            for p in sorted(todo, key=lambda x: -sum(a[q] >= 0 for q in self.adj[x])):
                nb = {a[q] for q in self.adj[p] if a[q] >= 0}
                if not nb:
                    continue
                best, best_c = None, np.inf
                for b in nb:
                    cur_m = np.where(a == b)[0]
                    c = (self.terr_cost(b, np.append(cur_m, p), w) - self.terr_cost(b, cur_m, w)
                         + w.w_cont * self.cont_cost[p, b])
                    if c < best_c:
                        best, best_c = b, c
                a[p] = best; todo.discard(p); progress = True
            if not progress:       # isolated PINs -> nearest anchor
                for p in list(todo):
                    d = np.linalg.norm(self.xy[p] - self.home_xy, axis=1)
                    a[p] = int(np.argmin(d)); todo.discard(p)
        return self._ensure_nonempty(a)


# ----------------------------------------------------------------------------
def build_problems(pins: pd.DataFrame, cust: pd.DataFrame, emps: pd.DataFrame,
                   regions: list[str] | None = None) -> dict[str, RegionProblem]:
    c = add_customer_fields(cust) if "visit_hours" not in cust else cust
    pt = pin_table(c, pins)
    out = {}
    for reg, pr in pt.groupby("region", sort=False):
        if regions and reg not in regions:
            continue
        er = emps[emps.region == reg]
        if len(er) == 0:
            continue
        out[reg] = RegionProblem(pr.reset_index(drop=True), c[c.region == reg], er.reset_index(drop=True))
    return out


def solve_all(pins, cust, emps, weights: C.Weights | None = None, seed: int = C.SEED,
              init_owner: dict | None = None, regions: list[str] | None = None,
              max_sweeps: int | None = None):
    """Run the optimiser for every region.
    init_owner: optional {pin: emp_id} warm start (-> what-if / re-alignment).
    Returns (pin_owner DataFrame[pin, emp_id, region], list[SolveInfo], problems)."""
    weights = weights or C.Weights()
    probs = build_problems(pins, cust, emps, regions)
    rows, infos = [], []
    for reg, pb in probs.items():
        init = None
        if init_owner is not None:
            epos = {e: i for i, e in enumerate(pb.emp_ids)}
            init = np.array([epos.get(init_owner.get(p), -1) for p in pb.pin_ids], int)
        a, info = pb.solve(weights, seed=seed, init=init, max_sweeps=max_sweeps)
        infos.append(info)
        rows += [dict(pin=p, emp_id=pb.emp_ids[a[i]], region=reg) for i, p in enumerate(pb.pin_ids)]
    return pd.DataFrame(rows), infos, probs


def apply_pin_owner(cust: pd.DataFrame, pin_owner: pd.DataFrame, col: str = "new_emp") -> pd.DataFrame:
    c = cust.copy()
    c[col] = c.pin.map(dict(zip(pin_owner.pin, pin_owner.emp_id)))
    return c
