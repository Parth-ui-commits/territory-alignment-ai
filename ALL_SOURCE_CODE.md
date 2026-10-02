# Complete source code - Territory Alignment
All files in dependency order. Same content as the zip. Run `python run_pipeline.py` then `streamlit run app.py`.

---

## `requirements.txt`  (12 lines)

```text
pandas>=2.0
numpy>=1.26
scipy>=1.11
scikit-learn>=1.3
shapely>=2.0
folium>=0.15
streamlit>=1.35
plotly>=5.20
networkx>=3.0
pytest>=7.0
# optional (LLM narrative layer): anthropic>=0.40
tabulate>=0.9

```

---

## `config.py`  (78 lines)

```python
"""Central configuration: business rules, regions, optimiser weights.

Everything a hackathon judge might ask "why this number?" about lives here,
so it can be changed in one place and defended in one slide.
"""
from dataclasses import dataclass, field

SEED = 42

# --------------------------------------------------------------------------
# Business model
# --------------------------------------------------------------------------
# Potential class -> (visits per month, hours per visit incl. prep/admin,
#                     mean annual revenue potential in INR lakh, spread)
POTENTIAL_CLASSES = {
    "A": dict(visits=4.0, hours=1.50, mean_lakh=14.0, sigma=0.35, share=0.10),
    "B": dict(visits=2.0, hours=1.25, mean_lakh=6.0,  sigma=0.35, share=0.25),
    "C": dict(visits=1.0, hours=1.00, mean_lakh=2.5,  sigma=0.40, share=0.40),
    "D": dict(visits=0.5, hours=0.75, mean_lakh=0.8,  sigma=0.45, share=0.25),
}

# Employee grade -> (monthly productive field hours, target-potential weight)
GRADES = {
    "Senior": dict(capacity_hours=150.0, potential_weight=1.30),
    "Mid":    dict(capacity_hours=140.0, potential_weight=1.00),
    "Junior": dict(capacity_hours=125.0, potential_weight=0.75),
}

ROAD_CIRCUITY = 1.30          # road km = straight-line km * 1.30
BEAT_CYCLES_MAX = 4           # a territory is swept at most weekly (4x / month)

# Utilisation thresholds used for automatic imbalance detection
OVERLOAD_UTIL = 1.10
UNDERLOAD_UTIL = 0.80
HIGH_POT_RATIO = 1.15         # potential / target potential
LOW_POT_RATIO = 0.85
GEO_INEFFICIENT_TRAVEL_SHARE = 0.30   # travel hours / total hours

# --------------------------------------------------------------------------
# Geography: one Area-Manager region per metro cluster (synthetic PIN grid
# around real city centres; plug a real PIN CSV in via data_gen.load_real_pins)
# --------------------------------------------------------------------------
REGIONS = [
    # name, AM id, lat, lon, spread_km, PIN prefix, urban speed km/h, employees
    dict(city="Delhi NCR",  am="AM01", lat=28.6139, lon=77.2090, spread_km=17, pin_prefix="110", speed=24, n_emp=8),
    dict(city="Mumbai",     am="AM02", lat=19.0760, lon=72.8777, spread_km=15, pin_prefix="400", speed=20, n_emp=8),
    dict(city="Bengaluru",  am="AM03", lat=12.9716, lon=77.5946, spread_km=14, pin_prefix="560", speed=22, n_emp=7),
    dict(city="Hyderabad",  am="AM04", lat=17.3850, lon=78.4867, spread_km=14, pin_prefix="500", speed=26, n_emp=7),
    dict(city="Chennai",    am="AM05", lat=13.0827, lon=80.2707, spread_km=13, pin_prefix="600", speed=26, n_emp=6),
    dict(city="Kolkata",    am="AM06", lat=22.5726, lon=88.3639, spread_km=13, pin_prefix="700", speed=21, n_emp=6),
]
PINS_PER_REGION = 100
CUSTOMERS_PER_EMP = 54        # average legacy book size (-> ~90 % network utilisation)

# --------------------------------------------------------------------------
# Optimiser
# --------------------------------------------------------------------------
@dataclass
class Weights:
    """Objective  J = w_wl*WorkloadDev + w_pot*PotentialDev + w_geo*Compactness
                      + w_cont*RelationshipLoss + w_cap*OverCapacity"""
    w_wl: float = 30.0      # workload deviation from capacity-proportional ideal
    w_pot: float = 12.0     # potential deviation from target-proportional ideal
    w_geo: float = 6.0      # territory compactness (workload-weighted distance)
    w_cont: float = 16.0    # relationship continuity lost (relationship-weighted)
    w_cap: float = 40.0     # hinge penalty above 100 % utilisation
    max_sweeps: int = 60
    n_restarts: int = 1

PRESETS = {
    "Max continuity":     Weights(w_cont=60.0),
    "Balanced (default)": Weights(),
    "Max balance":        Weights(w_cont=2.0, w_wl=45.0, w_pot=20.0),
}

# How much we care about keeping a relationship (continuity cost weight)
RELATION_CLASS_WEIGHT = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.5}
RELATION_TENURE_CAP_YEARS = 8.0     # tenure bonus saturates at 8 years

```

---

## `geo.py`  (126 lines)

```python
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

```

---

## `data_gen.py`  (236 lines)

```python
"""Synthetic-but-realistic India dataset + loaders for real data.

Produces four tables:
  pins       one row per PIN code      (pin, region, am, lat, lon, area_name)
  employees  field force               (emp_id, name, am, grade, capacity, home...)
  customers  customer book             (cust_id, pin, lat, lon, class, potential,
                                        visits_pm, hours_pm, legacy_emp, years)
  managers   Area-Manager / NM roster

The *legacy* alignment is deliberately messy so the optimiser has real
problems to fix: skewed books, overlapping territories (same PIN served by
several reps), orphaned customers (rep left -> 'VACANT') and scattered
relationships.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

import config as C
from geo import project_km, pairwise_km

FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Karan", "Rahul", "Amit", "Sanjay", "Vikram",
         "Neha", "Priya", "Anjali", "Pooja", "Sneha", "Kavita", "Meera", "Divya", "Isha", "Riya",
         "Imran", "Farhan", "Sameer", "Deepak", "Manish", "Suresh", "Ramesh", "Anil", "Nitin", "Varun"]
LAST = ["Sharma", "Verma", "Gupta", "Iyer", "Nair", "Reddy", "Rao", "Das", "Banerjee", "Chatterjee",
        "Patil", "Desai", "Shah", "Mehta", "Khan", "Singh", "Kapoor", "Joshi", "Menon", "Pillai"]

DATA_DIR = Path(__file__).parent / "data"


def _make_pins(rng, region) -> pd.DataFrame:
    n = C.PINS_PER_REGION
    lat0, lon0, spread = region["lat"], region["lon"], region["spread_km"]
    # 65 % in a dense core, 35 % around 3 satellite sub-centres
    centres = [np.zeros(2)] + [rng.normal(0, spread * 0.55, 2) for _ in range(3)]
    pts = []
    tries = 0
    while len(pts) < n and tries < 20000:
        tries += 1
        if rng.random() < 0.65:
            p = rng.normal(0, spread * 0.42, 2)
        else:
            p = centres[rng.integers(1, 4)] + rng.normal(0, spread * 0.22, 2)
        if np.linalg.norm(p) > spread * 1.6:
            continue
        if pts and np.min(np.linalg.norm(np.array(pts) - p, axis=1)) < 0.9:  # >= 0.9 km apart
            continue
        pts.append(p)
    pts = np.array(pts)
    lat = lat0 + pts[:, 1] / 110.574
    lon = lon0 + pts[:, 0] / (111.320 * np.cos(np.radians(lat0)))
    df = pd.DataFrame({"lat": lat, "lon": lon})
    df = df.sort_values(["lat", "lon"]).reset_index(drop=True)
    df["pin"] = [f"{region['pin_prefix']}{i + 1:03d}" for i in range(len(df))]
    df["region"] = region["city"]
    df["am"] = region["am"]
    df["area_name"] = [f"{region['city']} Zone-{i + 1:03d}" for i in range(len(df))]
    xy = project_km(df.lat, df.lon, lat0, lon0)
    df["dist_center_km"] = np.linalg.norm(xy, axis=1)
    return df[["pin", "region", "am", "area_name", "lat", "lon", "dist_center_km"]]


def generate(seed: int = C.SEED, customers_per_emp: int = C.CUSTOMERS_PER_EMP,
             save: bool = True) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pins_l, emps_l, cust_l = [], [], []
    emp_counter = 1
    cust_counter = 1

    for reg in C.REGIONS:
        pins = _make_pins(rng, reg)
        pins_l.append(pins)
        xy_pin = project_km(pins.lat, pins.lon, reg["lat"], reg["lon"])

        # ------------------------------------------------ employees
        n_emp = reg["n_emp"]
        grades = rng.choice(list(C.GRADES), size=n_emp, p=[0.25, 0.50, 0.25])
        home_idx = rng.choice(len(pins), size=n_emp, replace=False)
        emp_rows = []
        for k in range(n_emp):
            g = grades[k]
            hp = pins.iloc[home_idx[k]]
            emp_rows.append(dict(
                emp_id=f"E{emp_counter:03d}",
                name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                am=reg["am"], region=reg["city"], grade=g,
                capacity_hours=C.GRADES[g]["capacity_hours"],
                potential_weight=C.GRADES[g]["potential_weight"],
                home_pin=hp.pin, home_lat=hp.lat, home_lon=hp.lon,
                speed_kmph=reg["speed"]))
            emp_counter += 1
        emps = pd.DataFrame(emp_rows)
        emps_l.append(emps)

        # ------------------------------------------------ customers per PIN
        n_total = n_emp * customers_per_emp
        # denser near the centre (business districts)
        lam = np.exp(-pins.dist_center_km.values / (reg["spread_km"] * 0.8)) + 0.15
        lam = lam / lam.sum() * n_total
        counts = rng.poisson(lam)
        pin_idx = np.repeat(np.arange(len(pins)), counts)
        n_c = len(pin_idx)
        jitter = rng.normal(0, 0.45, (n_c, 2))                     # km around PIN centroid
        cxy = xy_pin[pin_idx] + jitter
        clat = reg["lat"] + cxy[:, 1] / 110.574
        clon = reg["lon"] + cxy[:, 0] / (111.320 * np.cos(np.radians(reg["lat"])))
        # class probabilities tilt to A/B near the centre
        dc = pins.dist_center_km.values[pin_idx]
        tilt = np.exp(-dc / (reg["spread_km"] * 0.6))
        base = np.array([C.POTENTIAL_CLASSES[c]["share"] for c in "ABCD"])
        classes = []
        for t in tilt:
            p = base * np.array([1 + 1.6 * t, 1 + 0.6 * t, 1.0, 1 - 0.4 * t])
            p = np.clip(p, 1e-6, None); p /= p.sum()
            classes.append(rng.choice(list("ABCD"), p=p))
        classes = np.array(classes)
        pot = np.array([rng.lognormal(np.log(C.POTENTIAL_CLASSES[c]["mean_lakh"]),
                                      C.POTENTIAL_CLASSES[c]["sigma"]) for c in classes])
        cust = pd.DataFrame({
            "cust_id": [f"C{cust_counter + i:05d}" for i in range(n_c)],
            "pin": pins.pin.values[pin_idx], "region": reg["city"], "am": reg["am"],
            "lat": clat, "lon": clon, "pot_class": classes,
            "potential_lakh": np.round(pot, 2),
            "visits_pm": [C.POTENTIAL_CLASSES[c]["visits"] for c in classes],
            "hours_per_visit": [C.POTENTIAL_CLASSES[c]["hours"] for c in classes],
        })
        cust_counter += n_c

        # ------------------------------------------------ messy legacy alignment
        exy = project_km(emps.home_lat, emps.home_lon, reg["lat"], reg["lon"])
        D = pairwise_km(cxy, exy)                                  # (customers, emps)
        bias = rng.normal(0, 0.35, n_emp)                          # some reps historically "own" more
        bias[rng.integers(n_emp)] += 0.55                          # one empire-builder
        bias[rng.integers(n_emp)] -= 0.55                          # one starved rep
        score = -D / 7.0 + bias[None, :]
        prob = np.exp(score - score.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)
        pick = np.array([rng.choice(n_emp, p=p) for p in prob])
        cust["legacy_emp"] = emps.emp_id.values[pick]
        # departed rep -> orphaned book (coverage gap)
        if rng.random() < 0.85:
            seed_pin = rng.integers(len(pins))
            d_seed = np.linalg.norm(cxy - xy_pin[seed_pin], axis=1)
            orphan_n = int(rng.integers(18, 34))
            cust.loc[np.argsort(d_seed)[:orphan_n], "legacy_emp"] = "VACANT"
        # relationship tenure (years); orphans keep tenure but have no live owner
        cust["relationship_years"] = np.round(rng.gamma(2.0, 1.8, n_c), 1)
        cust_l.append(cust)

    pins_all = pd.concat(pins_l, ignore_index=True)
    emps_all = pd.concat(emps_l, ignore_index=True)
    cust_all = pd.concat(cust_l, ignore_index=True)

    managers = pd.DataFrame(
        [dict(mgr_id=r["am"], role="Area Manager", region=r["city"], reports_to="NM01",
              name=f"{rng.choice(FIRST)} {rng.choice(LAST)}") for r in C.REGIONS] +
        [dict(mgr_id="NM01", role="National Manager", region="India", reports_to="",
              name="National Head")])

    out = dict(pins=pins_all, employees=emps_all, customers=cust_all, managers=managers)
    if save:
        DATA_DIR.mkdir(exist_ok=True)
        for k, v in out.items():
            v.to_csv(DATA_DIR / f"{k}.csv", index=False)
    return out


def load(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the 4 CSVs (generate them first if absent)."""
    if not (Path(data_dir) / "customers.csv").exists():
        return generate()
    return {k: pd.read_csv(Path(data_dir) / f"{k}.csv", dtype={"pin": str, "home_pin": str})
            for k in ("pins", "employees", "customers", "managers")}


# ------------------------------------------------------------------ real data
def load_real_pins(csv_path: str, state: str | None = None, city_latlon=None,
                   radius_km: float = 40.0) -> pd.DataFrame:
    """Load a real India PIN dataset (e.g. data.gov.in 'All India Pincode
    Directory' or the Kaggle/GitHub PIN-geocode CSVs).

    Required columns (case-insensitive): pincode, latitude, longitude.
    Optional: officename/areaname, statename, district.
    Rows with missing / (0,0) / outside-India coordinates are dropped and PINs
    are de-duplicated to one centroid each (median of office coordinates)."""
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"pincode", "latitude", "longitude"}
    if not need <= set(df.columns):
        raise ValueError(f"PIN file must contain columns {need}; found {list(df.columns)}")
    df["lat"] = pd.to_numeric(df.latitude, errors="coerce")
    df["lon"] = pd.to_numeric(df.longitude, errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df.lat.between(6, 37.5)) & (df.lon.between(68, 98))]     # India bounding box
    if state and "statename" in df.columns:
        df = df[df.statename.str.lower() == state.lower()]
    df["pin"] = df.pincode.str.strip().str.zfill(6)
    g = df.groupby("pin").agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
    if city_latlon is not None:
        from geo import haversine_km
        d = haversine_km(g.lat, g.lon, city_latlon[0], city_latlon[1])
        g = g[d <= radius_km]
    g["area_name"] = g.pin
    return g.reset_index(drop=True)


def load_customers_csv(csv_path: str) -> pd.DataFrame:
    """Customer file contract (see README): cust_id, pin, lat, lon, pot_class,
    potential_lakh, legacy_emp, relationship_years.  Missing optional columns
    are filled with defaults so a partial client file still runs."""
    df = pd.read_csv(csv_path, dtype={"pin": str, "cust_id": str})
    need = {"cust_id", "pin", "lat", "lon", "pot_class"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"customers file is missing columns: {sorted(miss)}")
    df["pot_class"] = df["pot_class"].str.upper().str.strip()
    bad = ~df.pot_class.isin(C.POTENTIAL_CLASSES)
    if bad.any():
        raise ValueError(f"{bad.sum()} rows have pot_class outside A/B/C/D")
    df["visits_pm"] = df.pot_class.map(lambda c: C.POTENTIAL_CLASSES[c]["visits"])
    df["hours_per_visit"] = df.pot_class.map(lambda c: C.POTENTIAL_CLASSES[c]["hours"])
    if "potential_lakh" not in df:
        df["potential_lakh"] = df.pot_class.map(lambda c: C.POTENTIAL_CLASSES[c]["mean_lakh"])
    if "legacy_emp" not in df:
        df["legacy_emp"] = "VACANT"
    df["legacy_emp"] = df.legacy_emp.fillna("VACANT")
    if "relationship_years" not in df:
        df["relationship_years"] = 1.0
    return df


if __name__ == "__main__":
    d = generate()
    for k, v in d.items():
        print(k, v.shape)

```

---

## `workload.py`  (170 lines)

```python
"""Ground-truth evaluation engine.

Given ANY customer->employee assignment (legacy, baseline or optimised) this
module computes the workload / potential / travel / geography metrics with the
same formulas, so comparisons are apples-to-apples.

Workload (hours / month) =  sum(visits * hours_per_visit)   [visit effort]
                          + travel_hours                    [beat-tour effort]

Travel uses a *frequency-layered beat plan*: cycle k (k=1..4) is a closed tour
from the rep's home through every PIN that needs >= k visits a month.
Tour length = NN + 2-opt on PIN centroids, x road-circuity, / city speed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C
from geo import project_km, tour_length_km, delaunay_adjacency, n_components


def relationship_weight(pot_class: pd.Series, years: pd.Series) -> pd.Series:
    cw = pot_class.map(C.RELATION_CLASS_WEIGHT).astype(float)
    tenure = 1.0 + np.minimum(years.astype(float), C.RELATION_TENURE_CAP_YEARS) / 4.0
    return cw * tenure


def add_customer_fields(cust: pd.DataFrame) -> pd.DataFrame:
    c = cust.copy()
    c["visit_hours"] = c.visits_pm * c.hours_per_visit
    c["rel_weight"] = relationship_weight(c.pot_class, c.relationship_years)
    return c


def pin_table(cust: pd.DataFrame, pins: pd.DataFrame) -> pd.DataFrame:
    """Aggregate customers to PIN level (the atomic unit of a territory)."""
    c = add_customer_fields(cust) if "visit_hours" not in cust else cust
    g = c.groupby("pin").agg(n_cust=("cust_id", "size"), visit_hours=("visit_hours", "sum"),
                             potential_lakh=("potential_lakh", "sum"),
                             stop_freq=("visits_pm", lambda s: int(min(C.BEAT_CYCLES_MAX,
                                                                        np.ceil(s.max())))))
    t = pins.merge(g, on="pin", how="left")
    t[["n_cust", "visit_hours", "potential_lakh", "stop_freq"]] = \
        t[["n_cust", "visit_hours", "potential_lakh", "stop_freq"]].fillna(0)
    t["n_cust"] = t.n_cust.astype(int)
    t["stop_freq"] = t.stop_freq.astype(int)
    return t


def _beat_travel_km(emp_home_xy, stop_xy, stop_freq) -> float:
    """Frequency-layered beat plan -> monthly straight-line km."""
    if len(stop_xy) == 0:
        return 0.0
    total = 0.0
    for k in range(1, C.BEAT_CYCLES_MAX + 1):
        sel = stop_freq >= k
        if sel.any():
            total += tour_length_km(stop_xy[sel], emp_home_xy)
    return total


def employee_metrics(cust: pd.DataFrame, emps: pd.DataFrame, pins: pd.DataFrame,
                     owner_col: str, pin_owner: dict | None = None) -> pd.DataFrame:
    """Per-employee metrics for the assignment stored in cust[owner_col]
    (NaN / 'VACANT' / unknown id = unassigned)."""
    c = add_customer_fields(cust) if "visit_hours" not in cust else cust.copy()
    c["_owner"] = c[owner_col].where(c[owner_col].isin(emps.emp_id), other=np.nan)
    pins_idx = pins.set_index("pin")
    rows = []
    region_total_pot = c.groupby("region").potential_lakh.sum()

    # adjacency per region (for fragmentation count)
    adj_cache = {}
    for reg, pr in pins.groupby("region"):
        pr = pr.reset_index(drop=True)
        xy = project_km(pr.lat, pr.lon)
        adj_cache[reg] = (list(pr.pin), delaunay_adjacency(xy))

    # which PINs are shared by >1 employee (overlap)
    pin_owner_n = c.dropna(subset=["_owner"]).groupby("pin")._owner.nunique()

    for reg, er in emps.groupby("region"):
        wsum = er.potential_weight.sum()
        for _, e in er.iterrows():
            ce = c[c._owner == e.emp_id]
            n = len(ce)
            reg_pins = pins[pins.region == reg]
            lat0, lon0 = reg_pins.lat.mean(), reg_pins.lon.mean()
            home_xy = project_km([e.home_lat], [e.home_lon], lat0, lon0)[0]
            if n:
                pt = pins_idx.loc[ce.pin.unique(), ["lat", "lon"]]
                stop_xy = project_km(pt.lat, pt.lon, lat0, lon0)
                f = ce.groupby("pin").visits_pm.max().reindex(pt.index).apply(
                    lambda v: min(C.BEAT_CYCLES_MAX, int(np.ceil(v)))).values
                trav_km = _beat_travel_km(home_xy, stop_xy, f) * C.ROAD_CIRCUITY
                cxy = project_km(ce.lat, ce.lon, lat0, lon0)
                centroid = np.average(cxy, axis=0, weights=ce.visit_hours.clip(lower=1e-6))
                avg_d = float(np.average(np.linalg.norm(cxy - centroid, axis=1),
                                         weights=ce.visit_hours.clip(lower=1e-6)))
                home_d = float(np.linalg.norm(home_xy - centroid))
                pin_list = list(pt.index)
                pins_ord, adj = adj_cache[reg]
                pos = {p: i for i, p in enumerate(pins_ord)}
                frag_pins = [p for p, o in pin_owner.items() if o == e.emp_id] if pin_owner else pin_list
                frags = n_components([pos[p] for p in frag_pins if p in pos], adj)
                overlap = int(sum(pin_owner_n.get(p, 1) > 1 for p in pin_list))
            else:
                trav_km, avg_d, home_d, frags, overlap, pin_list = 0.0, 0.0, 0.0, 0, 0, []
            vis_h = float(ce.visit_hours.sum())
            trav_h = trav_km / e.speed_kmph
            work = vis_h + trav_h
            target_pot = region_total_pot.get(reg, 0.0) * e.potential_weight / wsum
            pot = float(ce.potential_lakh.sum())
            rows.append(dict(
                emp_id=e.emp_id, name=e["name"], am=e.am, region=reg, grade=e.grade,
                capacity_hours=e.capacity_hours,
                n_customers=n, n_A=int((ce.pot_class == "A").sum()), n_B=int((ce.pot_class == "B").sum()),
                n_C=int((ce.pot_class == "C").sum()), n_D=int((ce.pot_class == "D").sum()),
                n_pins=len(pin_list), visit_hours=round(vis_h, 1), travel_km=round(trav_km, 1),
                travel_hours=round(trav_h, 1), workload_hours=round(work, 1),
                utilization=round(work / e.capacity_hours, 3),
                potential_lakh=round(pot, 1), target_potential_lakh=round(target_pot, 1),
                potential_ratio=round(pot / target_pot, 3) if target_pot else np.nan,
                potential_per_hour=round(pot / work, 2) if work else 0.0,
                avg_cust_dist_km=round(avg_d, 2), home_to_centroid_km=round(home_d, 2),
                fragments=frags, overlap_pins=overlap))
    m = pd.DataFrame(rows)
    m["flags"] = m.apply(flag_row, axis=1)
    return m


def flag_row(r) -> str:
    f = []
    if r.utilization > C.OVERLOAD_UTIL: f.append("OVERLOADED")
    if r.utilization < C.UNDERLOAD_UTIL: f.append("UNDERLOADED")
    if r.potential_ratio > C.HIGH_POT_RATIO: f.append("HIGH_POTENTIAL")
    if r.potential_ratio < C.LOW_POT_RATIO: f.append("LOW_POTENTIAL")
    if r.workload_hours and r.travel_hours / r.workload_hours > C.GEO_INEFFICIENT_TRAVEL_SHARE:
        f.append("GEO_INEFFICIENT")
    if r.fragments > 1: f.append("FRAGMENTED")
    if r.overlap_pins > 0: f.append("OVERLAP")
    return ",".join(f)


def summary_kpis(m: pd.DataFrame, cust: pd.DataFrame, owner_col: str,
                 legacy_col: str = "legacy_emp") -> dict:
    """Headline numbers used for the baseline comparison table."""
    n_total = len(cust)
    assigned = cust[owner_col].isin(m.emp_id)
    live_legacy = cust[legacy_col].isin(m.emp_id)
    kept = (cust[owner_col] == cust[legacy_col]) & live_legacy & assigned
    rw = cust.rel_weight if "rel_weight" in cust else add_customer_fields(cust).rel_weight
    owners_per_pin = cust[assigned].groupby("pin")[owner_col].nunique()
    return {
        "Utilisation std-dev": float(m.utilization.std(ddof=0)),
        "Utilisation min": float(m.utilization.min()),
        "Utilisation max": float(m.utilization.max()),
        "Overloaded reps (>110%)": int((m.utilization > C.OVERLOAD_UTIL).sum()),
        "Underloaded reps (<80%)": int((m.utilization < C.UNDERLOAD_UTIL).sum()),
        "Potential-ratio std-dev": float(m.potential_ratio.std(ddof=0)),
        "Total travel (km/month)": float(m.travel_km.sum()),
        "Avg travel share of workload": float((m.travel_hours.sum() / m.workload_hours.sum())),
        "Avg customer-to-centroid km": float(np.average(m.avg_cust_dist_km, weights=m.n_customers.clip(lower=1))),
        "PIN codes shared by >1 rep": int((owners_per_pin > 1).sum()),
        "Fragmented territories": int((m.fragments > 1).sum()),
        "Unassigned customers (coverage gap)": int((~assigned).sum()),
        "Customers keeping same rep (%)": float(100 * kept.sum() / max(1, live_legacy.sum())),
        "Relationship-weight kept (%)": float(100 * rw[kept].sum() / max(1e-9, rw[live_legacy].sum())),
        "Customers total": n_total,
    }

```

---

## `optimizer.py`  (362 lines)

```python
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

```

---

## `continuity.py`  (80 lines)

```python
"""Key-account relationship exceptions.

PIN-level territories are clean (no overlap, contiguous) but cap relationship
continuity (~50 % on our data) because legacy books are scattered.  This layer
lets high-value customers (A/B class by default) STAY with their existing rep
when that is cheap:

  * the legacy rep is still active
  * customer is within `max_km` of the rep's (new) territory
  * rep still has spare capacity after absorbing the visit + detour travel
  * the donor rep is not pushed below `min_donor_util`

Greedy in descending relationship weight.  Each exception is logged with a
reason so it can be shown / audited.  Set enabled=False for pure territories.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C
from geo import project_km


def apply_exceptions(cust: pd.DataFrame, emps: pd.DataFrame, pins: pd.DataFrame,
                     metrics: pd.DataFrame, owner_col: str = "new_emp", out_col: str = "final_emp",
                     classes=("A", "B"), max_km: float = 4.0, util_cap: float | None = None,
                     min_donor_util: float = 0.85, max_exceptions_per_rep: int = 8,
                     enabled: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = cust.copy()
    c[out_col] = c[owner_col]
    c["is_exception"] = False
    log = []
    if not enabled:
        return c, pd.DataFrame(log)

    util = metrics.set_index("emp_id").utilization.to_dict()
    cap = emps.set_index("emp_id").capacity_hours.to_dict()
    speed = emps.set_index("emp_id").speed_kmph.to_dict()
    mean_util = float(metrics.workload_hours.sum() / metrics.capacity_hours.sum())
    util_cap = util_cap if util_cap is not None else min(1.0, mean_util + 0.07)
    live = set(emps.emp_id)
    n_exc = {e: 0 for e in live}

    # PIN coordinates (km) per owner, from the pin-level assignment
    pin_xy = {}
    for reg, pr in pins.groupby("region"):
        lat0, lon0 = pr.lat.mean(), pr.lon.mean()
        xy = project_km(pr.lat, pr.lon, lat0, lon0)
        for p, v in zip(pr.pin, xy):
            pin_xy[p] = (reg, v)
    owner_pins = c.groupby(owner_col).pin.unique().to_dict()

    cand = c[(c.legacy_emp.isin(live)) & (c.legacy_emp != c[owner_col]) &
             (c.pot_class.isin(classes))].sort_values("rel_weight", ascending=False)
    for idx, r in cand.iterrows():
        rep, donor = r.legacy_emp, r[owner_col]
        if n_exc[rep] >= max_exceptions_per_rep or pd.isna(donor):
            continue
        rep_pins = [p for p in owner_pins.get(rep, []) if pin_xy[p][0] == r.region]
        if not rep_pins:
            continue
        lat0 = pins[pins.region == r.region].lat.mean(); lon0 = pins[pins.region == r.region].lon.mean()
        cxy = project_km([r.lat], [r.lon], lat0, lon0)[0]
        d_min = float(min(np.linalg.norm(pin_xy[p][1] - cxy) for p in rep_pins))
        if d_min > max_km:
            continue
        add_h = r.visit_hours + r.visits_pm * 2 * d_min * C.ROAD_CIRCUITY / speed[rep]
        new_rep_util = util[rep] + add_h / cap[rep]
        new_donor_util = util[donor] - r.visit_hours / cap[donor]
        if new_rep_util > util_cap or new_donor_util < min_donor_util:
            continue
        util[rep], util[donor] = new_rep_util, new_donor_util
        n_exc[rep] += 1
        c.loc[idx, out_col] = rep
        c.loc[idx, "is_exception"] = True
        log.append(dict(cust_id=r.cust_id, pin=r.pin, pot_class=r.pot_class,
                        relationship_years=r.relationship_years, kept_with=rep, territory_owner=donor,
                        dist_to_rep_territory_km=round(d_min, 2),
                        rep_util_after=round(new_rep_util, 3), donor_util_after=round(new_donor_util, 3)))
    return c, pd.DataFrame(log)

```

---

## `baselines.py`  (44 lines)

```python
"""Simple baselines the optimiser must beat (customer-level assignments).

B1  Nearest-home   : every customer goes to the closest rep's home (Voronoi).
B2  K-means        : geographic k-means on customers, clusters matched to reps
                     by Hungarian assignment on distance to home.  Compact but
                     ignores workload, potential, capacity and relationships.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from geo import project_km, pairwise_km


def nearest_home(cust: pd.DataFrame, emps: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=cust.index, dtype=object)
    for reg, er in emps.groupby("region"):
        idx = cust.index[cust.region == reg]
        if len(idx) == 0:
            continue
        lat0, lon0 = cust.loc[idx, "lat"].mean(), cust.loc[idx, "lon"].mean()
        cxy = project_km(cust.loc[idx, "lat"], cust.loc[idx, "lon"], lat0, lon0)
        exy = project_km(er.home_lat, er.home_lon, lat0, lon0)
        out.loc[idx] = er.emp_id.values[pairwise_km(cxy, exy).argmin(1)]
    return out


def kmeans_baseline(cust: pd.DataFrame, emps: pd.DataFrame, seed: int = 0) -> pd.Series:
    out = pd.Series(index=cust.index, dtype=object)
    for reg, er in emps.groupby("region"):
        idx = cust.index[cust.region == reg]
        if len(idx) == 0:
            continue
        lat0, lon0 = cust.loc[idx, "lat"].mean(), cust.loc[idx, "lon"].mean()
        cxy = project_km(cust.loc[idx, "lat"], cust.loc[idx, "lon"], lat0, lon0)
        exy = project_km(er.home_lat, er.home_lon, lat0, lon0)
        km = KMeans(n_clusters=len(er), n_init=10, random_state=seed).fit(cxy)
        cost = pairwise_km(km.cluster_centers_, exy)
        ci, ei = linear_sum_assignment(cost)
        mapping = {int(c): er.emp_id.values[e] for c, e in zip(ci, ei)}
        out.loc[idx] = [mapping[int(l)] for l in km.labels_]
    return out

```

---

## `pipeline.py`  (71 lines)

```python
"""Orchestration: data -> optimise -> exceptions -> metrics, with warm-start
support for what-if / dynamic re-alignment."""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

import config as C
import workload as W
import optimizer as O
import continuity as K
import baselines as B


@dataclass
class Alignment:
    pins: pd.DataFrame
    emps: pd.DataFrame
    cust: pd.DataFrame            # has legacy_emp, new_emp (pin-level), final_emp (after exceptions)
    pin_owner: dict
    metrics: pd.DataFrame         # per employee, on final_emp
    metrics_pin_level: pd.DataFrame
    exceptions: pd.DataFrame
    weights: C.Weights
    infos: list = field(default_factory=list)
    problems: dict = field(default_factory=dict)


def align(pins, emps, cust, weights: C.Weights | None = None, exceptions: bool = True,
          init_owner: dict | None = None, regions: list[str] | None = None,
          base: Alignment | None = None, seed: int = C.SEED, max_sweeps: int | None = None,
          exception_kwargs: dict | None = None) -> Alignment:
    weights = weights or C.Weights()
    c = W.add_customer_fields(cust) if "visit_hours" not in cust else cust.copy()
    po_df, infos, probs = O.solve_all(pins, c, emps, weights, seed, init_owner, regions, max_sweeps)
    pin_owner = dict(base.pin_owner) if (base is not None and regions) else {}
    pin_owner.update(dict(zip(po_df.pin, po_df.emp_id)))
    c["new_emp"] = c.pin.map(pin_owner)
    m_pin = W.employee_metrics(c, emps, pins, "new_emp", pin_owner)
    c, exc_log = K.apply_exceptions(c, emps, pins, m_pin, enabled=exceptions, **(exception_kwargs or {}))
    m = W.employee_metrics(c, emps, pins, "final_emp", pin_owner)
    return Alignment(pins, emps, c, pin_owner, m, m_pin, exc_log, weights, infos, probs)


def compare_methods(al: Alignment) -> tuple[pd.DataFrame, dict]:
    """Legacy vs baselines vs optimised, all scored by the same engine."""
    c = al.cust.copy()
    c["b_nearest"] = B.nearest_home(c, al.emps)
    c["b_kmeans"] = B.kmeans_baseline(c, al.emps)
    cols = {"Legacy (as-is)": "legacy_emp", "B1 Nearest-home": "b_nearest",
            "B2 K-means": "b_kmeans", "Optimised (ours)": "final_emp"}
    rows, mets = {}, {}
    for name, col in cols.items():
        m = al.metrics if col == "final_emp" else W.employee_metrics(c, al.emps, al.pins, col)
        mets[name] = m
        rows[name] = W.summary_kpis(m, c, col)
    return pd.DataFrame(rows), mets


def tradeoff_sweep(pins, emps, cust, w_cont_values=(0, 2, 4, 8, 16, 30, 60), exceptions: bool = True) -> pd.DataFrame:
    """Pareto data: how much balance do we give up for each step of continuity?"""
    from dataclasses import replace
    rows = []
    for wc in w_cont_values:
        al = align(pins, emps, cust, replace(C.Weights(), w_cont=float(wc)), exceptions=exceptions)
        k = W.summary_kpis(al.metrics, al.cust, "final_emp")
        rows.append(dict(w_cont=wc, util_std=k["Utilisation std-dev"], util_max=k["Utilisation max"],
                         util_min=k["Utilisation min"], potential_ratio_std=k["Potential-ratio std-dev"],
                         travel_km=k["Total travel (km/month)"], customers_kept_pct=k["Customers keeping same rep (%)"],
                         relationship_kept_pct=k["Relationship-weight kept (%)"],
                         shared_pins=k["PIN codes shared by >1 rep"]))
    return pd.DataFrame(rows)

```

---

## `metrics.py`  (74 lines)

```python
"""Manager roll-ups (Employee -> Area Manager -> National Manager) and
automatic imbalance detection."""
from __future__ import annotations
import numpy as np
import pandas as pd
import config as C


def area_manager_view(m: pd.DataFrame, cust: pd.DataFrame, owner_col: str, managers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for am, g in m.groupby("am"):
        cc = cust[cust.am == am]
        gap = int((~cc[owner_col].isin(m.emp_id)).sum())
        mg = managers[managers.mgr_id == am]
        rows.append(dict(
            am=am, am_name=mg.name.iloc[0] if len(mg) else am, region=g.region.iloc[0],
            reps=len(g), customers=int(g.n_customers.sum()), unassigned_customers=gap,
            pins=int(g.n_pins.sum()), potential_lakh=round(g.potential_lakh.sum(), 1),
            workload_hours=round(g.workload_hours.sum(), 1), capacity_hours=round(g.capacity_hours.sum(), 1),
            utilization=round(g.workload_hours.sum() / g.capacity_hours.sum(), 3),
            util_std=round(g.utilization.std(ddof=0), 3),
            potential_per_rep=round(g.potential_lakh.mean(), 1),
            potential_ratio_std=round(g.potential_ratio.std(ddof=0), 3),
            overloaded=int((g.utilization > C.OVERLOAD_UTIL).sum()),
            underloaded=int((g.utilization < C.UNDERLOAD_UTIL).sum()),
            travel_km=round(g.travel_km.sum(), 0)))
    return pd.DataFrame(rows)


def national_view(am: pd.DataFrame) -> dict:
    tot_w, tot_c = am.workload_hours.sum(), am.capacity_hours.sum()
    return {
        "reps": int(am.reps.sum()), "area_managers": int(len(am)), "customers": int(am.customers.sum()),
        "unassigned_customers": int(am.unassigned_customers.sum()),
        "potential_lakh": float(round(am.potential_lakh.sum(), 1)),
        "network_utilization": float(round(tot_w / tot_c, 3)),
        "util_spread_across_AMs": float(round(am.utilization.std(ddof=0), 3)),
        "potential_per_rep_spread_across_AMs": float(round(am.potential_per_rep.std(ddof=0), 1)),
        "overloaded_reps": int(am.overloaded.sum()), "underloaded_reps": int(am.underloaded.sum()),
        "travel_km": int(am.travel_km.sum()),
    }


def detect_imbalance(m: pd.DataFrame, am: pd.DataFrame) -> list[dict]:
    """Human-readable alerts, worst first."""
    out = []
    for _, r in m.iterrows():
        if r.utilization > C.OVERLOAD_UTIL:
            out.append(dict(level="rep", id=r.emp_id, severity=r.utilization,
                            msg=f"{r.emp_id} ({r['name']}) is overloaded at {r.utilization:.0%} of capacity."))
        if r.utilization < C.UNDERLOAD_UTIL:
            out.append(dict(level="rep", id=r.emp_id, severity=1 - r.utilization,
                            msg=f"{r.emp_id} ({r['name']}) is underloaded at {r.utilization:.0%} - spare capacity."))
        if r.potential_ratio > C.HIGH_POT_RATIO:
            out.append(dict(level="rep", id=r.emp_id, severity=r.potential_ratio - 1,
                            msg=f"{r.emp_id} holds {r.potential_ratio:.0%} of their fair potential share (high-potential territory)."))
        if r.potential_ratio < C.LOW_POT_RATIO:
            out.append(dict(level="rep", id=r.emp_id, severity=1 - r.potential_ratio,
                            msg=f"{r.emp_id} holds only {r.potential_ratio:.0%} of their fair potential share (low-potential territory)."))
        if r.workload_hours and r.travel_hours / r.workload_hours > C.GEO_INEFFICIENT_TRAVEL_SHARE:
            out.append(dict(level="rep", id=r.emp_id, severity=r.travel_hours / r.workload_hours,
                            msg=f"{r.emp_id} spends {r.travel_hours / r.workload_hours:.0%} of time travelling (geographically inefficient)."))
        if r.fragments > 1:
            out.append(dict(level="rep", id=r.emp_id, severity=r.fragments,
                            msg=f"{r.emp_id}'s territory is split into {r.fragments} disconnected pieces."))
    net = am.workload_hours.sum() / am.capacity_hours.sum()
    for _, r in am.iterrows():
        if abs(r.utilization - net) > 0.10:
            out.append(dict(level="area_manager", id=r.am, severity=abs(r.utilization - net),
                            msg=f"{r.am} ({r.region}) runs at {r.utilization:.0%} vs network {net:.0%} - consider moving headcount."))
        if r.unassigned_customers:
            out.append(dict(level="area_manager", id=r.am, severity=r.unassigned_customers / max(1, r.customers),
                            msg=f"{r.am} has {r.unassigned_customers} customers with no active owner (coverage gap)."))
    return sorted(out, key=lambda x: -x["severity"])

```

---

## `explain.py`  (149 lines)

```python
"""Explainability: every assignment decision gets a human-readable reason.

All reasons are computed from the actual numbers (distances, utilisation
before/after, relationship weight, exception log) - nothing is hallucinated.
An OPTIONAL LLM layer (llm_narrative) can rephrase these facts; it never
invents them.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

import config as C
import workload as W
from geo import haversine_km


def legacy_metrics(al) -> pd.DataFrame:
    return W.employee_metrics(al.cust, al.emps, al.pins, "legacy_emp")


def _centroids(al) -> dict:
    pins = al.pins.set_index("pin")
    own = pd.Series(al.pin_owner)
    g = pins.loc[own.index].assign(owner=own.values).groupby("owner")[["lat", "lon"]].mean()
    return {e: (r.lat, r.lon) for e, r in g.iterrows()}


def customer_reason(al, cust_id: str, legacy_m: pd.DataFrame | None = None) -> dict:
    legacy_m = legacy_m if legacy_m is not None else legacy_metrics(al)
    r = al.cust[al.cust.cust_id == cust_id]
    if r.empty:
        return dict(decision="unknown", reasons=[f"Customer {cust_id} not found."])
    r = r.iloc[0]
    lm, nm = legacy_m.set_index("emp_id"), al.metrics.set_index("emp_id")
    cents = _centroids(al)
    leg, fin, own = r.legacy_emp, r.final_emp, r.new_emp
    reasons = []
    d_to = lambda e: float(haversine_km(r.lat, r.lon, *cents[e])) if e in cents else np.nan

    if leg not in lm.index:
        decision = "ASSIGNED (previously unowned)"
        reasons.append("Previous owner is no longer active, so this customer was an unowned coverage gap.")
    elif fin == leg and not r.is_exception:
        decision = "KEPT"
        reasons.append(f"Customer sits inside {leg}'s optimised territory, so the existing relationship is preserved.")
    elif fin == leg and r.is_exception:
        decision = "KEPT (relationship exception)"
        ex = al.exceptions[al.exceptions.cust_id == cust_id]
        km = ex.dist_to_rep_territory_km.iloc[0] if len(ex) else np.nan
        reasons.append(f"Key-account exception: {r.pot_class}-class customer with {r.relationship_years:.1f} years of "
                       f"history stays with {leg} although the PIN belongs to {own}'s territory.")
        reasons.append(f"{leg}'s territory is only {km:.1f} km away and {leg} has spare capacity "
                       f"(utilisation now {nm.loc[leg, 'utilization']:.0%}).")
    else:
        decision = "MOVED"
        reasons.append(f"PIN {r.pin} was allocated to {fin}'s contiguous territory.")
        dl, dn = d_to(leg), d_to(fin)
        if not np.isnan(dl) and dl > dn:
            reasons.append(f"Customer is {dn:.1f} km from {fin}'s territory centre vs {dl:.1f} km from {leg}'s - less travel.")
        if lm.loc[leg, "utilization"] > C.OVERLOAD_UTIL:
            reasons.append(f"{leg} was overloaded ({lm.loc[leg, 'utilization']:.0%}); now {nm.loc[leg, 'utilization']:.0%}.")
        elif lm.loc[leg, "utilization"] < C.UNDERLOAD_UTIL:
            reasons.append(f"{leg} was underloaded ({lm.loc[leg, 'utilization']:.0%}) and the territory shape required this PIN elsewhere.")
        if lm.loc[fin, "utilization"] < C.UNDERLOAD_UTIL + 0.1:
            reasons.append(f"{fin} had spare capacity ({lm.loc[fin, 'utilization']:.0%} -> {nm.loc[fin, 'utilization']:.0%}).")
        reasons.append(f"Continuity cost accepted: {r.pot_class}-class, {r.relationship_years:.1f} yrs "
                       f"(relationship weight {r.rel_weight:.1f}).")
    return dict(cust_id=cust_id, pin=r.pin, legacy=leg, final=fin, decision=decision, reasons=reasons)


def territory_summary(al, emp_id: str, legacy_m: pd.DataFrame | None = None) -> dict:
    legacy_m = legacy_m if legacy_m is not None else legacy_metrics(al)
    nm = al.metrics.set_index("emp_id")
    if emp_id not in nm.index:
        return dict(error=f"{emp_id} not found")
    e, lm = nm.loc[emp_id], legacy_m.set_index("emp_id").loc[emp_id]
    mine = al.cust[al.cust.final_emp == emp_id]
    gained = mine[mine.legacy_emp != emp_id].groupby("legacy_emp").size().sort_values(ascending=False)
    lost = al.cust[(al.cust.legacy_emp == emp_id) & (al.cust.final_emp != emp_id)].groupby("final_emp").size().sort_values(ascending=False)
    kept = int((mine.legacy_emp == emp_id).sum())
    pins = sorted(p for p, o in al.pin_owner.items() if o == emp_id)
    return dict(
        emp_id=emp_id, name=e["name"], region=e.region, am=e.am, grade=e.grade,
        pins=pins, n_pins=len(pins), customers=int(e.n_customers),
        class_mix={k: int(e[f"n_{k}"]) for k in "ABCD"},
        utilization_before=float(lm.utilization), utilization_after=float(e.utilization),
        potential_ratio_before=float(lm.potential_ratio), potential_ratio_after=float(e.potential_ratio),
        travel_hours_before=float(lm.travel_hours), travel_hours_after=float(e.travel_hours),
        kept=kept, gained=int(gained.sum()), lost=int(lost.sum()),
        gained_from=gained.head(3).to_dict(), lost_to=lost.head(3).to_dict(),
        exceptions_held=int(((mine.is_exception) & (mine.final_emp == emp_id)).sum()),
        flags=e.flags)


def territory_text(s: dict) -> str:
    if "error" in s:
        return s["error"]
    t = (f"**{s['emp_id']} - {s['name']}** ({s['grade']}, {s['region']}, {s['am']}): {s['n_pins']} PINs, "
         f"{s['customers']} customers (A/B/C/D = {s['class_mix']['A']}/{s['class_mix']['B']}/{s['class_mix']['C']}/{s['class_mix']['D']}). "
         f"Utilisation {s['utilization_before']:.0%} -> {s['utilization_after']:.0%}; potential vs fair share "
         f"{s['potential_ratio_before']:.0%} -> {s['potential_ratio_after']:.0%}; travel {s['travel_hours_before']:.0f}h -> {s['travel_hours_after']:.0f}h/month. "
         f"Keeps {s['kept']} existing customers, gains {s['gained']}, loses {s['lost']}.")
    if s["gained_from"]:
        t += " Main sources of new customers: " + ", ".join(f"{k} ({v})" for k, v in s["gained_from"].items()) + "."
    if s["lost_to"]:
        t += " Main recipients of released customers: " + ", ".join(f"{k} ({v})" for k, v in s["lost_to"].items()) + "."
    return t


def headline(al, cmp: pd.DataFrame) -> str:
    a, b = cmp["Legacy (as-is)"], cmp["Optimised (ours)"]
    return (f"Utilisation spread (std-dev) falls from {a['Utilisation std-dev']:.2f} to {b['Utilisation std-dev']:.2f}; "
            f"overloaded reps {int(a['Overloaded reps (>110%)'])} -> {int(b['Overloaded reps (>110%)'])}, "
            f"underloaded {int(a['Underloaded reps (<80%)'])} -> {int(b['Underloaded reps (<80%)'])}. "
            f"Monthly travel drops {100 * (1 - b['Total travel (km/month)'] / a['Total travel (km/month)']):.0f}% "
            f"({a['Total travel (km/month)']:,.0f} -> {b['Total travel (km/month)']:,.0f} km). "
            f"Coverage gaps {int(a['Unassigned customers (coverage gap)'])} -> 0 customers, fragmented territories "
            f"{int(a['Fragmented territories'])} -> {int(b['Fragmented territories'])}. "
            f"Relationship continuity kept: {b['Customers keeping same rep (%)']:.0f}% of customers "
            f"({b['Relationship-weight kept (%)']:.0f}% by relationship value).")


def flows(al, top: int = 15) -> pd.DataFrame:
    ch = al.cust[(al.cust.legacy_emp != al.cust.final_emp)]
    f = ch.groupby(["legacy_emp", "final_emp"]).agg(customers=("cust_id", "size"),
                                                   potential_lakh=("potential_lakh", "sum")).reset_index()
    return f.sort_values("customers", ascending=False).head(top).round(1)


def llm_narrative(facts: dict, question: str | None = None, model: str | None = None) -> str | None:
    """OPTIONAL. Rephrase pre-computed facts with Claude if ANTHROPIC_API_KEY is set.
    Returns None when the SDK/key is missing or the call fails (callers fall back
    to the deterministic text above).  Not exercised by the offline test-suite."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"), max_tokens=700,
            system=("You explain sales-territory alignment decisions to managers. Use ONLY the JSON facts "
                    "provided; never invent numbers. Be concise."),
            messages=[{"role": "user", "content": (question or "Summarise the key decisions.") +
                       "\n\nFACTS:\n" + json.dumps(facts, default=str)[:12000]}])
        return msg.content[0].text
    except Exception:
        return None

```

---

## `whatif.py`  (128 lines)

```python
"""Dynamic re-alignment & what-if simulation.

Key idea: for a what-if the CURRENT alignment becomes the new relationship
baseline (legacy_emp := final_emp), and the optimiser is warm-started from the
current PIN ownership - so only the necessary PINs move (minimal disruption).
Only the affected Area-Manager region is re-solved.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd

import config as C
import pipeline as P
import workload as W
from geo import project_km


def _rebase(al: P.Alignment, drop_emp: str | None = None) -> pd.DataFrame:
    c = al.cust.copy()
    c["legacy_emp"] = c.final_emp                       # current alignment = relationships to preserve
    if drop_emp:
        c.loc[c.legacy_emp == drop_emp, "legacy_emp"] = "VACANT"
    return c.drop(columns=[x for x in ("new_emp", "final_emp", "is_exception") if x in c])


def employee_leaves(al: P.Alignment, emp_id: str, weights: C.Weights | None = None) -> tuple[P.Alignment, dict]:
    if emp_id not in set(al.emps.emp_id):
        raise ValueError(f"Unknown employee {emp_id}")
    reg = al.emps.set_index("emp_id").loc[emp_id, "region"]
    emps2 = al.emps[al.emps.emp_id != emp_id].reset_index(drop=True)
    cust2 = _rebase(al, drop_emp=emp_id)
    init = {p: o for p, o in al.pin_owner.items() if o != emp_id}      # departed rep's PINs -> unassigned (-1)
    new = P.align(al.pins, emps2, cust2, weights or al.weights, init_owner=init, regions=[reg],
                  base=al, max_sweeps=80)
    return new, diff(al, new, f"{emp_id} leaves", reg)


def employee_joins(al: P.Alignment, region: str, home_pin: str | None = None, grade: str = "Mid",
                   name: str = "New Joiner", weights: C.Weights | None = None,
                   relax_continuity: float = 4.0) -> tuple[P.Alignment, dict]:
    """Add a rep. Adding headcount is a deliberate rebalancing act, so the
    continuity weight is divided by `relax_continuity` unless weights are given."""
    if weights is None:
        weights = replace(al.weights, w_cont=max(1.0, al.weights.w_cont / relax_continuity))
    pins_r = al.pins[al.pins.region == region].reset_index(drop=True)
    if pins_r.empty:
        raise ValueError(f"Unknown region {region}")
    if home_pin is None:                                # default: PIN of the busiest territory's centre
        load = al.cust[al.cust.region == region].groupby("final_emp").visit_hours.sum()
        busiest = load.idxmax()
        mine = pins_r[pins_r.pin.map(al.pin_owner) == busiest]
        home_pin = mine.iloc[len(mine) // 2].pin
    hp = pins_r[pins_r.pin == home_pin]
    if hp.empty:
        raise ValueError(f"PIN {home_pin} is not in region {region}")
    hp = hp.iloc[0]
    new_id = f"E{int(al.emps.emp_id.str[1:].astype(int).max()) + 1:03d}"
    row = dict(emp_id=new_id, name=name, am=pins_r.am.iloc[0], region=region, grade=grade,
               capacity_hours=C.GRADES[grade]["capacity_hours"], potential_weight=C.GRADES[grade]["potential_weight"],
               home_pin=hp.pin, home_lat=hp.lat, home_lon=hp.lon,
               speed_kmph=al.emps[al.emps.region == region].speed_kmph.iloc[0])
    emps2 = pd.concat([al.emps, pd.DataFrame([row])], ignore_index=True)
    cust2 = _rebase(al)
    # seed: home PIN + its 2 nearest PINs go to the newcomer, everything else stays
    xy = project_km(pins_r.lat, pins_r.lon)
    i0 = int(np.where(pins_r.pin == home_pin)[0][0])
    near = np.argsort(np.linalg.norm(xy - xy[i0], axis=1))[:3]
    init = dict(al.pin_owner)
    for i in near:
        init[pins_r.pin[i]] = new_id
    new = P.align(al.pins, emps2, cust2, weights, init_owner=init, regions=[region],
                  base=al, max_sweeps=150)
    return new, diff(al, new, f"{new_id} joins {region}", region)


def add_customers(al: P.Alignment, region: str, n: int = 40, around_pin: str | None = None,
                  seed: int = 7, weights: C.Weights | None = None) -> tuple[P.Alignment, dict]:
    """Simulate a demand spike: n new customers appear around a PIN."""
    rng = np.random.default_rng(seed)
    pins_r = al.pins[al.pins.region == region]
    center = pins_r[pins_r.pin == around_pin].iloc[0] if around_pin else pins_r.sample(1, random_state=seed).iloc[0]
    base_id = int(al.cust.cust_id.str[1:].astype(int).max())
    rows = []
    for i in range(n):
        k = rng.choice(list("ABCD"), p=[0.15, 0.30, 0.35, 0.20])
        rows.append(dict(cust_id=f"C{base_id + 1 + i:05d}", pin=center.pin, region=region, am=center.am,
                         lat=center.lat + rng.normal(0, 0.004), lon=center.lon + rng.normal(0, 0.004),
                         pot_class=k, potential_lakh=round(float(rng.lognormal(np.log(C.POTENTIAL_CLASSES[k]["mean_lakh"]), 0.35)), 2),
                         visits_pm=C.POTENTIAL_CLASSES[k]["visits"], hours_per_visit=C.POTENTIAL_CLASSES[k]["hours"],
                         legacy_emp="VACANT", relationship_years=0.0))
    cust2 = pd.concat([_rebase(al), pd.DataFrame(rows)], ignore_index=True)
    cust2 = W.add_customer_fields(cust2)
    new = P.align(al.pins, al.emps, cust2, weights or al.weights, init_owner=dict(al.pin_owner), regions=[region],
                  base=al, max_sweeps=80)
    return new, diff(al, new, f"+{n} customers near {center.pin}", region)


def diff(before: P.Alignment, after: P.Alignment, title: str, region: str | None = None) -> dict:
    """Before/after comparison scoped to the affected region (other regions are untouched)."""
    b = before.cust.set_index("cust_id").final_emp
    a = after.cust.set_index("cust_id").final_emp
    common = b.index.intersection(a.index)
    if region is None:                                  # infer: region with most changes
        ch = after.cust.set_index("cust_id").loc[common]
        ch = ch[b[common] != a[common]]
        region = ch.region.mode().iloc[0] if len(ch) else after.emps.region.iloc[0]
    in_reg = after.cust.set_index("cust_id").loc[common].region == region
    moved = int(((b[common] != a[common]) & in_reg).sum())
    mb = before.metrics[before.metrics.region == region]
    ma = after.metrics[after.metrics.region == region]
    cap_gap_h = float(ma.workload_hours.sum() - ma.capacity_hours.sum())
    avg_cap = float(ma.capacity_hours.mean())
    reco = (f"{region} is over capacity by {cap_gap_h:.0f} h/month - hire about {int(np.ceil(cap_gap_h / avg_cap))} more rep(s)."
            if cap_gap_h > 0 else
            f"{region} has {-cap_gap_h:.0f} h/month of spare capacity after the change.")
    return dict(title=title, region=region, customers_moved=moved,
                share_moved=moved / max(1, int(in_reg.sum())),
                reps_before=len(mb), reps_after=len(ma),
                region_util_mean_before=float(mb.workload_hours.sum() / mb.capacity_hours.sum()),
                region_util_mean_after=float(ma.workload_hours.sum() / ma.capacity_hours.sum()),
                region_util_std_before=float(mb.utilization.std(ddof=0)),
                region_util_std_after=float(ma.utilization.std(ddof=0)),
                max_util_before=float(mb.utilization.max()), max_util_after=float(ma.utilization.max()),
                overloaded_before=int((mb.utilization > C.OVERLOAD_UTIL).sum()),
                overloaded_after=int((ma.utilization > C.OVERLOAD_UTIL).sum()),
                unassigned_after=int((~after.cust.final_emp.isin(after.emps.emp_id)).sum()),
                recommendation=reco)

```

---

## `nlq.py`  (105 lines)

```python
"""Natural-language interface over the alignment.

Deterministic intent router (regex + entity extraction) -> every answer is
computed from the data, so it is auditable and works offline.  If an
ANTHROPIC_API_KEY is set, unrecognised questions fall through to an optional
LLM that may only use the structured facts we pass it.
"""
from __future__ import annotations
import re
import pandas as pd

import config as C
import explain as X
import metrics as M
import whatif as WI
import pipeline as P

HELP = ("Try: 'who is overloaded?', 'show territory of E012', 'why is customer C00122 with E006?', "
        "'who covers PIN 400045?', 'what if E005 leaves?', 'add a rep in Mumbai', 'simulate 60 new customers in Bengaluru', "
        "'compare with baseline', 'summary for AM02', 'national summary', 'highest potential territories', 'show alerts'.")


def ask(q: str, al: P.Alignment, cmp: pd.DataFrame | None = None, legacy_m: pd.DataFrame | None = None,
        managers: pd.DataFrame | None = None):
    """Returns (answer_text, optional DataFrame, optional new Alignment for what-ifs)."""
    ql = q.lower().strip()
    legacy_m = legacy_m if legacy_m is not None else X.legacy_metrics(al)
    emp = re.search(r"\be\d{3}\b", ql)
    cust = re.search(r"\bc\d{5}\b", ql)
    pin = re.search(r"\b\d{6}\b", ql)
    am_id = re.search(r"\bam\d{2}\b", ql)
    emp = emp.group(0).upper() if emp else None
    cust = cust.group(0).upper() if cust else None
    pin = pin.group(0) if pin else None
    region = next((r for r in al.emps.region.unique() if r.lower().split()[0] in ql), None)
    m = al.metrics
    cols = ["emp_id", "name", "region", "grade", "n_customers", "workload_hours", "utilization", "potential_lakh", "potential_ratio"]

    # ---------------- what-ifs
    if emp and re.search(r"\b(leave|leaves|leaving|resign|quit|left|unavailable)\b", ql):
        new, d = WI.employee_leaves(al, emp)
        return (f"**{d['title']}** - {d['customers_moved']} customers moved ({d['share_moved']:.0%} of {d['region']}). "
                f"Region utilisation {d['region_util_mean_before']:.0%} -> {d['region_util_mean_after']:.0%}, max rep "
                f"{d['max_util_before']:.0%} -> {d['max_util_after']:.0%}. {d['recommendation']}"), None, new
    if re.search(r"\b(join|joins|hire|add (a |an )?(new )?(rep|employee|executive))\b", ql) and region:
        new, d = WI.employee_joins(al, region)
        return (f"**{d['title']}** - {d['customers_moved']} customers moved ({d['share_moved']:.0%} of {d['region']}). "
                f"Region utilisation spread {d['region_util_std_before']:.3f} -> {d['region_util_std_after']:.3f}; "
                f"average utilisation {d['region_util_mean_before']:.0%} -> {d['region_util_mean_after']:.0%}. {d['recommendation']}"), None, new
    mm = re.search(r"(\d+)\s+(new\s+)?customers", ql)
    if mm and region and re.search(r"(simulate|add|spike|new)", ql):
        new, d = WI.add_customers(al, region, int(mm.group(1)))
        return (f"**{d['title']}** - {d['customers_moved']} customers re-assigned. Region utilisation "
                f"{d['region_util_mean_before']:.0%} -> {d['region_util_mean_after']:.0%}; overloaded reps "
                f"{d['overloaded_before']} -> {d['overloaded_after']}. {d['recommendation']}"), None, new

    # ---------------- explanations / lookups
    if cust:
        r = X.customer_reason(al, cust, legacy_m)
        return f"**{cust}** -> {r['final']} ({r['decision']}). " + " ".join(r["reasons"]), None, None
    if pin and re.search(r"(cover|owner|owns|who|territory|pin)", ql):
        o = al.pin_owner.get(pin)
        if not o:
            return f"PIN {pin} is not in the dataset.", None, None
        n = int((al.cust.pin == pin).sum())
        return f"PIN {pin} belongs to {o}'s territory ({n} customers in this PIN).", al.cust[al.cust.pin == pin][
            ["cust_id", "pot_class", "potential_lakh", "legacy_emp", "final_emp"]], None
    if emp:
        s = X.territory_summary(al, emp, legacy_m)
        return X.territory_text(s), None, None

    # ---------------- imbalance / lists
    if "overload" in ql:
        t = m[m.utilization > C.OVERLOAD_UTIL][cols]
        return (f"{len(t)} rep(s) above {C.OVERLOAD_UTIL:.0%} utilisation." if len(t) else
                f"No rep exceeds {C.OVERLOAD_UTIL:.0%} utilisation (max {m.utilization.max():.0%})."), t, None
    if "underload" in ql or "spare capacity" in ql:
        t = m[m.utilization < C.UNDERLOAD_UTIL][cols]
        return (f"{len(t)} rep(s) below {C.UNDERLOAD_UTIL:.0%} utilisation." if len(t) else
                f"No rep is below {C.UNDERLOAD_UTIL:.0%} utilisation (min {m.utilization.min():.0%})."), t, None
    if "potential" in ql and re.search(r"(high|top|best|most)", ql):
        return "Top territories by potential:", m.sort_values("potential_lakh", ascending=False).head(8)[cols], None
    if "potential" in ql and re.search(r"(low|bottom|least|worst)", ql):
        return "Lowest-potential territories:", m.sort_values("potential_lakh").head(8)[cols], None
    if re.search(r"(alert|imbalance|problem|issue|flag)", ql):
        am = M.area_manager_view(m, al.cust, "final_emp", managers if managers is not None else pd.DataFrame(dict(mgr_id=[], name=[])))
        al_list = M.detect_imbalance(m, am)[:8]
        return ("Top alerts:\n" + "\n".join("- " + a["msg"] for a in al_list)) if al_list else "No imbalance alerts.", None, None

    # ---------------- roll-ups
    mg = managers if managers is not None else pd.DataFrame(dict(mgr_id=[], name=[]))
    if am_id or region:
        am = M.area_manager_view(m, al.cust, "final_emp", mg)
        row = am[am.am == am_id.group(0).upper()] if am_id else am[am.region == region]
        if len(row):
            return f"Area-Manager summary for {row.iloc[0].region}:", row.T.reset_index().rename(columns={"index": "metric", row.index[0]: "value"}), None
    if re.search(r"(national|overall|network|summary)", ql):
        am = M.area_manager_view(m, al.cust, "final_emp", mg)
        nv = M.national_view(am)
        return "National roll-up: " + ", ".join(f"{k}={v}" for k, v in nv.items()), am, None
    if re.search(r"(compare|baseline|better|improv|vs)", ql) and cmp is not None:
        return X.headline(al, cmp), cmp.round(2).reset_index(names="metric"), None

    txt = X.llm_narrative(dict(kpis=cmp.round(3).to_dict() if cmp is not None else {}), q)
    return (txt or "I couldn't map that question to an analysis. " + HELP), None, None

```

---

## `mapviz.py`  (154 lines)

```python
"""India map layer (Folium / OpenStreetMap tiles).

Territory boundaries are APPROXIMATED the standard way when true PIN polygons
are unavailable: a Voronoi tessellation of PIN centroids (one cell per PIN,
clipped to the region's footprint), dissolved by owner.  With real PIN boundary
shapefiles, replace `pin_cells()` and nothing else changes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import folium
from folium.plugins import MiniMap, Fullscreen
import shapely
from shapely.geometry import MultiPoint, Point, mapping
from shapely.ops import unary_union

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a0a0", "#9a6324",
           "#b00020", "#808000", "#000075", "#e377c2", "#17becf"]
CLASS_RADIUS = {"A": 6, "B": 4.5, "C": 3.2, "D": 2.2}
_CELL_CACHE: dict = {}


def pin_cells(pins_r: pd.DataFrame) -> dict:
    """pin -> shapely Polygon (lon/lat)."""
    key = (str(pins_r.region.iloc[0]), len(pins_r))
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    pts = [Point(lo, la) for la, lo in zip(pins_r.lat, pins_r.lon)]
    mp = MultiPoint(pts)
    footprint = mp.convex_hull.buffer(0.018)
    vor = shapely.voronoi_polygons(mp, extend_to=footprint.buffer(0.5))
    cells = {}
    for poly in vor.geoms:
        for pin, pt in zip(pins_r.pin, pts):
            if poly.contains(pt):
                cells[pin] = poly.intersection(footprint)
                break
    _CELL_CACHE[key] = cells
    return cells


def emp_colors(emps: pd.DataFrame) -> dict:
    col = {}
    for reg, er in emps.groupby("region", sort=False):
        for i, e in enumerate(er.emp_id):
            col[e] = PALETTE[i % len(PALETTE)]
    col["VACANT"] = "#7f7f7f"
    return col


def _base_map(pins: pd.DataFrame, region: str | None) -> folium.Map:
    p = pins if region is None else pins[pins.region == region]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True, zoom_start=5)
    m.fit_bounds([[p.lat.min(), p.lon.min()], [p.lat.max(), p.lon.max()]])
    Fullscreen().add_to(m)
    return m


def make_map(al, view: str = "optimised", region: str | None = None, emp_id: str | None = None,
             show_customers: bool = True, show_pins: bool = True) -> folium.Map:
    """view = 'optimised' (final alignment) | 'legacy' (as-is)."""
    pins, emps, cust = al.pins, al.emps, al.cust
    col = emp_colors(emps)
    m = _base_map(pins, region)
    owner_col = "final_emp" if view == "optimised" else "legacy_emp"
    regs = [region] if region else list(pins.region.unique())
    emp_info = emps.set_index("emp_id")
    met = (al.metrics if view == "optimised" else _legacy_metrics(al)).set_index("emp_id")

    g_pins = folium.FeatureGroup(name="PIN cells (colour = owner)", show=(view == "optimised" and show_pins))
    g_terr = folium.FeatureGroup(name="Territory outlines", show=True)
    g_cust = folium.FeatureGroup(name="Customers", show=show_customers)
    g_home = folium.FeatureGroup(name="Rep home bases", show=True)

    for reg in regs:
        pr = pins[pins.region == reg].reset_index(drop=True)
        cells = pin_cells(pr)
        if view == "optimised":
            owner = pr.pin.map(al.pin_owner)
            for pin, o in zip(pr.pin, owner):
                if emp_id and o != emp_id:
                    continue
                if pin in cells and o in col:
                    folium.GeoJson(mapping(cells[pin]), style_function=lambda _, c=col[o]: dict(
                        fillColor=c, color="#ffffff", weight=0.6, fillOpacity=0.30),
                        tooltip=f"PIN {pin} -> {o}").add_to(g_pins)
            for e, grp in pd.DataFrame(dict(pin=pr.pin, o=owner)).groupby("o"):
                if emp_id and e != emp_id:
                    continue
                geom = unary_union([cells[p] for p in grp.pin if p in cells])
                r = met.loc[e] if e in met.index else None
                tip = (f"<b>{e}</b> {emp_info.loc[e, 'name']}<br>{len(grp)} PINs, {int(r.n_customers)} customers<br>"
                       f"Utilisation {r.utilization:.0%} | Potential {r.potential_lakh:.0f}L ({r.potential_ratio:.0%} of fair)<br>"
                       f"Travel {r.travel_hours:.0f} h/mo") if r is not None else e
                folium.GeoJson(mapping(geom), style_function=lambda _, c=col[e]: dict(
                    fillColor=c, color=c, weight=3, fillOpacity=0.0), tooltip=folium.Tooltip(tip, sticky=True)).add_to(g_terr)
        else:   # legacy: no clean territories exist -> show each rep's convex hull (they overlap!)
            cc = cust[cust.region == reg]
            for e, grp in cc.groupby("legacy_emp"):
                if e == "VACANT" or (emp_id and e != emp_id) or len(grp) < 3:
                    continue
                hull = MultiPoint([Point(lo, la) for la, lo in zip(grp.lat, grp.lon)]).convex_hull
                if hull.geom_type != "Polygon":
                    continue
                r = met.loc[e] if e in met.index else None
                tip = f"<b>{e}</b> (legacy)<br>{len(grp)} customers, {r.n_pins:.0f} PINs<br>Utilisation {r.utilization:.0%}" if r is not None else e
                folium.GeoJson(mapping(hull), style_function=lambda _, c=col[e]: dict(
                    fillColor=c, color=c, weight=2, fillOpacity=0.07, dashArray="5 5"),
                    tooltip=folium.Tooltip(tip, sticky=True)).add_to(g_terr)

    if show_customers:
        cc = cust[cust.region.isin(regs)]
        if emp_id:
            cc = cc[cc[owner_col] == emp_id]
        for r in cc.itertuples():
            o = getattr(r, owner_col)
            c = col.get(o, "#7f7f7f")
            exc = bool(getattr(r, "is_exception", False)) and view == "optimised"
            nm = emp_info.loc[o, "name"] if o in emp_info.index else "UNASSIGNED"
            folium.CircleMarker((r.lat, r.lon), radius=CLASS_RADIUS[r.pot_class], color="#000000" if exc else c,
                                weight=2 if exc else 0.5, fill=True, fill_color=c, fill_opacity=0.85,
                                tooltip=f"{r.cust_id} | class {r.pot_class} | {r.potential_lakh:.1f}L | PIN {r.pin}<br>"
                                        f"Owner: {o} ({nm}){' - KEY-ACCOUNT EXCEPTION' if exc else ''}"
                                ).add_to(g_cust)
    for e, r in emp_info.iterrows():
        if r.region in regs and (not emp_id or e == emp_id):
            folium.Marker((r.home_lat, r.home_lon), tooltip=f"{e} {r['name']} (home base, {r.grade})",
                          icon=folium.Icon(color="black", icon="user", prefix="fa")).add_to(g_home)
    for g in (g_pins, g_terr, g_cust, g_home):
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    return m


def india_overview(al) -> folium.Map:
    """One bubble per Area-Manager region with roll-up numbers."""
    from metrics import area_manager_view
    am = area_manager_view(al.metrics, al.cust, "final_emp",
                           pd.DataFrame(dict(mgr_id=[], name=[])))
    m = folium.Map(location=[22.5, 79], zoom_start=5, tiles="OpenStreetMap")
    cen = al.pins.groupby("region")[["lat", "lon"]].mean()
    for _, r in am.iterrows():
        la, lo = cen.loc[r.region]
        color = "#d62728" if r.utilization > 1.05 else "#2ca02c" if r.utilization >= 0.8 else "#ff7f0e"
        folium.CircleMarker((la, lo), radius=10 + r.reps * 2, color=color, fill=True, fill_opacity=0.55,
                            tooltip=f"<b>{r.region}</b> ({r.am})<br>{r.reps} reps, {r.customers} customers<br>"
                                    f"Utilisation {r.utilization:.0%}, potential {r.potential_lakh:.0f}L").add_to(m)
    return m


def _legacy_metrics(al):
    import workload as W
    return W.employee_metrics(al.cust, al.emps, al.pins, "legacy_emp")

```

---

## `run_pipeline.py`  (155 lines)

```python
"""One-command end-to-end run:   python run_pipeline.py [--regenerate] [--preset NAME] [--no-exceptions]

Writes everything a judge needs to outputs/ :
  maps/*.html           India overview + per-region optimised & legacy maps + all-India map
  employee_metrics.csv  per-rep workload / potential / travel / flags (optimised)
  customer_assignments.csv  every customer: legacy rep, new rep, exception flag
  pin_territories.csv   PIN -> rep
  comparison.csv        legacy vs baselines vs optimised
  am_rollup.csv, national.json, alerts.txt, tradeoff.csv, exceptions.csv
  charts/*.png          utilisation before/after, trade-off frontier, flows
  REPORT.md             headline numbers, baseline table, limitations
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import data_gen, pipeline as P, explain as X, metrics as M, mapviz as MV

OUT = Path(__file__).parent / "outputs"

LIMITATIONS = [
    ("PIN-code atomicity caps relationship continuity",
     "Territories are unions of whole PIN codes, so customers inside one PIN cannot be split between reps. On this "
     "dataset the theoretical ceiling for continuity is ~50% of customers; on the default seed we reach ~36% (40% by value) because "
     "geography and balance also matter. The key-account exception layer recovers part of the gap at the price of "
     "explicit PIN overlaps. A client that wants more continuity must accept more overlap or lower balance."),
    ("Granularity and capacity shortfalls cannot be optimised away",
     "A PIN with a very heavy customer book is indivisible, which limits how finely workload can be balanced (~+/-5% "
     "here). More importantly, when a rep leaves or demand spikes the region may simply lack capacity - on the default seed the what-if "
     "runs show 1 overloaded rep after a departure and 3 after a +60-customer spike. No re-alignment fixes that; the "
     "system can only flag it and recommend headcount."),
    ("Synthetic data, approximate boundaries and a simple travel model",
     "PIN centroids/customers are synthetic around real city centres; PIN boundaries are approximated by Voronoi "
     "cells, not India Post polygons; travel uses straight-line x1.3 circuity at a flat city speed (no road network, "
     "traffic or public transport). Absolute hours/km are therefore indicative - the relative comparison between "
     "methods is what we claim."),
    ("Heuristic, not a proven optimum; weights are judgement calls",
     "Local search can stall in local minima and carries no optimality gap. The objective weights, relationship "
     "weights (class x tenure), visit frequencies and visit durations are assumptions that a real deployment must "
     "calibrate with the business."),
    ("Quality degrades if one region is made very large",
     "The solver is a local search. In the benchmark (benchmark_scaling.py) one region of 100 PINs / 8 reps solves in "
     "0.4 s with utilisation std-dev 0.05, but at 400-800 PINs / 32-64 reps in a single region it converges to a local "
     "minimum (std-dev ~0.085, a few reps at ~60% utilisation, more restarts did not help). The lowest-utilised reps "
     "there live 10-12 km from their territory centre, which is the likely cause (not isolated by an ablation). The "
     "design answer is the Area-Manager hierarchy: keep each solve at ~6-10 reps and scale by solving regions in "
     "parallel - cost grows linearly with the number of regions."),
    ("Static snapshot",
     "Potential is taken as given (no forecasting), territories are optimised for one month's workload, and the "
     "system ignores leave calendars, rep skills/language, customer time windows and rep preferences."),
]


def charts(al, cmp, mets, sweep, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    # 1. utilisation before / after
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for a, name in zip(ax, ["Legacy (as-is)", "Optimised (ours)"]):
        m = mets[name].sort_values(["region", "emp_id"])
        colors = ["#d62728" if u > 1.1 else "#ff7f0e" if u < 0.8 else "#2ca02c" for u in m.utilization]
        a.bar(range(len(m)), m.utilization, color=colors)
        a.axhline(1.0, color="k", lw=1, ls="--"); a.axhline(0.8, color="gray", lw=0.8, ls=":")
        a.set_title(f"{name}  (std-dev {m.utilization.std(ddof=0):.2f})"); a.set_xlabel("42 reps, grouped by region")
        a.set_xticks([])
    ax[0].set_ylabel("Workload / capacity")
    plt.suptitle("Rep utilisation: red >110%, orange <80%, green healthy")
    plt.tight_layout(); plt.savefig(out / "utilisation_before_after.png", dpi=130); plt.close()
    # 2. baseline bars
    keys = ["Utilisation std-dev", "Potential-ratio std-dev", "Total travel (km/month)", "Customers keeping same rep (%)"]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for a, k in zip(ax, keys):
        a.bar(range(len(cmp.columns)), cmp.loc[k].values, color=["#999", "#bbb", "#bbb", "#1f77b4"])
        a.set_xticks(range(len(cmp.columns))); a.set_xticklabels([c.split(" ")[0] if c[0] == "B" else c.split(" ")[0] for c in cmp.columns], fontsize=8)
        a.set_title(k, fontsize=10)
    plt.tight_layout(); plt.savefig(out / "baseline_comparison.png", dpi=130); plt.close()
    # 3. trade-off frontier
    if sweep is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(sweep.relationship_kept_pct, sweep.util_std, "o-")
        for _, r in sweep.iterrows():
            ax.annotate(f"w={r.w_cont:g}", (r.relationship_kept_pct, r.util_std), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Relationship value kept with existing rep (%)"); ax.set_ylabel("Utilisation std-dev (lower = better balance)")
        ax.set_title("Continuity vs balance trade-off"); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(out / "tradeoff_frontier.png", dpi=130); plt.close()


def write_report(al, cmp, sweep, elapsed, out: Path):
    ex = al.exceptions
    md = ["# Territory Alignment - run report", "", "## Headline", X.headline(al, cmp), "",
          f"Runtime: {elapsed:.1f}s for {len(al.emps)} reps, {len(al.pins)} PINs, {len(al.cust)} customers "
          f"(6 regions solved independently - the decomposition is what scales to India).", "",
          "## Baseline comparison (same scoring engine for every method)", "",
          cmp.round(3).to_markdown(), "",
          "## Key-account relationship exceptions",
          f"{len(ex)} A/B-class customers stay with their legacy rep outside the PIN territory "
          f"(max 4 km away, rep has spare capacity).", ""]
    if sweep is not None:
        md += ["## Continuity vs balance trade-off", "", sweep.round(3).to_markdown(index=False), ""]
    md += ["## Limitations / failure cases", ""]
    for i, (t, d) in enumerate(LIMITATIONS, 1):
        md += [f"{i}. **{t}.** {d}", ""]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="regenerate the synthetic dataset")
    ap.add_argument("--preset", default="Balanced (default)", choices=list(C.PRESETS))
    ap.add_argument("--no-exceptions", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    (OUT / "maps").mkdir(exist_ok=True)
    d = data_gen.generate() if args.regenerate else data_gen.load()
    t0 = time.time()
    al = P.align(d["pins"], d["employees"], d["customers"], C.PRESETS[args.preset], exceptions=not args.no_exceptions)
    elapsed = time.time() - t0
    cmp, mets = P.compare_methods(al)
    am = M.area_manager_view(al.metrics, al.cust, "final_emp", d["managers"])
    sweep = None if args.no_sweep else P.tradeoff_sweep(d["pins"], d["employees"], d["customers"])

    # tables
    al.metrics.to_csv(OUT / "employee_metrics.csv", index=False)
    al.cust[["cust_id", "pin", "region", "am", "lat", "lon", "pot_class", "potential_lakh", "legacy_emp",
             "new_emp", "final_emp", "is_exception"]].to_csv(OUT / "customer_assignments.csv", index=False)
    pd.DataFrame(dict(pin=list(al.pin_owner), emp_id=list(al.pin_owner.values()))).to_csv(OUT / "pin_territories.csv", index=False)
    cmp.to_csv(OUT / "comparison.csv"); am.to_csv(OUT / "am_rollup.csv", index=False)
    al.exceptions.to_csv(OUT / "exceptions.csv", index=False)
    if sweep is not None:
        sweep.to_csv(OUT / "tradeoff.csv", index=False)
    (OUT / "national.json").write_text(json.dumps(M.national_view(am), indent=2))
    (OUT / "alerts.txt").write_text("\n".join(a["msg"] for a in M.detect_imbalance(al.metrics, am)))
    # maps
    MV.india_overview(al).save(OUT / "maps" / "india_overview.html")
    MV.make_map(al, "optimised").save(OUT / "maps" / "all_india_optimised.html")
    for reg in d["pins"].region.unique():
        tag = reg.lower().replace(" ", "_")
        MV.make_map(al, "optimised", reg).save(OUT / "maps" / f"{tag}_optimised.html")
        MV.make_map(al, "legacy", reg).save(OUT / "maps" / f"{tag}_legacy.html")
    charts(al, cmp, mets, sweep, OUT / "charts")
    write_report(al, cmp, sweep, elapsed, OUT)

    print("=" * 78); print(X.headline(al, cmp)); print("=" * 78)
    print(cmp.round(2).to_string()); print(f"\nAll outputs written to {OUT}")


if __name__ == "__main__":
    main()

```

---

## `app.py`  (213 lines)

```python
"""Streamlit dashboard:  streamlit run app.py"""
from __future__ import annotations
from dataclasses import replace
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

import config as C
import data_gen, pipeline as P, explain as X, metrics as M, mapviz as MV, whatif as WI, nlq
from run_pipeline import LIMITATIONS

st.set_page_config(page_title="AI Territory Alignment - India", layout="wide", page_icon="🗺️")


def show_html(html: str, height: int):
    """st.iframe replaces the deprecated components.v1.html (falls back on old Streamlit).
    The HTML is generated by our own Folium code from the dataset, never from free-text user input."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


# ----------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading dataset...")
def get_data():
    return data_gen.load()


@st.cache_resource(show_spinner="Optimising territories for all regions...")
def get_base(weights_key: tuple, exceptions: bool):
    d = get_data()
    w = C.Weights(*weights_key)
    return P.align(d["pins"], d["employees"], d["customers"], w, exceptions=exceptions)


@st.cache_resource(show_spinner="Running trade-off sweep (about 20 s)...")
def get_sweep():
    d = get_data()
    return P.tradeoff_sweep(d["pins"], d["employees"], d["customers"])


d = get_data()

# ----------------------------------------------------------------- sidebar
st.sidebar.title("Territory AI")
preset = st.sidebar.selectbox("Objective preset", list(C.PRESETS), index=1)
pw = C.PRESETS[preset]
with st.sidebar.expander("Fine-tune objective weights"):
    w_wl = st.slider("Workload balance", 0.0, 80.0, float(pw.w_wl), 1.0)
    w_pot = st.slider("Potential balance", 0.0, 40.0, float(pw.w_pot), 1.0)
    w_geo = st.slider("Compactness (travel)", 0.0, 20.0, float(pw.w_geo), 0.5)
    w_cont = st.slider("Relationship continuity", 0.0, 80.0, float(pw.w_cont), 1.0)
use_exc = st.sidebar.checkbox("Key-account relationship exceptions", value=True)
weights = C.Weights(w_wl=w_wl, w_pot=w_pot, w_geo=w_geo, w_cont=w_cont, w_cap=pw.w_cap,
                    max_sweeps=pw.max_sweeps, n_restarts=pw.n_restarts)
wkey = (weights.w_wl, weights.w_pot, weights.w_geo, weights.w_cont, weights.w_cap, weights.max_sweeps, weights.n_restarts)
key = (wkey, use_exc)

base = get_base(wkey, use_exc)
if st.session_state.get("key") != key:                    # settings changed -> drop any adopted scenario
    st.session_state.update(key=key, current=base, scenario_diff=None)
al: P.Alignment = st.session_state["current"]
if st.sidebar.button("Reset to base alignment"):
    st.session_state.update(current=base, scenario_diff=None)
    st.rerun()
if st.session_state.get("scenario_diff"):
    st.sidebar.info("Showing adopted scenario: " + st.session_state["scenario_diff"]["title"])

legacy_m = X.legacy_metrics(al)
cmp, mets = P.compare_methods(al)
am = M.area_manager_view(al.metrics, al.cust, "final_emp", d["managers"])

st.title("🗺️ Intelligent India Territory Alignment & Workforce Optimisation")
st.caption(f"{len(al.emps)} reps · {len(al.pins)} PIN codes · {len(al.cust):,} customers · {al.pins.region.nunique()} Area-Manager regions")

tabs = st.tabs(["Overview", "Map", "Reps & explanations", "Managers", "Baselines", "What-if", "Ask", "Limitations"])

# ----------------------------------------------------------------- Overview
with tabs[0]:
    st.success(X.headline(al, cmp))
    a, b = cmp["Legacy (as-is)"], cmp["Optimised (ours)"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Utilisation std-dev", f"{b['Utilisation std-dev']:.2f}", f"{b['Utilisation std-dev'] - a['Utilisation std-dev']:+.2f}", delta_color="inverse")
    c2.metric("Overloaded reps", int(b["Overloaded reps (>110%)"]), int(b["Overloaded reps (>110%)"] - a["Overloaded reps (>110%)"]), delta_color="inverse")
    c3.metric("Travel km/month", f"{b['Total travel (km/month)']:,.0f}", f"{b['Total travel (km/month)'] - a['Total travel (km/month)']:+,.0f}", delta_color="inverse")
    c4.metric("Coverage gap (customers)", int(b["Unassigned customers (coverage gap)"]), int(b["Unassigned customers (coverage gap)"] - a["Unassigned customers (coverage gap)"]), delta_color="inverse")
    c5.metric("Relationship value kept", f"{b['Relationship-weight kept (%)']:.0f}%")
    long = pd.concat([mets["Legacy (as-is)"].assign(method="Legacy"), al.metrics.assign(method="Optimised")])
    long = long.sort_values(["region", "emp_id"])
    fig = px.bar(long, x="emp_id", y="utilization", color="method", barmode="group", facet_row="method",
                 color_discrete_map={"Legacy": "#d62728", "Optimised": "#2ca02c"}, height=430,
                 hover_data=["name", "region", "n_customers", "potential_ratio"])
    fig.add_hline(y=1.0, line_dash="dash"); fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------- Map
with tabs[1]:
    mc1, mc2, mc3, mc4 = st.columns(4)
    view = mc1.radio("View", ["optimised", "legacy"], horizontal=True)
    region = mc2.selectbox("Region", list(al.pins.region.unique()))
    emp_opts = ["(all reps)"] + al.emps[al.emps.region == region].emp_id.tolist()
    emp_sel = mc3.selectbox("Highlight rep", emp_opts)
    show_c = mc4.checkbox("Show customers", True)
    fmap = MV.make_map(al, view, region, None if emp_sel == "(all reps)" else emp_sel, show_c)
    show_html(fmap.get_root().render(), 640)
    st.caption("Legacy view: dashed hulls show each rep's customer footprint - they overlap heavily. Optimised view: solid outlines are "
               "contiguous, non-overlapping PIN territories; black-ringed dots are key-account exceptions. Cells are Voronoi "
               "approximations of PIN boundaries (swap in India Post polygons when available).")
    with st.expander("India overview (one bubble per Area Manager)"):
        show_html(MV.india_overview(al).get_root().render(), 420)

# ----------------------------------------------------------------- Reps
with tabs[2]:
    cols = ["emp_id", "name", "region", "grade", "n_customers", "n_pins", "workload_hours", "travel_hours", "utilization",
            "potential_lakh", "target_potential_lakh", "potential_ratio", "flags"]
    st.dataframe(al.metrics[cols], width="stretch", hide_index=True)
    r1, r2 = st.columns(2)
    emp = r1.selectbox("Explain a rep's territory", al.metrics.emp_id)
    st.markdown(X.territory_text(X.territory_summary(al, emp, legacy_m)))
    cid = r2.text_input("Why was this customer assigned? (e.g. C00004)", "C00004")
    if cid:
        res = X.customer_reason(al, cid.strip().upper(), legacy_m)
        st.markdown(f"**{res['decision']}** → {res.get('final', '')}")
        for line in res["reasons"]:
            st.write("• " + line)
    st.subheader("Biggest customer flows (legacy → new)")
    st.dataframe(X.flows(al, 15), width="stretch", hide_index=True)
    if len(al.exceptions):
        with st.expander(f"{len(al.exceptions)} key-account relationship exceptions"):
            st.dataframe(al.exceptions, width="stretch", hide_index=True)

# ----------------------------------------------------------------- Managers
with tabs[3]:
    nv = M.national_view(am)
    st.subheader("National Manager view")
    n1, n2, n3, n4, n5 = st.columns(5)
    n1.metric("Reps", nv["reps"]); n2.metric("Customers", nv["customers"])
    n3.metric("Network utilisation", f"{nv['network_utilization']:.0%}")
    n4.metric("Spread across AMs", f"{nv['util_spread_across_AMs']:.3f}"); n5.metric("Potential (₹ lakh)", f"{nv['potential_lakh']:,.0f}")
    st.subheader("Area Manager view")
    st.dataframe(am, width="stretch", hide_index=True)
    fig2 = px.bar(am, x="region", y=["workload_hours", "capacity_hours"], barmode="group", height=330)
    st.plotly_chart(fig2, width="stretch")
    st.subheader("Automatic imbalance detection")
    alerts = M.detect_imbalance(al.metrics, am)
    st.write(f"{len(alerts)} alerts")
    for a_ in alerts[:12]:
        st.write(f"• [{a_['level']}] {a_['msg']}")

# ----------------------------------------------------------------- Baselines
with tabs[4]:
    st.subheader("Same scoring engine applied to every method")
    st.dataframe(cmp.round(3), width="stretch")
    st.caption("Honest reading: nearest-home and k-means use less travel than ours because they ignore workload, potential, "
               "capacity and relationships entirely (utilisation std-dev 0.33-0.61 vs 0.05).")
    if st.button("Run continuity-vs-balance trade-off sweep"):
        sw = get_sweep()
        st.dataframe(sw.round(3), width="stretch", hide_index=True)
        st.plotly_chart(px.line(sw, x="relationship_kept_pct", y="util_std", markers=True, text="w_cont",
                                labels=dict(relationship_kept_pct="Relationship value kept (%)", util_std="Utilisation std-dev")),
                        width="stretch")

# ----------------------------------------------------------------- What-if
with tabs[5]:
    st.caption("Re-aligns only the affected region, warm-started from the current alignment (minimal disruption).")
    kind = st.radio("Scenario", ["Rep leaves", "Rep joins", "Demand spike (new customers)"], horizontal=True)
    new_al, dif = None, None
    if kind == "Rep leaves":
        who = st.selectbox("Which rep?", al.emps.emp_id)
        if st.button("Simulate departure"):
            new_al, dif = WI.employee_leaves(al, who)
    elif kind == "Rep joins":
        reg = st.selectbox("Region", list(al.pins.region.unique()), key="join_reg")
        grade = st.selectbox("Grade", list(C.GRADES), index=1)
        if st.button("Simulate new joiner"):
            new_al, dif = WI.employee_joins(al, reg, grade=grade)
    else:
        reg = st.selectbox("Region", list(al.pins.region.unique()), key="spike_reg")
        n = st.slider("New customers", 10, 150, 60, 10)
        if st.button("Simulate demand spike"):
            new_al, dif = WI.add_customers(al, reg, n)
    if dif:
        st.session_state["pending"] = (new_al, dif)
    if st.session_state.get("pending"):
        pal, pd_ = st.session_state["pending"]
        st.success(f"**{pd_['title']}** - {pd_['customers_moved']} customers moved ({pd_['share_moved']:.0%} of {pd_['region']}).")
        k1, k2, k3 = st.columns(3)
        k1.metric(f"{pd_['region']} avg utilisation", f"{pd_['region_util_mean_after']:.0%}", f"{pd_['region_util_mean_after'] - pd_['region_util_mean_before']:+.0%}", delta_color="off")
        k2.metric("Utilisation spread", f"{pd_['region_util_std_after']:.3f}", f"{pd_['region_util_std_after'] - pd_['region_util_std_before']:+.3f}", delta_color="inverse")
        k3.metric("Overloaded reps", pd_["overloaded_after"], pd_["overloaded_after"] - pd_["overloaded_before"], delta_color="inverse")
        st.info(pd_["recommendation"])
        if st.button("Adopt this scenario as the current alignment"):
            st.session_state.update(current=pal, scenario_diff=pd_, pending=None)
            st.rerun()

# ----------------------------------------------------------------- Ask
with tabs[6]:
    st.caption(nlq.HELP)
    q = st.text_input("Ask about territories", placeholder="who is overloaded?")
    if q:
        txt, df, new = nlq.ask(q, al, cmp, legacy_m, d["managers"])
        st.markdown(txt)
        if df is not None and len(df):
            st.dataframe(df, width="stretch", hide_index=True)
        if new is not None:
            st.caption("This was a what-if. Use the What-if tab to adopt a scenario.")

# ----------------------------------------------------------------- Limitations
with tabs[7]:
    for i, (t, dsc) in enumerate(LIMITATIONS, 1):
        st.markdown(f"**{i}. {t}.** {dsc}")

```

---

## `benchmark_scaling.py`  (20 lines)

```python
"""Measure solve time / quality as ONE region grows.   python benchmark_scaling.py
Shows why the system decomposes India into Area-Manager regions (~8 reps each)."""
import copy, time
import numpy as np, pandas as pd
import config as C, data_gen, pipeline as P

base = copy.deepcopy(C.REGIONS)
rows = []
for n_pins, n_emp in [(100, 8), (200, 16), (400, 32), (800, 64)]:
    C.PINS_PER_REGION = n_pins
    C.REGIONS = [dict(base[0], n_emp=n_emp, spread_km=base[0]["spread_km"] * np.sqrt(n_pins / 100))]
    d = data_gen.generate(save=False)
    t = time.time()
    al = P.align(d["pins"], d["employees"], d["customers"])
    rows.append(dict(pins=n_pins, reps=n_emp, customers=len(d["customers"]), seconds=round(time.time() - t, 1),
                     util_std=round(al.metrics.utilization.std(ddof=0), 3),
                     util_min=round(al.metrics.utilization.min(), 2),
                     fragmented=int((al.metrics_pin_level.fragments > 1).sum())))
print(pd.DataFrame(rows).to_string(index=False))
print("\nNationwide: ~19k PIN codes / ~8 reps per Area Manager region => embarrassingly parallel across regions.")

```

---

## `tests/test_smoke.py`  (187 lines)

```python
"""Run:  pytest -q      (about 1-2 minutes)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import data_gen, geo, workload as W, optimizer as O, continuity as K, pipeline as P
import whatif as WI, explain as X, nlq, metrics as M, mapviz as MV, baselines as B


@pytest.fixture(scope="module")
def d():
    return data_gen.generate(save=False)


@pytest.fixture(scope="module")
def al(d):
    return P.align(d["pins"], d["employees"], d["customers"])


# ------------------------------------------------------------------ geo
def test_haversine_known_distance():
    assert abs(geo.haversine_km(28.6139, 77.2090, 19.0760, 72.8777) - 1148) < 10


def test_tour_square_and_trivial():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert geo.tour_length_km(sq) == pytest.approx(4.0)
    assert geo.tour_length_km(np.empty((0, 2))) == 0.0
    assert geo.tour_length_km(np.array([[0, 0], [3, 4]], float)) == pytest.approx(10.0)


def test_components_and_adjacency():
    adj = geo.delaunay_adjacency(np.random.default_rng(0).random((40, 2)) * 10)
    assert geo.n_components(list(range(40)), adj) == 1
    assert geo.n_components([], adj) == 0
    assert geo.n_components([0, 1, 2], [{1}, {0}, set()]) == 2


# ------------------------------------------------------------------ data
def test_dataset_integrity(d):
    c, e, p = d["customers"], d["employees"], d["pins"]
    assert c.cust_id.is_unique and e.emp_id.is_unique and p.pin.is_unique
    assert set(c.pot_class) <= set("ABCD") and c.potential_lakh.gt(0).all()
    assert set(c.pin) <= set(p.pin)
    assert set(c.legacy_emp) - {"VACANT"} <= set(e.emp_id)
    assert (c.legacy_emp == "VACANT").any()                      # coverage gap exists to be fixed
    assert p.lat.between(6, 37.5).all() and p.lon.between(68, 98).all()   # inside India bbox


def test_loaders_validate(tmp_path):
    bad = tmp_path / "c.csv"; pd.DataFrame(dict(cust_id=["C1"], pin=["110001"])).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        data_gen.load_customers_csv(str(bad))
    ok = tmp_path / "ok.csv"
    pd.DataFrame(dict(cust_id=["C1", "C2"], pin=["110001", "110002"], lat=[28.6, 28.7], lon=[77.2, 77.3],
                      pot_class=["a", "D"])).to_csv(ok, index=False)
    df = data_gen.load_customers_csv(str(ok))
    assert list(df.pot_class) == ["A", "D"] and (df.legacy_emp == "VACANT").all()
    pins = tmp_path / "p.csv"
    pd.DataFrame(dict(pincode=["110001", "110001", "110002", "999999"], latitude=["28.6", "28.62", "28.7", "0"],
                      longitude=["77.2", "77.22", "77.3", "0"])).to_csv(pins, index=False)
    out = data_gen.load_real_pins(str(pins))
    assert len(out) == 2 and out.pin.is_unique                  # dedup + (0,0) dropped


# ------------------------------------------------------------------ optimiser invariants
def test_every_pin_one_owner_and_all_customers_covered(al):
    assert set(al.pin_owner) == set(al.pins.pin)
    assert set(al.pin_owner.values()) <= set(al.emps.emp_id)
    assert al.cust.final_emp.isin(al.emps.emp_id).all()                 # no coverage gap
    assert al.cust.new_emp.isin(al.emps.emp_id).all()
    assert al.metrics.n_pins.min() >= 1                                  # nobody left empty-handed
    assert (al.cust.final_emp == al.cust.new_emp).sum() >= len(al.cust) * 0.9   # exceptions are the minority


def test_territories_contiguous_and_pin_level_has_no_overlap(al):
    assert (al.metrics_pin_level.fragments <= 1).all()
    assert (al.metrics_pin_level.overlap_pins == 0).all()


def test_workload_conservation(al):
    c = al.cust
    assert al.metrics.n_customers.sum() == len(c)
    assert al.metrics.visit_hours.sum() == pytest.approx(c.visit_hours.sum(), abs=len(al.metrics) * 0.06)


def test_beats_legacy_and_baselines(al):
    cmp, _ = P.compare_methods(al)
    o = cmp["Optimised (ours)"]
    for other in ["Legacy (as-is)", "B1 Nearest-home", "B2 K-means"]:
        assert o["Utilisation std-dev"] < 0.5 * cmp[other]["Utilisation std-dev"]
        assert o["Potential-ratio std-dev"] < cmp[other]["Potential-ratio std-dev"]
    assert o["Total travel (km/month)"] < cmp["Legacy (as-is)"]["Total travel (km/month)"]
    assert o["Overloaded reps (>110%)"] == 0 and o["Unassigned customers (coverage gap)"] == 0


def test_deterministic(d):
    a1 = P.align(d["pins"], d["employees"], d["customers"], regions=["Chennai"])
    a2 = P.align(d["pins"], d["employees"], d["customers"], regions=["Chennai"])
    assert a1.pin_owner == a2.pin_owner


def test_exceptions_respect_rules(al):
    ex, m = al.exceptions, al.metrics.set_index("emp_id")
    assert len(ex) > 0 and ex.dist_to_rep_territory_km.max() <= 4.0
    assert (ex.rep_util_after <= 1.0 + 1e-9).all() and (ex.donor_util_after >= 0.85 - 1e-9).all()
    assert set(ex.pot_class) <= {"A", "B"}
    assert al.cust.is_exception.sum() == len(ex)


def test_continuity_weight_monotone(d):
    lo = P.align(d["pins"], d["employees"], d["customers"], C.Weights(w_cont=0), exceptions=False, regions=["Mumbai"])
    hi = P.align(d["pins"], d["employees"], d["customers"], C.Weights(w_cont=60), exceptions=False, regions=["Mumbai"])
    kp = lambda a: W.summary_kpis(a.metrics, a.cust[a.cust.region == "Mumbai"], "final_emp")["Relationship-weight kept (%)"]
    assert kp(hi) >= kp(lo)


# ------------------------------------------------------------------ edge cases
def _subset(d, region, n_pins, emp_ids):
    pins = d["pins"][d["pins"].region == region].head(n_pins)
    emps = d["employees"][d["employees"].emp_id.isin(emp_ids)].reset_index(drop=True)
    cust = d["customers"][d["customers"].pin.isin(pins.pin)].copy()
    return pins.reset_index(drop=True), emps, cust.reset_index(drop=True)


def test_single_rep_region(d):
    reg = "Chennai"
    e1 = d["employees"][d["employees"].region == reg].emp_id.iloc[:1].tolist()
    pins, emps, cust = _subset(d, reg, 40, e1)
    al1 = P.align(pins, emps, cust)
    assert set(al1.pin_owner.values()) == set(e1) and al1.cust.final_emp.eq(e1[0]).all()


def test_tiny_region_three_pins(d):
    reg = "Kolkata"
    e2 = d["employees"][d["employees"].region == reg].emp_id.iloc[:2].tolist()
    pins, emps, cust = _subset(d, reg, 3, e2)
    al2 = P.align(pins, emps, cust)
    assert len(set(al2.pin_owner.values())) == 2 and al2.cust.final_emp.notna().all()


# ------------------------------------------------------------------ what-if
def test_leave_join_spike(al):
    new, dif = WI.employee_leaves(al, "E005")
    assert "E005" not in set(new.cust.final_emp) and "E005" not in set(new.emps.emp_id)
    assert new.cust.final_emp.isin(new.emps.emp_id).all() and dif["share_moved"] < 0.30
    other = al.cust[al.cust.region != dif["region"]].set_index("cust_id").final_emp
    assert (new.cust.set_index("cust_id").final_emp.reindex(other.index) == other).all()   # other regions untouched
    new, dif = WI.employee_joins(al, "Mumbai")
    assert new.metrics[new.metrics.emp_id == "E043"].n_pins.iloc[0] >= 1 and dif["reps_after"] == dif["reps_before"] + 1
    new, dif = WI.add_customers(al, "Bengaluru", 30)
    assert len(new.cust) == len(al.cust) + 30 and new.cust.final_emp.notna().all()
    with pytest.raises(ValueError):
        WI.employee_leaves(al, "E999")


# ------------------------------------------------------------------ explain / nlq / metrics / map
def test_explanations_and_nlq(al):
    cmp, _ = P.compare_methods(al)
    lm = X.legacy_metrics(al)
    kinds = {X.customer_reason(al, cid, lm)["decision"].split(" (")[0] for cid in al.cust.cust_id.sample(300, random_state=1)}
    assert {"KEPT", "MOVED"} <= kinds
    assert X.customer_reason(al, "C99999", lm)["decision"] == "unknown"
    assert "E021" in X.territory_text(X.territory_summary(al, "E021", lm))
    for q, needle in [("who is overloaded?", "rep"), ("show territory of E012", "E012"), ("who covers PIN 400045?", "400045"),
                      ("compare with baseline", "Utilisation"), ("national summary", "National"), ("blah blah", "Try")]:
        txt, _, _ = nlq.ask(q, al, cmp, lm)
        assert needle in txt


def test_rollups(al, d):
    am = M.area_manager_view(al.metrics, al.cust, "final_emp", d["managers"])
    nv = M.national_view(am)
    assert nv["reps"] == len(al.emps) and nv["customers"] == len(al.cust) and len(am) == 6
    assert isinstance(M.detect_imbalance(al.metrics, am), list)


def test_maps_render(al, tmp_path):
    for view in ("optimised", "legacy"):
        f = tmp_path / f"{view}.html"
        MV.make_map(al, view, "Mumbai").save(f)
        assert f.stat().st_size > 50_000 and "leaflet" in f.read_text(encoding="utf-8").lower()
    MV.india_overview(al).save(tmp_path / "o.html")

```
