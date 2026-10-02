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
