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
