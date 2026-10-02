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
