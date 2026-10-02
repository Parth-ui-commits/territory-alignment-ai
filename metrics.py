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
