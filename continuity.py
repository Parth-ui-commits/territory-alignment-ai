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
