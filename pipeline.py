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
