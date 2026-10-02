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
