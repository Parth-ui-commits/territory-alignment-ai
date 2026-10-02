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
