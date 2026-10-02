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
