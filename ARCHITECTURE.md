# Architecture - Intelligent India Territory Alignment & Workforce Optimisation

## 1. What the problem really asks
Given customers (with PIN, potential class, visit workload), a field force (capacity, home base, grade) and the
current (messy) customer->rep assignment, produce a **territory design + assignment** that:
balances workload and business potential, stays geographically compact, **preserves relationships**, rolls up
Employee -> Area Manager -> National Manager, flags imbalance, explains decisions, and is **measurably better than a
baseline**, shown on an **India map**.

| Problem-statement requirement | Where it lives |
|---|---|
| India map, PIN as unit, territory boundaries | `mapviz.py` (Folium/OSM, Voronoi PIN cells dissolved per rep) |
| Assign customers to employees, show who owns what | `optimizer.py` -> `pipeline.py` -> map tooltips, `customer_assignments.csv` |
| Workload + potential per employee | `workload.py` (visit hours + beat-tour travel; potential vs fair share) |
| Keep existing relationships | continuity term in objective + `continuity.py` key-account exceptions |
| Roll-ups AM / NM | `metrics.py` (`area_manager_view`, `national_view`) |
| Overloaded / underloaded / high / low potential / geo-inefficient | `workload.flag_row`, `metrics.detect_imbalance` |
| Reason for decisions | `explain.py` (computed from real numbers) |
| Baseline comparison | `baselines.py` + `pipeline.compare_methods` (same scoring engine) |
| >=3 limitations | `run_pipeline.LIMITATIONS` (6 documented, with evidence) |
| Optional: re-alignment, what-if, NL, dashboard, old-vs-new, imbalance detection | `whatif.py`, `nlq.py`, `app.py`, legacy view, `metrics.py` |

## 2. System architecture

```
 DATA LAYER                    CORE ENGINE (pure Python, no network needed)                 PRESENTATION
 ----------                    --------------------------------------------                 ------------
 PIN centroids  ----+
 (synthetic or real |     +-----------------+     +------------------------+
  India Post CSV)   +---> | workload.py     | --> | optimizer.py            |
 Customers (class,  |     | PIN aggregation |     | per-REGION problem      |      +-----------------------+
  potential, legacy |     | visit hours     |     | 1 capacity-aware Lloyd  | ---> | continuity.py         |
  owner, tenure)    |     | beat-tour travel|     |   init (continuity-     |      | key-account exceptions|
 Employees (grade,  |     | ground-truth    |     |   biased)               |      +-----------+-----------+
  capacity, home)   |     | evaluation      |     | 2 local search: moves + |                  |
 Managers (AM/NM)---+     +--------^--------+     |   swaps, exact delta,   |                  v
                                   |              |   contiguity guard      |      +-----------------------+
                                   |              | 3 fragment repair       |      | pipeline.Alignment    |
                                   |              +------------------------+      | pin_owner, cust,      |
                           same scoring engine                                      | metrics, exceptions   |
                                   |                                                +-----+------+---------+
                    +--------------+-----------+                                         |      |
                    | baselines.py             |                  +----------------------+      +----------------+
                    | B1 nearest-home          |                  v                                              v
                    | B2 k-means + Hungarian   |      metrics.py (Emp->AM->NM, alerts)              explain.py (reasons, flows)
                    +--------------------------+      whatif.py (leave / join / spike, warm start)  nlq.py (NL questions)
                                                                  |                                              |
                                                                  +------------------+---------------------------+
                                                                                     v
                                                          mapviz.py (Folium maps)   app.py (Streamlit, 8 tabs)
                                                          run_pipeline.py (CSV, PNG, HTML, REPORT.md)
```

**Key design decisions**
1. **PIN code = atomic unit.** Every PIN has exactly one owner -> no overlaps, no gaps, territories are naturally
   describable ("rep X owns these 14 PINs") and map to real boundaries when PIN polygons are available.
2. **Hierarchical decomposition.** India -> Area-Manager regions (~6-10 reps). Regions are solved independently, so
   cost grows linearly with the number of regions and parallelises trivially (see benchmark).
3. **One scoring engine.** Legacy, baselines and optimised output are all scored by `workload.employee_metrics`,
   so the comparison cannot be rigged by a different yardstick.
4. **Explainable by construction.** The optimiser is an explicit weighted objective; every reason shown to the user
   is computed from the same quantities (distance, utilisation before/after, relationship weight).
5. **Warm-start re-alignment.** What-ifs start from the current alignment and treat it as the new relationship
   baseline, so only necessary PINs move.

## 3. Data model

| Table | Key columns |
|---|---|
| `pins` | pin, region, am, area_name, lat, lon |
| `employees` | emp_id, name, am, region, grade (Senior/Mid/Junior), capacity_hours, potential_weight, home_pin/lat/lon, speed_kmph |
| `customers` | cust_id, pin, lat, lon, pot_class (A/B/C/D), potential_lakh, visits_pm, hours_per_visit, legacy_emp (or `VACANT`), relationship_years |
| `managers` | mgr_id, role, region, reports_to |

Business rules (all in `config.py`): class A/B/C/D = 4/2/1/0.5 visits a month, 1.5/1.25/1.0/0.75 h per visit,
mean potential 14/6/2.5/0.8 INR lakh; rep capacity 150/140/125 h (Senior/Mid/Junior); fair potential share
weighted 1.3/1.0/0.75; road circuity 1.3; thresholds: overloaded >110%, underloaded <80%.

**Dataset used in the demo (synthetic, seed 42):** 6 metros (Delhi NCR, Mumbai, Bengaluru, Hyderabad, Chennai,
Kolkata) x 100 PINs, 42 reps, 6 Area Managers, 2,244 customers. The legacy alignment is deliberately messy:
historically skewed books (rep utilisation 26%-174%), 133 orphaned customers of departed reps (`VACANT`), PINs
served by several reps. Real data plugs in through `data_gen.load_real_pins` / `load_customers_csv`.

## 4. Workflow (what `run_pipeline.py` / the dashboard does)
1. **Load/generate** pins, reps, customers (+ legacy owner).
2. **Aggregate to PIN level**: visit-hours, potential, stop frequency per PIN (`workload.pin_table`).
3. **Build a problem per region**: Delaunay PIN adjacency graph, relationship matrix R[pin, rep] (class x tenure
   weights of customers legacy-owned by each live rep), fair targets (capacity-proportional hours, weight-proportional
   potential).
4. **Initialise**: capacity-aware, continuity-biased Lloyd iterations seeded at blend(legacy centroid, home).
5. **Local search**: steepest-descent over boundary-PIN moves and boundary swaps with exact incremental cost on the
   two touched territories; moves that would fragment a territory are rejected.
6. **Fragment repair**, second short local search.
7. **Key-account exceptions**: A/B customers within 4 km of their legacy rep's territory stay with that rep if the
   rep has headroom and the donor stays >= 85% utilised (greedy by relationship weight; logged and explainable).
8. **Evaluate** with true beat-tour travel; compare with Legacy, B1 (nearest-home), B2 (k-means+Hungarian).
9. **Roll up** Employee -> AM -> NM, detect imbalance, generate explanations, render maps/tables/charts.
10. **What-if**: leave / join / demand spike re-solve only the affected region, warm-started.

## 5. The algorithm in detail

Objective per region (rep i, PIN p, n = reps):

```
J =  w_wl   * mean_i (util_i - U*)^2                      workload balance (util = hours / capacity)
   + w_pot  * mean_i (potential_i / fair_i - 1)^2         opportunity balance
   + w_cap  * mean_i max(0, util_i - 1)^2                 capacity breach
   + w_geo  * sum_i sum_p W_p |x_p - c_i| / (W_tot * R)   compactness (workload-weighted)
   + w_cont * (relationship weight moved away from live owners) / (total relationship weight)
```
Default weights 30 / 12 / 40 / 6 / 16. `util_i` includes estimated travel (BHH tour proxy
1.9*sqrt(m)*mean-dist + commute, calibrated x1.28 against the exact tour model). Ground-truth evaluation uses a
**frequency-layered beat plan**: cycle k = closed NN+2-opt tour from the rep's home through all PINs needing >= k
visits a month, x1.3 circuity / city speed.

Why this works: PIN-level ownership gives clean topology; the exact incremental delta makes each candidate move O(territory
size); the contiguity guard (connected-components test) prevents "archipelago" territories; swaps fix the common case
where two neighbouring reps are both at capacity.

**Continuity ceiling (an honest insight):** with PIN-atomic ownership the best conceivable continuity on this data is
50.1% of customers (each PIN keeps its majority legacy rep). We reach 36% (40% by value). The exception layer exists to
go beyond the ceiling selectively, at a controlled price in overlap/travel.

## 6. Results (default seed; `outputs/REPORT.md` is regenerated every run)

| Metric | Legacy | B1 nearest-home | B2 k-means | **Optimised** |
|---|---|---|---|---|
| Utilisation std-dev | 0.425 | 0.610 | 0.333 | **0.053** |
| Utilisation min / max | 0.26 / 1.74 | 0.06 / 2.49 | 0.27 / 1.53 | **0.83 / 1.00** |
| Overloaded (>110%) / underloaded (<80%) reps | 15 / 20 | 15 / 21 | 12 / 16 | **0 / 0** |
| Potential-ratio std-dev | 0.49 | 0.71 | 0.46 | **0.12** |
| Travel km / month | 16,862 | 6,607 | 7,000 | 8,441 |
| PINs shared by >1 rep | 448 | 73 | 46 | 57 |
| Fragmented territories | 42 | 2 | 0 | **0** |
| Coverage gap (customers) | 133 | 0 | 0 | **0** |
| Customers keeping same rep | 100% (by definition) | 33% | 28% | **36%** (40% by value) |

Reading it honestly: B1/B2 travel less than we do because they ignore workload, potential, capacity and relationships
entirely - their utilisation spread is 6-12x worse. Against legacy we cut travel by 50%.
Continuity-vs-balance trade-off (sweep of `w_cont` 0..60): relationship value kept 37% -> 44% while utilisation
std-dev goes 0.047 -> 0.078 - the business picks its point on the frontier (sidebar presets).

What-if (default seed): a rep leaving Delhi NCR moves ~17% of the region's customers, pushes region utilisation
92% -> 104% and the system recommends hiring ~1 rep; a +60-customer spike in Bengaluru leaves 3 reps overloaded and
also recommends ~1 hire; adding a rep in Mumbai moves ~20% of Mumbai's customers and brings the region's spread to 0.04.

## 7. Scalability (measured, `benchmark_scaling.py`)
| PINs in one region | Reps | Solve time | Util std-dev |
|---|---|---|---|
| 100 | 8 | 0.5 s | 0.047 |
| 200 | 16 | 1.4 s | 0.037 |
| 400 | 32 | 2.9 s | 0.084 |
| 800 | 64 | 5.4 s | 0.086 |

Time scales ~linearly, but quality drops for oversized regions (local minimum; restarts did not help). So the design is
~8 reps per Area-Manager region and horizontal scaling across regions: ~19,000 India PINs => a few hundred independent
solves, each ~seconds.

## 8. Limitations / failure cases (also in the app and REPORT.md)
1. PIN atomicity caps continuity (50% ceiling here); exceptions trade it against overlap.
2. Indivisible heavy PINs limit balance granularity; capacity shortfalls (departure, spike) cannot be fixed by
   re-alignment - only flagged with a hiring recommendation.
3. Synthetic data; Voronoi approximations of PIN boundaries; straight-line x1.3 travel, flat speeds, no traffic.
4. Heuristic (no optimality gap); weights and relationship/visit assumptions need business calibration.
5. Oversized single regions converge to poor local minima (benchmark above).
6. Static monthly snapshot: no forecasting of potential, leave calendars, skills, language, time windows.

## 9. Evaluation-criteria map
Problem understanding -> section 1/5; Technical approach -> OR districting + exact incremental local search + hierarchy;
Geospatial -> Folium OSM map, PIN cells, dissolved territory outlines; Workload & potential -> `workload.py`, fair-share
targets; Relationship continuity -> objective term + exceptions + measured trade-off; Evaluation -> 2 baselines + legacy,
18 automated tests, mutation-checked; Innovation -> key-account exceptions, continuity ceiling analysis, warm-start
what-if with hiring recommendation, NL interface; Execution -> `python run_pipeline.py` and `streamlit run app.py` both
run end-to-end.

## 10. Suggested 5-minute demo script
1. Legacy map of Mumbai (dashed overlapping hulls) vs optimised map (contiguous territories).
2. Overview tab: before/after KPI cards and utilisation chart.
3. Reps tab: pick a rep, show explanation; ask "why was C00004 moved?".
4. Baselines tab: table + honest reading; trade-off sweep.
5. What-if: "E005 leaves" -> hiring recommendation -> adopt scenario.
6. Ask tab: "who is overloaded?", "compare with baseline".
7. Close with limitations and the scaling story.
