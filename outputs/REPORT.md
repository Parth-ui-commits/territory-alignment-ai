# Territory Alignment - run report

## Headline
Utilisation spread (std-dev) falls from 0.43 to 0.05; overloaded reps 15 -> 0, underloaded 20 -> 0. Monthly travel drops 50% (16,862 -> 8,441 km). Coverage gaps 133 -> 0 customers, fragmented territories 42 -> 0. Relationship continuity kept: 36% of customers (40% by relationship value).

Runtime: 3.2s for 42 reps, 600 PINs, 2244 customers (6 regions solved independently - the decomposition is what scales to India).

## Baseline comparison (same scoring engine for every method)

|                                     |   Legacy (as-is) |   B1 Nearest-home |   B2 K-means |   Optimised (ours) |
|:------------------------------------|-----------------:|------------------:|-------------:|-------------------:|
| Utilisation std-dev                 |            0.425 |             0.61  |        0.333 |              0.053 |
| Utilisation min                     |            0.256 |             0.057 |        0.27  |              0.833 |
| Utilisation max                     |            1.74  |             2.492 |        1.534 |              0.999 |
| Overloaded reps (>110%)             |           15     |            15     |       12     |              0     |
| Underloaded reps (<80%)             |           20     |            21     |       16     |              0     |
| Potential-ratio std-dev             |            0.49  |             0.712 |        0.456 |              0.119 |
| Total travel (km/month)             |        16862.4   |          6606.5   |     7000     |           8441.1   |
| Avg travel share of workload        |            0.136 |             0.055 |        0.057 |              0.069 |
| Avg customer-to-centroid km         |            6.584 |             3.642 |        2.978 |              3.989 |
| PIN codes shared by >1 rep          |          448     |            73     |       46     |             57     |
| Fragmented territories              |           42     |             2     |        0     |              0     |
| Unassigned customers (coverage gap) |          133     |             0     |        0     |              0     |
| Customers keeping same rep (%)      |          100     |            33.444 |       28.138 |             36.428 |
| Relationship-weight kept (%)        |          100     |            34.257 |       28.509 |             40.329 |
| Customers total                     |         2244     |          2244     |     2244     |           2244     |

## Key-account relationship exceptions
65 A/B-class customers stay with their legacy rep outside the PIN territory (max 4 km away, rep has spare capacity).

## Continuity vs balance trade-off

|   w_cont |   util_std |   util_max |   util_min |   potential_ratio_std |   travel_km |   customers_kept_pct |   relationship_kept_pct |   shared_pins |
|---------:|-----------:|-----------:|-----------:|----------------------:|------------:|---------------------:|------------------------:|--------------:|
|        0 |      0.047 |      1.01  |      0.815 |                 0.109 |      8512.7 |               32.496 |                  36.751 |            92 |
|        2 |      0.052 |      1.019 |      0.845 |                 0.112 |      8206.2 |               34.107 |                  37.401 |            74 |
|        4 |      0.056 |      1.002 |      0.814 |                 0.113 |      8250.2 |               33.681 |                  37.535 |            67 |
|        8 |      0.054 |      1.009 |      0.774 |                 0.114 |      8373.5 |               36.191 |                  40.477 |            66 |
|       16 |      0.053 |      0.999 |      0.833 |                 0.119 |      8441.1 |               36.428 |                  40.329 |            57 |
|       30 |      0.068 |      1.037 |      0.739 |                 0.129 |      8359.4 |               38.181 |                  42.491 |            54 |
|       60 |      0.078 |      1.139 |      0.791 |                 0.142 |      8680.5 |               39.886 |                  44.469 |            52 |

## Limitations / failure cases

1. **PIN-code atomicity caps relationship continuity.** Territories are unions of whole PIN codes, so customers inside one PIN cannot be split between reps. On this dataset the theoretical ceiling for continuity is ~50% of customers; on the default seed we reach ~36% (40% by value) because geography and balance also matter. The key-account exception layer recovers part of the gap at the price of explicit PIN overlaps. A client that wants more continuity must accept more overlap or lower balance.

2. **Granularity and capacity shortfalls cannot be optimised away.** A PIN with a very heavy customer book is indivisible, which limits how finely workload can be balanced (~+/-5% here). More importantly, when a rep leaves or demand spikes the region may simply lack capacity - on the default seed the what-if runs show 1 overloaded rep after a departure and 3 after a +60-customer spike. No re-alignment fixes that; the system can only flag it and recommend headcount.

3. **Synthetic data, approximate boundaries and a simple travel model.** PIN centroids/customers are synthetic around real city centres; PIN boundaries are approximated by Voronoi cells, not India Post polygons; travel uses straight-line x1.3 circuity at a flat city speed (no road network, traffic or public transport). Absolute hours/km are therefore indicative - the relative comparison between methods is what we claim.

4. **Heuristic, not a proven optimum; weights are judgement calls.** Local search can stall in local minima and carries no optimality gap. The objective weights, relationship weights (class x tenure), visit frequencies and visit durations are assumptions that a real deployment must calibrate with the business.

5. **Quality degrades if one region is made very large.** The solver is a local search. In the benchmark (benchmark_scaling.py) one region of 100 PINs / 8 reps solves in 0.4 s with utilisation std-dev 0.05, but at 400-800 PINs / 32-64 reps in a single region it converges to a local minimum (std-dev ~0.085, a few reps at ~60% utilisation, more restarts did not help). The lowest-utilised reps there live 10-12 km from their territory centre, which is the likely cause (not isolated by an ablation). The design answer is the Area-Manager hierarchy: keep each solve at ~6-10 reps and scale by solving regions in parallel - cost grows linearly with the number of regions.

6. **Static snapshot.** Potential is taken as given (no forecasting), territories are optimised for one month's workload, and the system ignores leave calendars, rep skills/language, customer time windows and rep preferences.
