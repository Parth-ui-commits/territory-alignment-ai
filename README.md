# Intelligent India Territory Alignment & Workforce Optimisation (36-hour AI challenge)

Hybrid OR + analytics system: PIN-level territory design, capacity/potential/relationship-aware assignment,
Employee -> Area Manager -> National roll-ups, India map, baselines, explanations, what-if, NL queries.
See **ARCHITECTURE.md** for the full design, algorithm, results and limitations.

## Quick start (Python 3.10+)
```bash
pip install -r requirements.txt
python run_pipeline.py            # ~40 s: writes outputs/ (maps, CSVs, charts, REPORT.md)
streamlit run app.py              # interactive dashboard
pytest -q                         # 18 tests, ~20 s
python benchmark_scaling.py       # scaling table
```
Open `outputs/maps/mumbai_legacy.html` and `mumbai_optimised.html` in a browser for the before/after maps.
Options: `python run_pipeline.py --regenerate --preset "Max continuity" --no-exceptions --no-sweep`.

## Files
| File | Role |
|---|---|
| `config.py` | business rules, regions, objective weights/presets |
| `data_gen.py` | synthetic India dataset (+ loaders for real PIN / customer CSVs) |
| `geo.py` | haversine, projection, Delaunay PIN adjacency, tour length (NN+2-opt) |
| `workload.py` | ground-truth workload / potential / travel evaluation and KPIs |
| `optimizer.py` | the territory optimiser (init + local search + fragment repair) |
| `continuity.py` | key-account relationship exceptions |
| `baselines.py` | nearest-home and k-means baselines |
| `pipeline.py` | orchestration, `Alignment` object, method comparison, trade-off sweep |
| `metrics.py` | AM / NM roll-ups, imbalance detection |
| `explain.py` | reasons per customer/territory, flows, optional LLM narrative |
| `whatif.py` | rep leaves / joins / demand spike with warm-start re-alignment |
| `nlq.py` | natural-language questions (deterministic router, optional LLM fallback) |
| `mapviz.py` | Folium maps (PIN cells, territory outlines, customers, homes) |
| `run_pipeline.py` | one-command end-to-end run |
| `app.py` | Streamlit dashboard |
| `tests/test_smoke.py` | invariants, edge cases, what-ifs, NL, maps |

## Using your own data
* **PINs**: `data_gen.load_real_pins("pincode.csv", state="Karnataka")` - needs `pincode, latitude, longitude`
  (e.g. data.gov.in All-India Pincode Directory with geocodes). Dedupes to one centroid per PIN, drops invalid coords.
* **Customers**: `data_gen.load_customers_csv(...)` - required `cust_id, pin, lat, lon, pot_class`; optional
  `potential_lakh, legacy_emp, relationship_years`.
* **Employees / managers**: same columns as `data/employees.csv`, `data/managers.csv`; add a `region` and `am` per rep.
  Then call `pipeline.align(pins, employees, customers)`.
* Real PIN **polygons**: replace `mapviz.pin_cells()` (returns pin -> shapely polygon); nothing else changes.

## Optional LLM layer
Set `ANTHROPIC_API_KEY` (and `pip install anthropic`) to let unrecognised questions in the Ask tab be answered from the
computed facts only. Everything else works offline. This path is **not covered by the tests** (needs a key).

## Notes
* The dataset is synthetic (real city centres, synthetic PINs/customers); absolute hours/km are indicative.
* Streamlit >= 1.35. The map is embedded with `st.iframe` when available (the old `components.html` is deprecated) and falls back to `components.html` otherwise (fallback path untested on old versions).
