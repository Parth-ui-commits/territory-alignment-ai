"""Run:  pytest -q      (about 1-2 minutes)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import data_gen, geo, workload as W, optimizer as O, continuity as K, pipeline as P
import whatif as WI, explain as X, nlq, metrics as M, mapviz as MV, baselines as B


@pytest.fixture(scope="module")
def d():
    return data_gen.generate(save=False)


@pytest.fixture(scope="module")
def al(d):
    return P.align(d["pins"], d["employees"], d["customers"])


# ------------------------------------------------------------------ geo
def test_haversine_known_distance():
    assert abs(geo.haversine_km(28.6139, 77.2090, 19.0760, 72.8777) - 1148) < 10


def test_tour_square_and_trivial():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert geo.tour_length_km(sq) == pytest.approx(4.0)
    assert geo.tour_length_km(np.empty((0, 2))) == 0.0
    assert geo.tour_length_km(np.array([[0, 0], [3, 4]], float)) == pytest.approx(10.0)


def test_components_and_adjacency():
    adj = geo.delaunay_adjacency(np.random.default_rng(0).random((40, 2)) * 10)
    assert geo.n_components(list(range(40)), adj) == 1
    assert geo.n_components([], adj) == 0
    assert geo.n_components([0, 1, 2], [{1}, {0}, set()]) == 2


# ------------------------------------------------------------------ data
def test_dataset_integrity(d):
    c, e, p = d["customers"], d["employees"], d["pins"]
    assert c.cust_id.is_unique and e.emp_id.is_unique and p.pin.is_unique
    assert set(c.pot_class) <= set("ABCD") and c.potential_lakh.gt(0).all()
    assert set(c.pin) <= set(p.pin)
    assert set(c.legacy_emp) - {"VACANT"} <= set(e.emp_id)
    assert (c.legacy_emp == "VACANT").any()                      # coverage gap exists to be fixed
    assert p.lat.between(6, 37.5).all() and p.lon.between(68, 98).all()   # inside India bbox


def test_loaders_validate(tmp_path):
    bad = tmp_path / "c.csv"; pd.DataFrame(dict(cust_id=["C1"], pin=["110001"])).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        data_gen.load_customers_csv(str(bad))
    ok = tmp_path / "ok.csv"
    pd.DataFrame(dict(cust_id=["C1", "C2"], pin=["110001", "110002"], lat=[28.6, 28.7], lon=[77.2, 77.3],
                      pot_class=["a", "D"])).to_csv(ok, index=False)
    df = data_gen.load_customers_csv(str(ok))
    assert list(df.pot_class) == ["A", "D"] and (df.legacy_emp == "VACANT").all()
    pins = tmp_path / "p.csv"
    pd.DataFrame(dict(pincode=["110001", "110001", "110002", "999999"], latitude=["28.6", "28.62", "28.7", "0"],
                      longitude=["77.2", "77.22", "77.3", "0"])).to_csv(pins, index=False)
    out = data_gen.load_real_pins(str(pins))
    assert len(out) == 2 and out.pin.is_unique                  # dedup + (0,0) dropped


# ------------------------------------------------------------------ optimiser invariants
def test_every_pin_one_owner_and_all_customers_covered(al):
    assert set(al.pin_owner) == set(al.pins.pin)
    assert set(al.pin_owner.values()) <= set(al.emps.emp_id)
    assert al.cust.final_emp.isin(al.emps.emp_id).all()                 # no coverage gap
    assert al.cust.new_emp.isin(al.emps.emp_id).all()
    assert al.metrics.n_pins.min() >= 1                                  # nobody left empty-handed
    assert (al.cust.final_emp == al.cust.new_emp).sum() >= len(al.cust) * 0.9   # exceptions are the minority


def test_territories_contiguous_and_pin_level_has_no_overlap(al):
    assert (al.metrics_pin_level.fragments <= 1).all()
    assert (al.metrics_pin_level.overlap_pins == 0).all()


def test_workload_conservation(al):
    c = al.cust
    assert al.metrics.n_customers.sum() == len(c)
    assert al.metrics.visit_hours.sum() == pytest.approx(c.visit_hours.sum(), abs=len(al.metrics) * 0.06)


def test_beats_legacy_and_baselines(al):
    cmp, _ = P.compare_methods(al)
    o = cmp["Optimised (ours)"]
    for other in ["Legacy (as-is)", "B1 Nearest-home", "B2 K-means"]:
        assert o["Utilisation std-dev"] < 0.5 * cmp[other]["Utilisation std-dev"]
        assert o["Potential-ratio std-dev"] < cmp[other]["Potential-ratio std-dev"]
    assert o["Total travel (km/month)"] < cmp["Legacy (as-is)"]["Total travel (km/month)"]
    assert o["Overloaded reps (>110%)"] == 0 and o["Unassigned customers (coverage gap)"] == 0


def test_deterministic(d):
    a1 = P.align(d["pins"], d["employees"], d["customers"], regions=["Chennai"])
    a2 = P.align(d["pins"], d["employees"], d["customers"], regions=["Chennai"])
    assert a1.pin_owner == a2.pin_owner


def test_exceptions_respect_rules(al):
    ex, m = al.exceptions, al.metrics.set_index("emp_id")
    assert len(ex) > 0 and ex.dist_to_rep_territory_km.max() <= 4.0
    assert (ex.rep_util_after <= 1.0 + 1e-9).all() and (ex.donor_util_after >= 0.85 - 1e-9).all()
    assert set(ex.pot_class) <= {"A", "B"}
    assert al.cust.is_exception.sum() == len(ex)


def test_continuity_weight_monotone(d):
    lo = P.align(d["pins"], d["employees"], d["customers"], C.Weights(w_cont=0), exceptions=False, regions=["Mumbai"])
    hi = P.align(d["pins"], d["employees"], d["customers"], C.Weights(w_cont=60), exceptions=False, regions=["Mumbai"])
    kp = lambda a: W.summary_kpis(a.metrics, a.cust[a.cust.region == "Mumbai"], "final_emp")["Relationship-weight kept (%)"]
    assert kp(hi) >= kp(lo)


# ------------------------------------------------------------------ edge cases
def _subset(d, region, n_pins, emp_ids):
    pins = d["pins"][d["pins"].region == region].head(n_pins)
    emps = d["employees"][d["employees"].emp_id.isin(emp_ids)].reset_index(drop=True)
    cust = d["customers"][d["customers"].pin.isin(pins.pin)].copy()
    return pins.reset_index(drop=True), emps, cust.reset_index(drop=True)


def test_single_rep_region(d):
    reg = "Chennai"
    e1 = d["employees"][d["employees"].region == reg].emp_id.iloc[:1].tolist()
    pins, emps, cust = _subset(d, reg, 40, e1)
    al1 = P.align(pins, emps, cust)
    assert set(al1.pin_owner.values()) == set(e1) and al1.cust.final_emp.eq(e1[0]).all()


def test_tiny_region_three_pins(d):
    reg = "Kolkata"
    e2 = d["employees"][d["employees"].region == reg].emp_id.iloc[:2].tolist()
    pins, emps, cust = _subset(d, reg, 3, e2)
    al2 = P.align(pins, emps, cust)
    assert len(set(al2.pin_owner.values())) == 2 and al2.cust.final_emp.notna().all()


# ------------------------------------------------------------------ what-if
def test_leave_join_spike(al):
    new, dif = WI.employee_leaves(al, "E005")
    assert "E005" not in set(new.cust.final_emp) and "E005" not in set(new.emps.emp_id)
    assert new.cust.final_emp.isin(new.emps.emp_id).all() and dif["share_moved"] < 0.30
    other = al.cust[al.cust.region != dif["region"]].set_index("cust_id").final_emp
    assert (new.cust.set_index("cust_id").final_emp.reindex(other.index) == other).all()   # other regions untouched
    new, dif = WI.employee_joins(al, "Mumbai")
    assert new.metrics[new.metrics.emp_id == "E043"].n_pins.iloc[0] >= 1 and dif["reps_after"] == dif["reps_before"] + 1
    new, dif = WI.add_customers(al, "Bengaluru", 30)
    assert len(new.cust) == len(al.cust) + 30 and new.cust.final_emp.notna().all()
    with pytest.raises(ValueError):
        WI.employee_leaves(al, "E999")


# ------------------------------------------------------------------ explain / nlq / metrics / map
def test_explanations_and_nlq(al):
    cmp, _ = P.compare_methods(al)
    lm = X.legacy_metrics(al)
    kinds = {X.customer_reason(al, cid, lm)["decision"].split(" (")[0] for cid in al.cust.cust_id.sample(300, random_state=1)}
    assert {"KEPT", "MOVED"} <= kinds
    assert X.customer_reason(al, "C99999", lm)["decision"] == "unknown"
    assert "E021" in X.territory_text(X.territory_summary(al, "E021", lm))
    for q, needle in [("who is overloaded?", "rep"), ("show territory of E012", "E012"), ("who covers PIN 400045?", "400045"),
                      ("compare with baseline", "Utilisation"), ("national summary", "National"), ("blah blah", "Try")]:
        txt, _, _ = nlq.ask(q, al, cmp, lm)
        assert needle in txt


def test_rollups(al, d):
    am = M.area_manager_view(al.metrics, al.cust, "final_emp", d["managers"])
    nv = M.national_view(am)
    assert nv["reps"] == len(al.emps) and nv["customers"] == len(al.cust) and len(am) == 6
    assert isinstance(M.detect_imbalance(al.metrics, am), list)


def test_maps_render(al, tmp_path):
    for view in ("optimised", "legacy"):
        f = tmp_path / f"{view}.html"
        MV.make_map(al, view, "Mumbai").save(f)
        assert f.stat().st_size > 50_000 and "leaflet" in f.read_text(encoding="utf-8").lower()
    MV.india_overview(al).save(tmp_path / "o.html")
