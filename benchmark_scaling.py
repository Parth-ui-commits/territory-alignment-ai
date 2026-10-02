"""Measure solve time / quality as ONE region grows.   python benchmark_scaling.py
Shows why the system decomposes India into Area-Manager regions (~8 reps each)."""
import copy, time
import numpy as np, pandas as pd
import config as C, data_gen, pipeline as P

base = copy.deepcopy(C.REGIONS)
rows = []
for n_pins, n_emp in [(100, 8), (200, 16), (400, 32), (800, 64)]:
    C.PINS_PER_REGION = n_pins
    C.REGIONS = [dict(base[0], n_emp=n_emp, spread_km=base[0]["spread_km"] * np.sqrt(n_pins / 100))]
    d = data_gen.generate(save=False)
    t = time.time()
    al = P.align(d["pins"], d["employees"], d["customers"])
    rows.append(dict(pins=n_pins, reps=n_emp, customers=len(d["customers"]), seconds=round(time.time() - t, 1),
                     util_std=round(al.metrics.utilization.std(ddof=0), 3),
                     util_min=round(al.metrics.utilization.min(), 2),
                     fragmented=int((al.metrics_pin_level.fragments > 1).sum())))
print(pd.DataFrame(rows).to_string(index=False))
print("\nNationwide: ~19k PIN codes / ~8 reps per Area Manager region => embarrassingly parallel across regions.")
