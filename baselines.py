"""Simple baselines the optimiser must beat (customer-level assignments).

B1  Nearest-home   : every customer goes to the closest rep's home (Voronoi).
B2  K-means        : geographic k-means on customers, clusters matched to reps
                     by Hungarian assignment on distance to home.  Compact but
                     ignores workload, potential, capacity and relationships.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from geo import project_km, pairwise_km


def nearest_home(cust: pd.DataFrame, emps: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=cust.index, dtype=object)
    for reg, er in emps.groupby("region"):
        idx = cust.index[cust.region == reg]
        if len(idx) == 0:
            continue
        lat0, lon0 = cust.loc[idx, "lat"].mean(), cust.loc[idx, "lon"].mean()
        cxy = project_km(cust.loc[idx, "lat"], cust.loc[idx, "lon"], lat0, lon0)
        exy = project_km(er.home_lat, er.home_lon, lat0, lon0)
        out.loc[idx] = er.emp_id.values[pairwise_km(cxy, exy).argmin(1)]
    return out


def kmeans_baseline(cust: pd.DataFrame, emps: pd.DataFrame, seed: int = 0) -> pd.Series:
    out = pd.Series(index=cust.index, dtype=object)
    for reg, er in emps.groupby("region"):
        idx = cust.index[cust.region == reg]
        if len(idx) == 0:
            continue
        lat0, lon0 = cust.loc[idx, "lat"].mean(), cust.loc[idx, "lon"].mean()
        cxy = project_km(cust.loc[idx, "lat"], cust.loc[idx, "lon"], lat0, lon0)
        exy = project_km(er.home_lat, er.home_lon, lat0, lon0)
        km = KMeans(n_clusters=len(er), n_init=10, random_state=seed).fit(cxy)
        cost = pairwise_km(km.cluster_centers_, exy)
        ci, ei = linear_sum_assignment(cost)
        mapping = {int(c): er.emp_id.values[e] for c, e in zip(ci, ei)}
        out.loc[idx] = [mapping[int(l)] for l in km.labels_]
    return out
