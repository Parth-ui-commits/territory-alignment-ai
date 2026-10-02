"""India map layer (Folium / OpenStreetMap tiles).

Territory boundaries are APPROXIMATED the standard way when true PIN polygons
are unavailable: a Voronoi tessellation of PIN centroids (one cell per PIN,
clipped to the region's footprint), dissolved by owner.  With real PIN boundary
shapefiles, replace `pin_cells()` and nothing else changes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import folium
from folium.plugins import MiniMap, Fullscreen
import shapely
from shapely.geometry import MultiPoint, Point, mapping
from shapely.ops import unary_union

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a0a0", "#9a6324",
           "#b00020", "#808000", "#000075", "#e377c2", "#17becf"]
CLASS_RADIUS = {"A": 6, "B": 4.5, "C": 3.2, "D": 2.2}
_CELL_CACHE: dict = {}


def pin_cells(pins_r: pd.DataFrame) -> dict:
    """pin -> shapely Polygon (lon/lat)."""
    key = (str(pins_r.region.iloc[0]), len(pins_r))
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    pts = [Point(lo, la) for la, lo in zip(pins_r.lat, pins_r.lon)]
    mp = MultiPoint(pts)
    footprint = mp.convex_hull.buffer(0.018)
    vor = shapely.voronoi_polygons(mp, extend_to=footprint.buffer(0.5))
    cells = {}
    for poly in vor.geoms:
        for pin, pt in zip(pins_r.pin, pts):
            if poly.contains(pt):
                cells[pin] = poly.intersection(footprint)
                break
    _CELL_CACHE[key] = cells
    return cells


def emp_colors(emps: pd.DataFrame) -> dict:
    col = {}
    for reg, er in emps.groupby("region", sort=False):
        for i, e in enumerate(er.emp_id):
            col[e] = PALETTE[i % len(PALETTE)]
    col["VACANT"] = "#7f7f7f"
    return col


def _base_map(pins: pd.DataFrame, region: str | None) -> folium.Map:
    p = pins if region is None else pins[pins.region == region]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True, zoom_start=5)
    m.fit_bounds([[p.lat.min(), p.lon.min()], [p.lat.max(), p.lon.max()]])
    Fullscreen().add_to(m)
    return m


def make_map(al, view: str = "optimised", region: str | None = None, emp_id: str | None = None,
             show_customers: bool = True, show_pins: bool = True) -> folium.Map:
    """view = 'optimised' (final alignment) | 'legacy' (as-is)."""
    pins, emps, cust = al.pins, al.emps, al.cust
    col = emp_colors(emps)
    m = _base_map(pins, region)
    owner_col = "final_emp" if view == "optimised" else "legacy_emp"
    regs = [region] if region else list(pins.region.unique())
    emp_info = emps.set_index("emp_id")
    met = (al.metrics if view == "optimised" else _legacy_metrics(al)).set_index("emp_id")

    g_pins = folium.FeatureGroup(name="PIN cells (colour = owner)", show=(view == "optimised" and show_pins))
    g_terr = folium.FeatureGroup(name="Territory outlines", show=True)
    g_cust = folium.FeatureGroup(name="Customers", show=show_customers)
    g_home = folium.FeatureGroup(name="Rep home bases", show=True)

    for reg in regs:
        pr = pins[pins.region == reg].reset_index(drop=True)
        cells = pin_cells(pr)
        if view == "optimised":
            owner = pr.pin.map(al.pin_owner)
            for pin, o in zip(pr.pin, owner):
                if emp_id and o != emp_id:
                    continue
                if pin in cells and o in col:
                    folium.GeoJson(mapping(cells[pin]), style_function=lambda _, c=col[o]: dict(
                        fillColor=c, color="#ffffff", weight=0.6, fillOpacity=0.30),
                        tooltip=f"PIN {pin} -> {o}").add_to(g_pins)
            for e, grp in pd.DataFrame(dict(pin=pr.pin, o=owner)).groupby("o"):
                if emp_id and e != emp_id:
                    continue
                geom = unary_union([cells[p] for p in grp.pin if p in cells])
                r = met.loc[e] if e in met.index else None
                tip = (f"<b>{e}</b> {emp_info.loc[e, 'name']}<br>{len(grp)} PINs, {int(r.n_customers)} customers<br>"
                       f"Utilisation {r.utilization:.0%} | Potential {r.potential_lakh:.0f}L ({r.potential_ratio:.0%} of fair)<br>"
                       f"Travel {r.travel_hours:.0f} h/mo") if r is not None else e
                folium.GeoJson(mapping(geom), style_function=lambda _, c=col[e]: dict(
                    fillColor=c, color=c, weight=3, fillOpacity=0.0), tooltip=folium.Tooltip(tip, sticky=True)).add_to(g_terr)
        else:   # legacy: no clean territories exist -> show each rep's convex hull (they overlap!)
            cc = cust[cust.region == reg]
            for e, grp in cc.groupby("legacy_emp"):
                if e == "VACANT" or (emp_id and e != emp_id) or len(grp) < 3:
                    continue
                hull = MultiPoint([Point(lo, la) for la, lo in zip(grp.lat, grp.lon)]).convex_hull
                if hull.geom_type != "Polygon":
                    continue
                r = met.loc[e] if e in met.index else None
                tip = f"<b>{e}</b> (legacy)<br>{len(grp)} customers, {r.n_pins:.0f} PINs<br>Utilisation {r.utilization:.0%}" if r is not None else e
                folium.GeoJson(mapping(hull), style_function=lambda _, c=col[e]: dict(
                    fillColor=c, color=c, weight=2, fillOpacity=0.07, dashArray="5 5"),
                    tooltip=folium.Tooltip(tip, sticky=True)).add_to(g_terr)

    if show_customers:
        cc = cust[cust.region.isin(regs)]
        if emp_id:
            cc = cc[cc[owner_col] == emp_id]
        for r in cc.itertuples():
            o = getattr(r, owner_col)
            c = col.get(o, "#7f7f7f")
            exc = bool(getattr(r, "is_exception", False)) and view == "optimised"
            nm = emp_info.loc[o, "name"] if o in emp_info.index else "UNASSIGNED"
            folium.CircleMarker((r.lat, r.lon), radius=CLASS_RADIUS[r.pot_class], color="#000000" if exc else c,
                                weight=2 if exc else 0.5, fill=True, fill_color=c, fill_opacity=0.85,
                                tooltip=f"{r.cust_id} | class {r.pot_class} | {r.potential_lakh:.1f}L | PIN {r.pin}<br>"
                                        f"Owner: {o} ({nm}){' - KEY-ACCOUNT EXCEPTION' if exc else ''}"
                                ).add_to(g_cust)
    for e, r in emp_info.iterrows():
        if r.region in regs and (not emp_id or e == emp_id):
            folium.Marker((r.home_lat, r.home_lon), tooltip=f"{e} {r['name']} (home base, {r.grade})",
                          icon=folium.Icon(color="black", icon="user", prefix="fa")).add_to(g_home)
    for g in (g_pins, g_terr, g_cust, g_home):
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    return m


def india_overview(al) -> folium.Map:
    """One bubble per Area-Manager region with roll-up numbers."""
    from metrics import area_manager_view
    am = area_manager_view(al.metrics, al.cust, "final_emp",
                           pd.DataFrame(dict(mgr_id=[], name=[])))
    m = folium.Map(location=[22.5, 79], zoom_start=5, tiles="OpenStreetMap")
    cen = al.pins.groupby("region")[["lat", "lon"]].mean()
    for _, r in am.iterrows():
        la, lo = cen.loc[r.region]
        color = "#d62728" if r.utilization > 1.05 else "#2ca02c" if r.utilization >= 0.8 else "#ff7f0e"
        folium.CircleMarker((la, lo), radius=10 + r.reps * 2, color=color, fill=True, fill_opacity=0.55,
                            tooltip=f"<b>{r.region}</b> ({r.am})<br>{r.reps} reps, {r.customers} customers<br>"
                                    f"Utilisation {r.utilization:.0%}, potential {r.potential_lakh:.0f}L").add_to(m)
    return m


def _legacy_metrics(al):
    import workload as W
    return W.employee_metrics(al.cust, al.emps, al.pins, "legacy_emp")
