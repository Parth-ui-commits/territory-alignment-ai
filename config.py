"""Central configuration: business rules, regions, optimiser weights.

Everything a hackathon judge might ask "why this number?" about lives here,
so it can be changed in one place and defended in one slide.
"""
from dataclasses import dataclass, field

SEED = 42

# --------------------------------------------------------------------------
# Business model
# --------------------------------------------------------------------------
# Potential class -> (visits per month, hours per visit incl. prep/admin,
#                     mean annual revenue potential in INR lakh, spread)
POTENTIAL_CLASSES = {
    "A": dict(visits=4.0, hours=1.50, mean_lakh=14.0, sigma=0.35, share=0.10),
    "B": dict(visits=2.0, hours=1.25, mean_lakh=6.0,  sigma=0.35, share=0.25),
    "C": dict(visits=1.0, hours=1.00, mean_lakh=2.5,  sigma=0.40, share=0.40),
    "D": dict(visits=0.5, hours=0.75, mean_lakh=0.8,  sigma=0.45, share=0.25),
}

# Employee grade -> (monthly productive field hours, target-potential weight)
GRADES = {
    "Senior": dict(capacity_hours=150.0, potential_weight=1.30),
    "Mid":    dict(capacity_hours=140.0, potential_weight=1.00),
    "Junior": dict(capacity_hours=125.0, potential_weight=0.75),
}

ROAD_CIRCUITY = 1.30          # road km = straight-line km * 1.30
BEAT_CYCLES_MAX = 4           # a territory is swept at most weekly (4x / month)

# Utilisation thresholds used for automatic imbalance detection
OVERLOAD_UTIL = 1.10
UNDERLOAD_UTIL = 0.80
HIGH_POT_RATIO = 1.15         # potential / target potential
LOW_POT_RATIO = 0.85
GEO_INEFFICIENT_TRAVEL_SHARE = 0.30   # travel hours / total hours

# --------------------------------------------------------------------------
# Geography: one Area-Manager region per metro cluster (synthetic PIN grid
# around real city centres; plug a real PIN CSV in via data_gen.load_real_pins)
# --------------------------------------------------------------------------
REGIONS = [
    # name, AM id, lat, lon, spread_km, PIN prefix, urban speed km/h, employees
    dict(city="Delhi NCR",  am="AM01", lat=28.6139, lon=77.2090, spread_km=17, pin_prefix="110", speed=24, n_emp=8),
    dict(city="Mumbai",     am="AM02", lat=19.0760, lon=72.8777, spread_km=15, pin_prefix="400", speed=20, n_emp=8),
    dict(city="Bengaluru",  am="AM03", lat=12.9716, lon=77.5946, spread_km=14, pin_prefix="560", speed=22, n_emp=7),
    dict(city="Hyderabad",  am="AM04", lat=17.3850, lon=78.4867, spread_km=14, pin_prefix="500", speed=26, n_emp=7),
    dict(city="Chennai",    am="AM05", lat=13.0827, lon=80.2707, spread_km=13, pin_prefix="600", speed=26, n_emp=6),
    dict(city="Kolkata",    am="AM06", lat=22.5726, lon=88.3639, spread_km=13, pin_prefix="700", speed=21, n_emp=6),
]
PINS_PER_REGION = 100
CUSTOMERS_PER_EMP = 54        # average legacy book size (-> ~90 % network utilisation)

# --------------------------------------------------------------------------
# Optimiser
# --------------------------------------------------------------------------
@dataclass
class Weights:
    """Objective  J = w_wl*WorkloadDev + w_pot*PotentialDev + w_geo*Compactness
                      + w_cont*RelationshipLoss + w_cap*OverCapacity"""
    w_wl: float = 30.0      # workload deviation from capacity-proportional ideal
    w_pot: float = 12.0     # potential deviation from target-proportional ideal
    w_geo: float = 6.0      # territory compactness (workload-weighted distance)
    w_cont: float = 16.0    # relationship continuity lost (relationship-weighted)
    w_cap: float = 40.0     # hinge penalty above 100 % utilisation
    max_sweeps: int = 60
    n_restarts: int = 1

PRESETS = {
    "Max continuity":     Weights(w_cont=60.0),
    "Balanced (default)": Weights(),
    "Max balance":        Weights(w_cont=2.0, w_wl=45.0, w_pot=20.0),
}

# How much we care about keeping a relationship (continuity cost weight)
RELATION_CLASS_WEIGHT = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.5}
RELATION_TENURE_CAP_YEARS = 8.0     # tenure bonus saturates at 8 years
