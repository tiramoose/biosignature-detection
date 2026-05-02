"""
synth_population.py

Generates a synthetic population of rocky exoplanets
by sampling from priors defined in a config YAML file.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def sample_radii(cfg: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw planet radii in Earth radii."""
    p = cfg["planet"]["radius_earth"]
    return rng.uniform(p["min"], p["max"], n)


def sample_distances(cfg: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw distances uniform in volume: p(d) ∝ d²
    Sample by drawing uniform in d³, then cube-rooting.
    """
    p = cfg["planet"]["distance_pc"]
    d_min, d_max = p["min"], p["max"]
    u = rng.uniform(d_min**3, d_max**3, n)
    return u ** (1.0 / 3.0)


def sample_periods(cfg: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw orbital periods log-uniformly."""
    p = cfg["planet"]["orbital_period_days"]
    log_min = np.log10(p["min"])
    log_max = np.log10(p["max"])
    return 10 ** rng.uniform(log_min, log_max, n)


def sample_star_types(cfg: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw host star spectral types with occurrence weights."""
    s = cfg["star"]
    return rng.choice(s["types"], size=n, p=s["weights"])


def estimate_transit_probability(radius_star_rsun: np.ndarray,
                                  period_days: np.ndarray) -> np.ndarray:
    """
    Geometric transit probability ~ R_star / a
    Using Kepler's third law with M_star ~ 1 M_sun approximation for now.
    """
    AU_per_day_factor = (period_days / 365.25) ** (2.0 / 3.0)  # a in AU
    R_star_AU = radius_star_rsun * 0.00465047
    return np.clip(R_star_AU / AU_per_day_factor, 0, 1)


# Rough median stellar radii by type (solar radii)
STAR_RADII = {"M": 0.3, "K": 0.7, "G": 1.0, "F": 1.3}


def generate_population(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    n = cfg["simulation"]["n_planets"]
    seed = cfg["simulation"]["random_seed"]
    rng = np.random.default_rng(seed)

    star_types = sample_star_types(cfg, n, rng)
    star_radii = np.array([STAR_RADII[s] for s in star_types])

    periods = sample_periods(cfg, n, rng)
    transit_prob = estimate_transit_probability(star_radii, periods)

    pop = pd.DataFrame({
        "planet_radius_rearth": sample_radii(cfg, n, rng),
        "distance_pc":          sample_distances(cfg, n, rng),
        "orbital_period_days":  periods,
        "star_type":            star_types,
        "star_radius_rsun":     star_radii,
        "transit_probability":  transit_prob,
    })

    return pop


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "priors.yaml"
    pop = generate_population(str(config_path))
    print(pop.describe())
    print(f"\nStar type counts:\n{pop['star_type'].value_counts()}")
    out_path = Path(__file__).parent.parent / "results" / "toy_population.csv"
    pop.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
