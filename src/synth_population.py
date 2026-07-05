import numpy as np
import yaml
import json
import os
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict


# ---------------------------------------------------------------------------
# Planet + star dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SyntheticPlanet:
    planet_id: int
    star_teff_k: float          # Host star effective temperature [K]
    star_radius_rs: float       # Host star radius [solar radii]
    star_mass_ms: float         # Host star mass [solar masses]
    star_magnitude_j: float     # J-band apparent magnitude
    star_type: str              # M / K / G / F

    planet_radius_re: float     # Planet radius [Earth radii]
    planet_mass_me: float       # Planet mass [Earth masses; estimated]
    orbital_period_days: float
    transit_duration_hours: float
    equilibrium_temp_k: float   # Assuming Bond albedo 0.3
    in_habitable_zone: bool
    distance_pc: float


@dataclass
class PopulationConfig:
    n_planets: int = 1000
    seed: int = 42

    # Star priors
    star_teff_min_k: float = 2500.0
    star_teff_max_k: float = 5000.0
    star_teff_peak_k: float = 3200.0   # Peak of skewed distribution (M/K dwarfs)

    # Planet radius priors [Earth radii] — bimodal rocky/sub-Neptune
    radius_rocky_mean_re: float = 1.2
    radius_rocky_std_re: float = 0.3
    radius_subneptune_mean_re: float = 2.5
    radius_subneptune_std_re: float = 0.5
    rocky_fraction: float = 0.55

    # Distance prior [pc] — uniform volume element out to d_max
    distance_max_pc: float = 50.0

    # Orbital period prior [days] — log-uniform
    period_min_days: float = 1.0
    period_max_days: float = 400.0

    @classmethod
    def from_yaml(cls, path: str) -> "PopulationConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Stellar property helpers
# ---------------------------------------------------------------------------

def _teff_to_stellar_props(teff: float) -> Dict:
    """
    Approximate stellar radius, mass, luminosity, and type from Teff.
    Uses simple empirical scalings valid for M–K–G–F dwarfs.
    """
    if teff < 3000:
        rs = 0.12 + (teff - 2500) / 500 * 0.08
        ms = 0.08 + (teff - 2500) / 500 * 0.12
        star_type = "M"
    elif teff < 3700:
        rs = 0.20 + (teff - 3000) / 700 * 0.25
        ms = 0.20 + (teff - 3000) / 700 * 0.30
        star_type = "M"
    elif teff < 5000:
        rs = 0.45 + (teff - 3700) / 1300 * 0.45
        ms = 0.50 + (teff - 3700) / 1300 * 0.45
        star_type = "K"
    elif teff < 6000:
        rs = 0.90 + (teff - 5000) / 1000 * 0.20
        ms = 0.95 + (teff - 5000) / 1000 * 0.15
        star_type = "G"
    else:
        rs = 1.10 + (teff - 6000) / 500 * 0.15
        ms = 1.10 + (teff - 6000) / 500 * 0.10
        star_type = "F"

    luminosity = (rs ** 2) * (teff / 5778) ** 4  # in solar luminosities
    return {"rs": rs, "ms": ms, "luminosity": luminosity, "star_type": star_type}


def _equilibrium_temp(star_luminosity_lsun: float, distance_au: float, albedo: float = 0.3) -> float:
    """Planet equilibrium temperature in Kelvin."""
    L_W = star_luminosity_lsun * 3.828e26
    T_eq = ((L_W * (1 - albedo)) / (16 * np.pi * 5.67e-8 * (distance_au * 1.496e11) ** 2)) ** 0.25
    return T_eq


def _habitable_zone(star_luminosity_lsun: float) -> tuple:
    """
    Kopparapu et al. (2013) empirical HZ limits.
    Returns (inner_au, outer_au).
    """
    inner_au = 0.95 * np.sqrt(star_luminosity_lsun)
    outer_au = 1.67 * np.sqrt(star_luminosity_lsun)
    return inner_au, outer_au


def _j_magnitude(distance_pc: float, star_type: str, star_radius_rs: float) -> float:
    """
    Rough J-band apparent magnitude estimate.
    Based on absolute magnitudes for dwarf stars of each type.
    """
    abs_j = {"M": 7.5, "K": 5.5, "G": 4.0, "F": 3.0}
    M_j = abs_j.get(star_type, 6.0) - 2.5 * np.log10(max(star_radius_rs, 0.1) / 0.3)
    m_j = M_j + 5 * np.log10(distance_pc / 10.0)
    return float(m_j)


def _transit_duration(period_days: float, planet_radius_re: float,
                       star_radius_rs: float, star_mass_ms: float) -> float:
    """
    Approximate transit duration in hours (assumes circular orbit, i=90°).
    T14 ≈ (period/π) * arcsin(R_*/a) * (1 + k) where k = R_p/R_*
    """
    G = 6.674e-11
    P_s = period_days * 86400
    M_kg = star_mass_ms * 1.989e30
    a_m = ((G * M_kg * P_s ** 2) / (4 * np.pi ** 2)) ** (1 / 3)
    R_star_m = star_radius_rs * 6.96e8
    R_planet_m = planet_radius_re * 6.371e6
    sin_arg = min((R_star_m + R_planet_m) / a_m, 1.0)
    T_s = (P_s / np.pi) * np.arcsin(sin_arg)
    return T_s / 3600  # hours


def _radius_to_mass(radius_re: float) -> float:
    """
    Empirical mass-radius relation (Wolfgang & Lopez 2015; Chen & Kipping 2017).
    """
    if radius_re < 1.5:
        return radius_re ** 3.7        # rocky
    elif radius_re < 4.0:
        return 2.69 * radius_re ** 0.93  # sub-Neptune
    else:
        return 0.5 * radius_re ** 1.74  # Neptune/gas


# ---------------------------------------------------------------------------
# Population generator
# ---------------------------------------------------------------------------

def generate_population(config: PopulationConfig) -> List[SyntheticPlanet]:
    """
    Draw a synthetic exoplanet population from the configured priors.
    """
    rng = np.random.default_rng(config.seed)
    planets = []

    for i in range(config.n_planets):
        # --- Star ---
        # Skewed distribution peaked toward cool stars (M dwarfs dominate)
        teff = rng.triangular(
            config.star_teff_min_k, config.star_teff_peak_k, config.star_teff_max_k
        )
        star_props = _teff_to_stellar_props(teff)

        # --- Distance: uniform in volume (N ∝ d^2 dd) ---
        u = rng.uniform(0, 1)
        distance = config.distance_max_pc * u ** (1 / 3)

        # --- Orbital period: log-uniform ---
        log_period = rng.uniform(
            np.log10(config.period_min_days), np.log10(config.period_max_days)
        )
        period = 10 ** log_period

        # --- Semi-major axis from Kepler's 3rd law ---
        a_au = (star_props["ms"] * (period / 365.25) ** 2) ** (1 / 3)

        # --- Planet radius: bimodal rocky / sub-Neptune ---
        if rng.random() < config.rocky_fraction:
            radius = rng.normal(config.radius_rocky_mean_re, config.radius_rocky_std_re)
        else:
            radius = rng.normal(config.radius_subneptune_mean_re, config.radius_subneptune_std_re)
        radius = float(np.clip(radius, 0.5, 4.5))

        # --- Habitable zone check ---
        hz_inner, hz_outer = _habitable_zone(star_props["luminosity"])
        in_hz = hz_inner <= a_au <= hz_outer

        # --- Derived quantities ---
        teq = _equilibrium_temp(star_props["luminosity"], a_au)
        mass = _radius_to_mass(radius)
        t_dur = _transit_duration(period, radius, star_props["rs"], star_props["ms"])
        mag_j = _j_magnitude(distance, star_props["star_type"], star_props["rs"])

        planets.append(SyntheticPlanet(
            planet_id=i,
            star_teff_k=float(teff),
            star_radius_rs=float(star_props["rs"]),
            star_mass_ms=float(star_props["ms"]),
            star_magnitude_j=float(mag_j),
            star_type=star_props["star_type"],
            planet_radius_re=radius,
            planet_mass_me=float(mass),
            orbital_period_days=float(period),
            transit_duration_hours=float(t_dur),
            equilibrium_temp_k=float(teq),
            in_habitable_zone=bool(in_hz),
            distance_pc=float(distance),
        ))

    n_hz = sum(p.in_habitable_zone for p in planets)
    print(f"Generated {len(planets)} planets | {n_hz} in habitable zone ({100*n_hz/len(planets):.1f}%)")
    return planets


def population_to_csv(planets: List[SyntheticPlanet], path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(planets[0]).keys())
        writer.writeheader()
        for p in planets:
            writer.writerow(asdict(p))
    print(f"Saved population to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic exoplanet population generator")
    parser.add_argument("--config", default="config/priors.yaml", help="Path to priors config YAML")
    parser.add_argument("--n", type=int, default=None, help="Override number of planets")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--out", default="data/synthetic_population.csv", help="Output CSV path")
    args = parser.parse_args()

    config_path = args.config
    if os.path.exists(config_path):
        cfg = PopulationConfig.from_yaml(config_path)
        print(f"Loaded config from {config_path}")
    else:
        cfg = PopulationConfig()
        print(f"Config not found at {config_path}, using defaults.")

    if args.n is not None:
        cfg.n_planets = args.n
    if args.seed is not None:
        cfg.seed = args.seed

    planets = generate_population(cfg)
    population_to_csv(planets, args.out)
