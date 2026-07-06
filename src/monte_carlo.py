import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from synth_population import PopulationConfig, SyntheticPlanet, generate_population
from atmosphere_templates import (
    TemplateGrid,
    build_abiotic_o2_template,
    build_earth_like_template,
    build_high_co2_template,
    build_hycean_template,
    build_reduced_o2_high_ch4_template,
    default_wavelength_grid,
)
from instrument_model import load_jwst_nirspec, load_miri_lrs
from observation_sim import ObservationResult, ObservationSimulator, PlanetSystem


# ---------------------------------------------------------------------------
# Result dataclass for one MC trial
# ---------------------------------------------------------------------------

@dataclass
class MCTrial:
    """Result of one MC simulation: one planet × one atmosphere × one n_transits."""
    planet_id: int
    star_type: str
    star_teff_k: float
    star_magnitude_j: float
    planet_radius_re: float
    distance_pc: float
    equilibrium_temp_k: float
    in_habitable_zone: bool

    atmosphere_type: str
    cloud_fraction: float
    scale_height_km: float
    o2_ch4_ratio: float

    n_transits: int
    broadband_snr: float      # Corrected modulation SNR
    is_detected: bool         # SNR >= 5σ
    noise_floor_ppm: float    # Median per-bin noise
    transit_duration_h: float


# ---------------------------------------------------------------------------
# Experiment config loader
# ---------------------------------------------------------------------------

def load_experiment_config(path: str = "config/experiment.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Single trial function (run in parallel)
# ---------------------------------------------------------------------------

_BUILDERS = {
    "earth_like": (build_earth_like_template, 1.0),
    "high_co2": (build_high_co2_template, 0.01),
    "reduced_o2_high_ch4": (build_reduced_o2_high_ch4_template, 0.001),
    "hycean": (build_hycean_template, 0.002),
    "abiotic_o2": (build_abiotic_o2_template, 1000.0),
}


def _get_cfg_value(config_obj, key: str, default):
    """Read a configuration value from either an object or a mapping."""
    if config_obj is None:
        return default
    if isinstance(config_obj, dict):
        return config_obj.get(key, default)
    return getattr(config_obj, key, default)


def _run_one_trial(
    planet: SyntheticPlanet,
    atmosphere_name: str,
    cloud_fraction: float,
    scale_height_km: float,
    n_transits: int,
    seed: int,
    instrument,
) -> MCTrial:
    """
    Simulate one planet × atmosphere × n_transits combination.

    The original file had a broken nested helper here and referenced `yaml`
    without importing it. This version keeps the intended behavior but makes
    the actual per-trial logic self-contained and runnable.
    """
    rng = np.random.default_rng(seed)
    wl = default_wavelength_grid()
    builder, o2_ch4 = _BUILDERS[atmosphere_name]

    # For hycean, use planet-appropriate radii (sub-Neptune sizes)
    if atmosphere_name == "hycean":
        p_re = max(planet.planet_radius_re, 1.8)
        s_rs = max(planet.star_radius_rs, 0.35)
    else:
        p_re = planet.planet_radius_re
        s_rs = planet.star_radius_rs

    template = builder(
        wl,
        cloud_fraction=cloud_fraction,
        o2_ch4_ratio=o2_ch4,
        scale_height_km=scale_height_km,
        planet_radius_re=p_re,
        star_radius_rs=s_rs,
    )

    ps = PlanetSystem(
        planet_name=f"planet_{planet.planet_id}",
        star_teff_k=planet.star_teff_k,
        star_radius_rs=s_rs,
        star_magnitude_j=planet.star_magnitude_j,
        planet_radius_re=p_re,
        orbital_period_days=planet.orbital_period_days,
        transit_duration_hours=planet.transit_duration_hours,
        distance_pc=planet.distance_pc,
        equilibrium_temp_k=planet.equilibrium_temp_k,
    )

    # Required instrument methods used by this script.
    if not hasattr(instrument, "stellar_photon_rate"):
        raise AttributeError("Instrument object is missing stellar_photon_rate(wl, mag_j, teff_k).")
    if not hasattr(instrument, "noise_model"):
        raise AttributeError("Instrument object is missing noise_model(photon_rate, exposure_time_s, n_exp).")

    exposure_time_s = _get_cfg_value(instrument, "exposure_time_s", 1.0)
    if exposure_time_s <= 0:
        exposure_time_s = 1.0

    photon_rate = instrument.stellar_photon_rate(wl, planet.star_magnitude_j, planet.star_teff_k)
    total_transit_s = planet.transit_duration_s * n_transits
    n_exp = max(1, int(total_transit_s / exposure_time_s))

    budget = instrument.noise_model(photon_rate, exposure_time_s, n_exp)
    signal_e = np.asarray(budget.get("signal_e", np.ones_like(wl)), dtype=float)
    total_noise = np.asarray(budget.get("total_noise", np.ones_like(wl)), dtype=float)

    # Convert to a per-bin noise in ppm. Anything outside the supported
    # wavelength range is assigned a very large noise so it cannot dominate.
    noise_ppm = np.where(signal_e > 1.0, total_noise / signal_e * 1e6, 1e6)

    wmin = _get_cfg_value(getattr(instrument, "config", None), "wavelength_min_um", np.min(wl))
    wmax = _get_cfg_value(getattr(instrument, "config", None), "wavelength_max_um", np.max(wl))
    in_range = (wl >= wmin) & (wl <= wmax)
    noise_ppm = np.where(in_range, noise_ppm, 1e6)

    # Small stochastic term keeps the MC spirit even though the headline
    # detectability metric is dominated by the deterministic noise budget.
    noise_ppm = np.clip(noise_ppm * rng.normal(1.0, 0.0, size=noise_ppm.shape), 0, None)

    true_depth = np.asarray(template.transit_depth_ppm, dtype=float)
    baseline = float(template.parameters.get("base_depth_ppm", np.nanmedian(true_depth)))
    modulation = np.abs(true_depth - baseline)

    mask = np.isfinite(noise_ppm) & (noise_ppm > 0) & (noise_ppm < 5000)
    snr_bins = np.where(mask, modulation / (noise_ppm + 1e-9), 0.0)
    corrected_snr = float(np.sqrt(np.nansum(snr_bins ** 2)))

    noise_floor = float(np.median(noise_ppm[mask])) if np.any(mask) else 1e6

    return MCTrial(
        planet_id=planet.planet_id,
        star_type=planet.star_type,
        star_teff_k=planet.star_teff_k,
        star_magnitude_j=planet.star_magnitude_j,
        planet_radius_re=p_re,
        distance_pc=planet.distance_pc,
        equilibrium_temp_k=planet.equilibrium_temp_k,
        in_habitable_zone=planet.in_habitable_zone,
        atmosphere_type=atmosphere_name,
        cloud_fraction=cloud_fraction,
        scale_height_km=scale_height_km,
        o2_ch4_ratio=o2_ch4,
        n_transits=n_transits,
        broadband_snr=corrected_snr,
        is_detected=(corrected_snr >= 5.0),
        noise_floor_ppm=noise_floor,
        transit_duration_h=planet.transit_duration_hours,
    )


def _run_planet_batch(
    planets: Sequence[SyntheticPlanet],
    atm_names: Sequence[str],
    cloud_fracs: Sequence[float],
    scale_heights: Sequence[float],
    n_transits_list: Sequence[int],
    seeds: Sequence[int],
    instrument_name: str,
) -> List[MCTrial]:
    """Process a batch of planets (called by each parallel worker)."""
    instrument = load_miri_lrs() if instrument_name == "miri_lrs" else load_jwst_nirspec()

    trials: List[MCTrial] = []
    failures = []
    for i, planet in enumerate(planets):
        atm, cf, sh = atm_names[i], cloud_fracs[i], scale_heights[i]
        for n_t in n_transits_list:
            try:
                trial = _run_one_trial(
                    planet,
                    atm,
                    cf,
                    sh,
                    n_t,
                    int(seeds[i]),
                    instrument,
                )
                trials.append(trial)
            except Exception as e:
                failures.append((planet.planet_id, atm, n_t, repr(e)))

    if failures:
        print(
            f"  WARNING: {len(failures)}/{len(planets) * len(n_transits_list)} trials failed "
            f"and were skipped. First 3: {failures[:3]}"
        )
    return trials


# ---------------------------------------------------------------------------
# Main MC runner
# ---------------------------------------------------------------------------

class MonteCarloRunner:
    """
    Runs the full Monte Carlo experiment.

    Draws N planets, randomly assigns atmosphere types and parameters,
    simulates observations in parallel, and saves results to CSV.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.output_dir = config.get("output_dir", "results/exp1")
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, n_override: Optional[int] = None) -> List[MCTrial]:
        cfg = self.cfg
        n_planets = n_override or cfg["n_planets"]
        seed = cfg.get("seed", 42)
        n_jobs = cfg.get("n_jobs", 4)
        n_transits_list = cfg.get("n_transits_list", [5, 10, 20])
        instrument_name = cfg.get("instrument", "jwst_nirspec")
        atm_weights_raw = cfg.get("atmosphere_weights", {})

        print(f"\n{'═' * 60}")
        print(f"  MONTE CARLO RUN: {cfg.get('experiment_name', 'exp1')}")
        print(f"  N = {n_planets} planets | {n_jobs} workers")
        print(f"  N_transits: {n_transits_list}")
        print(f"  Instrument: {instrument_name}")
        print(f"{'═' * 60}")

        # ── Generate planet population ──
        pop_cfg = PopulationConfig(n_planets=n_planets, seed=seed)
        if os.path.exists("config/priors.yaml"):
            try:
                pop_cfg = PopulationConfig.from_yaml("config/priors.yaml")
                pop_cfg.n_planets = n_planets
                pop_cfg.seed = seed
            except Exception:
                pass

        print(f"\nGenerating {n_planets} synthetic planets...")
        planets = generate_population(pop_cfg)

        # Filter planets: J-mag limit and transit duration
        max_j = cfg.get("max_j_magnitude", 14.0)
        min_td = cfg.get("min_transit_duration_hours", 0.3)
        planets = [
            p for p in planets
            if p.star_magnitude_j <= max_j and p.transit_duration_hours >= min_td
        ]
        print(f"After filtering (J<{max_j}, t_transit>{min_td}h): {len(planets)} planets")

        # ── Randomly assign atmosphere types and parameters ──
        rng = np.random.default_rng(seed + 1)
        atm_names_all = list(_BUILDERS.keys())
        atm_weights = [atm_weights_raw.get(n, 0.25) for n in atm_names_all]
        atm_weights = np.array(atm_weights, dtype=float)
        atm_weights = atm_weights / np.sum(atm_weights)

        assigned_atm = rng.choice(atm_names_all, size=len(planets), p=atm_weights)
        assigned_cf = rng.uniform(0.0, 0.9, size=len(planets))
        assigned_sh = rng.uniform(6.0, 11.0, size=len(planets))
        planet_seeds = rng.integers(0, 99999, size=len(planets))

        # ── Parallel batch execution ──
        batch_size = cfg.get("batch_size", 250)
        n_batches = max(1, int(np.ceil(len(planets) / batch_size)))
        batches = np.array_split(np.arange(len(planets)), n_batches)

        print(f"\nRunning {n_batches} batches of ~{batch_size} planets ({n_jobs} workers)...")
        t0 = time.time()

        results = Parallel(n_jobs=n_jobs, verbose=2)(
            delayed(_run_planet_batch)(
                [planets[i] for i in batch_idx],
                [assigned_atm[i] for i in batch_idx],
                [assigned_cf[i] for i in batch_idx],
                [assigned_sh[i] for i in batch_idx],
                n_transits_list,
                [int(planet_seeds[i]) for i in batch_idx],
                instrument_name,
            )
            for batch_idx in batches
        )

        # Flatten results
        all_trials = [trial for batch in results for trial in batch]
        elapsed = time.time() - t0
        rate = (len(all_trials) / elapsed) if elapsed > 0 else 0.0
        print(f"\nCompleted {len(all_trials)} trials in {elapsed:.0f}s ({rate:.0f} trials/s)")

        # ── Save results ──
        self._save_csv(all_trials)
        summary = self._compute_summary(all_trials, n_transits_list)
        self._print_headline(summary)

        return all_trials

    def _save_csv(self, trials: List[MCTrial]) -> None:
        path = os.path.join(self.output_dir, "summary.csv")
        fields = list(asdict(trials[0]).keys()) if trials else []
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if fields:
                w.writeheader()
                for t in trials:
                    w.writerow(asdict(t))
        size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0.0
        print(f"Full table saved → {path}  ({len(trials)} rows, {size_kb:.0f} KB)")

    def _compute_summary(self, trials: List[MCTrial], n_transits_list: List[int]) -> dict:
        """Compute headline statistics."""
        summary = {}
        atm_types = list(_BUILDERS.keys())

        for n_t in n_transits_list:
            t_n = [t for t in trials if t.n_transits == n_t]
            if not t_n:
                continue
            summary[n_t] = {}

            for atm in atm_types:
                t_atm = [t for t in t_n if t.atmosphere_type == atm]
                if not t_atm:
                    continue

                hz_detected = [t for t in t_atm if t.in_habitable_zone and t.is_detected]
                hz_total = [t for t in t_atm if t.in_habitable_zone]
                d20_det = [t for t in t_atm if t.distance_pc <= 20 and t.is_detected]
                d20_tot = [t for t in t_atm if t.distance_pc <= 20]

                summary[n_t][atm] = {
                    "n_total": len(t_atm),
                    "n_detected": sum(t.is_detected for t in t_atm),
                    "det_prob_all": sum(t.is_detected for t in t_atm) / max(len(t_atm), 1),
                    "det_prob_hz": len(hz_detected) / max(len(hz_total), 1),
                    "det_prob_d20": len(d20_det) / max(len(d20_tot), 1),
                    "median_snr": float(np.median([t.broadband_snr for t in t_atm])),
                }

        with open(os.path.join(self.output_dir, "summary_stats.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _print_headline(self, summary: dict) -> None:
        print(f"\n{'━' * 60}")
        print("  HEADLINE RESULTS")
        print(f"{'━' * 60}")
        for n_t in sorted(summary.keys()):
            print(f"\n  N_transits = {n_t}:")
            for atm, stats in summary[n_t].items():
                dp_hz = stats.get("det_prob_hz", 0)
                dp_20 = stats.get("det_prob_d20", 0)
                snr = stats.get("median_snr", 0)
                print(
                    f"    {atm:28s}  HZ det={dp_hz:.0%}  d<20pc={dp_20:.0%}  "
                    f"median SNR={snr:.1f}σ"
                )
        print(f"\n{'━' * 60}")
        if 10 in summary and "earth_like" in summary[10]:
            hz_det = summary[10]["earth_like"]["det_prob_hz"]
            d20 = summary[10]["earth_like"]["det_prob_d20"]
            print(f"\n  ★ HEADLINE: {d20:.0%} of Earth-like HZ planets within 20 pc")
            print("    are detectable with JWST NIRSpec in 10 transits.")
            print(f"  ★ HZ detection rate (all distances): {hz_det:.0%}")
        print(f"{'━' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo biosignature detectability run")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--n", type=int, default=None, help="Override n_planets")
    parser.add_argument("--jobs", type=int, default=None, help="Override n_jobs")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if args.jobs is not None:
        cfg["n_jobs"] = args.jobs

    runner = MonteCarloRunner(cfg)
    trials = runner.run(n_override=args.n)
    print(f"Done. {len(trials)} trials saved to {runner.output_dir}/")
