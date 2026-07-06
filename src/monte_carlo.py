import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
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
    planet_id:           int
    star_type:           str
    star_teff_k:         float
    star_magnitude_j:    float
    planet_radius_re:    float
    distance_pc:         float
    equilibrium_temp_k:  float
    in_habitable_zone:   bool

    atmosphere_type:     str
    cloud_fraction:      float
    scale_height_km:     float
    o2_ch4_ratio:        float

    n_transits:          int
    broadband_snr:       float      # Corrected modulation SNR
    is_detected:         bool       # SNR >= 5σ
    noise_floor_ppm:     float      # Median per-bin noise
    transit_duration_h:  float


# ---------------------------------------------------------------------------
# Experiment config loader
# ---------------------------------------------------------------------------

def load_experiment_config(path: str = "config/experiment.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Single trial function (run in parallel)
# ---------------------------------------------------------------------------

_BUILDERS = {
    "earth_like":          (build_earth_like_template, 1.0),
    "high_co2":            (build_high_co2_template, 0.01),
    "reduced_o2_high_ch4": (build_reduced_o2_high_ch4_template, 0.001),
    "hycean":              (build_hycean_template, 0.002),
    "abiotic_o2":          (build_abiotic_o2_template, 1000.0),
}


def _run_one_trial(
    planet: SyntheticPlanet,
    atmosphere_name: str,
    cloud_fraction: float,
    scale_height_km: float,
    n_transits: int,
    seed: int,
    instrument_name: str = "jwst_nirspec",
) -> List[MCTrial]:
    """
    Simulate one planet × atmosphere × n_transits combination.
    Returns list of MCTrial (one per n_transits value if called with a list).
    """
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

    if instrument_name == "miri_lrs":
        instrument = load_miri_lrs()
    else:
        instrument = load_jwst_nirspec()

    def simulate_batch(self, planet, template, n_transits, n_trials, rng=None):
        rng = rng or self.rng
        wl = template.wavelengths_um
        true_depth = template.transit_depth_ppm
        photon_rate = self.instrument.stellar_photon_rate(wl, planet.star_magnitude_j, planet.star_teff_k)
        total_transit_s = planet.transit_duration_s * n_transits
        n_exp = max(1, int(total_transit_s / self.exposure_time_s))
        budget = self.instrument.noise_model(photon_rate, self.exposure_time_s, n_exp)
        sig = budget["signal_e"]
        noise_ppm = np.where(sig > 1.0, budget["total_noise"] / sig * 1e6, 1e6)
        in_range = (wl >= self.instrument.config.wavelength_min_um) & (wl <= self.instrument.config.wavelength_max_um)
        noise_ppm = np.where(in_range, noise_ppm, 1e6)

        # one call instead of n_trials calls, shape (n_trials, n_wl)
        noise_draws = rng.normal(0.0, noise_ppm, size=(n_trials, len(wl)))
        observed = true_depth[None, :] + noise_draws
        return observed, noise_ppm  # feed each row into retrieval/SNR code as before

    # Corrected SNR: spectral modulation / noise (not absolute depth)
    baseline = template.parameters["base_depth_ppm"]
    modulation = np.abs(obs.true_depth_ppm - baseline)
    mask = (obs.noise_ppm > 0) & (obs.noise_ppm < 5000)
    snr_bins = np.where(mask, modulation / (obs.noise_ppm + 1e-9), 0.0)
    corrected_snr = float(np.sqrt(np.nansum(snr_bins ** 2)))

    noise_floor = float(np.median(obs.noise_ppm[mask])) if mask.sum() > 0 else 1e6

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
    planets, atm_names, cloud_fracs, scale_heights, n_transits_list, seeds, instrument_name,
) -> List[MCTrial]:
    """Process a batch of planets (called by each parallel worker)."""
    # Build the instrument ONCE per batch, not once per trial — it's identical
    # every call (same fixed config), so the original code was reconstructing
    # it n_trials times for zero benefit. Requires threading `instrument` into
    # _run_one_trial as a parameter instead of a name string; trivial change.
    instrument = load_miri_lrs() if instrument_name == "miri_lrs" else load_jwst_nirspec()

    trials = []
    failures = []
    for i, planet in enumerate(planets):
        atm, cf, sh = atm_names[i], cloud_fracs[i], scale_heights[i]
        for n_t in n_transits_list:
            try:
                trial = _run_one_trial(planet, atm, cf, sh, n_t, seeds[i], instrument=instrument)
                trials.append(trial)
            except Exception as e:
                failures.append((planet.planet_id, atm, n_t, repr(e)))

    if failures:
        print(f"  WARNING: {len(failures)}/{len(planets)*len(n_transits_list)} trials failed "
              f"and were skipped. First 3: {failures[:3]}")
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

        print(f"\n{'═'*60}")
        print(f"  MONTE CARLO RUN: {cfg.get('experiment_name', 'exp1')}")
        print(f"  N = {n_planets} planets | {n_jobs} workers")
        print(f"  N_transits: {n_transits_list}")
        print(f"  Instrument: {instrument_name}")
        print(f"{'═'*60}")

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
        max_j  = cfg.get("max_j_magnitude", 14.0)
        min_td = cfg.get("min_transit_duration_hours", 0.3)
        planets = [p for p in planets
                   if p.star_magnitude_j <= max_j
                   and p.transit_duration_hours >= min_td]
        print(f"After filtering (J<{max_j}, t_transit>{min_td}h): {len(planets)} planets")

        # ── Randomly assign atmosphere types and parameters ──
        rng = np.random.default_rng(seed + 1)
        atm_names_all = list(_BUILDERS.keys())
        atm_weights   = [atm_weights_raw.get(n, 0.25) for n in atm_names_all]
        atm_weights   = np.array(atm_weights) / sum(atm_weights)

        assigned_atm = rng.choice(atm_names_all, size=len(planets), p=atm_weights)
        assigned_cf  = rng.uniform(0.0, 0.9, size=len(planets))
        assigned_sh  = rng.uniform(6.0, 11.0, size=len(planets))
        planet_seeds = rng.integers(0, 99999, size=len(planets))

        # ── Parallel batch execution ──
        batch_size = cfg.get("batch_size", 250)
        n_batches  = max(1, len(planets) // batch_size)
        batches    = np.array_split(np.arange(len(planets)), n_batches)

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
        print(f"\nCompleted {len(all_trials)} trials in {elapsed:.0f}s "
              f"({len(all_trials)/elapsed:.0f} trials/s)")

        # ── Save results ──
        self._save_csv(all_trials)
        summary = self._compute_summary(all_trials, n_transits_list)
        self._print_headline(summary)

        return all_trials

    def _save_csv(self, trials: List[MCTrial]) -> None:
        path = os.path.join(self.output_dir, "summary.csv")
        fields = list(asdict(trials[0]).keys()) if trials else []
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trials:
                w.writerow(asdict(t))
        size_kb = os.path.getsize(path) / 1024
        print(f"Full table saved → {path}  ({len(trials)} rows, {size_kb:.0f} KB)")

    def _compute_summary(
        self, trials: List[MCTrial], n_transits_list: List[int]
    ) -> dict:
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
                # Detection probability by distance bin
                hz_detected = [t for t in t_atm if t.in_habitable_zone and t.is_detected]
                hz_total    = [t for t in t_atm if t.in_habitable_zone]
                d20_det     = [t for t in t_atm if t.distance_pc <= 20 and t.is_detected]
                d20_tot     = [t for t in t_atm if t.distance_pc <= 20]

                summary[n_t][atm] = {
                    "n_total": len(t_atm),
                    "n_detected": sum(t.is_detected for t in t_atm),
                    "det_prob_all": sum(t.is_detected for t in t_atm) / max(len(t_atm), 1),
                    "det_prob_hz": len(hz_detected) / max(len(hz_total), 1),
                    "det_prob_d20": len(d20_det) / max(len(d20_tot), 1),
                    "median_snr": float(np.median([t.broadband_snr for t in t_atm])),
                }

        # Save summary JSON
        with open(os.path.join(self.output_dir, "summary_stats.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _print_headline(self, summary: dict) -> None:
        print(f"\n{'━'*60}")
        print(f"  HEADLINE RESULTS")
        print(f"{'━'*60}")
        for n_t in sorted(summary.keys()):
            print(f"\n  N_transits = {n_t}:")
            for atm, stats in summary[n_t].items():
                dp_hz = stats.get("det_prob_hz", 0)
                dp_20 = stats.get("det_prob_d20", 0)
                snr   = stats.get("median_snr", 0)
                print(f"    {atm:28s}  HZ det={dp_hz:.0%}  d<20pc={dp_20:.0%}  "
                      f"median SNR={snr:.1f}σ")
        print(f"\n{'━'*60}")
        # The headline number
        if 10 in summary and "earth_like" in summary[10]:
            hz_det = summary[10]["earth_like"]["det_prob_hz"]
            d20    = summary[10]["earth_like"]["det_prob_d20"]
            print(f"\n  ★ HEADLINE: {d20:.0%} of Earth-like HZ planets within 20 pc")
            print(f"    are detectable with JWST NIRSpec in 10 transits.")
            print(f"  ★ HZ detection rate (all distances): {hz_det:.0%}")
        print(f"{'━'*60}\n")


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
    if args.jobs:
        cfg["n_jobs"] = args.jobs

    runner = MonteCarloRunner(cfg)
    trials = runner.run(n_override=args.n)
    print(f"Done. {len(trials)} trials saved to {runner.output_dir}/")
