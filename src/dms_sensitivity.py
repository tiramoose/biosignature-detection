import numpy as np
import os, sys, csv
from dataclasses import dataclass
from typing import List
sys.path.insert(0, os.path.dirname(__file__))
from atmosphere_templates import build_hycean_template, default_wavelength_grid
from instrument_model import load_jwst_nirspec
from observation_sim import ObservationSimulator, PlanetSystem

@dataclass
class DMSSensitivityPoint:
    dms_strength: float
    n_transits: int
    median_snr: float
    detection_prob: float
    n_trials: int

class DMSSensitivityRunner:
    DMS_STRENGTHS = [0.0, 0.02, 0.05, 0.08, 0.15, 0.25]
    N_TRANSITS = [5, 10, 20, 40, 80]

    def __init__(self, n_noise_trials: int=20, rng_seed: int=11):
        self.n_noise_trials = n_noise_trials
        self.rng = np.random.default_rng(rng_seed)
        self.instrument = load_jwst_nirspec()
        self.wl = default_wavelength_grid()

    def _dms_band_snr(self, sim: ObservationSimulator, planet: PlanetSystem, dms_strength: float, n_transits: int) -> float:
        tmpl_with = build_hycean_template(self.wl, planet_radius_re=planet.planet_radius_re, star_radius_rs=planet.star_radius_rs, dms_strength=dms_strength)
        tmpl_without = build_hycean_template(self.wl, planet_radius_re=planet.planet_radius_re, star_radius_rs=planet.star_radius_rs, dms_strength=0.0)
        dms_only_signal = tmpl_with.transit_depth_ppm - tmpl_without.transit_depth_ppm
        obs = sim.simulate(planet, tmpl_with, n_transits=n_transits)
        band_mask = (self.wl > 3.1) & (self.wl < 3.7) & (obs.noise_ppm > 0) & (obs.noise_ppm < 100000.0)
        if not np.any(band_mask):
            return 0.0
        snr_bins = np.where(band_mask, dms_only_signal / obs.noise_ppm, 0.0)
        return float(np.sqrt(np.nansum(snr_bins ** 2)))

    def run(self, planet: PlanetSystem, verbose: bool=True) -> List[DMSSensitivityPoint]:
        results = []
        for dms_s in self.DMS_STRENGTHS:
            for n_t in self.N_TRANSITS:
                snrs = []
                for _ in range(self.n_noise_trials):
                    sim = ObservationSimulator(self.instrument, rng=self.rng)
                    snrs.append(self._dms_band_snr(sim, planet, dms_s, n_t))
                snrs = np.array(snrs)
                point = DMSSensitivityPoint(dms_strength=dms_s, n_transits=n_t, median_snr=float(np.median(snrs)), detection_prob=float(np.mean(snrs >= 5.0)), n_trials=self.n_noise_trials)
                results.append(point)
                if verbose:
                    print(f'  dms_strength={dms_s:.2f}  N={n_t:>3}  median SNR={point.median_snr:5.1f}  P(detect)={point.detection_prob:.0%}')
        return results

    def save(self, results: List[DMSSensitivityPoint], path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dms_strength', 'n_transits', 'median_snr', 'detection_prob', 'n_trials'])
            for r in results:
                w.writerow([r.dms_strength, r.n_transits, f'{r.median_snr:.3f}', f'{r.detection_prob:.3f}', r.n_trials])
        print(f'DMS sensitivity table saved -> {path}')
if __name__ == '__main__':
    runner = DMSSensitivityRunner(n_noise_trials=15)
    k218b = PlanetSystem.k2_18b()
    print(f'Running DMS sensitivity sweep for {k218b.planet_name}...')
    results = runner.run(k218b)
    runner.save(results, 'results/sensitivity/dms_sweep.csv')
    default_strength_results = [r for r in results if r.dms_strength == 0.08]
    print("\nAt the pipeline's original DMS strength (0.08):")
    for r in default_strength_results:
        flag = '  <-- 5-sigma crossed' if r.detection_prob >= 0.5 else ''
        print(f'  {r.n_transits:>3} transits: median SNR={r.median_snr:.1f}, P(detect)={r.detection_prob:.0%}{flag}')
