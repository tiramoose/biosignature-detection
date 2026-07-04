import numpy as np
import os
import sys
import csv
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from atmosphere_templates import (
    build_earth_like_template, build_high_co2_template,
    build_reduced_o2_high_ch4_template, default_wavelength_grid,
)
from instrument_model import load_jwst_nirspec, InstrumentModel, InstrumentConfig
from observation_sim import ObservationSimulator, PlanetSystem


@dataclass
class SensitivityPoint:
    varied_param: str
    param_value: float
    atmosphere_type: str
    n_transits: int
    distance_pc: float
    det_prob: float
    median_snr: float
    n_trials: int
    bias_recovery: float = 1.0
    bias_fp_rate: float = 0.0


class SensitivityRunner:
    CLOUD_FRACS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99]
    NOISE_FLOORS = [10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
    SCALE_HEIGHTS = [3.0, 5.0, 7.0, 8.5, 10.0, 12.0]
    N_TRANSITS = [1, 2, 5, 10, 20, 50]
    DISTANCES_PC = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 65.0, 80.0, 100.0]
    ATM_TYPES = ["earth_like", "high_co2", "reduced_o2_high_ch4"]
    N_TRIALS = 20

    BUILDERS = {
        "earth_like": (build_earth_like_template, 1.0),
        "high_co2": (build_high_co2_template, 0.01),
        "reduced_o2_high_ch4": (build_reduced_o2_high_ch4_template, 0.001),
    }

    def __init__(self, base_planet: Optional[PlanetSystem] = None, seed: int = 99):
        self.rng = np.random.default_rng(seed)
        self.wl = default_wavelength_grid()
        self.base_planet = base_planet or PlanetSystem.trappist1e()

    def _detection_rate(self, planet, atm_type, cloud_fraction, scale_height_km, n_transits,
                         instrument, n_trials=20) -> Tuple[float, float]:
        builder, o2_ch4 = self.BUILDERS[atm_type]
        template = builder(
            self.wl, cloud_fraction=cloud_fraction, o2_ch4_ratio=o2_ch4,
            scale_height_km=scale_height_km, planet_radius_re=planet.planet_radius_re,
            star_radius_rs=planet.star_radius_rs,
        )
        detections = 0
        snrs = []
        for _ in range(n_trials):
            sim = ObservationSimulator(instrument, rng=np.random.default_rng(int(self.rng.integers(0, 999999))))
            obs = sim.simulate(planet, template, n_transits)
            snr = obs.detection_snr
            snrs.append(snr)
            if snr >= 5.0:
                detections += 1
        return detections / n_trials, float(np.median(snrs))

    def _make_instrument(self, noise_floor_ppm: float) -> InstrumentModel:
        base = load_jwst_nirspec()
        cfg = base.config
        modified = InstrumentConfig(
            name=f"JWST NIRSpec (floor={noise_floor_ppm:.0f}ppm)",
            wavelength_min_um=cfg.wavelength_min_um, wavelength_max_um=cfg.wavelength_max_um,
            spectral_resolution=cfg.spectral_resolution, collecting_area_m2=cfg.collecting_area_m2,
            throughput_peak=cfg.throughput_peak, read_noise_electrons=cfg.read_noise_electrons,
            dark_current_e_per_s=cfg.dark_current_e_per_s,
            n_pixels_per_resolution_element=cfg.n_pixels_per_resolution_element,
            pixel_scale_arcsec=cfg.pixel_scale_arcsec, detector_gain=cfg.detector_gain,
            saturation_electrons=cfg.saturation_electrons, stellar_noise_floor_ppm=noise_floor_ppm,
            notes=cfg.notes,
        )
        return InstrumentModel(modified)

    def sweep_cloud_fraction(self, atm_type="earth_like", n_transits=10) -> List[SensitivityPoint]:
        jwst = load_jwst_nirspec()
        results = []
        for cf in self.CLOUD_FRACS:
            dp, snr = self._detection_rate(self.base_planet, atm_type, cf, 8.5, n_transits, jwst, self.N_TRIALS)
            results.append(SensitivityPoint("cloud_fraction", cf, atm_type, n_transits,
                                             self.base_planet.distance_pc, dp, snr, self.N_TRIALS))
        return results

    def sweep_noise_floor(self, atm_type="earth_like", n_transits=10) -> List[SensitivityPoint]:
        results = []
        for nf in self.NOISE_FLOORS:
            instrument = self._make_instrument(nf)
            dp, snr = self._detection_rate(self.base_planet, atm_type, 0.5, 8.5, n_transits, instrument, self.N_TRIALS)
            results.append(SensitivityPoint("noise_floor_ppm", nf, atm_type, n_transits,
                                             self.base_planet.distance_pc, dp, snr, self.N_TRIALS))
        return results

    def sweep_scale_height(self, atm_type="earth_like", n_transits=10) -> List[SensitivityPoint]:
        jwst = load_jwst_nirspec()
        results = []
        for sh in self.SCALE_HEIGHTS:
            dp, snr = self._detection_rate(self.base_planet, atm_type, 0.5, sh, n_transits, jwst, self.N_TRIALS)
            results.append(SensitivityPoint("scale_height_km", sh, atm_type, n_transits,
                                             self.base_planet.distance_pc, dp, snr, self.N_TRIALS))
        return results

    def sweep_n_transits(self, atm_type="earth_like", cloud_fraction=0.5) -> List[SensitivityPoint]:
        jwst = load_jwst_nirspec()
        results = []
        for n_t in self.N_TRANSITS:
            dp, snr = self._detection_rate(self.base_planet, atm_type, cloud_fraction, 8.5, n_t, jwst, self.N_TRIALS)
            results.append(SensitivityPoint("n_transits", float(n_t), atm_type, n_t,
                                             self.base_planet.distance_pc, dp, snr, self.N_TRIALS))
        return results

    def heatmap_cloud_distance(self, atm_type="earth_like", n_transits=10,
                                cloud_fracs=None, distances_pc=None) -> Tuple[np.ndarray, List, List]:
        cloud_fracs = cloud_fracs or self.CLOUD_FRACS
        distances_pc = distances_pc or self.DISTANCES_PC
        jwst = load_jwst_nirspec()
        grid = np.zeros((len(cloud_fracs), len(distances_pc)))
        for i, cf in enumerate(cloud_fracs):
            for j, d in enumerate(distances_pc):
                planet = PlanetSystem.synthetic(
                    distance_pc=d, star_teff_k=self.base_planet.star_teff_k,
                    star_radius_rs=self.base_planet.star_radius_rs,
                    planet_radius_re=self.base_planet.planet_radius_re,
                    j_mag_at_10pc=self.base_planet.star_magnitude_j
                                  - 5 * np.log10(self.base_planet.distance_pc / 10.0),
                )
                dp, _ = self._detection_rate(planet, atm_type, cf, 8.5, n_transits, jwst, n_trials=12)
                grid[i, j] = dp
        return grid, cloud_fracs, distances_pc

    def bias_check(self, n_trials_per_point=15, cloud_fracs=None, scale_heights=None, n_transits=10) -> Dict:
        cloud_fracs = cloud_fracs or [0.0, 0.3, 0.6, 0.9, 0.97, 0.99]
        scale_heights = scale_heights or [3.0, 6.0, 8.5, 11.0]
        jwst = load_jwst_nirspec()
        results = {}
        for atm in self.ATM_TYPES:
            for cf in cloud_fracs:
                for sh in scale_heights:
                    dp, snr = self._detection_rate(self.base_planet, atm, cf, sh, n_transits, jwst, n_trials_per_point)
                    key = f"{atm}__cf{cf:.2f}__sh{sh:.1f}"
                    results[key] = {
                        "atm_type": atm, "cloud_fraction": cf, "scale_height_km": sh,
                        "n_transits": n_transits, "recovery_rate": dp, "median_snr": snr,
                        "n_trials": n_trials_per_point, "pass": dp >= 0.5,
                    }
        return results

    def run_all(self) -> Dict:
        all_results = {}
        all_results["cloud_sweep_el"] = self.sweep_cloud_fraction("earth_like", 10)
        all_results["cloud_sweep_co2"] = self.sweep_cloud_fraction("high_co2", 10)
        all_results["noise_sweep"] = self.sweep_noise_floor("earth_like", 10)
        all_results["sh_sweep"] = self.sweep_scale_height("earth_like", 10)
        all_results["nt_sweep"] = self.sweep_n_transits("earth_like", 0.5)
        all_results["nt_sweep_cloudy"] = self.sweep_n_transits("earth_like", 0.9)
        grid, cfs, dists = self.heatmap_cloud_distance("earth_like", n_transits=10)
        all_results["heatmap_el"] = {"grid": grid, "cloud_fracs": cfs, "distances_pc": dists}
        grid2, cfs2, dists2 = self.heatmap_cloud_distance("high_co2", n_transits=10)
        all_results["heatmap_co2"] = {"grid": grid2, "cloud_fracs": cfs2, "distances_pc": dists2}
        all_results["bias_check"] = self.bias_check()
        return all_results

    def save(self, results: Dict, output_dir: str = "results/sensitivity") -> None:
        os.makedirs(output_dir, exist_ok=True)
        for key in [k for k in results if "sweep" in k]:
            path = os.path.join(output_dir, f"{key}.csv")
            rows = results[key]
            if rows:
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=asdict(rows[0]).keys())
                    w.writeheader()
                    for r in rows:
                        w.writerow(asdict(r))
        for key in ["heatmap_el", "heatmap_co2"]:
            if key in results:
                path = os.path.join(output_dir, f"{key}.json")
                d = results[key].copy()
                d["grid"] = d["grid"].tolist()
                with open(path, "w") as f:
                    json.dump(d, f)
        if "bias_check" in results:
            path = os.path.join(output_dir, "bias_check.json")
            with open(path, "w") as f:
                json.dump(results["bias_check"], f, indent=2)


if __name__ == "__main__":
    runner = SensitivityRunner()
    results = runner.run_all()
    runner.save(results)
