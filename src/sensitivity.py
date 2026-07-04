import numpy as np
import os, sys, csv, json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
sys.path.insert(0, os.path.dirname(__file__))
from atmosphere_templates import build_earth_like_template, build_high_co2_template, build_reduced_o2_high_ch4_template, default_wavelength_grid
from instrument_model import load_jwst_nirspec, InstrumentModel, InstrumentConfig
from observation_sim import ObservationSimulator, PlanetSystem
from retrieval import TemplateRetrieval
from atmosphere_templates import TemplateGrid

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
    CLOUD_FRACS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    NOISE_FLOORS = [10.0, 20.0, 30.0, 50.0]
    SCALE_HEIGHTS = [5.0, 7.0, 8.5, 10.0, 12.0]
    N_TRANSITS = [5, 10, 20, 50]
    DISTANCES_PC = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    ATM_TYPES = ['earth_like', 'high_co2', 'reduced_o2_high_ch4']
    N_TRIALS = 20
    BUILDERS = {'earth_like': (build_earth_like_template, 1.0), 'high_co2': (build_high_co2_template, 0.01), 'reduced_o2_high_ch4': (build_reduced_o2_high_ch4_template, 0.001)}

    def __init__(self, base_planet: Optional[PlanetSystem]=None, seed: int=99):
        self.rng = np.random.default_rng(seed)
        self.wl = default_wavelength_grid()
        self.base_planet = base_planet or PlanetSystem.trappist1e()
        self._retrieval_cache: Dict[Tuple[float, float], TemplateRetrieval] = {}

    def _retrieval_engine(self, planet: PlanetSystem) -> TemplateRetrieval:
        key = (float(planet.planet_radius_re), float(planet.star_radius_rs))
        if key not in self._retrieval_cache:
            grid = TemplateGrid(wavelengths=self.wl)
            grid.build_grid(template_names=list(self.ATM_TYPES), cloud_fractions=[0.0, 0.3, 0.6, 0.9], scale_heights_km=[6.0, 8.5, 11.0], planet_radius_re=planet.planet_radius_re, star_radius_rs=planet.star_radius_rs)
            self._retrieval_cache[key] = TemplateRetrieval(grid)
        return self._retrieval_cache[key]

    def _detection_rate(self, planet: PlanetSystem, atm_type: str, cloud_fraction: float, scale_height_km: float, n_transits: int, instrument: InstrumentModel, n_trials: int=20) -> Tuple[float, float]:
        builder, o2_ch4 = self.BUILDERS[atm_type]
        template = builder(self.wl, cloud_fraction=cloud_fraction, o2_ch4_ratio=o2_ch4, scale_height_km=scale_height_km, planet_radius_re=planet.planet_radius_re, star_radius_rs=planet.star_radius_rs)
        retrieval = self._retrieval_engine(planet)
        recoveries = 0
        snrs = []
        for _ in range(n_trials):
            sim = ObservationSimulator(instrument=instrument, rng=np.random.default_rng(int(self.rng.integers(0, 99999))))
            obs = sim.simulate(planet, template, n_transits)
            try:
                result = retrieval.fit(obs.wavelengths_um, obs.observed_depth_ppm, obs.noise_ppm)
                snrs.append(result.detection_snr)
                if result.is_detected and result.preferred_atmosphere == atm_type:
                    recoveries += 1
            except Exception:
                snrs.append(0.0)
        return (recoveries / n_trials, float(np.median(snrs)))

    def _make_instrument(self, noise_floor_ppm: float) -> InstrumentModel:
        base = load_jwst_nirspec()
        cfg = base.config
        modified = InstrumentConfig(name=f'JWST NIRSpec (floor={noise_floor_ppm:.0f}ppm)', wavelength_min_um=cfg.wavelength_min_um, wavelength_max_um=cfg.wavelength_max_um, spectral_resolution=cfg.spectral_resolution, collecting_area_m2=cfg.collecting_area_m2, throughput_peak=cfg.throughput_peak, read_noise_electrons=cfg.read_noise_electrons, dark_current_e_per_s=cfg.dark_current_e_per_s, n_pixels_per_resolution_element=cfg.n_pixels_per_resolution_element, pixel_scale_arcsec=cfg.pixel_scale_arcsec, detector_gain=cfg.detector_gain, saturation_electrons=cfg.saturation_electrons, stellar_noise_floor_ppm=noise_floor_ppm, notes=cfg.notes)
        return InstrumentModel(modified)

    def sweep_cloud_fraction(self, atm_type: str='earth_like', n_transits: int=10) -> List[SensitivityPoint]:
        print(f'  Cloud fraction sweep ({atm_type}, N={n_transits})...')
        jwst = load_jwst_nirspec()
        results = []
        for cf in self.CLOUD_FRACS:
            dp, snr = self._detection_rate(self.base_planet, atm_type, cf, 8.5, n_transits, jwst, n_trials=self.N_TRIALS)
            results.append(SensitivityPoint(varied_param='cloud_fraction', param_value=cf, atmosphere_type=atm_type, n_transits=n_transits, distance_pc=self.base_planet.distance_pc, det_prob=dp, median_snr=snr, n_trials=self.N_TRIALS))
            print(f'    cf={cf:.2f}: rec={dp:.0%}  SNR={snr:.1f}σ')
        return results

    def sweep_noise_floor(self, atm_type: str='earth_like', n_transits: int=10) -> List[SensitivityPoint]:
        print(f'  Noise floor sweep ({atm_type}, N={n_transits})...')
        results = []
        for nf in self.NOISE_FLOORS:
            instrument = self._make_instrument(nf)
            dp, snr = self._detection_rate(self.base_planet, atm_type, 0.5, 8.5, n_transits, instrument, n_trials=self.N_TRIALS)
            results.append(SensitivityPoint(varied_param='noise_floor_ppm', param_value=nf, atmosphere_type=atm_type, n_transits=n_transits, distance_pc=self.base_planet.distance_pc, det_prob=dp, median_snr=snr, n_trials=self.N_TRIALS))
            print(f'    floor={nf:.0f}ppm: rec={dp:.0%}  SNR={snr:.1f}σ')
        return results

    def sweep_scale_height(self, atm_type: str='earth_like', n_transits: int=10) -> List[SensitivityPoint]:
        print(f'  Scale height sweep ({atm_type}, N={n_transits})...')
        jwst = load_jwst_nirspec()
        results = []
        for sh in self.SCALE_HEIGHTS:
            dp, snr = self._detection_rate(self.base_planet, atm_type, 0.5, sh, n_transits, jwst, n_trials=self.N_TRIALS)
            results.append(SensitivityPoint(varied_param='scale_height_km', param_value=sh, atmosphere_type=atm_type, n_transits=n_transits, distance_pc=self.base_planet.distance_pc, det_prob=dp, median_snr=snr, n_trials=self.N_TRIALS))
            print(f'    H={sh:.1f}km: rec={dp:.0%}  SNR={snr:.1f}σ')
        return results

    def sweep_n_transits(self, atm_type: str='earth_like', cloud_fraction: float=0.5) -> List[SensitivityPoint]:
        print(f'  N_transits sweep ({atm_type}, cf={cloud_fraction})...')
        jwst = load_jwst_nirspec()
        results = []
        for n_t in self.N_TRANSITS:
            dp, snr = self._detection_rate(self.base_planet, atm_type, cloud_fraction, 8.5, n_t, jwst, n_trials=self.N_TRIALS)
            results.append(SensitivityPoint(varied_param='n_transits', param_value=float(n_t), atmosphere_type=atm_type, n_transits=n_t, distance_pc=self.base_planet.distance_pc, det_prob=dp, median_snr=snr, n_trials=self.N_TRIALS))
            print(f'    N_t={n_t}: rec={dp:.0%}  SNR={snr:.1f}σ')
        return results

    def heatmap_cloud_distance(self, atm_type: str='earth_like', n_transits: int=10, cloud_fracs: Optional[List[float]]=None, distances_pc: Optional[List[float]]=None) -> np.ndarray:
        if cloud_fracs is None:
            cloud_fracs = [0.0, 0.2, 0.4, 0.6, 0.8]
        if distances_pc is None:
            distances_pc = [5.0, 10.0, 15.0, 20.0, 30.0]
        print(f'  2D heatmap: cloud×distance ({atm_type}, N={n_transits})')
        jwst = load_jwst_nirspec()
        grid = np.zeros((len(cloud_fracs), len(distances_pc)))
        for i, cf in enumerate(cloud_fracs):
            for j, d in enumerate(distances_pc):
                planet = PlanetSystem.synthetic(distance_pc=d, star_teff_k=self.base_planet.star_teff_k, star_radius_rs=self.base_planet.star_radius_rs, planet_radius_re=self.base_planet.planet_radius_re, j_mag_at_10pc=self.base_planet.star_magnitude_j - 5 * np.log10(self.base_planet.distance_pc / 10.0))
                dp, _ = self._detection_rate(planet, atm_type, cf, 8.5, n_transits, jwst, n_trials=12)
                grid[i, j] = dp
                print(f'    cf={cf:.1f}, d={d:.0f}pc → {dp:.0%}')
        return (grid, cloud_fracs, distances_pc)

    def bias_check(self, n_trials_per_point: int=15, cloud_fracs: Optional[List[float]]=None, scale_heights: Optional[List[float]]=None, n_transits: int=10) -> Dict:
        if cloud_fracs is None:
            cloud_fracs = [0.0, 0.3, 0.6, 0.9]
        if scale_heights is None:
            scale_heights = [6.0, 8.5, 11.0]
        print(f'  Bias check: {len(self.ATM_TYPES)} atm × {len(cloud_fracs)} cf × {len(scale_heights)} sh × {n_trials_per_point} trials...')
        jwst = load_jwst_nirspec()
        results = {}
        total = len(self.ATM_TYPES) * len(cloud_fracs) * len(scale_heights)
        done = 0
        for atm in self.ATM_TYPES:
            for cf in cloud_fracs:
                for sh in scale_heights:
                    dp, snr = self._detection_rate(self.base_planet, atm, cf, sh, n_transits, jwst, n_trials=n_trials_per_point)
                    key = f'{atm}__cf{cf:.1f}__sh{sh:.1f}'
                    results[key] = {'atm_type': atm, 'cloud_fraction': cf, 'scale_height_km': sh, 'n_transits': n_transits, 'recovery_rate': dp, 'median_snr': snr, 'n_trials': n_trials_per_point, 'pass': dp >= 0.5}
                    done += 1
                    if done % 6 == 0:
                        print(f'    [{done}/{total}] {atm[:12]:12s} cf={cf:.1f} sh={sh:.0f}km → {dp:.0%} ({snr:.0f}σ)')
        all_rates = [v['recovery_rate'] for v in results.values()]
        failing = [k for k, v in results.items() if not v['pass']]
        print(f'\n  Bias check summary:')
        print(f'    Mean recovery rate: {np.mean(all_rates):.1%}')
        print(f'    Min recovery rate:  {np.min(all_rates):.1%}')
        print(f'    Failing points (<50%): {len(failing)}')
        if failing:
            for k in failing[:5]:
                v = results[k]
                print(f"      → {k}: {v['recovery_rate']:.0%}  (cf={v['cloud_fraction']:.1f}, sh={v['scale_height_km']:.0f}km)")
        return results

    def run_all(self, verbose: bool=True) -> Dict:
        print(f"\n{'═' * 60}")
        print(f'  SENSITIVITY ANALYSIS — {self.base_planet.planet_name}')
        print(f"{'═' * 60}\n")
        all_results = {}
        print('1. Cloud fraction sweep (earth_like, N=10):')
        all_results['cloud_sweep_el'] = self.sweep_cloud_fraction('earth_like', 10)
        print('\n2. Cloud fraction sweep (high_co2, N=10):')
        all_results['cloud_sweep_co2'] = self.sweep_cloud_fraction('high_co2', 10)
        print('\n3. Noise floor sweep (earth_like, N=10):')
        all_results['noise_sweep'] = self.sweep_noise_floor('earth_like', 10)
        print('\n4. Scale height sweep (earth_like, N=10):')
        all_results['sh_sweep'] = self.sweep_scale_height('earth_like', 10)
        print('\n5. N_transits sweep (earth_like, cf=0.5):')
        all_results['nt_sweep'] = self.sweep_n_transits('earth_like', 0.5)
        print('\n6. N_transits sweep (earth_like, cf=0.8 — pessimistic clouds):')
        all_results['nt_sweep_cloudy'] = self.sweep_n_transits('earth_like', 0.8)
        print('\n7. 2D heatmap: cloud × distance (earth_like, N=10):')
        grid, cfs, dists = self.heatmap_cloud_distance('earth_like', n_transits=10)
        all_results['heatmap_el'] = {'grid': grid, 'cloud_fracs': cfs, 'distances_pc': dists}
        print('\n8. 2D heatmap: cloud × distance (high_co2, N=10):')
        grid2, cfs2, dists2 = self.heatmap_cloud_distance('high_co2', n_transits=10)
        all_results['heatmap_co2'] = {'grid': grid2, 'cloud_fracs': cfs2, 'distances_pc': dists2}
        print('\n9. Systematic bias check:')
        all_results['bias_check'] = self.bias_check()
        print(f"\n{'═' * 60}")
        print(f'  SENSITIVITY ANALYSIS COMPLETE')
        print(f"{'═' * 60}")
        return all_results

    def save(self, results: Dict, output_dir: str='results/sensitivity') -> None:
        os.makedirs(output_dir, exist_ok=True)
        sweep_keys = [k for k in results if 'sweep' in k]
        for key in sweep_keys:
            path = os.path.join(output_dir, f'{key}.csv')
            rows = results[key]
            if rows:
                with open(path, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=asdict(rows[0]).keys())
                    w.writeheader()
                    for r in rows:
                        w.writerow(asdict(r))
                print(f'Saved → {path}')
        for key in ['heatmap_el', 'heatmap_co2']:
            if key in results:
                path = os.path.join(output_dir, f'{key}.json')
                d = results[key].copy()
                d['grid'] = d['grid'].tolist()
                with open(path, 'w') as f:
                    json.dump(d, f)
                print(f'Saved → {path}')
        if 'bias_check' in results:
            path = os.path.join(output_dir, 'bias_check.json')
            with open(path, 'w') as f:
                json.dump(results['bias_check'], f, indent=2)
            print(f'Saved → {path}')
if __name__ == '__main__':
    runner = SensitivityRunner()
    results = runner.run_all()
    runner.save(results)
