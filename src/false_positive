import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os, sys
 
sys.path.insert(0, os.path.dirname(__file__))
from atmosphere_templates import (
    AtmosphereTemplate, TemplateGrid, default_wavelength_grid,
    build_earth_like_template, build_high_co2_template,
    build_reduced_o2_high_ch4_template, build_hycean_template,
)
from observation_sim import ObservationSimulator, PlanetSystem
from retrieval import TemplateRetrieval
 
 
# ---------------------------------------------------------------------------
# Single trial result
# ---------------------------------------------------------------------------
 
@dataclass
class InjectionTrial:
    """Result of a single injection/recovery trial."""
    true_atmosphere:      str     # Injected atmosphere type
    retrieved_atmosphere: str     # Retrieval best-fit type
    true_cloud_fraction:  float
    true_scale_height_km: float
    n_transits:           int
    detection_snr:        float   # Retrieval SNR
    broadband_snr:        float   # Forward-model SNR (from observation_sim)
    chi2_reduced:         float
    delta_bic_correct:    float   # ΔBIC for correct model (vs. next best)
    is_detected:          bool    # SNR ≥ 5σ
    is_correct:           bool    # Retrieved == True
 
    @property
    def is_true_positive(self) -> bool:
        return self.is_detected and self.is_correct
 
    @property
    def is_false_positive(self) -> bool:
        """Detected something that wasn't there (wrong type)."""
        return self.is_detected and not self.is_correct
 
    @property
    def is_false_negative(self) -> bool:
        """Real signal present but not detected."""
        return not self.is_detected
 
 
# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
 
@dataclass
class InjectionRecoveryTable:
    """
    Full injection/recovery results across all atmosphere types and parameters.
    """
    trials: List[InjectionTrial]
    atmosphere_types: List[str]
 
    @property
    def n_trials(self) -> int:
        return len(self.trials)
 
    def confusion_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        N_types × N_types confusion matrix.
        Rows = true (injected) atmosphere.
        Cols = retrieved atmosphere.
        Normalized to row fractions.
        """
        types = self.atmosphere_types
        idx = {t: i for i, t in enumerate(types)}
        matrix = np.zeros((len(types), len(types)), dtype=int)
        for trial in self.trials:
            ti = idx.get(trial.true_atmosphere, -1)
            ri = idx.get(trial.retrieved_atmosphere, -1)
            if ti >= 0 and ri >= 0:
                matrix[ti, ri] += 1
        return matrix, types
 
    def detection_completeness(self) -> Dict[str, float]:
        """
        Detection completeness per atmosphere type.
        = fraction of injected signals that were detected at ≥5σ.
        """
        from collections import defaultdict
        detected = defaultdict(int)
        total    = defaultdict(int)
        for t in self.trials:
            total[t.true_atmosphere] += 1
            if t.is_detected:
                detected[t.true_atmosphere] += 1
        return {atm: detected[atm] / max(total[atm], 1) for atm in self.atmosphere_types}
 
    def false_positive_rate(self) -> Dict[str, float]:
        """
        False positive rate per atmosphere type.
        = fraction of detections that retrieved the WRONG atmosphere.
        """
        from collections import defaultdict
        fp = defaultdict(int)
        det = defaultdict(int)
        for t in self.trials:
            if t.is_detected:
                det[t.true_atmosphere] += 1
                if not t.is_correct:
                    fp[t.true_atmosphere] += 1
        return {atm: fp[atm] / max(det[atm], 1) for atm in self.atmosphere_types}
 
    def completeness_vs_ntransits(
        self, atm_type: str, n_transit_bins: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Completeness as a function of number of transits observed."""
        if n_transit_bins is None:
            all_n = sorted(set(t.n_transits for t in self.trials))
            n_transit_bins = all_n
 
        completeness = []
        for n in n_transit_bins:
            relevant = [t for t in self.trials
                        if t.true_atmosphere == atm_type and t.n_transits == n]
            if relevant:
                completeness.append(sum(t.is_detected for t in relevant) / len(relevant))
            else:
                completeness.append(np.nan)
        return np.array(n_transit_bins), np.array(completeness)
 
    def print_summary(self) -> None:
        print(f"\n{'═'*65}")
        print(f"  INJECTION/RECOVERY SUMMARY  ({self.n_trials} trials)")
        print(f"{'═'*65}")
 
        completeness = self.detection_completeness()
        fpr = self.false_positive_rate()
 
        print(f"\n  {'Atmosphere':28s}  {'Completeness':14s}  {'False Pos Rate':14s}")
        print(f"  {'─'*28}  {'─'*14}  {'─'*14}")
        for atm in self.atmosphere_types:
            c = completeness.get(atm, 0)
            f = fpr.get(atm, 0)
            bar_c = '█' * int(c * 15)
            print(f"  {atm:28s}  {c:6.1%}  {bar_c:15s}  {f:6.1%}")
 
        print(f"\n  Confusion matrix (rows=true, cols=retrieved):")
        matrix, types = self.confusion_matrix()
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm = np.where(row_sums > 0, matrix / row_sums, 0)
        short = [t[:10] for t in types]
        header = "  " + " " * 12 + "  ".join(f"{s:>10}" for s in short)
        print(header)
        for i, row_name in enumerate(types):
            row_str = "  ".join(f"{norm[i,j]:10.2f}" for j in range(len(types)))
            prefix = "→ " if i == np.argmax(norm[i]) else "  "
            print(f"  {prefix}{row_name:10s}  {row_str}")
 
        print(f"\n{'═'*65}")
 
 
# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
 
class InjectionRecoveryRunner:
    """
    Runs injection/recovery tests over the full parameter space.
 
    For each atmosphere type and parameter combination, injects the signal
    into a noisy simulated observation, then runs the retrieval.
    Repeats N times per combination to build statistics.
    """
 
    def __init__(
        self,
        grid: TemplateGrid,
        simulator: ObservationSimulator,
        retrieval: TemplateRetrieval,
        rng: Optional[np.random.Generator] = None,
    ):
        self.grid = grid
        self.simulator = simulator
        self.retrieval = retrieval
        self.rng = rng or np.random.default_rng()
 
    def run(
        self,
        planet: PlanetSystem,
        n_transits_list: List[int] = [5, 10, 20],
        cloud_fractions: List[float] = [0.0, 0.5, 0.8],
        scale_heights_km: List[float] = [7.0, 8.5, 10.0],
        n_noise_trials: int = 5,
        verbose: bool = True,
    ) -> InjectionRecoveryTable:
        """
        Run the full injection/recovery grid.
 
        Parameters
        ----------
        planet           : target planet+star system
        n_transits_list  : list of transit counts to test
        cloud_fractions  : list of cloud fractions to inject
        scale_heights_km : list of scale heights to inject
        n_noise_trials   : number of independent noise realizations per parameter point
        verbose          : print progress
 
        Returns
        -------
        InjectionRecoveryTable with all trial results
        """
        wl = default_wavelength_grid()
        builders = {
            "earth_like":          build_earth_like_template,
            "high_co2":            build_high_co2_template,
            "reduced_o2_high_ch4": build_reduced_o2_high_ch4_template,
        }
 
        # Pick o2_ch4 defaults per type
        o2_ch4_defaults = {
            "earth_like": 1.0,
            "high_co2": 0.01,
            "reduced_o2_high_ch4": 0.001,
        }
 
        trials = []
        total = (len(builders) * len(cloud_fractions) * len(scale_heights_km)
                 * len(n_transits_list) * n_noise_trials)
        done = 0
 
        for atm_name, builder in builders.items():
            for cf in cloud_fractions:
                for sh in scale_heights_km:
                    ratio = o2_ch4_defaults[atm_name]
 
                    # Build the true template
                    true_tmpl = builder(
                        wl, cloud_fraction=cf, o2_ch4_ratio=ratio,
                        scale_height_km=sh,
                        planet_radius_re=planet.planet_radius_re,
                        star_radius_rs=planet.star_radius_rs,
                    )
 
                    for n_t in n_transits_list:
                        for _ in range(n_noise_trials):
                            done += 1
 
                            # Simulate observation (new noise realization each time)
                            obs = self.simulator.simulate(planet, true_tmpl, n_t)
 
                            # Run retrieval
                            try:
                                ret = self.retrieval.fit(
                                    obs.wavelengths_um,
                                    obs.observed_depth_ppm,
                                    obs.noise_ppm,
                                    wavelength_range=(0.6, 5.3),
                                )
                                retrieved_name = ret.preferred_atmosphere
                                det_snr = ret.detection_snr
                                chi2_red = ret.best_chi2_reduced
 
                                # ΔBIC between correct model and best-fit
                                correct_bic = ret.bic_scores.get(atm_name, np.inf)
                                best_bic = min(ret.bic_scores.values())
                                delta_bic_correct = correct_bic - best_bic
 
                            except Exception as e:
                                retrieved_name = "unknown"
                                det_snr = 0.0
                                chi2_red = np.inf
                                delta_bic_correct = np.inf
 
                            trial = InjectionTrial(
                                true_atmosphere=atm_name,
                                retrieved_atmosphere=retrieved_name,
                                true_cloud_fraction=cf,
                                true_scale_height_km=sh,
                                n_transits=n_t,
                                detection_snr=det_snr,
                                broadband_snr=obs.detection_snr,
                                chi2_reduced=chi2_red,
                                delta_bic_correct=delta_bic_correct,
                                is_detected=(det_snr >= 5.0),
                                is_correct=(retrieved_name == atm_name),
                            )
                            trials.append(trial)
 
                            if verbose and done % 25 == 0:
                                print(f"  [{done:4d}/{total}] {atm_name:25s} "
                                      f"cf={cf:.1f} sh={sh:.0f}km N={n_t:3d}t "
                                      f"→ {retrieved_name:25s} SNR={det_snr:.1f}σ "
                                      f"{'✓' if atm_name == retrieved_name else '✗'}")
 
        atm_types = list(builders.keys())
        return InjectionRecoveryTable(trials=trials, atmosphere_types=atm_types)
 
 
def save_fp_table(table: InjectionRecoveryTable, path: str) -> None:
    """Save false positive results to CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "true_atmosphere", "retrieved_atmosphere", "cloud_fraction",
            "scale_height_km", "n_transits", "detection_snr", "broadband_snr",
            "chi2_reduced", "delta_bic_correct", "is_detected", "is_correct",
            "is_true_positive", "is_false_positive",
        ])
        for t in table.trials:
            w.writerow([
                t.true_atmosphere, t.retrieved_atmosphere, t.true_cloud_fraction,
                t.true_scale_height_km, t.n_transits,
                f"{t.detection_snr:.3f}", f"{t.broadband_snr:.3f}",
                f"{t.chi2_reduced:.3f}", f"{t.delta_bic_correct:.3f}",
                int(t.is_detected), int(t.is_correct),
                int(t.is_true_positive), int(t.is_false_positive),
            ])
    print(f"False positive table saved → {path}  ({len(table.trials)} trials)")
 
 
# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    from instrument_model import load_jwst_nirspec
 
    print("Building template grid...")
    grid = TemplateGrid()
    grid.build_grid(
        planet_radius_re=0.92, star_radius_rs=0.1192,
        cloud_fractions=[0.0, 0.5],
        scale_heights_km=[7.0, 8.5, 10.0],
    )
 
    jwst = load_jwst_nirspec()
    sim  = ObservationSimulator(jwst, rng=np.random.default_rng(0))
    ret  = TemplateRetrieval(grid)
    runner = InjectionRecoveryRunner(grid, sim, ret, rng=np.random.default_rng(1))
 
    planet = PlanetSystem.trappist1e()
    print("Running injection/recovery (quick test, 3 noise trials)...")
    table = runner.run(planet, n_transits_list=[10, 20], n_noise_trials=3, verbose=True)
    table.print_summary()
    save_fp_table(table, "results/false_positive_table.csv")
