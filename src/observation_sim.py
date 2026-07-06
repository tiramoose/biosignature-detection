import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from atmosphere_templates import (
    AtmosphereTemplate,
    default_wavelength_grid,
    get_default_templates,
)
from instrument_model import InstrumentModel, load_jwst_nirspec


# ---------------------------------------------------------------------------
# Planet system dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlanetSystem:
    """
    Star + planet system descriptor.

    Everything the simulator needs to know about the astrophysical target:
    the host star's brightness (sets photon rate), the planet's size
    (sets transit depth), and the transit geometry (sets how long you
    collect in-transit light per orbit).
    """
    planet_name: str
    star_teff_k: float
    star_radius_rs: float
    star_magnitude_j: float
    planet_radius_re: float
    orbital_period_days: float
    transit_duration_hours: float
    distance_pc: float
    equilibrium_temp_k: float

    @property
    def transit_duration_s(self) -> float:
        return self.transit_duration_hours * 3600.0

    @property
    def in_habitable_zone(self) -> bool:
        """Rough habitable zone check: 200 K < T_eq < 320 K."""
        return 200.0 < self.equilibrium_temp_k < 320.0

    # ---- Pre-built real targets ----------------------------------------

    @classmethod
    def trappist1e(cls) -> "PlanetSystem":
        """
        TRAPPIST-1e — the canonical habitable-zone rocky exoplanet target.
        M-dwarf host makes it ideal for transit spectroscopy with JWST.
        """
        return cls(
            planet_name="TRAPPIST-1e",
            star_teff_k=2566.0,
            star_radius_rs=0.1192,
            star_magnitude_j=11.35,
            planet_radius_re=0.92,
            orbital_period_days=6.10,
            transit_duration_hours=0.96,
            distance_pc=12.43,
            equilibrium_temp_k=251.0,
        )

    @classmethod
    def k2_18b(cls) -> "PlanetSystem":
        """
        K2-18b — sub-Neptune 'hycean world' candidate.
        JWST detected CH4 and CO2 in its atmosphere (Madhusudhan et al. 2023).
        Excellent validation target for our CH4 templates.
        """
        return cls(
            planet_name="K2-18b",
            star_teff_k=3457.0,
            star_radius_rs=0.4445,
            star_magnitude_j=9.76,
            planet_radius_re=2.37,
            orbital_period_days=32.94,
            transit_duration_hours=2.77,
            distance_pc=38.0,
            equilibrium_temp_k=255.0,
        )

    @classmethod
    def lhs_1140b(cls) -> "PlanetSystem":
        """
        LHS 1140b — rocky super-Earth in the habitable zone.
        One of the best rocky planet targets for JWST atmospheric characterization.
        """
        return cls(
            planet_name="LHS 1140b",
            star_teff_k=3216.0,
            star_radius_rs=0.2138,
            star_magnitude_j=9.61,
            planet_radius_re=1.727,
            orbital_period_days=24.74,
            transit_duration_hours=1.85,
            distance_pc=14.99,
            equilibrium_temp_k=235.0,
        )

    @classmethod
    def trappist1d(cls) -> "PlanetSystem":
        """TRAPPIST-1d — inner edge of habitable zone, compact period."""
        return cls(
            planet_name="TRAPPIST-1d",
            star_teff_k=2566.0,
            star_radius_rs=0.1192,
            star_magnitude_j=11.35,
            planet_radius_re=0.788,
            orbital_period_days=4.05,
            transit_duration_hours=0.726,
            distance_pc=12.43,
            equilibrium_temp_k=288.0,
        )

    @classmethod
    def synthetic(
        cls,
        distance_pc: float,
        star_teff_k: float = 3000.0,
        star_radius_rs: float = 0.2,
        planet_radius_re: float = 1.0,
        j_mag_at_10pc: float = 12.0,
    ) -> "PlanetSystem":
        """
        Create a generic synthetic planet at arbitrary distance.
        Used for detection-horizon sweeps.
        """
        dm = 5.0 * np.log10(distance_pc / 10.0)
        teq = 250.0 * (0.2 / star_radius_rs) ** 0.5 * (5778 / star_teff_k) ** 0.5
        return cls(
            planet_name=f"synthetic_{distance_pc:.0f}pc",
            star_teff_k=star_teff_k,
            star_radius_rs=star_radius_rs,
            star_magnitude_j=j_mag_at_10pc + dm,
            planet_radius_re=planet_radius_re,
            orbital_period_days=10.0,
            transit_duration_hours=1.5,
            distance_pc=distance_pc,
            equilibrium_temp_k=float(teq),
        )


# ---------------------------------------------------------------------------
# Observation result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ObservationResult:
    """
    Output of a single simulated transit observation.

    Contains the true input spectrum, the noisy observed spectrum,
    the noise budget, and pre-computed SNR metrics.
    """
    planet: PlanetSystem
    atmosphere_type: str
    n_transits: int
    instrument_name: str

    wavelengths_um: np.ndarray
    true_depth_ppm: np.ndarray
    observed_depth_ppm: np.ndarray
    noise_ppm: np.ndarray
    snr_per_bin: np.ndarray
    noise_budget: Dict[str, np.ndarray]
    baseline_ppm: float = 0.0

    @property
    def detection_snr(self) -> float:
        """
        Broadband detection SNR via matched filter (quadrature over all bins).
        This is the SNR for detecting spectral modulation vs. a flat spectrum,
        i.e. the atmosphere-only signal above the featureless (Rp/Rs)^2 baseline.
        """
        return float(np.sqrt(np.nansum(self.snr_per_bin ** 2)))

    @property
    def is_detected(self) -> bool:
        """5σ detection threshold (standard in exoplanet literature)."""
        return self.detection_snr >= 5.0

    @property
    def biosignature_snr(self) -> Dict[str, float]:
        """
        Integrated SNR within each key biosignature wavelength window.

        For each feature, combines per-bin SNRs in quadrature across
        the feature's wavelength range.

        Returns dict of feature_name → integrated SNR.
        """
        windows = {
            "O2_A_band": (0.74, 0.79),
            "H2O_1.4um": (1.30, 1.47),
            "H2O_1.9um": (1.80, 1.95),
            "CH4_1.7um": (1.60, 1.75),
            "CH4_2.3um": (2.15, 2.50),
            "CO2_4.3um": (4.00, 4.65),
            "CO2_2.0um": (1.90, 2.10),
        }
        results = {}
        for name, (lo, hi) in windows.items():
            mask = (self.wavelengths_um >= lo) & (self.wavelengths_um <= hi)
            if mask.sum() > 0:
                results[name] = float(np.sqrt(np.nansum(self.snr_per_bin[mask] ** 2)))
            else:
                results[name] = 0.0
        return results

    @property
    def median_noise_ppm(self) -> float:
        """Median noise per spectral bin across the full bandpass."""
        valid = self.noise_ppm[
            (self.wavelengths_um >= 0.6)
            & (self.noise_ppm < 1e5)
        ]
        return float(np.median(valid)) if len(valid) > 0 else float(np.median(self.noise_ppm))

    def summary(self) -> str:
        bio = self.biosignature_snr
        detected = "✓ DETECTED" if self.is_detected else "✗ not detected"
        lines = [
            f"{'─' * 55}",
            f"  Planet     : {self.planet.planet_name}",
            f"  Atmosphere : {self.atmosphere_type}",
            f"  Instrument : {self.instrument_name}",
            f"  N transits : {self.n_transits}",
            f"  Broadband SNR : {self.detection_snr:.1f}σ  [{detected}]",
            f"  Median noise  : {self.median_noise_ppm:.0f} ppm/bin",
            "  Biosignature feature SNRs:",
        ]
        for feat, snr in bio.items():
            bar = "█" * int(min(snr, 20))
            if snr >= 5:
                bar += " ← detectable"
            lines.append(f"    {feat:16s}: {snr:5.1f}σ  {bar}")
        lines.append(f"{'─' * 55}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class ObservationSimulator:
    """
    Runs end-to-end transit spectroscopy simulations.

    Core method: simulate()
      planet + atmosphere template + instrument → ObservationResult

    Also provides:
      run_grid()          — batch over planets × templates × n_transits
      detection_horizon() — SNR vs. distance curve for a given template
    """

    def __init__(
        self,
        instrument: InstrumentModel,
        rng: Optional[np.random.Generator] = None,
        exposure_time_s: float = 88.0,
    ):
        self.instrument = instrument
        self.rng = rng if rng is not None else np.random.default_rng()
        self.exposure_time_s = exposure_time_s

    def simulate(
        self,
        planet: PlanetSystem,
        template: AtmosphereTemplate,
        n_transits: int = 10,
        verbose: bool = False,
    ) -> ObservationResult:
        """
        Simulate a transit spectroscopy observation.
        """
        wl = template.wavelengths_um
        true_depth = template.transit_depth_ppm.copy()
        baseline_ppm = float(template.parameters.get("base_depth_ppm", np.median(true_depth)))

        # Step 1: Stellar photon rate (photons/s/bin)
        photon_rate = self.instrument.stellar_photon_rate(
            wl,
            star_magnitude_j=planet.star_magnitude_j,
            star_teff_k=planet.star_teff_k,
        )

        # Step 2: Noise budget over total in-transit time
        total_transit_s = planet.transit_duration_s * n_transits
        n_exp = max(1, int(total_transit_s / self.exposure_time_s))
        budget = self.instrument.noise_model(photon_rate, self.exposure_time_s, n_exp)

        # Step 3: Convert noise from electrons → ppm
        sig = budget["signal_e"]
        noise_ppm = np.where(
            sig > 1.0,
            budget["total_noise"] / sig * 1e6,
            1e6,
        )

        # Mask out-of-instrument-range bins
        inst = self.instrument.config
        in_range = (wl >= inst.wavelength_min_um) & (wl <= inst.wavelength_max_um)
        noise_ppm = np.where(in_range, noise_ppm, 1e6)

        # Step 4: Draw noisy observed spectrum
        noise_draw = self.rng.normal(0.0, noise_ppm)
        observed_depth = true_depth + noise_draw

        # Step 5: Per-bin SNR, on the modulation above baseline
        modulation_ppm = np.abs(true_depth - baseline_ppm)
        snr_per_bin = np.where(
            (noise_ppm > 0) & (noise_ppm < 1e5) & in_range,
            modulation_ppm / noise_ppm,
            0.0,
        )

        if verbose:
            print(f"\n[simulate] {planet.planet_name} × {template.name} × {n_transits} transits")
            print(f"  In-transit time   : {total_transit_s / 3600:.1f} h  ({n_exp} exposures)")
            wl_mid = np.argmin(np.abs(wl - 1.38))
            print("  At 1.38 μm (H2O window):")
            print(f"    Photon rate  : {photon_rate[wl_mid]:.2e} ph/s/bin")
            print(f"    Signal       : {sig[wl_mid]:.2e} e⁻")
            print(f"    Shot noise   : {budget['shot_noise'][wl_mid]:.2e} e⁻")
            print(f"    Total noise  : {budget['total_noise'][wl_mid]:.2e} e⁻  ({noise_ppm[wl_mid]:.0f} ppm)")

        return ObservationResult(
            planet=planet,
            atmosphere_type=template.name,
            n_transits=n_transits,
            instrument_name=inst.name,
            wavelengths_um=wl,
            true_depth_ppm=true_depth,
            observed_depth_ppm=observed_depth,
            noise_ppm=noise_ppm,
            snr_per_bin=snr_per_bin,
            noise_budget=budget,
            baseline_ppm=baseline_ppm,
        )

    def simulate_batch(
        self,
        planet: PlanetSystem,
        template: AtmosphereTemplate,
        n_transits: int,
        n_trials: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Draw n_trials independent noise realizations for the same
        (planet, template, n_transits) in one vectorized call.
        """
        rng = rng if rng is not None else self.rng
        wl = template.wavelengths_um
        true_depth = template.transit_depth_ppm

        photon_rate = self.instrument.stellar_photon_rate(
            wl,
            star_magnitude_j=planet.star_magnitude_j,
            star_teff_k=planet.star_teff_k,
        )
        total_transit_s = planet.transit_duration_s * n_transits
        n_exp = max(1, int(total_transit_s / self.exposure_time_s))
        budget = self.instrument.noise_model(photon_rate, self.exposure_time_s, n_exp)

        sig = budget["signal_e"]
        noise_ppm = np.where(sig > 1.0, budget["total_noise"] / sig * 1e6, 1e6)
        inst = self.instrument.config
        in_range = (wl >= inst.wavelength_min_um) & (wl <= inst.wavelength_max_um)
        noise_ppm = np.where(in_range, noise_ppm, 1e6)

        noise_draws = rng.normal(0.0, noise_ppm, size=(n_trials, len(wl)))
        observed = true_depth[None, :] + noise_draws
        return observed, noise_ppm, true_depth

    def run_grid(
        self,
        planets: List[PlanetSystem],
        templates: Dict[str, AtmosphereTemplate],
        n_transits_list: List[int] = [1, 5, 10, 20, 50],
        verbose_every: int = 15,
    ) -> List[ObservationResult]:
        """
        Run a grid of simulations over all combinations of:
          planets × atmosphere templates × n_transit counts.
        """
        results = []
        total = len(planets) * len(templates) * len(n_transits_list)
        i = 0
        for planet in planets:
            for atm_name, tmpl in templates.items():
                for n_t in n_transits_list:
                    r = self.simulate(planet, tmpl, n_t)
                    results.append(r)
                    i += 1
                    if i % verbose_every == 0 or i == total:
                        print(
                            f"  [{i:4d}/{total}] {planet.planet_name:20s} × "
                            f"{atm_name:25s} × {n_t:3d} transits "
                            f"→ SNR = {r.detection_snr:.1f}σ"
                        )
        return results

    def detection_horizon(
        self,
        template: AtmosphereTemplate,
        distances_pc: np.ndarray,
        star_teff_k: float = 3000.0,
        star_radius_rs: float = 0.2,
        j_mag_at_10pc: float = 12.0,
        planet_radius_re: float = 1.0,
        transit_duration_hours: float = 1.5,
        n_transits: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute detection SNR as a function of stellar distance.
        """
        snrs = []
        for d in distances_pc:
            planet = PlanetSystem.synthetic(
                distance_pc=d,
                star_teff_k=star_teff_k,
                star_radius_rs=star_radius_rs,
                planet_radius_re=planet_radius_re,
                j_mag_at_10pc=j_mag_at_10pc,
            )
            planet.transit_duration_hours = transit_duration_hours
            r = self.simulate(planet, template, n_transits)
            snrs.append(r.detection_snr)
        return distances_pc, np.array(snrs)

    def n_transits_for_detection(
        self,
        planet: PlanetSystem,
        template: AtmosphereTemplate,
        target_snr: float = 5.0,
        n_trials: List[int] = [1, 2, 5, 10, 20, 50, 100, 200],
    ) -> int:
        """
        Find the minimum number of transits to reach target_snr.
        Returns -1 if not achievable within n_trials range.
        """
        for n_t in n_trials:
            r = self.simulate(planet, template, n_t)
            if r.detection_snr >= target_snr:
                return n_t
        return -1


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("OBSERVATION SIMULATOR — END-TO-END DEMO")
    print("=" * 60)

    jwst = load_jwst_nirspec()
    sim = ObservationSimulator(instrument=jwst, rng=np.random.default_rng(42))

    templates = get_default_templates(star_radius_rs=0.1192, planet_radius_re=0.92)
    planet = PlanetSystem.trappist1e()

    for atm_name, tmpl in templates.items():
        result = sim.simulate(planet, tmpl, n_transits=10, verbose=False)
        print(result.summary())

    print("\nDetection horizon (earth_like, 10 transits, M-dwarf host):")
    distances = np.array([5, 10, 15, 20, 30, 40, 50])
    _, snrs = sim.detection_horizon(templates["earth_like"], distances, n_transits=10)
    for d, s in zip(distances, snrs):
        bar = "█" * int(min(s / 2, 25))
        flag = " ← DETECTED" if s >= 5 else ""
        print(f"  {d:3.0f} pc : {s:6.1f}σ  {bar}{flag}")
