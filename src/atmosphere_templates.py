import numpy as np
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
 
 
# ---------------------------------------------------------------------------
# Instrument config dataclass
# ---------------------------------------------------------------------------
 
@dataclass
class InstrumentConfig:
    """
    All the hardware numbers for one instrument mode.
    Loaded from a JSON file in config/instruments/.
    """
    name: str
    wavelength_min_um: float        # Blue wavelength cutoff [μm]
    wavelength_max_um: float        # Red wavelength cutoff [μm]
    spectral_resolution: float      # R = λ/Δλ (resolving power)
    collecting_area_m2: float       # Effective mirror area [m²]
    throughput_peak: float          # Peak system efficiency (0–1): mirrors + detector QE
    read_noise_electrons: float     # Per-read detector read noise [e⁻]
    dark_current_e_per_s: float     # Thermal dark current [e⁻/s/pixel]
    n_pixels_per_resolution_element: int  # Pixels per spectral bin
    pixel_scale_arcsec: float       # Plate scale [arcsec/pixel]
    detector_gain: float            # Conversion factor [e⁻/ADU]
    saturation_electrons: float     # Full-well capacity [e⁻]
    stellar_noise_floor_ppm: float  # Systematic photometric stability floor [ppm]
    notes: str = ""
 
    @classmethod
    def from_json(cls, path: str) -> "InstrumentConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)
 
    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        d = {k: v for k, v in self.__dict__.items()}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"Instrument config saved → {path}")
 
 
# ---------------------------------------------------------------------------
# Main instrument model class
# ---------------------------------------------------------------------------
 
class InstrumentModel:
    """
    Computes photon rates, noise sources, and SNR for transit spectroscopy.
 
    The core calculation chain is:
        stellar flux → photon rate → noise model → SNR per bin
 
    Three noise regimes to be aware of:
        1. Shot-noise limited (bright stars): noise ∝ sqrt(N_photons)
        2. Read-noise limited (faint stars / short exposures): noise ∝ sqrt(N_reads)
        3. Systematic floor (any star): noise floor set by instrument stability (~20 ppm)
           This floor dominates for very bright targets with many transits.
    """
 
    def __init__(self, config: InstrumentConfig):
        self.config = config
 
    @classmethod
    def from_json(cls, path: str) -> "InstrumentModel":
        return cls(InstrumentConfig.from_json(path))
 
    # ------------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------------
 
    def throughput(self, wavelengths_um: np.ndarray) -> np.ndarray:
        """
        Wavelength-dependent system throughput (0–1).
 
        Accounts for: primary mirror reflectivity, grating/prism efficiency,
        detector quantum efficiency. Modeled as a smooth Gaussian envelope
        peaked at the center of the bandpass.
 
        Returns zeros outside the instrument's wavelength range.
        """
        wl_min = self.config.wavelength_min_um
        wl_max = self.config.wavelength_max_um
        wl_peak = (wl_min + wl_max) / 2.0
        wl_sigma = (wl_max - wl_min) / 3.0
 
        envelope = self.config.throughput_peak * np.exp(
            -0.5 * ((wavelengths_um - wl_peak) / wl_sigma) ** 2
        )
        in_range = (wavelengths_um >= wl_min) & (wavelengths_um <= wl_max)
        return np.where(in_range, np.clip(envelope, 0.0, 1.0), 0.0)
 
    def dlambda(self, wavelengths_um: np.ndarray) -> np.ndarray:
        """
        Spectral bin width at each wavelength: Δλ = λ / R  [μm].
        Each bin is one resolution element wide.
        """
        return wavelengths_um / self.config.spectral_resolution
 
    # ------------------------------------------------------------------
    # Stellar photon rate
    # ------------------------------------------------------------------
 
    def stellar_sed_calibration(self, wavelengths_um: np.ndarray) -> np.ndarray:
        """
        Wavelength-dependent calibration factor correcting the blackbody
        approximation toward real M-dwarf photospheric SEDs.
 
        WHY THIS EXISTS:
          A blackbody is a poor approximation to a real M-dwarf spectrum
          longward of ~2 um, where photospheric H2O and CO molecular bands
          suppress the emergent flux well below the blackbody prediction.
          Cross-validation against the JWST ETC (Pandeia v3.0) for TRAPPIST-1
          (Teff=2566K, J=11.35) shows our blackbody model agrees with ETC
          photon rates to ~5-10% at J-band (1.25 um) but overestimates by
          factors of ~1.5-2x at 2 um, ~3x at 3 um, and ~3-4x at 4.3 um.
          See notebooks/instrument_validation.ipynb for the full derivation.
 
        This function returns a smooth multiplicative correction, anchored
        at 1.0 at 1.25 um (where the blackbody is accurate) and decreasing
        toward longer wavelengths following the calibration factors derived
        from ETC cross-validation. It is intentionally conservative (i.e. it
        REDUCES claimed sensitivity), so applying it never overstates
        detectability.
 
        Calibration anchor points (wavelength_um: factor):
            1.25 : 1.00   (ETC agreement, no correction)
            2.00 : 0.70
            3.00 : 0.40
            4.30 : 0.30
            5.30 : 0.28   (held flat beyond 4.3 um)
 
        Returns
        -------
        calib : array of multiplicative factors, same shape as wavelengths_um,
                to be multiplied onto the raw blackbody photon rate.
        """
        anchor_wl  = np.array([0.6, 1.25, 2.0, 3.0, 4.3, 5.3])
        anchor_cal = np.array([1.00, 1.00, 0.70, 0.40, 0.30, 0.28])
        calib = np.interp(wavelengths_um, anchor_wl, anchor_cal)
        return calib
 
    def stellar_photon_rate(
        self,
        wavelengths_um: np.ndarray,
        star_magnitude_j: float,
        star_teff_k: float = 3500.0,
        apply_sed_calibration: bool = True,
    ) -> np.ndarray:
        """
        Stellar photon rate arriving at the detector [photons / s / spectral bin].
 
        Method:
          1. Compute a Planck blackbody SED for the star's Teff
          2. Anchor the absolute flux to the J-band apparent magnitude
          3. Multiply by spectral bin width, collecting area, and throughput
          4. (Optional, default ON) Apply the ETC-derived SED calibration
             correction from stellar_sed_calibration()
 
        Parameters
        ----------
        wavelengths_um    : wavelength array [μm]
        star_magnitude_j  : J-band (1.25 μm) apparent magnitude of the host star
        star_teff_k       : stellar effective temperature [K]
        apply_sed_calibration : if True (default), apply the wavelength-dependent
            calibration correction derived from JWST ETC cross-validation
            (see stellar_sed_calibration()). Set False only to reproduce the
            uncalibrated pre-Week-7 numbers for comparison purposes.
        """
        # --- Planck function B_λ (relative SED shape) ---
        h = 6.626e-34   # J·s
        c = 3.0e8       # m/s
        k = 1.381e-23   # J/K
        wl_m = wavelengths_um * 1e-6
 
        exponent = np.clip((h * c) / (wl_m * k * star_teff_k), 0.01, 700.0)
        B_lambda = wl_m ** -5 / (np.exp(exponent) - 1.0)
 
        # --- Anchor to J-band magnitude ---
        # J-band zero point: ~1600 Jy (Vega system)
        j_zp_jy = 1600.0
        F_j_jy = j_zp_jy * 10.0 ** (-star_magnitude_j / 2.5)  # [Jy]
 
        # Convert Jy → erg/s/cm²/Hz at 1.25 μm
        F_j_cgs = F_j_jy * 1e-23  # [erg/s/cm²/Hz]
 
        # Convert to per-μm using dν/dλ = c/λ² [c in μm/s]
        c_um_s = 3e14
        F_j_per_um = F_j_cgs * c_um_s / (1.25 ** 2)  # [erg/s/cm²/μm]
 
        # Scale full SED to this J-band flux
        j_idx = np.argmin(np.abs(wavelengths_um - 1.25))
        F_lambda = B_lambda / B_lambda[j_idx] * F_j_per_um  # [erg/s/cm²/μm]
 
        # --- Photon energy at each wavelength [erg] ---
        # E = hc/λ (in CGS: h=6.626e-27, c=3e10, λ in cm)
        E_photon_erg = (6.626e-27 * 3e10) / (wl_m * 100.0)
 
        # --- Photon rate per unit area per μm [photons/s/cm²/μm] ---
        photon_flux_per_um = F_lambda / E_photon_erg
 
        # --- Integrate over spectral bin width [photons/s/cm²/bin] ---
        photon_flux_per_bin = photon_flux_per_um * self.dlambda(wavelengths_um)
 
        # --- Scale to telescope collecting area ---
        area_cm2 = self.config.collecting_area_m2 * 1e4
        raw_rate = photon_flux_per_bin * area_cm2
 
        # --- Apply wavelength-dependent throughput ---
        rate = raw_rate * self.throughput(wavelengths_um)
 
        # --- Apply ETC-derived SED calibration correction ---
        if apply_sed_calibration:
            rate = rate * self.stellar_sed_calibration(wavelengths_um)
 
        return np.maximum(rate, 0.0)
 
 
    # ------------------------------------------------------------------
    # Noise model
    # ------------------------------------------------------------------
 
    def noise_model(
        self,
        photon_rate: np.ndarray,
        exposure_time_s: float,
        n_exposures: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        Full noise budget for a given observation.
 
        Noise sources included:
          - Shot noise: Poisson fluctuations in stellar photon counts
          - Read noise: detector amplifier noise per read
          - Dark current: thermally generated electrons
          - Sky background: zodiacal dust emission (~0.1% of stellar flux)
          - Stellar noise floor: instrument photometric stability limit (20 ppm)
            This is a systematic, not random, so it adds in quadrature separately.
 
        Parameters
        ----------
        photon_rate     : stellar photon rate [photons/s/bin], from stellar_photon_rate()
        exposure_time_s : duration of one individual exposure [s]
        n_exposures     : number of exposures to co-add
 
        Returns
        -------
        dict with keys:
          signal_e            : total collected signal [electrons]
          shot_noise          : √(signal) [e⁻]
          read_noise          : read noise contribution [e⁻]
          dark_noise          : dark current contribution [e⁻]
          sky_noise           : sky background contribution [e⁻]
          stellar_floor_e     : systematic noise floor [e⁻]
          total_random_noise  : quadrature sum of stochastic terms [e⁻]
          total_noise         : total noise including systematic floor [e⁻]
        """
        total_time_s = exposure_time_s * n_exposures
        n_pix = self.config.n_pixels_per_resolution_element
 
        # --- Signal ---
        signal_e = photon_rate * total_time_s
 
        # --- Shot noise: σ = √N ---
        shot_noise = np.sqrt(np.maximum(signal_e, 0.0))
 
        # --- Read noise: σ_read × √(n_exposures × n_pixels) ---
        read_rms = self.config.read_noise_electrons * np.sqrt(n_exposures * n_pix)
        read_noise = np.full_like(photon_rate, read_rms)
 
        # --- Dark current: σ = √(dark_rate × t × n_pixels) ---
        dark_e = self.config.dark_current_e_per_s * total_time_s * n_pix
        dark_noise = np.full_like(photon_rate, np.sqrt(dark_e))
 
        # --- Sky/zodiacal background: ~0.1% of stellar signal for bright targets ---
        sky_e = photon_rate * 0.001 * total_time_s
        sky_noise = np.sqrt(np.maximum(sky_e, 0.0))
 
        # --- Systematic stellar noise floor (non-random) ---
        # JWST NIRSpec has ~20 ppm stability floor from e.g. detector non-linearity,
        # thermal drifts, pointing jitter. Sets ultimate precision limit.
        floor_ppm = self.config.stellar_noise_floor_ppm
        stellar_floor_e = signal_e * floor_ppm * 1e-6
 
        # --- Total noise ---
        total_random = np.sqrt(
            shot_noise**2 + read_noise**2 + dark_noise**2 + sky_noise**2
        )
        total_noise = np.sqrt(total_random**2 + stellar_floor_e**2)
 
        return {
            "signal_e":           signal_e,
            "shot_noise":         shot_noise,
            "read_noise":         read_noise,
            "dark_noise":         dark_noise,
            "sky_noise":          sky_noise,
            "stellar_floor_e":    stellar_floor_e,
            "total_random_noise": total_random,
            "total_noise":        total_noise,
        }
 
    # ------------------------------------------------------------------
    # SNR calculator
    # ------------------------------------------------------------------
 
    def snr_per_bin(
        self,
        photon_rate: np.ndarray,
        transit_depth_ppm: np.ndarray,
        transit_duration_s: float,
        n_transits: int = 10,
        exposure_time_s: float = 88.0,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Per-wavelength-bin signal-to-noise ratio for detecting transit depth.
 
        SNR_bin = (depth_ppm × 1e-6 × signal_e) / total_noise_e
 
        Parameters
        ----------
        photon_rate       : from stellar_photon_rate() [photons/s/bin]
        transit_depth_ppm : wavelength-dependent transit depth [ppm]
        transit_duration_s: time spent in-transit per event [s]
        n_transits        : total number of transits observed
        exposure_time_s   : individual exposure length [s]
 
        Returns
        -------
        snr  : per-bin SNR array
        budget : full noise budget dict
        """
        total_in_transit_s = transit_duration_s * n_transits
        n_exp = max(1, int(total_in_transit_s / exposure_time_s))
 
        budget = self.noise_model(photon_rate, exposure_time_s, n_exp)
 
        # Transit signal in electrons: depth × total stellar counts
        transit_signal_e = (transit_depth_ppm * 1e-6) * budget["signal_e"]
 
        snr = np.where(
            budget["total_noise"] > 0,
            transit_signal_e / budget["total_noise"],
            0.0,
        )
        return snr, budget
 
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
 
    def summary(self) -> str:
        c = self.config
        return (
            f"{'─'*50}\n"
            f"Instrument : {c.name}\n"
            f"Wavelengths: {c.wavelength_min_um:.2f} – {c.wavelength_max_um:.2f} μm\n"
            f"Resolution : R = {c.spectral_resolution:.0f}  "
            f"(Δλ ≈ {1.25/c.spectral_resolution*1000:.1f} nm at 1.25 μm)\n"
            f"Aperture   : {c.collecting_area_m2:.1f} m²  "
            f"(D ≈ {(4*c.collecting_area_m2/np.pi)**0.5:.1f} m effective)\n"
            f"Throughput : {c.throughput_peak*100:.0f}% peak\n"
            f"Read noise : {c.read_noise_electrons:.1f} e⁻/read\n"
            f"Dark current: {c.dark_current_e_per_s:.3f} e⁻/s/pixel\n"
            f"Noise floor: {c.stellar_noise_floor_ppm:.0f} ppm (systematic)\n"
            f"Notes      : {c.notes}\n"
            f"{'─'*50}"
        )
 
 
# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------
 
def load_jwst_nirspec() -> InstrumentModel:
    """
    Load JWST NIRSpec Prism mode configuration.
    Falls back to hardcoded defaults if config file not found.
    """
    path = "config/instruments/jwst_nirspec.json"
    if os.path.exists(path):
        return InstrumentModel.from_json(path)
    # Hardcoded defaults matching config/instruments/jwst_nirspec.json
    return InstrumentModel(InstrumentConfig(
        name="JWST NIRSpec Prism",
        wavelength_min_um=0.60,
        wavelength_max_um=5.30,
        spectral_resolution=100.0,
        collecting_area_m2=25.4,
        throughput_peak=0.30,
        read_noise_electrons=15.0,
        dark_current_e_per_s=0.01,
        n_pixels_per_resolution_element=2,
        pixel_scale_arcsec=0.10,
        detector_gain=1.0,
        saturation_electrons=65000.0,
        stellar_noise_floor_ppm=20.0,
        notes="NIRSpec PRISM/CLEAR, R~100, 0.6-5.3 μm. See JWST ETC for precision modeling.",
    ))
 
 
def load_elt_harmoni() -> InstrumentModel:
    """
    Load a hypothetical ELT HARMONI-like configuration (concept only).
    The 39-meter ELT will enable higher-resolution studies of nearby M-dwarf systems.
    """
    path = "config/instruments/elt_example.json"
    if os.path.exists(path):
        return InstrumentModel.from_json(path)
    return InstrumentModel(InstrumentConfig(
        name="ELT HARMONI (Concept)",
        wavelength_min_um=0.47,
        wavelength_max_um=2.45,
        spectral_resolution=3500.0,
        collecting_area_m2=978.0,
        throughput_peak=0.18,
        read_noise_electrons=3.0,
        dark_current_e_per_s=0.001,
        n_pixels_per_resolution_element=3,
        pixel_scale_arcsec=0.004,
        detector_gain=1.0,
        saturation_electrons=100000.0,
        stellar_noise_floor_ppm=10.0,
        notes="Conceptual ELT HARMONI-like config. 39m primary, R~3500. Not yet validated.",
    ))
 
 
def load_miri_lrs() -> InstrumentModel:
    """
    Load JWST MIRI Low Resolution Spectrometer (LRS) configuration.
 
    Why MIRI matters for biosignatures:
      MIRI LRS covers 5–14 μm — the mid-infrared regime where several
      key biosignature and discriminator features live that NIRSpec cannot reach:
        - CO2 bending mode at 15 μm (just beyond LRS; MIRI photometry can probe it)
        - O3 at 9.6 μm (ozone; strongest mid-IR biosignature)
        - N2O at 7.8 μm (nitrous oxide; produced by denitrifying bacteria)
        - SO2 at 7.3 μm (volcanic; false-positive discriminator)
        - H2O at 6.3 μm (another water detection window)
 
    MIRI is ~100× less sensitive than NIRSpec per photon due to higher
    thermal background in the mid-IR, so it requires many more transits.
    Treat this as a stub for multi-instrument studies in Weeks 5–6.
 
    Ref: Kendrew et al. (2015), PASP 127, 623; Wright et al. (2023), PASP 135, 048003.
    """
    path = "config/instruments/miri_lrs.json"
    if os.path.exists(path):
        return InstrumentModel.from_json(path)
    return InstrumentModel(InstrumentConfig(
        name="JWST MIRI LRS",
        wavelength_min_um=5.00,
        wavelength_max_um=14.00,
        spectral_resolution=100.0,       # R~100 for LRS slitless mode
        collecting_area_m2=25.4,
        throughput_peak=0.12,            # Lower than NIRSpec; thermal background dominates
        read_noise_electrons=30.0,       # Si:As IBC detector; higher read noise
        dark_current_e_per_s=10.0,       # Thermal background: ~10 e⁻/s/pixel at 5-14 μm
        n_pixels_per_resolution_element=2,
        pixel_scale_arcsec=0.11,
        detector_gain=5.5,
        saturation_electrons=250000.0,
        stellar_noise_floor_ppm=50.0,    # Higher floor in mid-IR due to thermal background
        notes=(
            "MIRI LRS slitless mode. R~100, 5-14 um. Si:As IBC detector. "
            "Thermal background dominates over shot noise for most targets. "
            "Ref: Kendrew+2015; Wright+2023. Use JWST ETC for precise noise budget."
        ),
    ))
 
 
def save_miri_lrs_config() -> None:
    """Write the MIRI LRS config JSON to disk."""
    miri = load_miri_lrs()
    miri.config.to_json("config/instruments/miri_lrs.json")
 
 
# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
 
    jwst = load_jwst_nirspec()
    print(jwst.summary())
 
    wl = np.logspace(np.log10(0.6), np.log10(5.3), 300)
 
    # Example: TRAPPIST-1 host star
    rate = jwst.stellar_photon_rate(wl, star_magnitude_j=11.35, star_teff_k=2566)
    print(f"\nTRAPPIST-1 photon rate:")
    print(f"  Peak : {rate.max():.2e} photons/s/bin at {wl[np.argmax(rate)]:.2f} μm")
    print(f"  Median (in-band): {np.median(rate[rate > 0]):.2e} photons/s/bin")
 
    budget = jwst.noise_model(rate, exposure_time_s=88.0, n_exposures=500)
    sig = budget["signal_e"]
    noise = budget["total_noise"]
    noise_ppm = np.where(sig > 0, noise / sig * 1e6, 1e6)
    print(f"\nNoise at 1.4 μm H2O band:")
    idx = np.argmin(np.abs(wl - 1.38))
    print(f"  Signal : {sig[idx]:.2e} e⁻")
    print(f"  Noise  : {noise[idx]:.2e} e⁻  ({noise_ppm[idx]:.0f} ppm)")
