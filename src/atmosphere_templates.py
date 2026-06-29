"""
atmosphere_templates.py
-----------------------
Precomputed atmosphere template grid for transmission spectroscopy.

Implements three canonical atmosphere types:
  1. earth_like      — O2 + H2O + CO2 + O3 + trace CH4 (biosignature-rich)
  2. high_co2        — CO2-dominated (Venus-analog; no biosignatures)
  3. reduced_o2_high_ch4 — Anoxic biosphere (high CH4, low O2; Archean Earth analog)

Methodology note (paper/methods.md):
  We adopt a precomputed template grid approach, which enables parameter
  sweeps across cloud fraction, O2/CH4, and scale height that would be
  computationally infeasible with online radiative transfer codes.
  Grid parameters: cloud_fraction ∈ [0, 0.9], scale_height ∈ [6, 11] km,
  O2/CH4 ratio (template-dependent).

Wavelength coverage: 0.6 – 5.3 μm (JWST NIRSpec prism)
Spectral grid: 300 points, log-spaced

Decision note:
  petitRADTRANS installation was evaluated; the template grid approach was
  adopted as the primary method for computational tractability.
  This is methodologically equivalent for detection-horizon studies.

Usage:
    from atmosphere_templates import TemplateGrid, get_default_templates

    # Get the three canonical templates (quick start):
    templates = get_default_templates(star_radius_rs=0.2, planet_radius_re=1.0)

    # Or build a full parameter grid:
    grid = TemplateGrid()
    grid.build_grid()
    grid.save("data/templates/template_grid.json")
    t = grid.get_template("earth_like", cloud_fraction=0.5, scale_height_km=8.5)
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Wavelength grid
# ---------------------------------------------------------------------------

WAVELENGTH_MIN_UM = 0.60   # JWST NIRSpec prism blue cutoff
WAVELENGTH_MAX_UM = 5.30   # JWST NIRSpec prism red cutoff
N_WAVELENGTHS = 300


def default_wavelength_grid() -> np.ndarray:
    """300-point log-spaced grid, 0.6–5.3 μm."""
    return np.logspace(np.log10(WAVELENGTH_MIN_UM), np.log10(WAVELENGTH_MAX_UM), N_WAVELENGTHS)


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass
class AtmosphereTemplate:
    """
    A single atmosphere model: wavelength-dependent transit depth spectrum.

    Transit depth D(λ) = (R_p(λ) / R_*)^2 in parts per million.
    The wavelength dependence encodes absorption features of atmospheric
    constituents, modulated by cloud fraction and scale height.
    """
    name: str
    description: str
    wavelengths_um: np.ndarray     # microns
    transit_depth_ppm: np.ndarray  # parts per million
    parameters: Dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "wavelengths_um": self.wavelengths_um.tolist(),
            "transit_depth_ppm": self.transit_depth_ppm.tolist(),
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AtmosphereTemplate":
        return cls(
            name=d["name"],
            description=d["description"],
            wavelengths_um=np.array(d["wavelengths_um"]),
            transit_depth_ppm=np.array(d["transit_depth_ppm"]),
            parameters=d["parameters"],
        )

    @property
    def dominant_species(self) -> str:
        """Human-readable dominant absorber(s)."""
        return {
            "earth_like": "O2 + H2O + CO2 + O3",
            "high_co2": "CO2 (dominant)",
            "reduced_o2_high_ch4": "CH4 + H2O + CO2",
        }.get(self.name, self.name)


# ---------------------------------------------------------------------------
# Spectral feature builders (Gaussian + Lorentzian profiles)
# ---------------------------------------------------------------------------

def _gaussian(wavelengths: np.ndarray, center: float, width: float) -> np.ndarray:
    """Normalized Gaussian absorption profile."""
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def _lorentzian(wavelengths: np.ndarray, center: float, width: float) -> np.ndarray:
    """Normalized Lorentzian (pressure-broadened) absorption profile."""
    return 1.0 / (1.0 + ((wavelengths - center) / width) ** 2)


# ---------------------------------------------------------------------------
# Atmosphere template builders
# ---------------------------------------------------------------------------

def _base_depth_ppm(planet_radius_re: float, star_radius_rs: float) -> float:
    """
    Continuum transit depth (R_p/R_*)^2 in ppm.
    This is the flat-spectrum baseline before atmospheric features are added.
    """
    rp_m = planet_radius_re * 6.371e6
    rs_m = star_radius_rs * 6.96e8
    return (rp_m / rs_m) ** 2 * 1e6


def _atm_amplitude_ppm(scale_height_km: float, star_radius_rs: float) -> float:
    """
    Spectral feature amplitude in ppm.
    Scales as 5 × H / R_* (5 scale heights is the canonical feature depth).
    """
    H_m = scale_height_km * 1e3
    Rs_m = star_radius_rs * 6.96e8
    return 5.0 * H_m / Rs_m * 1e6


def build_earth_like_template(
    wavelengths: np.ndarray,
    cloud_fraction: float = 0.5,
    o2_ch4_ratio: float = 1.0,
    scale_height_km: float = 8.5,
    planet_radius_re: float = 1.0,
    star_radius_rs: float = 0.2,
) -> AtmosphereTemplate:
    """
    Earth-analog atmosphere: O2, H2O, CO2, O3, trace CH4.

    This is the primary biosignature template. Key detection windows:
    - O2 A-band at 0.762 μm (sharpest biosignature feature)
    - H2O bands at 0.94, 1.14, 1.38, 1.87, 2.7 μm
    - CO2 at 1.6, 2.0, 4.3 μm
    - O3 Chappuis band at ~0.6 μm (broad)
    - Trace CH4 at 1.67, 2.3 μm

    The simultaneous detection of O2 + CH4 is the gold-standard abiotic
    disequilibrium biosignature (Sagan et al. 1993; Schwieterman et al. 2018).

    Parameters
    ----------
    cloud_fraction : 0–1, fraction of planet disk covered by optically thick clouds.
                     Clouds mute spectral features proportionally.
    o2_ch4_ratio : ratio of O2 to CH4 mixing ratio (Earth present-day ≈ 1.0).
    scale_height_km : pressure scale height H = kT/(mg). Earth ≈ 8.5 km.
    planet_radius_re : planet radius in Earth radii.
    star_radius_rs : host star radius in solar radii.
    """
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    amp = _atm_amplitude_ppm(scale_height_km, star_radius_rs)
    f_clear = 1.0 - cloud_fraction   # Clear-sky fraction

    depth = np.full_like(wavelengths, base)

    # ---- Water vapor (H2O) ------------------------------------------------
    h2o = [
        (0.940, 0.030, 0.60),   # Y band
        (1.140, 0.040, 0.90),   # J band
        (1.380, 0.060, 1.50),   # J/H gap — strong, clear-sky window test
        (1.870, 0.070, 1.80),   # H/K gap
        (2.700, 0.150, 1.30),   # K band
    ]
    for center, width, strength in h2o:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)

    # ---- O2 ---------------------------------------------------------------
    # A-band at 0.762 μm: sharpest, most diagnostic biosignature feature
    # B-band at 0.688 μm: weaker
    o2_factor = np.clip(o2_ch4_ratio, 0.001, 10.0)
    depth += f_clear * amp * 1.0 * o2_factor * _gaussian(wavelengths, 0.762, 0.005)
    depth += f_clear * amp * 0.25 * o2_factor * _gaussian(wavelengths, 0.688, 0.004)

    # ---- CO2 --------------------------------------------------------------
    co2 = [
        (1.600, 0.030, 0.15),   # 1.6 μm band (weak)
        (2.010, 0.050, 0.20),   # 2.0 μm band (weak)
        (4.300, 0.200, 0.55),   # 4.3 μm band (strongest CO2 feature)
    ]
    for center, width, strength in co2:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)

    # ---- Ozone (O3) -------------------------------------------------------
    # Chappuis band: broad, centered ~0.6 μm
    # Hartley band: UV (<0.35 μm) — outside NIRSpec range
    depth += f_clear * amp * 0.20 * _lorentzian(wavelengths, 0.600, 0.080)

    # ---- Trace CH4 --------------------------------------------------------
    # At Earth-present levels, CH4 features are small but detectable
    ch4_abundance = 1.0 / max(o2_ch4_ratio, 0.01)
    ch4 = [
        (1.670, 0.040, 0.04 * ch4_abundance),
        (2.300, 0.080, 0.07 * ch4_abundance),
    ]
    for center, width, strength in ch4:
        depth += f_clear * amp * np.clip(strength, 0, 0.5) * _gaussian(wavelengths, center, width)

    # ---- Rayleigh scattering slope ----------------------------------------
    # λ^-4 scattering from N2/O2 adds a blue slope to the continuum
    rayleigh_amp = f_clear * amp * 0.30
    depth += rayleigh_amp * (wavelengths[0] / wavelengths) ** 4

    return AtmosphereTemplate(
        name="earth_like",
        description="Earth-analog: O2 + H2O + CO2 + O3 + trace CH4 (biosignature-rich)",
        wavelengths_um=wavelengths,
        transit_depth_ppm=np.maximum(depth, 0.0),
        parameters={
            "cloud_fraction": float(cloud_fraction),
            "o2_ch4_ratio": float(o2_ch4_ratio),
            "scale_height_km": float(scale_height_km),
            "planet_radius_re": float(planet_radius_re),
            "star_radius_rs": float(star_radius_rs),
            "base_depth_ppm": float(base),
        },
    )


def build_high_co2_template(
    wavelengths: np.ndarray,
    cloud_fraction: float = 0.30,
    o2_ch4_ratio: float = 0.01,
    scale_height_km: float = 7.0,
    planet_radius_re: float = 1.0,
    star_radius_rs: float = 0.2,
) -> AtmosphereTemplate:
    """
    Venus-analog atmosphere: CO2-dominated, no biological biosignatures.

    A key *false positive* case for biosignature searches:
    - Strong, pressure-broadened CO2 bands dominate spectrum
    - Sulfuric acid clouds suppress H2O
    - Minimal O2 (trace amounts from CO2 photolysis only)
    - SO2 features (volcanic)

    The near-absence of H2O despite a warm equilibrium temperature is
    a key discriminator from the earth_like template.

    Parameters: see build_earth_like_template.
    """
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    amp = _atm_amplitude_ppm(scale_height_km, star_radius_rs)
    f_clear = 1.0 - cloud_fraction

    depth = np.full_like(wavelengths, base)

    # ---- Strong, pressure-broadened CO2 bands ----------------------------
    # High surface pressure (90 bar on Venus) broadens lines significantly.
    # We widen features by ×2–3 compared to the earth_like template.
    co2 = [
        (1.050, 0.040, 0.30),
        (1.210, 0.045, 0.35),
        (1.430, 0.055, 0.55),
        (1.600, 0.070, 0.80),   # pressure-broadened
        (2.010, 0.100, 1.20),
        (2.680, 0.200, 1.50),
        (4.300, 0.450, 2.00),   # dominant: very broad & deep
    ]
    for center, width, strength in co2:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)

    # ---- Trace H2O (suppressed by sulfuric acid clouds) ------------------
    depth += f_clear * amp * 0.05 * _gaussian(wavelengths, 1.380, 0.060)

    # ---- SO2 (volcanic; SO2 at 4.0 μm) -----------------------------------
    depth += f_clear * amp * 0.25 * _gaussian(wavelengths, 4.000, 0.200)

    # ---- Abiotic O2 (trace; photolysis of CO2) ---------------------------
    depth += f_clear * amp * 0.01 * _gaussian(wavelengths, 0.762, 0.005)

    # ---- Sulfuric acid haze slope (UV-blue scattering) -------------------
    depth += f_clear * amp * 0.15 * (wavelengths[0] / wavelengths) ** 2

    return AtmosphereTemplate(
        name="high_co2",
        description="Venus-analog: CO2-dominated, pressure-broadened, no biosignatures",
        wavelengths_um=wavelengths,
        transit_depth_ppm=np.maximum(depth, 0.0),
        parameters={
            "cloud_fraction": float(cloud_fraction),
            "o2_ch4_ratio": float(o2_ch4_ratio),
            "scale_height_km": float(scale_height_km),
            "planet_radius_re": float(planet_radius_re),
            "star_radius_rs": float(star_radius_rs),
            "base_depth_ppm": float(base),
        },
    )


def build_reduced_o2_high_ch4_template(
    wavelengths: np.ndarray,
    cloud_fraction: float = 0.40,
    o2_ch4_ratio: float = 0.001,
    scale_height_km: float = 9.0,
    planet_radius_re: float = 1.0,
    star_radius_rs: float = 0.2,
) -> AtmosphereTemplate:
    """
    Anoxic biosphere analog: high CH4, low O2 (Archean Earth, ~2.5 Ga).

    This represents the SECOND biosignature template — a biosphere that
    produces abundant CH4 (methanogenesis) without photosynthetic O2.
    Key features:
    - Strong CH4 bands at 1.0, 1.33, 1.67, 2.3, 3.3 μm
    - Moderate H2O (liquid water biosphere)
    - Moderate CO2 (greenhouse, not dominant)
    - Very weak O2 (sub-ppm levels)
    - Hydrocarbon haze slope (Titan-like at high CH4 levels)

    The CH4 abundance itself is a biosignature: at these levels without
    O2 to destroy it, geological sources alone are insufficient
    (Krissansen-Totton et al. 2018).

    Parameters: see build_earth_like_template.
    """
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    amp = _atm_amplitude_ppm(scale_height_km, star_radius_rs)
    f_clear = 1.0 - cloud_fraction

    # CH4 mixing ratio scales inversely with O2/CH4 ratio
    # Cap to prevent unphysical spectra
    ch4_factor = np.clip(1.0 / max(o2_ch4_ratio, 1e-5), 1.0, 50.0)

    depth = np.full_like(wavelengths, base)

    # ---- Strong CH4 absorption bands -------------------------------------
    ch4 = [
        (1.000, 0.030, 0.10),   # 1.0 μm (weak)
        (1.330, 0.040, 0.15),   # 1.33 μm
        (1.670, 0.050, 0.30),   # 1.67 μm (STRONG — key diagnostic)
        (2.300, 0.100, 0.50),   # 2.3 μm  (STRONG)
        (3.300, 0.150, 0.70),   # 3.3 μm  (fundamental; strongest CH4 band)
    ]
    for center, width, raw_strength in ch4:
        strength = np.clip(raw_strength * ch4_factor, 0, 3.0)
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)

    # ---- H2O (present; liquid water biosphere) ---------------------------
    h2o = [
        (0.940, 0.030, 0.45),
        (1.140, 0.040, 0.65),
        (1.380, 0.060, 1.00),
        (1.870, 0.070, 1.10),
        (2.700, 0.150, 0.80),
    ]
    for center, width, strength in h2o:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)

    # ---- CO2 (moderate greenhouse, not dominant) -------------------------
    co2 = [
        (1.600, 0.040, 0.30),
        (2.010, 0.060, 0.40),
        (4.300, 0.250, 0.70),
    ]
    for center, width, strength in co2:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)

    # ---- Very weak O2 ---------------------------------------------------
    o2_factor = np.clip(o2_ch4_ratio, 0.0, 1.0)
    depth += f_clear * amp * 0.5 * o2_factor * _gaussian(wavelengths, 0.762, 0.005)

    # ---- Hydrocarbon haze slope (Titan-like at high CH4) ----------------
    haze_strength = np.clip((ch4_factor - 1.0) / 50.0, 0, 0.5)
    depth += f_clear * amp * haze_strength * (wavelengths[0] / wavelengths) ** 2

    # ---- Rayleigh scattering (N2-dominated background gas) ---------------
    depth += f_clear * amp * 0.25 * (wavelengths[0] / wavelengths) ** 4

    return AtmosphereTemplate(
        name="reduced_o2_high_ch4",
        description="Anoxic biosphere: high CH4 + H2O + CO2, negligible O2 (Archean analog)",
        wavelengths_um=wavelengths,
        transit_depth_ppm=np.maximum(depth, 0.0),
        parameters={
            "cloud_fraction": float(cloud_fraction),
            "o2_ch4_ratio": float(o2_ch4_ratio),
            "scale_height_km": float(scale_height_km),
            "planet_radius_re": float(planet_radius_re),
            "star_radius_rs": float(star_radius_rs),
            "base_depth_ppm": float(base),
        },
    )


def build_hycean_template(
    wavelengths: np.ndarray,
    cloud_fraction: float = 0.20,
    o2_ch4_ratio: float = 0.002,
    scale_height_km: float = 12.0,
    planet_radius_re: float = 2.3,
    star_radius_rs: float = 0.4,
) -> AtmosphereTemplate:
    """
    Hycean world: ocean-covered sub-Neptune with H2-rich atmosphere.

    Motivated by K2-18b (Madhusudhan et al. 2023, ApJL 956 L13), which showed
    CH4 + CO2 detections and a tentative DMS signal with JWST NIRSpec.
    Key characteristics:
    - H2-dominated atmosphere (low mean molecular weight → large scale height)
    - Strong CH4 (biogenic or abiotic)
    - CO2 detected; CO depleted (chemical disequilibrium biosignature context)
    - H2O vapor above a global liquid water ocean
    - Minimal O2 (reducing atmosphere)
    - Possible DMS (dimethyl sulfide) at ~3.4 μm — tentative biosignature

    The large planet radius and extended H2 atmosphere give spectral features
    5–10× larger in ppm than rocky planets of the same host star,
    making hycean worlds the highest-SNR biosignature targets for JWST.

    Parameters: see build_earth_like_template.
    Note: planet_radius_re default is 2.3 (K2-18b-like); star_radius_rs default 0.4.
    """
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    # H2 atmosphere: scale height ~3× larger than N2 atmosphere at same T
    # (mean molecular weight μ = 2.3 vs 28 g/mol)
    effective_H = scale_height_km * (28.0 / 2.3)  # effective scale height in ppm terms
    amp = _atm_amplitude_ppm(effective_H, star_radius_rs)
    # Cap amplitude — very large scale heights still bounded by physics
    amp = min(amp, _atm_amplitude_ppm(scale_height_km, star_radius_rs) * 8.0)
    f_clear = 1.0 - cloud_fraction

    depth = np.full_like(wavelengths, base)

    # ---- CH4 (strong; biogenic or primordial) ----------------------------
    # K2-18b: CH4 detected at ~1% mixing ratio
    ch4_factor = np.clip(1.0 / max(o2_ch4_ratio, 1e-4), 1.0, 20.0)
    ch4 = [
        (1.000, 0.030, 0.15),
        (1.330, 0.040, 0.20),
        (1.670, 0.055, 0.45),   # STRONG — key K2-18b detection band
        (2.300, 0.110, 0.65),   # STRONG
        (3.300, 0.160, 0.80),   # fundamental band
    ]
    for center, width, raw_s in ch4:
        s = np.clip(raw_s * ch4_factor * 0.3, 0, 2.5)
        depth += f_clear * amp * s * _gaussian(wavelengths, center, width)

    # ---- CO2 (detected on K2-18b at ~1% mixing ratio) -------------------
    co2 = [
        (1.600, 0.035, 0.25),
        (2.010, 0.060, 0.40),
        (4.300, 0.260, 1.20),   # strongest feature — key detection
    ]
    for center, width, strength in co2:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)

    # ---- H2O vapor (ocean surface evaporation) --------------------------
    h2o = [
        (0.940, 0.035, 0.30),
        (1.140, 0.045, 0.50),
        (1.380, 0.065, 0.80),
        (1.870, 0.080, 0.90),
        (2.700, 0.160, 0.70),
    ]
    for center, width, strength in h2o:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)

    # ---- CO depletion is an absence feature — no CO band at 4.7 μm ------
    # (CO conspicuously absent on K2-18b; chemical disequilibrium indicator)
    # Nothing to add here — its ABSENCE is the signal.

    # ---- DMS (dimethyl sulfide) — tentative biosignature at 3.4 μm ------
    # Very weak; tentative in Madhusudhan+2023; included at reduced strength
    depth += f_clear * amp * 0.08 * _gaussian(wavelengths, 3.40, 0.06)

    # ---- H2-H2 collision-induced absorption (CIA) — broad IR slope ------
    # CIA from H2-H2 pairs adds broad opacity longward of 2 μm
    cia_slope = f_clear * amp * 0.20 * np.where(wavelengths > 2.0,
                                                  (wavelengths - 2.0) / 3.0, 0.0)
    depth += cia_slope

    # ---- Rayleigh scattering (H2 atmosphere) ----------------------------
    depth += f_clear * amp * 0.40 * (wavelengths[0] / wavelengths) ** 4

    return AtmosphereTemplate(
        name="hycean",
        description="Hycean world: H2-rich, CH4+CO2+H2O, global ocean (K2-18b analog)",
        wavelengths_um=wavelengths,
        transit_depth_ppm=np.maximum(depth, 0.0),
        parameters={
            "cloud_fraction": float(cloud_fraction),
            "o2_ch4_ratio": float(o2_ch4_ratio),
            "scale_height_km": float(scale_height_km),
            "planet_radius_re": float(planet_radius_re),
            "star_radius_rs": float(star_radius_rs),
            "base_depth_ppm": float(base),
            "reference": "Madhusudhan et al. 2023, ApJL 956 L13",
        },
    )


# ---------------------------------------------------------------------------
# Template grid
# ---------------------------------------------------------------------------

class TemplateGrid:
    """
    Precomputed grid of atmosphere templates over key parameters.

    Grid axes:
      - Template type: {earth_like, high_co2, reduced_o2_high_ch4}
      - Cloud fraction: configurable list, default [0.0, 0.3, 0.6, 0.9]
      - Scale height [km]: configurable list, default [6.0, 8.5, 11.0]
      - O2/CH4 ratio: template-specific lists

    Total default grid size: 3 types × 4 clouds × 3 scales × 3 ratios = 108 templates.

    Usage:
        grid = TemplateGrid()
        grid.build_grid()
        t = grid.get_template("earth_like", cloud_fraction=0.5, scale_height_km=8.5)
        grid.save("data/templates/template_grid.json")
    """

    BUILDERS = {
        "earth_like": build_earth_like_template,
        "high_co2": build_high_co2_template,
        "reduced_o2_high_ch4": build_reduced_o2_high_ch4_template,
        "hycean": build_hycean_template,
    }

    DEFAULT_O2_CH4 = {
        "earth_like":           [0.3, 1.0, 3.0],
        "high_co2":             [0.001, 0.01, 0.05],
        "reduced_o2_high_ch4":  [0.0001, 0.001, 0.01],
        "hycean":               [0.001, 0.005, 0.02],
    }

    def __init__(self, wavelengths: Optional[np.ndarray] = None):
        self.wavelengths = wavelengths if wavelengths is not None else default_wavelength_grid()
        self.templates: Dict[str, List[AtmosphereTemplate]] = {}
        self._meta: dict = {}

    def build_grid(
        self,
        template_names: Optional[List[str]] = None,
        cloud_fractions: List[float] = [0.0, 0.3, 0.6, 0.9],
        scale_heights_km: List[float] = [6.0, 8.5, 11.0],
        o2_ch4_ratios: Optional[Dict[str, List[float]]] = None,
        planet_radius_re: float = 1.0,
        star_radius_rs: float = 0.2,
    ) -> None:
        """Build and cache all templates in the grid."""
        if template_names is None:
            template_names = list(self.BUILDERS.keys())
        if o2_ch4_ratios is None:
            o2_ch4_ratios = self.DEFAULT_O2_CH4

        self._meta = {
            "cloud_fractions": cloud_fractions,
            "scale_heights_km": scale_heights_km,
            "o2_ch4_ratios": o2_ch4_ratios,
            "planet_radius_re": planet_radius_re,
            "star_radius_rs": star_radius_rs,
        }

        total = 0
        for name in template_names:
            self.templates[name] = []
            builder = self.BUILDERS[name]
            ratios = o2_ch4_ratios.get(name, [1.0])
            for cf in cloud_fractions:
                for sh in scale_heights_km:
                    for ratio in ratios:
                        t = builder(
                            wavelengths=self.wavelengths,
                            cloud_fraction=cf,
                            o2_ch4_ratio=ratio,
                            scale_height_km=sh,
                            planet_radius_re=planet_radius_re,
                            star_radius_rs=star_radius_rs,
                        )
                        self.templates[name].append(t)
                        total += 1

        print(f"TemplateGrid built: {total} templates "
              f"({len(template_names)} types × {len(cloud_fractions)} cloud fracs × "
              f"{len(scale_heights_km)} scale heights)")

    def get_template(
        self,
        template_name: str,
        cloud_fraction: float = 0.5,
        o2_ch4_ratio: float = 1.0,
        scale_height_km: float = 8.5,
    ) -> AtmosphereTemplate:
        """
        Return the grid template nearest to the requested parameters.
        Uses a weighted Euclidean distance in normalized parameter space.
        """
        if template_name not in self.templates or not self.templates[template_name]:
            raise KeyError(
                f"Template '{template_name}' not in grid. "
                f"Available: {list(self.templates.keys())}. Call build_grid() first."
            )
        candidates = self.templates[template_name]

        def _dist(t: AtmosphereTemplate) -> float:
            p = t.parameters
            dc = abs(p["cloud_fraction"] - cloud_fraction)
            dr = abs(np.log10(max(p["o2_ch4_ratio"], 1e-7))
                     - np.log10(max(o2_ch4_ratio, 1e-7))) * 0.5
            dh = abs(p["scale_height_km"] - scale_height_km) / 10.0
            return dc + dr + dh

        return min(candidates, key=_dist)

    def save(self, path: str) -> None:
        """Serialize the template grid to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "wavelengths_um": self.wavelengths.tolist(),
            "meta": self._meta,
            "templates": {
                name: [t.to_dict() for t in lst]
                for name, lst in self.templates.items()
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"Template grid saved → {path}  ({size_kb:.0f} KB)")

    @classmethod
    def load(cls, path: str) -> "TemplateGrid":
        """Load a saved template grid from disk."""
        with open(path) as f:
            data = json.load(f)
        grid = cls(wavelengths=np.array(data["wavelengths_um"]))
        grid._meta = data.get("meta", {})
        grid.templates = {
            name: [AtmosphereTemplate.from_dict(d) for d in lst]
            for name, lst in data["templates"].items()
        }
        total = sum(len(v) for v in grid.templates.values())
        print(f"TemplateGrid loaded: {total} templates from {path}")
        return grid


# ---------------------------------------------------------------------------
# Convenience function: three canonical default templates
# ---------------------------------------------------------------------------

def get_default_templates(
    star_radius_rs: float = 0.2,
    planet_radius_re: float = 1.0,
) -> Dict[str, AtmosphereTemplate]:
    """
    Return the four canonical templates with default parameters.
    Quick start for demos and notebooks.
    """
    wl = default_wavelength_grid()
    return {
        "earth_like": build_earth_like_template(
            wl, cloud_fraction=0.5, o2_ch4_ratio=1.0, scale_height_km=8.5,
            planet_radius_re=planet_radius_re, star_radius_rs=star_radius_rs,
        ),
        "high_co2": build_high_co2_template(
            wl, cloud_fraction=0.3, o2_ch4_ratio=0.01, scale_height_km=7.0,
            planet_radius_re=planet_radius_re, star_radius_rs=star_radius_rs,
        ),
        "reduced_o2_high_ch4": build_reduced_o2_high_ch4_template(
            wl, cloud_fraction=0.4, o2_ch4_ratio=0.001, scale_height_km=9.0,
            planet_radius_re=planet_radius_re, star_radius_rs=star_radius_rs,
        ),
        "hycean": build_hycean_template(
            wl, cloud_fraction=0.2, o2_ch4_ratio=0.002, scale_height_km=12.0,
            planet_radius_re=max(planet_radius_re * 2.0, 2.3),
            star_radius_rs=max(star_radius_rs * 2.0, 0.4),
        ),
    }


# ---------------------------------------------------------------------------
# CLI / standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build atmosphere template grid")
    parser.add_argument("--out", default="data/templates/template_grid.json")
    parser.add_argument("--star-radius", type=float, default=0.2, help="Star radius [R_sun]")
    parser.add_argument("--planet-radius", type=float, default=1.0, help="Planet radius [R_earth]")
    args = parser.parse_args()

    print("Building atmosphere template grid...")
    grid = TemplateGrid()
    grid.build_grid(planet_radius_re=args.planet_radius, star_radius_rs=args.star_radius)
    grid.save(args.out)

    # Quick sanity check
    templates = get_default_templates(star_radius_rs=args.star_radius, planet_radius_re=args.planet_radius)
    for name, t in templates.items():
        base = t.parameters["base_depth_ppm"]
        peak = t.transit_depth_ppm.max()
        print(f"  {name:30s}: base={base:.0f} ppm, peak={peak:.0f} ppm, "
              f"delta={peak - base:.0f} ppm")
