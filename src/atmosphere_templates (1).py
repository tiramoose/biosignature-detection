import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

WAVELENGTH_MIN_UM = 0.6
WAVELENGTH_MAX_UM = 5.3
N_WAVELENGTHS = 300


def default_wavelength_grid() -> np.ndarray:
    return np.logspace(np.log10(WAVELENGTH_MIN_UM), np.log10(WAVELENGTH_MAX_UM), N_WAVELENGTHS)


@dataclass
class AtmosphereTemplate:
    name: str
    description: str
    wavelengths_um: np.ndarray
    transit_depth_ppm: np.ndarray
    parameters: Dict


def _gaussian(wavelengths, center, width):
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def _lorentzian(wavelengths, center, width):
    return 1.0 / (1.0 + ((wavelengths - center) / width) ** 2)


def _base_depth_ppm(planet_radius_re, star_radius_rs):
    rp_m = planet_radius_re * 6371000.0
    rs_m = star_radius_rs * 696000000.0
    return (rp_m / rs_m) ** 2 * 1000000.0


def _atm_amplitude_ppm(scale_height_km, star_radius_rs):
    H_m = scale_height_km * 1000.0
    Rs_m = star_radius_rs * 696000000.0
    return 5.0 * H_m / Rs_m * 1000000.0


def build_earth_like_template(wavelengths, cloud_fraction=0.5, o2_ch4_ratio=1.0, scale_height_km=8.5,
                               planet_radius_re=1.0, star_radius_rs=0.2) -> AtmosphereTemplate:
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    amp = _atm_amplitude_ppm(scale_height_km, star_radius_rs)
    f_clear = 1.0 - cloud_fraction
    depth = np.full_like(wavelengths, base)
    h2o = [(0.94, 0.03, 0.6), (1.14, 0.04, 0.9), (1.38, 0.06, 1.5), (1.87, 0.07, 1.8), (2.7, 0.15, 1.3)]
    for center, width, strength in h2o:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)
    o2_factor = np.clip(o2_ch4_ratio, 0.001, 10.0)
    depth += f_clear * amp * 1.0 * o2_factor * _gaussian(wavelengths, 0.762, 0.005)
    depth += f_clear * amp * 0.25 * o2_factor * _gaussian(wavelengths, 0.688, 0.004)
    co2 = [(1.6, 0.03, 0.15), (2.01, 0.05, 0.2), (4.3, 0.2, 0.55)]
    for center, width, strength in co2:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)
    depth += f_clear * amp * 0.2 * _lorentzian(wavelengths, 0.6, 0.08)
    ch4_abundance = 1.0 / max(o2_ch4_ratio, 0.01)
    ch4 = [(1.67, 0.04, 0.04 * ch4_abundance), (2.3, 0.08, 0.07 * ch4_abundance)]
    for center, width, strength in ch4:
        depth += f_clear * amp * np.clip(strength, 0, 0.5) * _gaussian(wavelengths, center, width)
    rayleigh_amp = f_clear * amp * 0.3
    depth += rayleigh_amp * (wavelengths[0] / wavelengths) ** 4
    return AtmosphereTemplate(
        name="earth_like",
        description="Earth-analog atmosphere",
        wavelengths_um=wavelengths,
        transit_depth_ppm=np.maximum(depth, 0.0),
        parameters={"cloud_fraction": float(cloud_fraction), "o2_ch4_ratio": float(o2_ch4_ratio),
                    "scale_height_km": float(scale_height_km), "planet_radius_re": float(planet_radius_re),
                    "star_radius_rs": float(star_radius_rs), "base_depth_ppm": float(base)},
    )


def build_high_co2_template(wavelengths, cloud_fraction=0.3, o2_ch4_ratio=0.01, scale_height_km=7.0,
                            planet_radius_re=1.0, star_radius_rs=0.2) -> AtmosphereTemplate:
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    amp = _atm_amplitude_ppm(scale_height_km, star_radius_rs)
    f_clear = 1.0 - cloud_fraction
    depth = np.full_like(wavelengths, base)
    co2 = [(1.05, 0.04, 0.3), (1.21, 0.045, 0.35), (1.43, 0.055, 0.55), (1.6, 0.07, 0.8),
           (2.01, 0.1, 1.2), (2.68, 0.2, 1.5), (4.3, 0.45, 2.0)]
    for center, width, strength in co2:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)
    depth += f_clear * amp * 0.05 * _gaussian(wavelengths, 1.38, 0.06)
    depth += f_clear * amp * 0.25 * _gaussian(wavelengths, 4.0, 0.2)
    depth += f_clear * amp * 0.01 * _gaussian(wavelengths, 0.762, 0.005)
    depth += f_clear * amp * 0.15 * (wavelengths[0] / wavelengths) ** 2
    return AtmosphereTemplate(
        name="high_co2",
        description="Venus-analog atmosphere",
        wavelengths_um=wavelengths,
        transit_depth_ppm=np.maximum(depth, 0.0),
        parameters={"cloud_fraction": float(cloud_fraction), "o2_ch4_ratio": float(o2_ch4_ratio),
                    "scale_height_km": float(scale_height_km), "planet_radius_re": float(planet_radius_re),
                    "star_radius_rs": float(star_radius_rs), "base_depth_ppm": float(base)},
    )


def build_abiotic_o2_template(wavelengths, cloud_fraction=0.3, o2_ch4_ratio=1000.0, scale_height_km=7.0,
                              planet_radius_re=1.0, star_radius_rs=0.2) -> AtmosphereTemplate:
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    amp = _atm_amplitude_ppm(scale_height_km, star_radius_rs)
    f_clear = 1.0 - cloud_fraction
    depth = np.full_like(wavelengths, base)
    co2 = [(1.6, 0.03, 0.15), (2.01, 0.05, 0.2), (4.3, 0.2, 0.55)]
    for center, width, strength in co2:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)
    depth += f_clear * amp * 1.0 * _gaussian(wavelengths, 0.762, 0.005)
    depth += f_clear * amp * 0.25 * _gaussian(wavelengths, 0.688, 0.004)
    depth += f_clear * amp * 0.2 * _lorentzian(wavelengths, 0.6, 0.08)
    depth += f_clear * amp * 0.03 * _gaussian(wavelengths, 1.38, 0.06)
    depth += f_clear * amp * 0.3 * (wavelengths[0] / wavelengths) ** 4
    return AtmosphereTemplate(
        name="abiotic_o2",
        description="Photolytic abiotic O2 false positive",
        wavelengths_um=wavelengths,
        transit_depth_ppm=np.maximum(depth, 0.0),
        parameters={"cloud_fraction": float(cloud_fraction), "o2_ch4_ratio": float(o2_ch4_ratio),
                    "scale_height_km": float(scale_height_km), "planet_radius_re": float(planet_radius_re),
                    "star_radius_rs": float(star_radius_rs), "base_depth_ppm": float(base)},
    )


def build_reduced_o2_high_ch4_template(wavelengths, cloud_fraction=0.4, o2_ch4_ratio=0.001, scale_height_km=9.0,
                                       planet_radius_re=1.0, star_radius_rs=0.2) -> AtmosphereTemplate:
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    amp = _atm_amplitude_ppm(scale_height_km, star_radius_rs)
    f_clear = 1.0 - cloud_fraction
    ch4_factor = np.clip(1.0 / max(o2_ch4_ratio, 1e-05), 1.0, 50.0)
    depth = np.full_like(wavelengths, base)
    ch4 = [(1.0, 0.03, 0.1), (1.33, 0.04, 0.15), (1.67, 0.05, 0.3), (2.3, 0.1, 0.5), (3.3, 0.15, 0.7)]
    for center, width, raw_strength in ch4:
        strength = np.clip(raw_strength * ch4_factor, 0, 3.0)
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)
    h2o = [(0.94, 0.03, 0.45), (1.14, 0.04, 0.65), (1.38, 0.06, 1.0), (1.87, 0.07, 1.1), (2.7, 0.15, 0.8)]
    for center, width, strength in h2o:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)
    co2 = [(1.6, 0.04, 0.3), (2.01, 0.06, 0.4), (4.3, 0.25, 0.7)]
    for center, width, strength in co2:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)
    o2_factor = np.clip(o2_ch4_ratio, 0.0, 1.0)
    depth += f_clear * amp * 0.5 * o2_factor * _gaussian(wavelengths, 0.762, 0.005)
    haze_strength = np.clip((ch4_factor - 1.0) / 50.0, 0, 0.5)
    depth += f_clear * amp * haze_strength * (wavelengths[0] / wavelengths) ** 2
    depth += f_clear * amp * 0.25 * (wavelengths[0] / wavelengths) ** 4
    return AtmosphereTemplate(
        name="reduced_o2_high_ch4",
        description="Archean-analog atmosphere",
        wavelengths_um=wavelengths,
        transit_depth_ppm=np.maximum(depth, 0.0),
        parameters={"cloud_fraction": float(cloud_fraction), "o2_ch4_ratio": float(o2_ch4_ratio),
                    "scale_height_km": float(scale_height_km), "planet_radius_re": float(planet_radius_re),
                    "star_radius_rs": float(star_radius_rs), "base_depth_ppm": float(base)},
    )
