from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

R_EARTH_KM = 6371.0
R_SUN_KM = 695700.0
DEFAULT_WAVELENGTH_MIN_UM = 0.60
DEFAULT_WAVELENGTH_MAX_UM = 5.30
DEFAULT_WAVELENGTH_POINTS = 1600


@dataclass
class AtmosphereTemplate:
    name: str
    wavelengths_um: np.ndarray
    transit_depth_ppm: np.ndarray
    parameters: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelengths_um = np.asarray(self.wavelengths_um, dtype=float)
        self.transit_depth_ppm = np.asarray(self.transit_depth_ppm, dtype=float)
        if self.wavelengths_um.ndim != 1 or self.transit_depth_ppm.ndim != 1:
            raise ValueError("wavelengths_um and transit_depth_ppm must be 1D arrays")
        if self.wavelengths_um.shape != self.transit_depth_ppm.shape:
            raise ValueError("wavelength and depth arrays must have the same length")
        if not np.all(np.isfinite(self.wavelengths_um)) or not np.all(np.isfinite(self.transit_depth_ppm)):
            raise ValueError("template arrays must be finite")
        order = np.argsort(self.wavelengths_um)
        self.wavelengths_um = self.wavelengths_um[order]
        self.transit_depth_ppm = self.transit_depth_ppm[order]
        self.parameters = dict(self.parameters)

    @property
    def base_depth_ppm(self) -> float:
        if "base_depth_ppm" in self.parameters:
            return float(self.parameters["base_depth_ppm"])
        return float(np.nanmedian(self.transit_depth_ppm))

    def copy(self, **overrides) -> "AtmosphereTemplate":
        data = {
            "name": self.name,
            "wavelengths_um": self.wavelengths_um.copy(),
            "transit_depth_ppm": self.transit_depth_ppm.copy(),
            "parameters": dict(self.parameters),
        }
        data.update(overrides)
        return AtmosphereTemplate(**data)


def default_wavelength_grid(
    wavelength_min_um: float = DEFAULT_WAVELENGTH_MIN_UM,
    wavelength_max_um: float = DEFAULT_WAVELENGTH_MAX_UM,
    n_points: int = DEFAULT_WAVELENGTH_POINTS,
) -> np.ndarray:
    if wavelength_min_um <= 0 or wavelength_max_um <= wavelength_min_um:
        raise ValueError("invalid wavelength bounds")
    if n_points < 10:
        raise ValueError("n_points must be at least 10")
    return np.linspace(wavelength_min_um, wavelength_max_um, int(n_points))


def _planet_star_depth_ppm(planet_radius_re: float, star_radius_rs: float) -> float:
    planet_radius_re = max(float(planet_radius_re), 1e-6)
    star_radius_rs = max(float(star_radius_rs), 1e-6)
    return float(((planet_radius_re * R_EARTH_KM) / (star_radius_rs * R_SUN_KM)) ** 2 * 1e6)


def _reference_modulation_ppm(
    planet_radius_re: float,
    star_radius_rs: float,
    scale_height_km: float,
    n_scale_heights: float = 5.5,
) -> float:
    planet_radius_km = max(float(planet_radius_re), 1e-6) * R_EARTH_KM
    star_radius_km = max(float(star_radius_rs), 1e-6) * R_SUN_KM
    scale_height_km = max(float(scale_height_km), 0.1)
    return float(2.0 * planet_radius_km * scale_height_km * n_scale_heights / (star_radius_km**2) * 1e6)


def _feature_profile(wavelengths_um: np.ndarray, center_um: float, width_um: float) -> np.ndarray:
    width_um = max(float(width_um), 1e-3)
    return np.exp(-0.5 * ((wavelengths_um - center_um) / width_um) ** 2)


def _make_template(
    name: str,
    wavelengths_um: np.ndarray,
    planet_radius_re: float,
    star_radius_rs: float,
    scale_height_km: float,
    cloud_fraction: float,
    o2_ch4_ratio: float,
    family: str,
) -> AtmosphereTemplate:
    wl = np.asarray(wavelengths_um, dtype=float)
    base_depth_ppm = _planet_star_depth_ppm(planet_radius_re, star_radius_rs)
    ref_mod = _reference_modulation_ppm(planet_radius_re, star_radius_rs, scale_height_km)
    cloud_fraction = float(np.clip(cloud_fraction, 0.0, 0.95))
    scale_height_km = float(np.clip(scale_height_km, 2.0, 20.0))
    o2_ch4_ratio = float(np.clip(o2_ch4_ratio, 1e-4, 1e4))
    oxidizing = float(np.clip(np.log10(o2_ch4_ratio + 1e-12), -4.0, 4.0))
    oxidizing_strength = np.clip(0.5 + 0.16 * oxidizing, 0.15, 2.4)
    reducing_strength = np.clip(0.5 - 0.16 * oxidizing, 0.15, 2.4)
    haze_suppression = np.clip(1.0 - 0.75 * cloud_fraction, 0.18, 1.0)
    slope_factor = np.clip(1.0 - 0.55 * cloud_fraction, 0.30, 1.0)

    depth = np.full_like(wl, base_depth_ppm, dtype=float)

    blue_slope = ref_mod * 0.10 * slope_factor * ((0.80 / np.maximum(wl, 0.60)) ** 4)
    depth += np.clip(blue_slope, 0.0, ref_mod * 0.30)

    bands = []
    if family == "earth_like":
        bands = [
            (0.76, 0.012, 1.4 * oxidizing_strength * haze_suppression),
            (0.94, 0.020, 0.55 * haze_suppression),
            (1.13, 0.030, 0.75 * haze_suppression),
            (1.38, 0.060, 3.5 * haze_suppression),
            (1.60, 0.040, 0.40 * haze_suppression),
            (1.65, 0.042, 1.0 * reducing_strength * haze_suppression),
            (1.88, 0.085, 2.6 * haze_suppression),
            (2.01, 0.060, 1.6 * haze_suppression),
            (2.32, 0.070, 1.25 * reducing_strength * haze_suppression),
            (4.30, 0.135, 3.8 * haze_suppression),
        ]
    elif family == "high_co2":
        bands = [
            (0.76, 0.012, 0.25 * oxidizing_strength * haze_suppression),
            (1.15, 0.030, 0.30 * haze_suppression),
            (1.38, 0.060, 1.1 * haze_suppression),
            (1.60, 0.045, 0.30 * haze_suppression),
            (1.88, 0.080, 1.0 * haze_suppression),
            (2.01, 0.060, 4.6 * haze_suppression),
            (2.32, 0.070, 0.22 * reducing_strength * haze_suppression),
            (4.30, 0.120, 7.0 * haze_suppression),
        ]
    elif family == "reduced_o2_high_ch4":
        bands = [
            (0.76, 0.012, 0.18 * oxidizing_strength * haze_suppression),
            (0.94, 0.020, 0.50 * haze_suppression),
            (1.13, 0.030, 0.55 * haze_suppression),
            (1.38, 0.060, 2.4 * haze_suppression),
            (1.60, 0.040, 0.35 * haze_suppression),
            (1.65, 0.045, 3.8 * reducing_strength * haze_suppression),
            (1.88, 0.080, 2.1 * haze_suppression),
            (2.32, 0.075, 2.9 * reducing_strength * haze_suppression),
            (4.30, 0.135, 1.1 * haze_suppression),
        ]
    elif family == "abiotic_o2":
        bands = [
            (0.76, 0.012, 2.2 * oxidizing_strength * haze_suppression),
            (1.06, 0.020, 0.9 * oxidizing_strength * haze_suppression),
            (1.27, 0.030, 0.55 * oxidizing_strength * haze_suppression),
            (1.38, 0.060, 0.95 * haze_suppression),
            (1.60, 0.040, 0.30 * haze_suppression),
            (1.88, 0.080, 1.0 * haze_suppression),
            (2.01, 0.060, 1.3 * haze_suppression),
            (2.32, 0.075, 0.18 * reducing_strength * haze_suppression),
            (4.30, 0.130, 1.7 * haze_suppression),
        ]
    else:
        raise ValueError(f"unknown family: {family}")

    for center_um, width_um, strength in bands:
        amp = np.clip(ref_mod * float(strength), 0.0, base_depth_ppm * 0.15)
        depth += amp * _feature_profile(wl, center_um, width_um)

    depth = np.maximum(depth, 1.0)

    params = {
        "base_depth_ppm": base_depth_ppm,
        "planet_radius_re": float(planet_radius_re),
        "star_radius_rs": float(star_radius_rs),
        "scale_height_km": scale_height_km,
        "cloud_fraction": cloud_fraction,
        "o2_ch4_ratio": o2_ch4_ratio,
        "reference_modulation_ppm": ref_mod,
    }
    return AtmosphereTemplate(name=name, wavelengths_um=wl, transit_depth_ppm=depth, parameters=params)


def build_earth_like_template(
    wavelengths_um: np.ndarray,
    cloud_fraction: float = 0.35,
    o2_ch4_ratio: float = 1.0,
    scale_height_km: float = 8.5,
    planet_radius_re: float = 1.0,
    star_radius_rs: float = 0.2,
) -> AtmosphereTemplate:
    return _make_template(
        name="earth_like",
        wavelengths_um=wavelengths_um,
        planet_radius_re=planet_radius_re,
        star_radius_rs=star_radius_rs,
        scale_height_km=scale_height_km,
        cloud_fraction=cloud_fraction,
        o2_ch4_ratio=o2_ch4_ratio,
        family="earth_like",
    )


def build_high_co2_template(
    wavelengths_um: np.ndarray,
    cloud_fraction: float = 0.25,
    o2_ch4_ratio: float = 0.10,
    scale_height_km: float = 6.5,
    planet_radius_re: float = 1.0,
    star_radius_rs: float = 0.2,
) -> AtmosphereTemplate:
    return _make_template(
        name="high_co2",
        wavelengths_um=wavelengths_um,
        planet_radius_re=planet_radius_re,
        star_radius_rs=star_radius_rs,
        scale_height_km=scale_height_km,
        cloud_fraction=cloud_fraction,
        o2_ch4_ratio=o2_ch4_ratio,
        family="high_co2",
    )


def build_reduced_o2_high_ch4_template(
    wavelengths_um: np.ndarray,
    cloud_fraction: float = 0.30,
    o2_ch4_ratio: float = 0.15,
    scale_height_km: float = 8.0,
    planet_radius_re: float = 1.0,
    star_radius_rs: float = 0.2,
) -> AtmosphereTemplate:
    return _make_template(
        name="reduced_o2_high_ch4",
        wavelengths_um=wavelengths_um,
        planet_radius_re=planet_radius_re,
        star_radius_rs=star_radius_rs,
        scale_height_km=scale_height_km,
        cloud_fraction=cloud_fraction,
        o2_ch4_ratio=o2_ch4_ratio,
        family="reduced_o2_high_ch4",
    )


def build_abiotic_o2_template(
    wavelengths_um: np.ndarray,
    cloud_fraction: float = 0.20,
    o2_ch4_ratio: float = 10.0,
    scale_height_km: float = 7.5,
    planet_radius_re: float = 1.0,
    star_radius_rs: float = 0.2,
) -> AtmosphereTemplate:
    return _make_template(
        name="abiotic_o2",
        wavelengths_um=wavelengths_um,
        planet_radius_re=planet_radius_re,
        star_radius_rs=star_radius_rs,
        scale_height_km=scale_height_km,
        cloud_fraction=cloud_fraction,
        o2_ch4_ratio=o2_ch4_ratio,
        family="abiotic_o2",
    )


def get_default_templates(
    star_radius_rs: float = 0.2,
    planet_radius_re: float = 1.0,
    wavelengths_um: Optional[np.ndarray] = None,
) -> Dict[str, AtmosphereTemplate]:
    wl = default_wavelength_grid() if wavelengths_um is None else np.asarray(wavelengths_um, dtype=float)
    return {
        "earth_like": build_earth_like_template(
            wl,
            cloud_fraction=0.35,
            o2_ch4_ratio=1.0,
            scale_height_km=8.5,
            planet_radius_re=planet_radius_re,
            star_radius_rs=star_radius_rs,
        ),
        "high_co2": build_high_co2_template(
            wl,
            cloud_fraction=0.25,
            o2_ch4_ratio=0.10,
            scale_height_km=6.5,
            planet_radius_re=planet_radius_re,
            star_radius_rs=star_radius_rs,
        ),
        "reduced_o2_high_ch4": build_reduced_o2_high_ch4_template(
            wl,
            cloud_fraction=0.30,
            o2_ch4_ratio=0.15,
            scale_height_km=8.0,
            planet_radius_re=planet_radius_re,
            star_radius_rs=star_radius_rs,
        ),
        "abiotic_o2": build_abiotic_o2_template(
            wl,
            cloud_fraction=0.20,
            o2_ch4_ratio=10.0,
            scale_height_km=7.5,
            planet_radius_re=planet_radius_re,
            star_radius_rs=star_radius_rs,
        ),
    }


class TemplateGrid:
    def __init__(self) -> None:
        self.templates: Dict[str, List[AtmosphereTemplate]] = {}

    def build_grid(
        self,
        planet_radius_re: float = 1.0,
        star_radius_rs: float = 0.2,
        cloud_fractions: Optional[List[float]] = None,
        scale_heights_km: Optional[List[float]] = None,
        o2_ch4_ratios: Optional[List[float]] = None,
        wavelengths_um: Optional[np.ndarray] = None,
    ) -> Dict[str, List[AtmosphereTemplate]]:
        wl = default_wavelength_grid() if wavelengths_um is None else np.asarray(wavelengths_um, dtype=float)
        cloud_fractions = [0.0, 0.3, 0.6] if cloud_fractions is None else list(cloud_fractions)
        scale_heights_km = [7.0, 8.5, 10.0] if scale_heights_km is None else list(scale_heights_km)
        o2_ch4_ratios = [0.15, 1.0, 10.0] if o2_ch4_ratios is None else list(o2_ch4_ratios)

        families: Dict[str, List[AtmosphereTemplate]] = {
            "earth_like": [],
            "high_co2": [],
            "reduced_o2_high_ch4": [],
            "abiotic_o2": [],
        }

        for cf in cloud_fractions:
            for sh in scale_heights_km:
                for ratio in o2_ch4_ratios:
                    families["earth_like"].append(
                        build_earth_like_template(
                            wl,
                            cloud_fraction=cf,
                            o2_ch4_ratio=ratio,
                            scale_height_km=sh,
                            planet_radius_re=planet_radius_re,
                            star_radius_rs=star_radius_rs,
                        )
                    )
                    families["high_co2"].append(
                        build_high_co2_template(
                            wl,
                            cloud_fraction=cf,
                            o2_ch4_ratio=ratio,
                            scale_height_km=max(4.5, sh - 1.5),
                            planet_radius_re=planet_radius_re,
                            star_radius_rs=star_radius_rs,
                        )
                    )
                    families["reduced_o2_high_ch4"].append(
                        build_reduced_o2_high_ch4_template(
                            wl,
                            cloud_fraction=cf,
                            o2_ch4_ratio=ratio,
                            scale_height_km=sh,
                            planet_radius_re=planet_radius_re,
                            star_radius_rs=star_radius_rs,
                        )
                    )
                    families["abiotic_o2"].append(
                        build_abiotic_o2_template(
                            wl,
                            cloud_fraction=cf,
                            o2_ch4_ratio=ratio,
                            scale_height_km=sh,
                            planet_radius_re=planet_radius_re,
                            star_radius_rs=star_radius_rs,
                        )
                    )

        self.templates = families
        return self.templates

    def all_templates(self) -> List[AtmosphereTemplate]:
        out: List[AtmosphereTemplate] = []
        for template_list in self.templates.values():
            out.extend(template_list)
        return out

    def __len__(self) -> int:
        return sum(len(v) for v in self.templates.values())


__all__ = [
    "AtmosphereTemplate",
    "TemplateGrid",
    "default_wavelength_grid",
    "get_default_templates",
    "build_earth_like_template",
    "build_high_co2_template",
    "build_reduced_o2_high_ch4_template",
    "build_abiotic_o2_template",
]
