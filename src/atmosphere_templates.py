import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

WAVELENGTH_MIN_UM = 0.6
WAVELENGTH_MAX_UM = 5.3
N_WAVELENGTHS = 300


def default_wavelength_grid() -> np.ndarray:
    return np.logspace(
        np.log10(WAVELENGTH_MIN_UM),
        np.log10(WAVELENGTH_MAX_UM),
        N_WAVELENGTHS,
    )


@dataclass
class AtmosphereTemplate:
    name: str
    description: str
    wavelengths_um: np.ndarray
    transit_depth_ppm: np.ndarray
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
        return {
            "earth_like": "O2 + H2O + CO2 + O3",
            "high_co2": "CO2 (dominant)",
            "reduced_o2_high_ch4": "CH4 + H2O + CO2",
        }.get(self.name, self.name)


def _gaussian(wavelengths: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def _lorentzian(wavelengths: np.ndarray, center: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + ((wavelengths - center) / width) ** 2)


def _base_depth_ppm(planet_radius_re: float, star_radius_rs: float) -> float:
    rp_m = planet_radius_re * 6_371_000.0
    rs_m = star_radius_rs * 696_000_000.0
    return (rp_m / rs_m) ** 2 * 1_000_000.0


def _atm_amplitude_ppm(
    scale_height_km: float,
    planet_radius_re: float,
    star_radius_rs: float,
    n_scale_heights: float = 5.0,
) -> float:
    """
    Δδ = 2 N R_p H / R_s^2  (fully unit-consistent; see de Wit & Seager 2013).
    n_scale_heights: assumed vertical extent of the absorbing layer, in H.
    """
    h_m = scale_height_km * 1_000.0
    rp_m = planet_radius_re * 6_371_000.0
    rs_m = star_radius_rs * 696_000_000.0
    return 2.0 * n_scale_heights * rp_m * h_m / rs_m**2 * 1_000_000.0


def build_earth_like_template(
    wavelengths: np.ndarray,
    cloud_fraction: float = 0.5,
    o2_ch4_ratio: float = 1.0,
    scale_height_km: float = 8.5,
    planet_radius_re: float = 1.0,
    star_radius_rs: float = 0.2,
) -> AtmosphereTemplate:
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    amp = _atm_amplitude_ppm(scale_height_km, planet_radius_re, star_radius_rs)
    f_clear = 1.0 - cloud_fraction
    depth = np.full_like(wavelengths, base)

    h2o = [
        (0.94, 0.03, 0.6),
        (1.14, 0.04, 0.9),
        (1.38, 0.06, 1.5),
        (1.87, 0.07, 1.8),
        (2.7, 0.15, 1.3),
    ]
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
    cloud_fraction: float = 0.3,
    o2_ch4_ratio: float = 0.01,
    scale_height_km: float = 7.0,
    planet_radius_re: float = 1.0,
    star_radius_rs: float = 0.2,
) -> AtmosphereTemplate:
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    amp = _atm_amplitude_ppm(scale_height_km, planet_radius_re, star_radius_rs)
    f_clear = 1.0 - cloud_fraction
    depth = np.full_like(wavelengths, base)

    co2 = [
        (1.05, 0.04, 0.3),
        (1.21, 0.045, 0.35),
        (1.43, 0.055, 0.55),
        (1.6, 0.07, 0.8),
        (2.01, 0.1, 1.2),
        (2.68, 0.2, 1.5),
        (4.3, 0.45, 2.0),
    ]
    for center, width, strength in co2:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)

    depth += f_clear * amp * 0.05 * _gaussian(wavelengths, 1.38, 0.06)
    depth += f_clear * amp * 0.25 * _gaussian(wavelengths, 4.0, 0.2)
    depth += f_clear * amp * 0.01 * _gaussian(wavelengths, 0.762, 0.005)
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
    cloud_fraction: float = 0.4,
    o2_ch4_ratio: float = 0.001,
    scale_height_km: float = 9.0,
    planet_radius_re: float = 1.0,
    star_radius_rs: float = 0.2,
) -> AtmosphereTemplate:
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    amp = _atm_amplitude_ppm(scale_height_km, planet_radius_re, star_radius_rs)
    f_clear = 1.0 - cloud_fraction
    ch4_factor = np.clip(1.0 / max(o2_ch4_ratio, 1e-5), 1.0, 50.0)
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


def build_abiotic_o2_template(
    wavelengths: np.ndarray,
    cloud_fraction: float = 0.3,
    o2_ch4_ratio: float = 1000.0,
    scale_height_km: float = 7.0,
    planet_radius_re: float = 1.0,
    star_radius_rs: float = 0.2,
) -> AtmosphereTemplate:
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    amp = _atm_amplitude_ppm(scale_height_km, planet_radius_re, star_radius_rs)
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
        description="Abiotic O2 false positive: CO2-dominated, desiccated, photolytic O2/O3, CH4-free (Krissansen-Totton et al. 2021)",
        wavelengths_um=wavelengths,
        transit_depth_ppm=np.maximum(depth, 0.0),
        parameters={
            "cloud_fraction": float(cloud_fraction),
            "o2_ch4_ratio": float(o2_ch4_ratio),
            "scale_height_km": float(scale_height_km),
            "planet_radius_re": float(planet_radius_re),
            "star_radius_rs": float(star_radius_rs),
            "base_depth_ppm": float(base),
            "reference": "Krissansen-Totton et al. 2021, AGU Advances, 2, e2020AV000294",
        },
    )


def build_hycean_template(
    wavelengths: np.ndarray,
    cloud_fraction: float = 0.2,
    o2_ch4_ratio: float = 0.002,
    scale_height_km: float = 12.0,
    planet_radius_re: float = 2.3,
    star_radius_rs: float = 0.4,
    dms_strength: float = 0.08,
    dmds_strength: float = 0.0,
) -> AtmosphereTemplate:
    base = _base_depth_ppm(planet_radius_re, star_radius_rs)
    effective_h = scale_height_km * (28.0 / 2.3)
    amp = _atm_amplitude_ppm(effective_h, planet_radius_re, star_radius_rs)
    amp = min(amp, _atm_amplitude_ppm(scale_height_km, planet_radius_re, star_radius_rs))
    f_clear = 1.0 - cloud_fraction
    depth = np.full_like(wavelengths, base)

    ch4_factor = np.clip(1.0 / max(o2_ch4_ratio, 0.0001), 1.0, 20.0)
    ch4 = [(1.0, 0.03, 0.15), (1.33, 0.04, 0.2), (1.67, 0.055, 0.45), (2.3, 0.11, 0.65), (3.3, 0.16, 0.8)]
    for center, width, raw_s in ch4:
        s = np.clip(raw_s * ch4_factor * 0.3, 0, 2.5)
        depth += f_clear * amp * s * _gaussian(wavelengths, center, width)

    co2 = [(1.6, 0.035, 0.25), (2.01, 0.06, 0.4), (4.3, 0.26, 1.2)]
    for center, width, strength in co2:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)

    h2o = [(0.94, 0.035, 0.3), (1.14, 0.045, 0.5), (1.38, 0.065, 0.8), (1.87, 0.08, 0.9), (2.7, 0.16, 0.7)]
    for center, width, strength in h2o:
        depth += f_clear * amp * strength * _gaussian(wavelengths, center, width)

    if dms_strength > 0:
        depth += f_clear * amp * dms_strength * _gaussian(wavelengths, 3.4, 0.06)
    if dmds_strength > 0:
        depth += f_clear * amp * dmds_strength * _gaussian(wavelengths, 7.5, 0.15)

    cia_slope = f_clear * amp * 0.2 * np.where(wavelengths > 2.0, (wavelengths - 2.0) / 3.0, 0.0)
    depth += cia_slope
    depth += f_clear * amp * 0.4 * (wavelengths[0] / wavelengths) ** 4

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
            "dms_strength": float(dms_strength),
            "dmds_strength": float(dmds_strength),
            "reference": "Madhusudhan et al. 2023, ApJL 956 L13",
        },
    )


class TemplateGrid:
    BUILDERS = {
        "earth_like": build_earth_like_template,
        "high_co2": build_high_co2_template,
        "reduced_o2_high_ch4": build_reduced_o2_high_ch4_template,
        "hycean": build_hycean_template,
        "abiotic_o2": build_abiotic_o2_template,
    }

    DEFAULT_O2_CH4 = {
        "earth_like": [0.3, 1.0, 3.0],
        "high_co2": [0.001, 0.01, 0.05],
        "reduced_o2_high_ch4": [0.0001, 0.001, 0.01],
        "hycean": [0.001, 0.005, 0.02],
        "abiotic_o2": [500.0, 1000.0, 5000.0],
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
            if name not in self.BUILDERS:
                raise KeyError(f"Unknown template '{name}'. Available: {list(self.BUILDERS.keys())}.")
            if name not in o2_ch4_ratios:
                raise KeyError(
                    f"o2_ch4_ratios missing entry for '{name}'; "
                    f"pass an explicit value or omit '{name}' from template_names."
                )

            self.templates[name] = []
            builder = self.BUILDERS[name]
            ratios = o2_ch4_ratios[name]

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

        print(
            f"TemplateGrid built: {total} templates "
            f"({len(template_names)} types × {len(cloud_fractions)} cloud fracs × "
            f"{len(scale_heights_km)} scale heights)"
        )

    def get_template(
        self,
        template_name: str,
        cloud_fraction: float = 0.5,
        o2_ch4_ratio: float = 1.0,
        scale_height_km: float = 8.5,
    ) -> AtmosphereTemplate:
        if template_name not in self.templates or not self.templates[template_name]:
            raise KeyError(
                f"Template '{template_name}' not in grid. Available: {list(self.templates.keys())}. "
                "Call build_grid() first."
            )

        candidates = self.templates[template_name]

        def _dist(t: AtmosphereTemplate) -> float:
            p = t.parameters
            dc = abs(p["cloud_fraction"] - cloud_fraction)
            dr = abs(np.log10(max(p["o2_ch4_ratio"], 1e-7)) - np.log10(max(o2_ch4_ratio, 1e-7))) * 0.5
            dh = abs(p["scale_height_km"] - scale_height_km) / 10.0
            return dc + dr + dh

        return min(candidates, key=_dist)

    def save(self, path: str) -> None:
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


def get_default_templates(
    star_radius_rs: float = 0.2,
    planet_radius_re: float = 1.0,
) -> Dict[str, AtmosphereTemplate]:
    wl = default_wavelength_grid()
    return {
        "earth_like": build_earth_like_template(
            wl,
            cloud_fraction=0.5,
            o2_ch4_ratio=1.0,
            scale_height_km=8.5,
            planet_radius_re=planet_radius_re,
            star_radius_rs=star_radius_rs,
        ),
        "high_co2": build_high_co2_template(
            wl,
            cloud_fraction=0.3,
            o2_ch4_ratio=0.01,
            scale_height_km=7.0,
            planet_radius_re=planet_radius_re,
            star_radius_rs=star_radius_rs,
        ),
        "reduced_o2_high_ch4": build_reduced_o2_high_ch4_template(
            wl,
            cloud_fraction=0.4,
            o2_ch4_ratio=0.001,
            scale_height_km=9.0,
            planet_radius_re=planet_radius_re,
            star_radius_rs=star_radius_rs,
        ),
        "hycean": build_hycean_template(
            wl,
            cloud_fraction=0.2,
            o2_ch4_ratio=0.002,
            scale_height_km=12.0,
            planet_radius_re=max(planet_radius_re * 2.0, 2.3),
            star_radius_rs=max(star_radius_rs * 2.0, 0.4),
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build atmosphere template grid")
    parser.add_argument("--out", default="data/templates/template_grid.json")
    parser.add_argument("--star-radius", type=float, default=0.2, help="Star radius [R_sun]")
    parser.add_argument("--planet-radius", type=float, default=1.0, help="Planet radius [R_earth]")
    args = parser.parse_args()

    print("Building atmosphere template grid...")
    grid = TemplateGrid()
    grid.build_grid(planet_radius_re=args.planet_radius, star_radius_rs=args.star_radius)
    grid.save(args.out)

    templates = get_default_templates(
        star_radius_rs=args.star_radius,
        planet_radius_re=args.planet_radius,
    )
    for name, t in templates.items():
        base = t.parameters["base_depth_ppm"]
        peak = t.transit_depth_ppm.max()
        print(f"  {name:30s}: base={base:.0f} ppm, peak={peak:.0f} ppm, delta={peak - base:.0f} ppm")
