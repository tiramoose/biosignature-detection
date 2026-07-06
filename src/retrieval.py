import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from atmosphere_templates import AtmosphereTemplate, TemplateGrid


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    Output of a single template-matching retrieval.

    Contains the best-fit template, chi-squared values for all templates,
    BIC-based model ranking, and marginalized parameter constraints.
    """
    best_template: AtmosphereTemplate
    best_chi2: float
    best_chi2_reduced: float   # chi2 / degrees of freedom
    n_dof: int                 # degrees of freedom

    # All templates ranked by chi2
    all_chi2: Dict[str, List[Tuple[float, AtmosphereTemplate]]]
    # Format: {template_name: [(chi2_value, template), ...]} sorted ascending

    # Model comparison (BIC: lower = better)
    bic_scores: Dict[str, float]
    delta_bic: Dict[str, float]     # BIC_i - BIC_best
    model_weights: Dict[str, float]  # Bayesian model weights (sum to 1)

    # Detection confidence
    detection_snr: float     # SNR of best-fit atmosphere vs. flat spectrum
    flat_chi2: float         # chi2 for a featureless (flat) spectrum
    delta_chi2_flat: float   # chi2_flat - chi2_best (positive = detection)

    # Input data
    wavelengths_um: np.ndarray
    observed_depth_ppm: np.ndarray
    noise_ppm: np.ndarray
    best_fit_model_ppm: np.ndarray  # best-fit template interpolated to obs wavelengths

    @property
    def is_detected(self) -> bool:
        """5-sigma detection: delta_chi2 > 25 (Wilks' theorem, 1 free parameter)."""
        return self.detection_snr >= 5.0

    @property
    def preferred_atmosphere(self) -> str:
        return self.best_template.name

    @property
    def top_models(self) -> List[Tuple[str, float]]:
        """Top 3 models ranked by BIC, as (name, delta_BIC) pairs."""
        ranked = sorted(self.delta_bic.items(), key=lambda x: x[1])
        return ranked[:3]

    def summary(self) -> str:
        lines = [
            f"{'─' * 55}",
            "  RETRIEVAL RESULT",
            f"  Best fit    : {self.best_template.name}",
            f"  Chi2_red    : {self.best_chi2_reduced:.2f}  (ideal ~1.0)",
            f"  Detection   : {self.detection_snr:.1f}σ  "
            f"({'DETECTED' if self.is_detected else 'not detected'})",
            f"  Delta-chi2  : {self.delta_chi2_flat:.1f}  (vs. flat spectrum)",
            "",
            "  Model ranking (BIC):",
        ]
        for name, dbic in self.top_models:
            weight = self.model_weights.get(name, 0.0)
            lines.append(f"    {name:30s}  ΔBIC={dbic:+7.1f}  weight={weight:.3f}")
        lines.append(f"{'─' * 55}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retrieval engine
# ---------------------------------------------------------------------------

class TemplateRetrieval:
    """
    Chi-squared template matching retrieval.

    Fits an observed transmission spectrum against a precomputed template
    grid to determine: (1) the best-fit atmosphere type, (2) parameter
    constraints, (3) model comparison statistics.

    The retrieval is essentially a grid search over the parameter space
    defined by the TemplateGrid.
    """

    def __init__(self, grid: TemplateGrid):
        self.grid = grid

    def _interpolate_template(
        self,
        template: AtmosphereTemplate,
        wavelengths_obs: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate template depth onto the observed wavelength grid.
        Uses linear interpolation; assumes template grid is sufficiently dense.
        """
        return np.interp(
            wavelengths_obs,
            template.wavelengths_um,
            template.transit_depth_ppm,
            left=np.nan,
            right=np.nan,
        )

    def _chi2(
        self,
        observed: np.ndarray,
        model: np.ndarray,
        errors: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute chi-squared statistic."""
        if mask is None:
            mask = np.ones(len(observed), dtype=bool)

        valid = mask & np.isfinite(model) & np.isfinite(observed) & np.isfinite(errors) & (errors > 0)
        if valid.sum() < 3:
            return np.inf

        residuals = observed[valid] - model[valid]
        return float(np.sum((residuals / errors[valid]) ** 2))

    def _bic(self, chi2: float, n_data: int, n_params: int) -> float:
        """
        Bayesian Information Criterion (BIC).
        BIC = chi2 + k * ln(n)  (for Gaussian errors, BIC = -2*ln(L) + k*ln(n))
        Lower BIC = preferred model.
        """
        return chi2 + n_params * np.log(n_data)

    def _fit_template(
        self,
        observed_depth_ppm: np.ndarray,
        noise_ppm: np.ndarray,
        model_depth_ppm: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Fit a single template with one free multiplicative scale factor.

        Returns:
            chi2, scale, best_fit_model_ppm
        """
        valid = mask & np.isfinite(model_depth_ppm) & np.isfinite(observed_depth_ppm) & np.isfinite(noise_ppm) & (noise_ppm > 0)
        if valid.sum() < 3:
            return np.inf, np.nan, np.full_like(observed_depth_ppm, np.nan)

        obs_v = observed_depth_ppm[valid]
        mod_v = model_depth_ppm[valid]
        err_v = noise_ppm[valid]
        w = 1.0 / (err_v ** 2)

        denom = np.sum((mod_v ** 2) * w)
        if denom <= 0:
            return np.inf, np.nan, np.full_like(observed_depth_ppm, np.nan)

        numer = np.sum(mod_v * obs_v * w)
        scale = numer / denom
        scale = float(np.clip(scale, 0.0, np.inf))

        best_model = scale * model_depth_ppm
        chi2 = self._chi2(observed_depth_ppm, best_model, noise_ppm, mask=mask)
        return chi2, scale, best_model

    def fit(
        self,
        wavelengths_obs: np.ndarray,
        observed_depth_ppm: np.ndarray,
        noise_ppm: np.ndarray,
        wavelength_range: Optional[Tuple[float, float]] = None,
    ) -> RetrievalResult:
        """
        Fit all templates in the grid to the observed spectrum.
        """
        if wavelength_range is not None:
            mask = (
                (wavelengths_obs >= wavelength_range[0])
                & (wavelengths_obs <= wavelength_range[1])
                & np.isfinite(observed_depth_ppm)
                & np.isfinite(noise_ppm)
                & (noise_ppm > 0)
                & (noise_ppm < 1e5)
            )
        else:
            mask = (
                np.isfinite(observed_depth_ppm)
                & np.isfinite(noise_ppm)
                & (noise_ppm > 0)
                & (noise_ppm < 1e5)
            )

        n_data = int(mask.sum())
        if n_data < 3:
            raise ValueError("Not enough valid data points to perform retrieval.")

        n_params = 1  # one free parameter per template: depth normalization scale

        # Flat spectrum chi2 (null hypothesis: no atmosphere)
        flat_depth = float(np.median(observed_depth_ppm[mask]))
        flat_model = np.full_like(observed_depth_ppm, flat_depth)
        flat_chi2 = self._chi2(observed_depth_ppm, flat_model, noise_ppm, mask)

        # Fit all templates
        all_chi2: Dict[str, List[Tuple[float, AtmosphereTemplate]]] = {}
        best_chi2 = np.inf
        best_template: Optional[AtmosphereTemplate] = None
        best_model_ppm: Optional[np.ndarray] = None

        for atm_name, template_list in self.grid.templates.items():
            ranked: List[Tuple[float, AtmosphereTemplate]] = []

            for template in template_list:
                model = self._interpolate_template(template, wavelengths_obs)
                chi2, _, best_fit_model = self._fit_template(
                    observed_depth_ppm=observed_depth_ppm,
                    noise_ppm=noise_ppm,
                    model_depth_ppm=model,
                    mask=mask,
                )
                ranked.append((chi2, template))

                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_template = template
                    best_model_ppm = best_fit_model

            ranked.sort(key=lambda item: item[0])
            all_chi2[atm_name] = ranked

        if best_template is None or best_model_ppm is None:
            raise RuntimeError("No templates could be fit. Check grid and data overlap.")

        # BIC for each atmosphere type (using best chi2 within each type)
        bic_scores: Dict[str, float] = {}
        for atm_name, ranked in all_chi2.items():
            if ranked:
                best_type_chi2 = ranked[0][0]
                bic_scores[atm_name] = self._bic(best_type_chi2, n_data, n_params)

        best_bic = min(bic_scores.values())
        delta_bic = {name: bic - best_bic for name, bic in bic_scores.items()}

        # Bayesian model weights
        raw_weights = {name: np.exp(-0.5 * dbic) for name, dbic in delta_bic.items()}
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            model_weights = {name: 0.0 for name in raw_weights}
        else:
            model_weights = {name: w / total_weight for name, w in raw_weights.items()}

        # Detection SNR: how much better is best-fit vs flat?
        delta_chi2_flat = flat_chi2 - best_chi2
        detection_snr = float(np.sqrt(max(delta_chi2_flat, 0.0)))

        n_dof = max(n_data - n_params, 1)

        return RetrievalResult(
            best_template=best_template,
            best_chi2=best_chi2,
            best_chi2_reduced=best_chi2 / n_dof,
            n_dof=n_dof,
            all_chi2=all_chi2,
            bic_scores=bic_scores,
            delta_bic=delta_bic,
            model_weights=model_weights,
            detection_snr=detection_snr,
            flat_chi2=flat_chi2,
            delta_chi2_flat=delta_chi2_flat,
            wavelengths_um=wavelengths_obs,
            observed_depth_ppm=observed_depth_ppm,
            noise_ppm=noise_ppm,
            best_fit_model_ppm=best_model_ppm,
        )

    def confusion_matrix(
        self,
        sim_results: List[object],  # List[ObservationResult] from observation_sim
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute retrieval confusion matrix over a set of simulated observations.

        For each simulated observation, runs the retrieval and records:
        - True atmosphere type (from simulation)
        - Retrieved atmosphere type (from chi2 minimization)

        Returns N_types × N_types confusion matrix (rows=true, cols=retrieved).
        """
        type_names = list(self.grid.templates.keys())
        n_types = len(type_names)
        idx = {name: i for i, name in enumerate(type_names)}
        matrix = np.zeros((n_types, n_types), dtype=int)

        for i, obs in enumerate(sim_results):
            if getattr(obs, "noise_ppm", None) is None:
                continue

            try:
                result = self.fit(
                    obs.wavelengths_um,
                    obs.observed_depth_ppm,
                    obs.noise_ppm,
                )
                true_idx = idx.get(getattr(obs, "atmosphere_type", ""), -1)
                ret_idx = idx.get(result.preferred_atmosphere, -1)
                if true_idx >= 0 and ret_idx >= 0:
                    matrix[true_idx, ret_idx] += 1
            except Exception:
                pass

            if verbose and (i + 1) % 20 == 0:
                print(f"  Confusion matrix: {i + 1}/{len(sim_results)} done")

        return matrix, type_names


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from atmosphere_templates import default_wavelength_grid, build_earth_like_template
    from observation_sim import ObservationSimulator, PlanetSystem
    from instrument_model import load_jwst_nirspec

    print("Building template grid for retrieval demo...")
    grid = TemplateGrid()
    grid.build_grid(
        planet_radius_re=0.92,
        star_radius_rs=0.1192,
        cloud_fractions=[0.0, 0.3, 0.6],
        scale_heights_km=[7.0, 8.5, 10.0],
    )

    retrieval = TemplateRetrieval(grid)
    jwst = load_jwst_nirspec()
    sim = ObservationSimulator(jwst, rng=np.random.default_rng(42))
    planet = PlanetSystem.trappist1e()

    wl = default_wavelength_grid()
    true_tmpl = build_earth_like_template(
        wl,
        cloud_fraction=0.5,
        o2_ch4_ratio=1.0,
        scale_height_km=8.5,
        planet_radius_re=0.92,
        star_radius_rs=0.1192,
    )

    print("Simulating observation (10 transits)...")
    obs = sim.simulate(planet, true_tmpl, n_transits=10)

    print("Running retrieval...")
    result = retrieval.fit(obs.wavelengths_um, obs.observed_depth_ppm, obs.noise_ppm)
    print(result.summary())
    print(f"\nTrue atmosphere: {obs.atmosphere_type}")
    print(f"Retrieved:       {result.preferred_atmosphere}")
    print(f"Correct: {'YES' if obs.atmosphere_type == result.preferred_atmosphere else 'NO'}")
