import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, (center - half) / denom), min(1.0, (center + half) / denom))


def bootstrap_ci(data, statistic=np.median, n_boot: int = 5000,
                  alpha: float = 0.05, rng=None) -> tuple:
    """Percentile bootstrap CI for any summary statistic (median SNR, etc.)."""
    rng = rng or np.random.default_rng()
    data = np.asarray(data)
    idx = rng.integers(0, len(data), size=(n_boot, len(data)))
    samples = data[idx]
    boot_stats = statistic(samples, axis=1)
    lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(boot_stats)), float(lo), float(hi)


# -------------------------------------------------------------------

def sidak_equivalent_sigma(target_sigma: float, m_tests: int) -> float:
    alpha_family = 2 * stats.norm.sf(target_sigma)
    alpha_per_test = 1 - (1 - alpha_family) ** (1 / m_tests)
    return float(stats.norm.isf(alpha_per_test / 2))


def benjamini_hochberg(pvals, alpha: float = 0.05) -> np.ndarray:
    pvals = np.asarray(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(m, dtype=bool)
    kmax = np.max(np.where(passed))
    sig = np.zeros(m, dtype=bool)
    sig[order[:kmax + 1]] = True
    return sig


--------------------------------------------------------

def empirical_null_delta_chi2(grid, instrument, planet, baseline_ppm,
                               n_transits: int = 10, n_trials: int = 300,
                               rng=None) -> np.ndarray:
    from atmosphere_templates import AtmosphereTemplate
    from retrieval import TemplateRetrieval

    rng = rng or np.random.default_rng(0)
    wl = grid.wavelengths
    flat_tmpl = AtmosphereTemplate(
        name="flat_null", description="pure noise null",
        wavelengths_um=wl, transit_depth_ppm=np.full_like(wl, baseline_ppm),
        parameters={"base_depth_ppm": baseline_ppm},
    )
    sim_cls = instrument.__class__  # instrument is already an InstrumentModel instance
    from observation_sim import ObservationSimulator
    sim = ObservationSimulator(instrument, rng=rng)
    retr = TemplateRetrieval(grid)

    null_delta_chi2 = np.empty(n_trials)
    for i in range(n_trials):
        obs = sim.simulate(planet, flat_tmpl, n_transits=n_transits)
        result = retr.fit(obs.wavelengths_um, obs.observed_depth_ppm, obs.noise_ppm)
        null_delta_chi2[i] = result.delta_chi2_flat
    return null_delta_chi2


def empirical_sigma(observed_delta_chi2: float, null_delta_chi2: np.ndarray) -> tuple:
    n = len(null_delta_chi2)
    p_empirical = np.mean(null_delta_chi2 >= observed_delta_chi2)
    p_empirical = max(p_empirical, 1.0 / (n + 1))   # can't claim finer than 1/(n+1) with n trials
    return p_empirical, float(stats.norm.isf(p_empirical / 2))


# ---------------------------------------------------------------------------

def roc_auc_by_atmosphere(trials, atmosphere_types) -> dict:
    from sklearn.metrics import roc_curve, auc
    out = {}
    for atm in atmosphere_types:
        y_true = np.array([1 if t.true_atmosphere == atm else 0 for t in trials])
        y_score = np.array([t.detection_snr for t in trials])
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            out[atm] = float("nan")   # AUC undefined with only one class present
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        out[atm] = float(auc(fpr, tpr))
    return out


# ---------------------------------------------------------------------------

def residual_whiteness_check(residuals: np.ndarray, sigma: np.ndarray) -> dict:
    z = residuals / sigma
    ks_stat, ks_p = stats.kstest(z, "norm")
    lag1 = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    return {"ks_stat": float(ks_stat), "ks_p": float(ks_p), "lag1_autocorr": lag1}
