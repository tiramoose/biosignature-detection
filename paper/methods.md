# Methods

## 2.1 Synthetic Planet Population

We construct a Monte Carlo ensemble of N = 1,000 hypothetical rocky exoplanets
by drawing system parameters from prior distributions stored in a configuration
file (config/priors.yaml). Planet radii are drawn uniformly in [0.5, 2.5] R⊕,
capturing the super-Earth and sub-Neptune regime most relevant to biosignature
searches. Distances are sampled uniform in volume out to 50 pc, reflecting the
approximate completeness limit of high-resolution spectroscopic surveys. Orbital
periods are drawn log-uniformly over [10, 400] days to span the habitable zone
across M through F spectral types. Host star types are assigned with occurrence
weights [0.50, 0.25, 0.15, 0.10] for M, K, G, F stars respectively, consistent
with the stellar census of the solar neighborhood. Geometric transit probabilities
are estimated analytically from stellar radii and semi-major axes via Kepler's
third law. This toy population will be replaced with Kepler/TESS occurrence
rates in Weeks 3–4.
