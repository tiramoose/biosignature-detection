def _run_one_trial(
    planet: SyntheticPlanet,
    atmosphere_name: str,
    cloud_fraction: float,
    scale_height_km: float,
    n_transits: int,
    seed: int,
    *,
    instrument=None,
    instrument_name: str = "jwst_nirspec",
) -> MCTrial:
    """
    Simulate one planet × atmosphere × n_transits combination.
    """
    wl = default_wavelength_grid()
    builder, o2_ch4 = _BUILDERS[atmosphere_name]

    # For hycean, use planet-appropriate radii
    if atmosphere_name == "hycean":
        p_re = max(planet.planet_radius_re, 1.8)
        s_rs = max(planet.star_radius_rs, 0.35)
    else:
        p_re = planet.planet_radius_re
        s_rs = planet.star_radius_rs

    template = builder(
        wl,
        cloud_fraction=cloud_fraction,
        o2_ch4_ratio=o2_ch4,
        scale_height_km=scale_height_km,
        planet_radius_re=p_re,
        star_radius_rs=s_rs,
    )

    ps = PlanetSystem(
        planet_name=f"planet_{planet.planet_id}",
        star_teff_k=planet.star_teff_k,
        star_radius_rs=s_rs,
        star_magnitude_j=planet.star_magnitude_j,
        planet_radius_re=p_re,
        orbital_period_days=planet.orbital_period_days,
        transit_duration_hours=planet.transit_duration_hours,
        distance_pc=planet.distance_pc,
        equilibrium_temp_k=planet.equilibrium_temp_k,
    )

    if instrument is None:
        instrument = (
            load_miri_lrs()
            if instrument_name == "miri_lrs"
            else load_jwst_nirspec()
        )

    sim = ObservationSimulator(
        instrument=instrument,
        rng=np.random.default_rng(seed),
    )

    obs = sim.simulate(
        planet=ps,
        template=template,
        n_transits=n_transits,
    )

    # Corrected SNR: spectral modulation / noise
    baseline = template.parameters["base_depth_ppm"]
    modulation = np.abs(obs.true_depth_ppm - baseline)

    mask = (obs.noise_ppm > 0) & (obs.noise_ppm < 5000)

    snr_bins = np.where(
        mask,
        modulation / (obs.noise_ppm + 1e-9),
        0.0,
    )

    corrected_snr = float(np.sqrt(np.nansum(snr_bins ** 2)))

    noise_floor = (
        float(np.median(obs.noise_ppm[mask]))
        if mask.any()
        else 1e6
    )

    return MCTrial(
        planet_id=planet.planet_id,
        star_type=planet.star_type,
        star_teff_k=planet.star_teff_k,
        star_magnitude_j=planet.star_magnitude_j,
        planet_radius_re=p_re,
        distance_pc=planet.distance_pc,
        equilibrium_temp_k=planet.equilibrium_temp_k,
        in_habitable_zone=planet.in_habitable_zone,
        atmosphere_type=atmosphere_name,
        cloud_fraction=cloud_fraction,
        scale_height_km=scale_height_km,
        o2_ch4_ratio=o2_ch4,
        n_transits=n_transits,
        broadband_snr=corrected_snr,
        is_detected=(corrected_snr >= 5.0),
        noise_floor_ppm=noise_floor,
        transit_duration_h=planet.transit_duration_hours,
    )
