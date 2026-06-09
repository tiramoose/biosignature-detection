# Methods

## Week 2 

# synth_population.py 
generates fake catalog of thousands of planets by drawing on random numbers form realistic distributions. Factors include:
1. Star Size
2. Temperature
3. Distance
4. Planet Size

# atmosphere_templates.py 
each planet recieves three atmospheric qualities. 
1. Earth-like: has oxygen, water vapoer, C02, ozone. biosignature case.
2. High C02: dominated by C02, minimal-to-no oxygen or water. flase positive.
3. Reduced oxygen, high methane: archean sample from early planets. semi-biosignature case.
each gets differing cloud covers, scale heights, chemical ratios. produces 108 versions.
update 1: added hycean world (modeled after K2-18b) 

# instrument_model.py 
models telescope. given a star calculates
1. how many photos JWST collects at each wavelength
2. noise of the measure --> photon shot noise, detector read noise, thermal dark current, 20 parts-per-million stability floor
update 1: added second instrument that covers near-infared and mid-infared light. 

# jwst_nirspec.json & elt_example.json 
stores hardware data 

# observation_sim.py
when provided a planet, atmosphere, telescope calculate noise of observation, noise spectrum, and whether signal is observable. sweeps over distance. 

## Week 3 

# config/priors.yaml 
replaced planet factory with actual published stats 
1. Star temperatures follow M-dwarf-heavy distribution (Winters et al. 2019)
2. Planet radii reflect Fulton gap
3. Roughly 1 in 4 M-Dwarf stars have a rocky planet in habitable zone (Dressing & Charbonneau 2015)

# docs/prior_sources.md
citations for numbers 

# data/target_list.csv 
list of habitable zone planets to run pipeline against. have information including star brightness, planet size, distance, equilibrium, temperature, etc. 

updated atmosphere & instruments 

## Week 4 

# retrival.py
inference engine incorporating chi-squared matching and bayesion model weights for asessing atsmopheric fit 

# false_positive.py 
asessing false positive rate systematically. 
1. inject known atmosphere
2. add realistic noise from the instrument model
3. run retrival
4. check validity
5. repeat

# data/real_spectra/
1. k2_18b_nirspec.csv
actual k2-18b nirspec transmission spectrum from Madhusudhan et al (2023)
2. trappist1b_nirspec.csv
null-result control from lustig-yeageral et al. (2023) research


