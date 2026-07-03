Biosignature Project, in simple terms.

The Big Idea
Does an exoplanet (defined by a planet outside of our solar system) have life on it? JWST launched in 2021 is collecting light from planets that orbit other stars. Some of these planets, including TRAPPIST-1e, LHS 1140b, K2-18b, orbit in what’s known as the habitable zone (HZ) of their respective planets. The HZ zone is defined by a planet having liquid water in their atmosphere, which is a foundational criteria of life as we know it. The question this is trying to answer is simple: can JWST tell if any of these planets have a living biosphere?

This project uses quantitative parameters to answer this question for specific, promising planets. 

Detection Methods
When a planet passes through its sun, it blocks a tiny fraction of the starlight in what’s known as a transit event. The amount it blocks depends on wavelength. Different molecules in the atmosphere absorb different colors of light: oxygen absorbs at 0.762 micrometers, water at 1.38 micrometers, methane at 2.3 micrometers, CO2 at 4.3 micrometers. 

Plotting the amount of light blocked against wavelength gives you a bumpy line that functions as an atmospheric fingerprint: by comparing these signals to molecular signatures, you can figure out what’s in the atmosphere, in a technique known as transmission spectroscopy. JWST NIRSpec instrument does this by covering wavelengths from 0.6 to 5.3 micrometers; broad enough to cover signatures from all key molecules. 

Oxygen and Methane 
Oxygen and methane react with each other destructively, destroying one another in the process. Thus, if there’s no outside source to replenish their concentrations, we don’t expect to see the two molecules together. Conversely, when these two compounds exist together, that indicates that something is replenishing both the oxygen and methane concentrations. On Earth, that process is done through photosynthesis: plants create sugar and expel oxygen, while microbes expel methane. No known geological process produces both at detectable concentrations simultaneously. 

Finding oxygen gas and methane together is called a chemical equilibrium biosignature, and it’s among the strongest evidence for life. 

Five Atmospheres
There are five main atmospheres to compare observations against. 
Earth-like: Contains oxygen gas, water vapor, carbon dioxide, ozone, and trace amounts of methane. This is the optimal case, with an atmosphere that looks traditionally “alive”. 
Venus-analog: Dominated by high CO2 absorption. Little to no water vapor. Dead, barren, and uninhabitable—much like our very own Venus. Crucially, the presence of CO2 makes this type of planet a dangerous false positive indicator, with CO2 itself being a biosignature. 
Archean/High CH4: Much like a primordial Earth, this type of planet contains high amounts of methane, but barely any oxygen. Often, the levels of methane that are present in the atmospheres of these planets are at high enough levels where geology alone can’t sustain it—leading to questions of habitability. 
Hycean: Ocean-covered sub-Neptune with a hydrogen-dominated atmosphere, modeled after K2-18b—a planet JWST detected methane and carbon dioxide in during 2023. Hydrogen atmospheres are light, and puff out much higher, making spectral features often five to ten times larger than rocky planets; much easier to detect.
Abiotic-O2: Oxygen gas can build up abiotically through CO2/H2O photolysis, leading to a weak, misleading biosignature case. This is a point of contention for many astrobiologists and serves as the primary false positive adjunct of the project. 

Each atmosphere is parametrized by a number of variable conditions. 

Cloud Fraction: How much cloud mutes features of the planet 
Scale Height: “Puffiness” of the atmosphere, a feature of the gravity and temperature of the planet 
O2/CH4 Ratio: Cleanest biosignature and important indicator

This program precomputes 108 variations of each type of variable. 

JWST Squints
It’s important to realistically model JWST itself before beginning any sort of detection. This is a tricky process with many moving parts. For example, light gets lost in mirrors. The detector adds random noise. A star’s brightness fluctuates. 

Thus, the simulated telescope has four distinct noise sources:
Shot Noise: Photons arrive randomly. All stars, regardless of stability, produce slightly different counts of photons by pure chance. Noise = sqrt(number of photons)
Read Noise: Each time a detector reads out, electronics add a small bit of random error. It’s roughly 15 electrons for NIRSPEC
Dark Current: In complete darkness, the detector generates fake signals even in complete darkness from thermal electrons. 
Stellar Noise Floor: A 20-parts-per-million limit from thermal drifts, pointing jitter, and detector imperfections that no amount of telescope time can properly negate. 

Something I discovered as the project and coding went on is that regarding stars as perfect blackbodies (an idealized object that absorbs all electromagnetic light) overestimates stellar brightness at wavelengths above 2 nanometers. In reality, M-dwarf stars have molecular absorption in their own atmosphere that suppresses emitted light. After cross-checking with the official JWST Exposure Time Calculator and deriving correction factors—called SED calibration—the photon rates aligned back with reality. 

Simulating JWST’s Eyes 
Now, combine it together! This simulator takes a planet, an atmosphere, and the telescope model to produce a realistic simulated observation—similar to a noisy spectrum that JWST would return after watching a planet transit “N” amount of times. 

Specifically:
Computes how many photons per second the star delivers to the detector at each wavelength
Calculates total observation time (transit duration * number of transits)
Runs the noise model to get noise in parts per million per wavelength bin
Draws a random noise realization and adds it to the true spectrum 
Computes a signal-to-noise ratio (SNR)

This SNR calculation is important. Using the corrected version that only accounts for the spectral modulation (bumps above flat baseline) divided by noise, rather than absolute transit depth, avoids including a flat component that carries no atmosphere information and inflates SNR by order of magnitude. 

Run it Back 
If the observation simulator asks “given this atmosphere, what’s the spectrum?” then the retrieval asks “given the spectrum, what atmosphere do we expect?” 

The program compares observed spectrum against every template in the 108 template grid for each atmosphere type. For each template, it computes a chi-squared model—the sum of squared differences between observation and model, divided by the noise squared. Smaller chi-squared means better fit, and vice versa. Whichever model has the best chi-squared value wins. 

After finding the best fitting template for each atmosphere type, the program uses a Bayesian Information Criterion to rank the five types against each other. It converts scores into probability weights while keeping in mind how probable each atmospheric type is given what was observed.

Bayesian Nested Sampling 
For the standard case—an Earth-twin at 10 parsecs—the program runs a more power analysis using a package called dynesty. This package explores the full continuous parameter space and computes the Bayesian evidence for each atmosphere type. 

The evidence is a single number that represents how well a model explains the data when averaged over all possible parameter values. Comparing the two pieces of evidence gives the Bayes factor, which is how many times more likely model A is compared to model B.

This program gave a Bayes factor of +170 when comparing Earth_like vs Venus-analog evidence. For reference, “very strong evidence” is anything above 5, giving solid validation for a 5σ detection threshold used throughout the project. 
False Positive Framework 
False positives are how often a pipeline gets the wrong answer by doing injection/recovery testing. First, inject a known atmospheric signal—Venus-analog, 60% cloud cover, at a 12 km scale height—and combine realistic noise. Run the retrieval: what does the pipeline say? If it says anything besides Venus-analog, then that’s an example of a false positive. 

After running 405 trials across all atmospheric types, cloud cover ranges, scale height, and transits counts, the results are compiled into a confusion matrix: a grid showing how often each atmospheric type gets mistaken for each other type. The Venus-to-Earth false positive rate was 2.6%—for reference, a random guess yields a roughly 33% of accuracy. So, the pipeline improves accuracy by more than 11-fold. 

Monte Carlo Population Analysis 
Statistical claims require large, diverse representative populations. This project generates roughly 2,000 synthetic planets by drawing from literature-based distributions. For example, start temperatures are weighted toward M-dwarfs (matching the stellar census found in Winters et al. 2019); planet radii follow the Fulton gap bimodal distribution (Fulton et al. 2017). In addition, each planet is randomly assigned atmosphere type, cloud fraction, and scale height. The simulated JWST observations occur at 5, 10, and 20 transits utilizing 4 parallel CPU workers. 

Earth-like: 100% habitable zone detection, 96.6% within 20 pc, median SNR 33σ at 10 transits. 
High CO2: 100% HZ detection, 100% within 20 pc, median SNR 44σ 
Reduced O2/High CH4: 100% across all categories, median SNRσ 
Hycean: 100% across all categories, median SNR 439σ 

Real Target Forecasts
The pipeline also ran against all 11 confirmed transiting habitable zone candidates using the parameters from your real target list.
TRAPPIST-1e: 54σ at 5 transits, 70σ at 10 transits, 86σ at 20 transits
LHS 1140b: 56σ at 5 transits, 62σ at 10 transits. 66σ at 20 transits
K2-18b: 386σ at 5 transits (hycean atmosphere, huge features)
TOI-700d: 24σ at 5 transits, 29σ at 10 transits 

Every target shows transit count of at least 1—all 11 known HZ candidates are detectable in a single transit under nominal cloud assumptions. Under more pessimistic cloud coverage (80%) 9 out of 11 remain detectable within 20 transits. 

In addition, the pipeline is validated against two published JWST results. For K2-18b, the retrieval correctly identified the hycean atmosphere, consistent with Madhusudhan et al. 2023. For TRAPPIST-1b, the retrieval correctly returns a null result, consistent with Lustig-Yeager et al. 2023.

The Sensitivity Analysis 
Sensitivity analysis involves stress testing assumptions. For this project, I systematically varied four parameters:
Cloud Fraction: Detection remains at 100% for up to 60% cloud cover; it falls to about 40% at 95% cloud coverage. At 80% coverage, which is considered the pessimistic scenario, detection probability is still around 80% in transits. 
Noise Floor: Detection is flat from 10 to 30 parts per million, which is the full NIRSpec specification range. This only downgrades above 50 ppm, which is outside the expected range. 
Scale Height: This relies heavily on scale height being below 7 km—high gravity planets with thin atmospheres, which therefore have higher scale heights, are intrinsically harder to identify. 
Transit Count: Sharp transition from marginal to confident detection between 5 and 10 transits for normal clouds. 

Finally, the bias check ran 360 injection/recovery trials across the full parameter space and ound mean recovery rate of 100%. 

Bonus Add-Ons
Confusion Matrix: A visual, 3x3 grid of retrieval accuracy for the Venus-to-Earth false positive cell. 
Detection Horizon: SNR vs distance for all five atmosphere types; real target positions are marked. 
Cloud Threshold: For each of the five priority targets, finding the exact cloud coverage at which detection breaks below 5σ. 

Codebase & Literature 
10 python source files
10 Jupyter notebooks
4 config files
Real JWST spectra data from two published papers
A PHOENIX stellar model comparison 
24+ Figures

These files are available at github.com/tiramoose/biosignature-detection

In addition, all of the numbers are based on 19 papers across biosignature theory, JWST instrument characterization, stellar population statistics, planet occurrence rates, and statistical methods. They can be found in docs/prior_sources.md. 


 



