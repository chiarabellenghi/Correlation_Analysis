----------------------------------------
Introduction
----------------------------------------
IceCube has performed a search for point-like sources of neutrinos using seven years
of IceCube data, supplanting the previously available four year analysis. The new sample
includes previously analyzed data from 2008-2012, but adds 2012-2015 data as well as
contributions from track-like events starting inside of the detector, improving the
sensitivity to sources in the southern hemisphere. This data release provides tables
of the effective area and point spread functions for these additional data the 2012-2015
through-going track events and for the 2010-2015 starting track events as well as the
pre-trials local p-values for each point on the sky for the augmented complete sample.

No statistically significant excess was observed when searching the full sky. Further
searches checking for clustering of hotspots and investigating two source catalogs also
returned no significant observations. A detailed description of the sample and the methods
used to create the p-value map can be found in the corresponding paper:

"All-sky Search for Time-integrated Neutrino Emission from Astrophysical Sources with
7 yr of IceCube Data", IceCube Collaboration: M.G. Aartsen et al. Published in The
Astrophysical Journal, 835 (2017) 151, doi: 10.3847/1538-4357/835/2/151


----------------------------------------
Dataset citation
----------------------------------------
Suggested citation for this dataset:
IceCube Collaboration (2019): All-sky search for point sources with seven years of IceCube
data. IceCube Neutrino Observatory. Dataset. DOI: 


----------------------------------------
Data files
----------------------------------------
IC2012-2015_PS_effA.dat - Tabulated total effective areas as a function of neutrino
energy and declination for the 2012-2015 years.

IC2012-2015_PS_PSF.dat - Measurements of the median angular resolution for simulated
through-going muon neutrinos and antineutrinos events as a function of neutrino energy. 

startingEvents_effA.dat - Tabulated total effective areas for the starting track sample
added to the Southern sky (declination < 0). The effective area is given for the full
hemisphere, since the subsample is largely declination-independent.

startingEvents_PSF.dat - The median angular resolution for simulated starting muon
neutrino and antineutrino events as a function of neutrino energy.

effective_areas.pdf - Neutrino effective areas shown graphically using the information
in this data release.

angular_resolutions.pdf - Neutrino angular resolutions shown graphically using information
contained in this data release

7yrPS_log10p_maps.fits - A FITS file containing the pre-trial log10(p) for each tested
point on the sky using through-going tracks from 2008-2015 and starting tracks from
2010-2015. The map is an nside=512 HEALPix map produced and saved with the healpy package.
The values can be loaded using the healpy.read_map function. 


----------------------------------------
For any questions about this data release, please write to data@icecube.wisc.edu
----------------------------------------

----------------------------------------
Last Updated: 6 February, 2020
----------------------------------------


