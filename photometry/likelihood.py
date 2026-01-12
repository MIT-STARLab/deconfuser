#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: S. N. Hasler

Likelihood function for use in photometry deconfusion

"""
import numpy as np
import deconfuser.sample_planets as sample_planets
import photometry.photometry as phot

def likelihood(parameter, observed_sample, Detector, nbins=20): 
    '''
    Function to calculate likelihood of observed sample given the parameter value. 
    https://en.wikipedia.org/wiki/Likelihood_function#Definition
        
    Parameters
    ----------
        parameter : float 
            parameter of the likelihood function
        observed_sample : float
            observed sample point
        nbins : int
            number of bins for histogram / estimated pdf
        Detector : Detector object with parameters including t and noise_distribution method
    Returns
    -------
        L : float
            likelihood of parameter given observed sample
    '''
    dist = np.asarray(Detector.noise_distribution(parameter)) # generate distribution of possible options
    hist = np.histogram(dist, nbins)                          # histogram of distribution
    bin_edges = hist[1][:-1]                                  # bin edges of histogram [e- counts]
    # normalize distribution to get estimated pdf
    pdf = hist[0] / float(len(dist)) 

    # -------------- Remove dependence on bin size --------------
    bin_width = np.diff(hist[1])                        # [e-]
    updated_pdf = pdf / bin_width
    
    # Find where the value falls within the distribution
    diff_arr = np.absolute(bin_edges - observed_sample) # find nearest location to noisy count
    id_nearest = diff_arr.argmin()                      # get index of nearest value
    L = updated_pdf[id_nearest]                         # L of observed_sample
    L = L * bin_width[0]                                # rescale likelihood to bin width of histogram  

    return L

# Compute likelihood of one orbit in a confused system option
def get_L_orbit(n_detections, a, e, i, o, O, M0, ts, noisy_counts, Star, Planet, Detector):
    '''
    Function to calculate the likelihood of a single planet's orbit in a system.

    Parameters
    ----------
    n_detections : int
        Number of detections on the system (equivalent to n_epochs in test_deconfuser scripts).
    a : float
        Planet-star separation [AU].
    e : float
        Eccentricity of orbit option.
    i : float
        Inclination of orbit option [rad].
    o : float
        Argument of periapsis for orbit option [rad].
    O : float
        Argument of ascending node of orbit option [rad].
    M0 : float
        Mean anomaly of orbti option [rad].
    ts : numpy.ndarray
        Array of detection times [years].
    noisy_counts : np.ndarray
        Noisy detections of simulated or detected system [e-]
    Star : # TODO: finish docstring
    Planet : 
    Detector : 

    Returns
    -------
    L_orbit : float
        Likelihood of orbit option.
    L_detections : np.ndarray
        Likelihood of each detection.
        
    '''
    L_detections_orbit = np.zeros((n_detections))
    
    #  Get detection coordinates 
    xs, ys, zs = sample_planets.get_observations(a, e, i, o, O, M0, ts, Star.mu.value)

    # Calculate phase and intensity information   
    phases, phase_func, fpfs, photon_rates = phot.get_planet_count_rate(Planet, Star, Detector, 
                                                                              xs=xs, ys=ys, zs=zs)
    
    # For all detections, calculate likelihood
    for detection in range(n_detections):
        rate = photon_rates[detection]                  # get calculated photon rate of each detection accounting for integration time
        noisy = noisy_counts[detection]                 # get matching noisy detection
        L_detections_orbit[detection] = likelihood(rate, noisy, Detector) # calculate L of detection 
        
    L_orbit = np.prod(L_detections_orbit) # L of orbit option
    
    return L_orbit, L_detections_orbit
