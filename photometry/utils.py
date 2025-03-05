# S. Hasler
# Script to house helper functions for deconfusion work

import numpy as np
import astropy.units as u
from astropy.constants import G, M_sun
import pandas as pd

def parallax_as(d, a=1.0):
    '''
    Calculate parallax given a system distance (d) in pc
    and the observer's distance from the Sun in AU

    Parameters
    ----------
    d : float
        System distance in parsecs
    a : float, optional
        Observer's distance from Sun in AU, by default 1.0

    Returns
    -------
    a / d : float
        parallax in arcseconds
    '''
    return a / d 

def get_E(a, e, M0, ts, mu=4*np.pi**2, tol=1e-10):
    '''
    Calculate eccentric anomaly (E) at each time of detection

    Parameters
    ----------
    a : float
        semi-major axis
    e : float
        eccentricity
    M0 : float
        mean anomaly
    ts : np.ndarray
        Times of observation in fractions of years
    mu : _type_, float
        gravitational parameter in units consistent with a and t, by default 4*np.pi**2 [AU^3/year^2]
    tol : float, optional
        tolerance of the solution of Kepler's equation, by default 1e-10

    Returns
    -------
    np.ndarray
        Eccentric anomaly at each time of detection
    '''
    # Reshape times for following calculations
    t = np.reshape([ts], (1,-1))

    #initial guess for eccentric anomaly is mean anomaly
    E = M = np.sqrt(mu/a**3)*t + M0

    #Newton's method for solving Keplers equation
    while True:
        E_update = (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
        E = E - E_update
        if np.max(np.abs(E_update)) < tol: break
    return E


def get_nu(E, e):
    '''
    Calculate the true anomaly from eccentric anomaly and eccentricity

    Parameters
    ----------
    E : float or np.ndarray
        Eccentric anomaly [rad]
    e : float
        eccentricity [0, 1]

    Returns
    -------
    float
        true anomaly [rad]
        
    '''
    nu = 2 * np.arctan( np.sqrt( (1 + e) / (1 - e) ) * np.tan(E / 2) )

    return nu


def delta_RA_dec(a, e, E, i, nu, omega_p, Omega, plx):
    '''
    Calcualte relative right ascension (RA) and declination (dec) for 
    a given planet detection

    Based on orbitize solutions

    Parameters
    ----------
    a : float
        Planet-star separation [au]
    e : float
        eccentricity [0, 1]
    E : float
        eccentric anomaly [rad]
    i : float
        Inclination [rad]
    nu : float
        true anomaly [rad]
    omega_p : float
        argument of periastron [rad]
    Omega : float
        longitude of ascending node [rad]
    plx : float
        parallax [mas]

    Returns
    -------
    delta_RA, delta_dec
        Relative RA and declination of planet detection
    '''

    radius = a * ( 1.0 - e * np.cos(E) ) # orbital radius

    delta_RA = radius * ( (np.cos(i/2))**2 * np.sin(nu + omega_p + Omega) - (np.sin(i/2))**2 * np.sin(nu + omega_p - Omega) ) * plx
    delta_dec = radius * ( (np.cos(i/2))**2 * np.cos(nu + omega_p + Omega) + (np.sin(i/2))**2 * np.cos(nu + omega_p - Omega) ) * plx

    return delta_RA, delta_dec

def convert_AU_to_arcsec(planet_separation_AU, system_distance_pc):
    '''
    Convert planet separation in AU to arcseconds
    
    '''
    planet_separation_AU *= u.AU 
    system_distance_pc *= u.pc
    planet_sep_km = planet_separation_AU.to(u.km)
    system_dist_km = system_distance_pc.to(u.km)

    theta = 206265 * (planet_sep_km / system_dist_km)

    return theta

def tau_from_M0(M0, a, M_tot, ts_mjd, mu=4*np.pi**2):
    '''
    Calculate tau for orbitize comparison from mean anomaly
    Essentially inverse of orbitize.kepler.tau_to_manom

    Parameters
    ----------
    M0 : list of floats
        List of mean anomalies for each planet
    a : list of floats
        List of semi-major axes for each planet
    ts_mjd : list of floats
        List of time of detection in MJD
    mu : float
        Stellar gravitational parameter [AU^3 / yr^2]
    '''

    # Calculate tau for comparison to orbitize

    tau_ref_epoch = ts_mjd[0] # set reference epoch

    # Calculate orbital period in days
    period = [np.sqrt( 4 * np.pi**2 * (sma * u.AU)**3 / (G * (M_tot * M_sun))).to(u.day).value for sma in a]
    
    # Calculate fractional date normalized to the orbital period
    frac_date = [(ts_mjd[i] - tau_ref_epoch) / period[i] for i in range(len(period))]
    frac_date = [date % 1 for date in frac_date]

    # Calculate tau from mean anomaly (and make sure it's normalied between 0-1)
    tau = [(frac_date[i] - M0[i]) / (2 * np.pi) for i in range(len(M0))]
    tau = [t % 1 for t in tau]

    return tau

def get_truth_from_ranked_df(single_system_ranked_df, ts_mjd=[59215.0, 59397.5, 59580.0], M_tot=1.0000090104680466):
    '''
    Returns a list of dicts with true simulated orbital parameters from each planet in a system
    from a single-system-dataframe
    For use in comparing orbitize fits to deconfuser fits.

    Parameters
    ----------
    single_system_ranked_df : pandas.DataFrame
        dataframe from ranked output, filtered for a single system and truth values
    mu : float, optional
        Stellar gravitational parameter for system in AU^3 / yr^2. Default for Sun = 4*pi^2
    ts_mjd : list of floats, optional
        Observation times in MJD, default = 3 observations spread equally over 1 year
    M_tot : float
        Total system mass in units of M_sun

    Returns
    -------
    list of dicts
        List of n_planet dicts, which includes the true orbital parameters of each simulated
        planet in the system.
    '''
    # Get truth parameters of simulated system
    a = pd.to_numeric(single_system_ranked_df['a_original1'], errors='coerce').to_list()
    e = pd.to_numeric(single_system_ranked_df['e_original1'], errors='coerce').to_list()
    i = pd.to_numeric(single_system_ranked_df['i_original1'], errors='coerce').to_list()
    o = pd.to_numeric(single_system_ranked_df['o_original1'], errors='coerce').to_list()
    O = pd.to_numeric(single_system_ranked_df['O_original1'], errors='coerce').to_list()
    M0 = pd.to_numeric(single_system_ranked_df['M0_original1'], errors='coerce').to_list()

    tau = tau_from_M0(M0, a, M_tot, ts_mjd)
    
    planet = pd.to_numeric(single_system_ranked_df['planet_original1'], errors='coerce').to_list()
    
    true_system_vals = []

    for j in range(len(a)):
        true_params = {}
        true_params['planet'] = planet[j]
        true_params['a'] = a[j]
        true_params['e'] = e[j]
        true_params['i'] = i[j]
        true_params['o'] = o[j]
        true_params['O'] = O[j]
        true_params['tau'] = tau[j] 

        true_system_vals.append(true_params)

    return true_system_vals

def get_top_deconf_from_ranked(single_system_ranked_df, system, ts_mjd=[59215.0, 59397.5, 59580.0], M_tot=1.0000090104680466):
    '''
    Get the top ranked orbitl parameters from the ranked dataframe.
    For use in comparing orbitize fits to deconfuser fits.

    Parameters
    ----------
    single_system_ranked_df : pandas.DataFrame
        dataframe from ranked output
    system : int
        System number that we're looking at
    M_tot : float
        Total system mass in units of M_sun
    '''
    
    # Get orbital parameters for top ranked option returned by deconfuser + photometry
    a_deconf = single_system_ranked_df[(single_system_ranked_df['top_ranked_partition1'] == True) & (single_system_ranked_df['system_original1'] == system)]['a_original1'].to_numpy()
    e_deconf = single_system_ranked_df[(single_system_ranked_df['top_ranked_partition1'] == True) & (single_system_ranked_df['system_original1'] == system)]['e_original1'].to_numpy()
    i_deconf = single_system_ranked_df[(single_system_ranked_df['top_ranked_partition1'] == True) & (single_system_ranked_df['system_original1'] == system)]['i_original1'].to_numpy()
    o_deconf = single_system_ranked_df[(single_system_ranked_df['top_ranked_partition1'] == True) & (single_system_ranked_df['system_original1'] == system)]['o_original1'].to_numpy()
    O_deconf = single_system_ranked_df[(single_system_ranked_df['top_ranked_partition1'] == True) & (single_system_ranked_df['system_original1'] == system)]['O_original1'].to_numpy()
    M0_deconf = single_system_ranked_df[(single_system_ranked_df['top_ranked_partition1'] == True) & (single_system_ranked_df['system_original1'] == system)]['M0_original1'].to_numpy()
    planet = single_system_ranked_df[(single_system_ranked_df['top_ranked_partition1'] == True) & (single_system_ranked_df['system_original1'] == system)]['planet_original1'].to_numpy()

    a_deconf = [eval(x)[0] for x in a_deconf]
    e_deconf = [eval(x)[0] for x in e_deconf]
    i_deconf = [np.rad2deg(eval(x)[0]) for x in i_deconf]
    o_deconf = [np.rad2deg(eval(x)[0]) for x in o_deconf]
    O_deconf = [np.rad2deg(eval(x)[0]) for x in O_deconf]
    M0_deconf = [np.rad2deg(eval(x)[0]) for x in M0_deconf]
    tau_deconf = tau_from_M0(M0_deconf, a_deconf, M_tot, ts_mjd)

    deconf_vals = []

    for j in range(len(a_deconf)):
        deconf_params = {}
        deconf_params['planet'] = planet[j]
        deconf_params['a'] = a_deconf[j]
        deconf_params['e'] = e_deconf[j]
        deconf_params['i'] = i_deconf[j]
        deconf_params['o'] = o_deconf[j]
        deconf_params['O'] = O_deconf[j]
        deconf_params['tau'] = tau_deconf[j]  

        deconf_vals.append(deconf_params)

    for planet in range(len(deconf_vals)):
        if deconf_vals[planet]['i'] > 90: # set i range between 0 - 90
            deconf_vals[planet]['i'] = 180 - deconf_vals[planet]['i'] 
        if deconf_vals[planet]['o'] < 0: # change negative angles to positive
            deconf_vals[planet]['o'] = deconf_vals[planet]['o'] + 360
        if deconf_vals[planet]['O'] < 0: #  change negative angles to positive
            deconf_vals[planet]['O'] = deconf_vals[planet]['O'] + 360

    return deconf_vals