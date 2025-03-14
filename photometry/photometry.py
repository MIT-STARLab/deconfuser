'''
@author: S. N. Hasler

Photometry functions for use with deconfuser.
'''
import numpy as np
import astropy.constants as const
import astropy.units as u

class Star:
    def __init__(self, T, R_star, d_system, mu):
        '''
        Holds system's star parameters.

        Parameters
        ----------
        T : float
            Stellar effective temperature [units: K]
        R_star : float
            Stellar radius [units: m]
        d_system : float
            Distance of system from observer [units: parsecs]
        mu : float
            Stellar gravitational parameter in units of AU^3 / yr^2 (for consistency with deconfuser)
        '''
        self.T = T * u.K
        self.R_star = R_star * u.m 
        self.d_system = d_system * u.parsec
        self.mu = mu * (u.AU**3 / u.yr**2)

    def blackbody_spec(self, wavelength):
        '''
        Calculate blackbody spectrum value of star at given wavelength. 

        Parameters
        ----------
        wavelength : numpy.ndarray or numpy.float64
            Wavelength or wavelength range to consider [m].

        Returns
        -------
        B_lambda_star : numpy.ndarray or numpy.float64
            Blackbody spectrum value of star at wavelength of interest.

        '''
        wavelength *= u.m   # add units to wavelength
        h = const.h         # Planck's constant
        c = const.c         # speed of light
        k = const.k_B       # boltzmann constant
        
        B_lambda_star = ( (2 * h * c**2) / wavelength**5) * \
                    (1 / ( np.exp((h*c) / (wavelength * k * self.T)) - 1 ) ) / u.sr # blackbody spectrum of star [units: J / s / m^3 / sr] 

        return B_lambda_star
    
    def stellar_flux(self, B_lambda_star):
        '''
        Calculate flux contribution from star. 

        Parameters
        ----------
        B_lambda_star : numpy.ndarray or numpy.float64
            Blackbody spectrum of star. Generated with blackbody_spec()

        Returns
        -------
        F_star : numpy.ndarray or numpy.float64
            Stellar flux density.

        '''
        system_distance = self.d_system.to(u.m) # convert to meters
        F_star = (((np.pi * u.sr) * B_lambda_star * ( self.R_star / system_distance )**2)) 

        return F_star

class Planet:
    def __init__(self, R_p, Ag):
        '''
        Planet object 

        Parameters
        ----------
        R_p : float
            Planet's radius [m]
        Ag : float
            geometric albedo
        '''
        self.R_p = R_p * u.m
        self.Ag = Ag

    def choose_random_Rp(self, R_min, R_max, n_planets):
        '''
        Choose random planet radii from a uniform distribution
        Radii in units of R_Earth

        Parameters
        ----------
        R_min : float
            Minimum radius to sample from
        R_max : float
            Maximum radius to sample from
        n_planets : int
            Number of planets to sample radii for
        Returns 
        -------
        Rp : np.ndarray
            Array of length n_planets with planet radii in R_Earth
        '''
        Rp = np.random.uniform(R_min, R_max, n_planets)
        return Rp
    
    def choose_random_Ag(self, Ag_min, Ag_max, n_planets):
        '''
        Choose random geometric albedo values from a uniform distribution

        Parameters
        ----------
        Ag_min : float
            Minimum Ag value to sample from
        Ag_max : float
            Maximum Ag value to sample from
        n_planets : int
            Number of planets to sample Ag for
        '''
        Ag = np.random.uniform(Ag_min, Ag_max, n_planets)
        return Ag

class Detector:
    '''
    Class for the detecting instrument.
    '''
    def __init__(self, qe, cic, dark_current, read_noise, gain, fwc, conversion_gain, t,
                 D, throughput, f_pa, wavelength, bandwidth):
        '''
        Detector parameters

        Parameters
        ----------
        qe : float
            Quantum efficiency of detector. 
        cic : float
            Clock-induced charge of detector [e-]
        dark_current : float
            Detector dark current [e-]
        read_noise : int
            Read noise of detector [e-/sec]
        gain : int
            Gain of emccd detector
        fwc : int
            Fell-well capacity of detector
        conversion_gain : float
            Conversion value for e- to ADU
        t : int
            Integration time. [s]
        D : float
            Main aperture diameter [m]
        throughput : float
            Total throughput (generally wavelength dependent). For calculating planet photon rate at detector.
        f_pa : float
            Describes the fraction of light from the planet that falls within the photometric aperture (see Robinson+2016, Eqn 12)
            For calculating planet photon rate at detector.
        wavelength : float
            Wavelength of observation [m]
        bandwidth : float
            bandwidth of wavelength band [m]

        '''
        self.qe = qe
        self.cic = cic
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.gain = gain
        self.fwc = fwc
        self.conversion_gain = conversion_gain
        self.t = t
        self.D = D
        self.throughput = throughput
        self.f_pa = f_pa
        self.wavelength = wavelength
        self.bandwidth = bandwidth

    def add_noise(self, detected_rate):
        '''
        Function to add detector noise to the count rate that reaches the detector.

        Parameters
        ----------
        detected_rate : float
            Photon count rate from object reaching the detector. [photons/s]
        
        Returns
        -------
        numpy.ndarray
            Output electron count from detector 
        '''
        N = detected_rate * self.t # [photons]
        input_photons = np.array([N])

        # Add shot noise
        with_shot_noise = np.random.poisson(input_photons)
        expected_counts = with_shot_noise * self.qe # [e-]

        # Add dark current
        base_dark_current = self.dark_current * self.t + self.cic # [e-/pxl]
        expected_counts = expected_counts + base_dark_current

        # add gain for emccd -- reference: https://github.com/nasa-jpl/lowfssim/blob/a76d89e3e6c5286674da490492ccc59f5b754965/lowfsc/emccd.py#L201
        # and https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0053671#pone.0053671-Basden1 (Eqn 16)
        k = expected_counts
        zero_mask = k == 0 # temporarily remove 0 to avoid divide by 0 errors
        k[zero_mask] = 1
        theta = self.gain - 1 + 1/k # gamma dist. scale parameter
        count_add_gain = np.random.gamma(k, theta) + k - 1 # gamma dist. for gain
        count_add_gain[zero_mask] = 0 # set values back to 0

        # read noise
        rn = np.random.normal(0, self.read_noise)
        with_noise = count_add_gain + rn

        # if noise > fwc, set to fwc
        with_noise[with_noise > self.fwc] = self.fwc

        with_noise /= self.conversion_gain # [ADU]

        return with_noise
    
    def noise_distribution(self, detected_rate, dist_size=1e5):
        '''
        Function to generate noise distribution from input source count

        Parameters
        ----------
        detected_rate : float
            Photon count rate from object reaching the detector. [photons/s]
        dist_size : int
            Size of distribution to generate (default: 1e5)

        Returns
        -------
        with_noise : np.ndarray
            Output distribution of possible e- counts for input planet count rate
        
        '''
        N = detected_rate * self.t # pixel signal
        input_photons = np.array([N]) # convert to array for the following

        # Add shot noise
        with_shot_noise = np.random.poisson(input_photons, int(dist_size))
        expected_counts = with_shot_noise * self.qe # [e-]

        # Add dark current
        base_dark_current = self.dark_current * self.t + self.cic # [e-/pxl]
        expected_counts = expected_counts + base_dark_current 

        # Add gain for emccd
        k = expected_counts
        zero_mask = k == 0 # temporarily remove 0 to avoid didvide by zero errors
        k[zero_mask] = 1
        theta = self.gain - 1 + 1/k # gamma distribution scale parameter
        counts_add_gain = np.random.gamma(k, theta) + k - 1 # gamma distribution for gain
        counts_add_gain[zero_mask] = 0 # set values back

        # read noise
        read_noise_dist = np.random.normal(0, self.read_noise, int(dist_size))
        with_noise = counts_add_gain + read_noise_dist

        # if > fwc, set to fwc
        with_noise[with_noise > self.fwc] = self.fwc

        with_noise /= self.conversion_gain # [ADU]

        return with_noise

def get_planet_count_rate(Planet, Star, Detector, xs, ys, zs):
    '''
    Function to calculate planetary phase angle given x,y,z coordinates on-sky. 
    Calculates the planet-star flux ratio and converts planet flux density
    to planet photon count rate on the detector.

    Parameters
    ----------
    Planet : 
        Object with planet parameters (Ag, R_p)
    Star : 
        Object with stellar parameters (R_star, d_system)
    Detector : detector.Detector
        Detector object with all parameters defined.
    xs : numpy.ndarray
        x-values for planet location from deconfuser.sample_planets.
            Example: array([x_planet_1, x_planet_2, ..., x_planet_N])
    ys : numpy.ndarray
        y-values for planet location from deconfuser.sample_planets.
            Example: array([y_planet_1, y_planet_2, ..., y_planet_N])
    zs : numpy.ndarray
        z-values for planet location from deconfuser.sample_planets.
            Example: array([z_planet_1, z_planet_2, ..., z_planet_N])

    Returns
    -------
    phases : list of floats
        Phase angles at each planet location.
    phase_function : list of floats
        Value of phase function at each planet location.
    fpfs : list of floats
        Value of planet-star flux ratio at each planet location.
    planet_counts : list of floats
        Value of planet count rate at each planet location.

    '''
    # Set constants
    h = const.h # Planck's constant
    c = const.c # speed of light 
    
    # Set empty lists for appending later
    phases, phase_function, fpfs, separation, Fp, planet_counts = [], [], [], [], [], []
    
    # --------- Set constants ---------
    observer_distance_AU = Star.d_system.to(u.AU)  # units: AU
    d_system = Star.d_system.to(u.m)               # distance to system in meters

    # --------- Convert coordinates to orbital separation from star (star @ origin (0,0,0)) ---------
    for i in range(0,len(xs[0])):
        x_planet = xs[0][i] * u.AU                 # all coordinates from deconfuser are in units of AU
        y_planet = ys[0][i] * u.AU
        z_planet = zs[0][i] * u.AU
    
        separation.append(np.sqrt(x_planet**2 + y_planet**2 + z_planet**2)) # planet separation from star

    # --------- Calculate star values ---------
    B_lambda_star = Star.blackbody_spec(wavelength=Detector.wavelength) 
    F_star = Star.stellar_flux(B_lambda_star=B_lambda_star)             # stellar flux density

    # --------- Calculate phase angle, lambert phase function, flux ratio, planet count rate ---------
    for detection in range(0, len(xs[0])): # For each planet detection
        # Orbital phase angle
        planet_vector = (-xs[0][detection], -ys[0][detection], -zs[0][detection]) # planet vector = (0-x_planet, 0-y_planet, 0-z_planet)
        observer_vector = (0 - xs[0][detection], 0 - ys[0][detection], -observer_distance_AU.value - zs[0][detection])  # observer location = (0,0,-observer_distance) [AU], observer vector = (0 - x_planet, 0 - y_planet, -observer_distance - z_planet)
        planet_mag = np.linalg.norm(planet_vector) # get magnitude of vector
        obs_mag = np.linalg.norm(observer_vector)

        # --------- Phase angle ---------
        phase_angle = np.arccos(np.dot(planet_vector, observer_vector) / (planet_mag * obs_mag)) # calculate phase angle
        if xs[0][detection] < 0:
            phase_angle = phase_angle - 2*phase_angle # convert to negative angle for plotting whole orbit
        phases.append(np.degrees(phase_angle))

        # --------- Lambert phase function ---------
        lambert_phase = (np.sin(np.absolute(phase_angle)) + \
                            (np.pi - np.absolute(phase_angle)) * np.cos(np.absolute(phase_angle))) / np.pi
        phase_function.append(lambert_phase)
        
        # --------- Planet flux density ---------
        # See Robinson+2016
        F_planet = np.pi * Planet.Ag * lambert_phase * B_lambda_star*u.sr * (Star.R_star / (separation[detection].to(u.m)))**2 * (Planet.R_p / d_system)**2 
        Fp.append(F_planet.value)

         # --------- Flux ratio ---------
        flux_ratio = Planet.Ag * ((Planet.R_p / (separation[detection].to(u.m)))**2) * lambert_phase
        fpfs.append(flux_ratio.value)
    
        # --------- Convert to planet counts ---------
        c_p = np.pi * Detector.qe * Detector.f_pa * Detector.throughput * (Detector.wavelength*u.m / (h * c)) * F_planet * Detector.bandwidth*u.m * (Detector.D*u.m / 2)**2
        planet_counts.append(c_p.value)
        
    return phases, phase_function, fpfs, planet_counts 

def separate_xyzs(xyzs_array):
    '''
    Function to separate x, y, and z coordinates of planet detections
    
    Parameters
    ----------
    xyzs_array : list of lists 
        List of x-, y-, z-coordinate groupings
        
    Returns 
    -------
    xs, ys, zs : np.arrays of x, y, and z coordinate values
    
    '''
    xs = np.array([[xyzs_array[i][0] for i in range(0,len(xyzs_array))]])
    ys = np.array([[xyzs_array[i][1] for i in range(0,len(xyzs_array))]])
    zs = np.array([[xyzs_array[i][2] for i in range(0,len(xyzs_array))]])
    
    return xs, ys, zs

def get_detections_counts(n_planets, n_detections, xyzs, Planet, Star, Detector): # TODO: rename function to something like "simulate_noisy_detection" ?
    '''
    Generates noisy planet detections.
    Accepts detection coordinates, calculates phase/brightness, adds detector noise.

    Parameters
    ----------
    n_planets : int
        Number of planets in the system.
    n_detections : int
        Number of detections of the system.
    xyzs : numpy.ndarray
        Array of X, Y, Z coordinates for each detection. Format: 
            [[X1_1, Y1_1, Z1_1], [X2_1, Y2_1, Z2_1], ..., [XN_M, YN_M, ZN_M]], 
            where N is the number/time of detection and M is the number of 
            planet in the system.
    Planet : 
        Planet object containg information about planet (Ag, R_p)
    Star :
        Star object containing information about host star (R_star, distance)
    Detector : 
        Detector object containing detecting instrument parameters.
    wavelength : float, optional
        Wavelength of observation. 
    bandwidth : float, optional
        Bandwidth of observation. 

    Returns
    -------
    noisy_counts_sys : list
        "Simulated detections". Planet detections with detector noise added [e-].
    photon_rates_sys : list
        Calculated photon rates per planet detection [photons/s].

    '''
    
    noisy_counts_sys = []
    photon_rates_sys = []
    
    for planet in range(n_planets):
        # --------- Handle detection coordinates ----------
        xyzs_planet = xyzs[planet]
        xs, ys, zs = separate_xyzs(xyzs_planet) 
        
        # ----------- Calculate phase and intensity information -----------
        phases, phase_func, fpfs, photon_rates = get_planet_count_rate(Planet, Star, Detector, xs=xs, 
                                                                       ys=ys, zs=zs)
        
        # ----------- append detections' photon rates to one list ---------
        photon_rates_sys.append(photon_rates)

        # ----------- Calculate noisy detections ---------
        noisy_counts = [Detector.add_noise(rate) for rate in photon_rates]
        noisy_counts = np.reshape(np.asarray(noisy_counts), (1,n_detections))
        noisy_counts_sys.append(noisy_counts[0]) 
        
    return noisy_counts_sys, photon_rates_sys  