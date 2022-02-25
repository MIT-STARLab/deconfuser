import numpy as np
 
def random_planet_elements(n_planets, min_a, max_a, max_e, sep_a=0, min_i=0, max_i=np.pi, spread_i_O=0):
    """
    Uniformly sample orbital elements of planets.
    The inclination of the system-plane is sampled first, and then the inclunation of each planet around that plane

    Parameters
    ----------
    n_planets:  number of planets
    min_a:      lower bound on semi-major axes
    max_a:      upper bound on semi-major axes
    max_e:      upper bound on eccentricities
    sep_a:      minimum seperation between adjacent semi-major axes
    min_i:      lower bound on inclination of the plane of the system
    max_i:      upper bound on inclination of the plane of the system
    spread_i_O: spread of inclinations and ascending nodes around the plane of the system

    Returns
    -------
    np.array of semi-major axes
    np.array of eccentricities
    np.array of inclinations
    np.array of arguments of periapsis
    np.array of arguments of ascending nodes
    np.array of mean anomalies
    """

    assert(n_planets*sep_a < max_a - min_a) #make sure there's enough space to fit all planets

    #sample plaent semi-major axes uniformly until they are all spaced more than sep_a apart
    a = np.zeros(n_planets)
    while any(abs(a[i]-a[j]) <= sep_a for i in range(n_planets) for j in range(i)):
        a = np.random.random(n_planets)*(max_a - min_a) + min_a

    #uniform distributions of eccentricities, arguments of periapsis and mean anomalies
    e = np.random.random(n_planets)*max_e
    o = np.random.random(n_planets)*2*np.pi
    M0 = np.random.random(n_planets)*2*np.pi

    #uniform distributions of system inclination and ascending node and uniform distribution for each plaent on top of that
    i = np.random.random()*(max_i - min_i) + min_i + (0.5 - np.random.random(n_planets))*spread_i_O
    O = np.random.random()*2*np.pi + (0.5 - np.random.random(n_planets))*spread_i_O

    return a,e,i,o,O,M0
 
def get_observations(a, e, i, o, O, M0, t, mu, tol=1e-10):
    """
    Get the xy coordinates of the planets in the image plane.

    Parameters
    ----------
    a:   float or array of semi-major axes
    e:   float or array of eccentricities
    i:   float or array of inclinations
    o:   float or array of arguments of periapsis
    O:   float or array of arguments of ascending nodes
    M0:  float or array of mean anomalies
    t:   float or array of observation times
    mu:  gravitational parameter in units consistent with a and t
    tol: tolerance of the solution of Kepler's equation

    Returns
    -------
    np.array of shape (len(a), len(t))
        x coordiantes

    np.array of shape (len(a), len(t))
        y coordiantes
    """

    #reshape all inputs such that 0th axis corresponds to time and 1st axis corresponds to planet
    a, e, i, o, O, M0, mu = map(lambda p: np.reshape([p], (-1,1)), (a, e, i, o, O, M0, mu))
    t = np.reshape([t], (1,-1))

    #initial guess for eccentric anomaly is mean anomaly
    E = M = np.sqrt(mu/a**3)*t + M0

    #Newton's method for solving Keplers equation
    while True:
        E_update = (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
        E = E - E_update
        if np.max(np.abs(E_update)) < tol: break

    #find x,y coordinates in orbit plane
    xs = (np.cos(E) - e)*a
    ys = (np.sin(E)*np.sqrt(1 - e**2))*a

    #rotate from orbit plane to image plane
    xs, ys = xs*np.cos(o) - ys*np.sin(o), (xs*np.sin(o) + ys*np.cos(o))*np.cos(i)
    return xs*np.cos(O) - ys*np.sin(O), xs*np.sin(O) + ys*np.cos(O)
