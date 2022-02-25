import numpy as np
import sample_planets
import orbit_fitting

#testing parameters
min_a = 0.25
max_a = 4
max_e = 0.7
tol = 0.1
mu = 4*np.pi**2

for i in range(1000):
    #choose semi-majot axis first
    a = np.random.random()*(max_a - min_a) + min_a

    #choose observation times
    n_ts = np.random.randint(3,5)
    if np.random.random() > 0.5:#non-uniformly spaced
        ts = np.arange(n_ts)*(1.25 - 0.5*np.random.random(n_ts))
    else:#uniformly spaced
        ts = np.arange(n_ts)*(0.25 + np.random.random())

    #check to which region of semi-major axes a belongs (see documantation of orbit_fitting.get_a_regions)
    a_regions = list(orbit_fitting.get_a_regions(ts, mu, min_a))
    if a_regions[-1] < a:
        a_regions.append(a + 2*tol)
    a_i = max(np.searchsorted(a_regions, a), 1)

    #create a grdi search object for that region
    gs = orbit_fitting.OrbitGridSearch(mu, ts, max_e, a_regions[a_i-1], a_regions[a_i], tol)

    #sample several planets with given semi-major axis and random orientation
    _,e,i,o,O,M0 = sample_planets.random_planet_elements(64, a, a+0.5*tol, max_e, 0, 0, 2*np.pi, 2*np.pi)
    xs,ys = sample_planets.get_observations(a, e, i, o, O, M0, ts, mu)

    #fit each planet with and orbit and make sure the error is below tolerance (true error of best fit is zero)
    for j in range(xs.shape[1]):
        xys = np.stack([xs[j], ys[j]], axis=1)
        fit_err = gs.fit(xys, only_error=True)

        if fit_err > tol:
            print("Error greater than tolerance", (fit_err, tol))
            print("xys = ", list(xys))
            print("ts = ", list(ts))
            print("min_a, max_a, max_e = ", (a_regions[a_i-1], a_regions[a_i], max_e))
            print("planet = ", list(planets[0]))

