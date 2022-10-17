import numpy as np

class OrbitGridSearch:
    """
    A class for grid-searching over orbital elements that fit 2D points (in an image) with an orbit.
    Because this is exhaustive search, the minimum error is guaranteed to be found within specified tolerance.
    The search is over a 3D space, (a, ex, ey), where:
     - a is the semi-majot axis bounded by min_a < a < max_a (min/max a are computed in the OrbitFitter class)
     - (ex, ey) is the eccentricity vetor in the orbit plane; it is bounded by max_e > sqrt(ex^2 + ey^2)
    The 3 other orbital elements are found implicitly: a linear transofmration between the orbit plane and the image plane is fitted via least squares.

    The spacing of the grid in the (ex, ey, a) is determined by the tolerance (tol) parameter:
     - the RMS fitting error in the image plane is guaranteed to be less than tol + O(tol^2) of best fit
     - the size of the grid (search space) is proportional to tol^(-3)
     - the grid spacing is determined empirically


    Usage example:
        #grid search for 3 equally spaced observations 4 months apart
        mu = 4*np.pi**2 #in AU^3/year^2
        ts = np.arange(0,1,1.0/3) #in years
        of = orbit_fitting.OrbitGridSearch(mu, ts, max_e=0.3, min_a=0.8, max_a=1.2, tol=0.1)

        #planet with a = 1, e = 0.3 and random orientation
        a,e = 1,0.3
        i,o,O,M0 = 2*np.pi*np.random.random(4)
        xs,ys = sample_planets.get_observations(a, e, i, o, O, M0, ts, mu)
        xys = np.concatenate([xs,ys]).T

        #fit orbital elements and print RMS errors
        err, (a, e, i, o, O, M0) = of.fit(xys)
        xs_fit,ys_fit = sample_planets.get_observations(a, e, i, o, O, M0, ts, mu)
        print(err, np.sqrt(np.mean((xs-xs_fit)**2 + (ys-ys_fit)**2)))
    """
    def __init__(self, mu, ts, max_e, min_a, max_a, tol, kepler_iter=32):     
        """
        Constructor - precomputes points in orbit plane for all combinations of (a,ex,ey) to be searched over

        Parameters
        ----------
        mu:          gravitatoinal parameter
        ts:          times at which detections will be given
        max_e:       upper bound on the eccentricity of orbits
        min_a:       lower bound on the semi-major axis of orbits (units must be consistent with mu and possible_ts)
        max_a:       upper bound on the semi-major axis of orbits (in the same units of min_a)
        tol:         astrometry error tollerance (RMS across all detections in the same units of min_a)
        kepler_iter: maximum number of iterations when solving Kepler's equation
        """

        self.ts_mean = np.mean(ts)
        self.ts = np.array(ts) - self.ts_mean #mean time is subtracted
        self.mu = mu
        self.max_e = max_e
        self.tol = tol/2 #safety factor based on testing
        self.kepler_iter = kepler_iter

        a = min_a
        all_as = []
        all_exs = []
        all_eys = []
        while True: #increase a until it is larger than max_a
            #get sensitiviry of RMS error to a, ex and ey
            sa, sex, sey = self._get_error_sensitivites(a)
            a += self.tol/sa #increase a accordingly

            if a >= max_a - self.tol/sa:
                if len(all_as): break
                #here if only one grid point is neccesary between min_a and max_a
                a = 0.5*(min_a + max_a)

            #choose grid spacing for ex, ey (probably unnesecarily dense towards ex=ey=0)
            dex = min(self.tol/sex, self.max_e)
            dey = min(self.tol/sey, self.max_e)
            #rectangular grid in ex, ey
            ex_grid = np.linspace(-self.max_e, self.max_e, int(np.ceil(2*self.max_e/dex))//2*2 + 1)
            ey_grid = np.linspace(-self.max_e, self.max_e, int(np.ceil(2*self.max_e/dey))//2*2 + 1)

            #only select ex,ey within radius max_e
            EX,EY = np.meshgrid(ex_grid,ey_grid)
            valid_e = np.where(EX**2 + EY**2 < self.max_e**2)

            #append all grid points for the current a
            all_exs.append(EX[valid_e])
            all_eys.append(EY[valid_e])
            all_as.append(a*np.ones(len(valid_e[0])))

        #store all grid points in the (a, ex, ey) space
        self.exs = np.concatenate(all_exs).reshape((-1,1))
        self.eys = np.concatenate(all_eys).reshape((-1,1))
        self.as_ = np.concatenate(all_as).reshape((-1,1))

        #mean anomalies of shape(#grid points, #observation epochs)
        dMs = np.sqrt(self.mu/self.as_**3)*self.ts.reshape((1,-1))

        #solve Kepler's equation to find locations of points in orbits plane (for all elements in parallel; ormalized by a)
        self._xs, self._ys = _modified_kepler_newton(dMs, self.exs, self.eys, self.kepler_iter, 0.5*self.tol/max_a)

        #precompute pseudo inversematrix for leasts-squares fitting points in orbits plane to data points in image plane
        A = np.sum(self._xs*self._xs, axis=1, keepdims = True)
        B = np.sum(self._xs*self._ys, axis=1, keepdims = True)
        D = np.sum(self._ys*self._ys, axis=1, keepdims = True)
        det = A*D - B*B

        self._pinv1 = (D*self._xs - B*self._ys)/det
        self._pinv2 = (A*self._ys - B*self._xs)/det

    def fit(self, xys, only_error=False):
        """
        Fits an orbit image-plane points xys at times self.ts + self.ts_mean.
        This is done by axshaustive search over a 3D space of orbital elements (a, ex, ey).
        The 3 remaining elements corresponding to a linear transormation which is fitted.
        Least-squares fiting the linear transormation, however, gives 4 degrees of freedom.
        One of the DOF is the first singular value which is adjusted to match the semi-major axis, a.

        Because of numerical noise, this process breaks down at low inclinations.
        For this reason, a 0 inclination orbit is fitted separately.

        Parameters
        ----------
        xys:        detections in image plane; np.array of shape (?,2)
        only_error: whether to only return fit error

        Returns
        -------
        float
            fit error

        tuple of 6 floats (optional)
            orbital parameters of best fit (a, e, i, o, O, M0)
        """
        xs = xys[:,0].reshape((1,-1))
        ys = xys[:,1].reshape((1,-1))

        #multiply by pseudo-inverse to find the 2x2 transofrmations from orbit plane (normalized by a) to image plane which minimzes squared error
        #this is done in parallel for all grid points
        m11 = np.einsum("ij->i", self._pinv1*xs).reshape((-1,1))
        m12 = np.einsum("ij->i", self._pinv1*ys).reshape((-1,1))
        m21 = np.einsum("ij->i", self._pinv2*xs).reshape((-1,1))
        m22 = np.einsum("ij->i", self._pinv2*ys).reshape((-1,1))

        #find 1st singular values
        m2_1112 = m11*m11 + m12*m12
        m2_2122 = m21*m21 + m22*m22
        dm2 = m2_1112 - m2_2122
        sm2 = m11*m21 + m12*m22
        s1 = np.sqrt(0.5*(m2_1112 + m2_2122 + np.sqrt(dm2*dm2 + 4*sm2*sm2)))

        #correct 1st singular values to be the semi-major axes (i.e., scale the transofmarion) and compute errors
        scale = self.as_/s1
        err = np.einsum("ij->i", ((m11*self._xs + m21*self._ys)*scale - xs)**2 + ((m12*self._xs + m22*self._ys)*scale - ys)**2)

        best_err = np.sqrt(err.min()/len(xys))

        #in the small inclination cases, the above scaling is very sensitive to noise
        #find rotation only transofmation as well which might be better
        l1 = np.einsum("ij->i", self._ys*xs - self._xs*ys).reshape((-1,1))
        l2 = np.einsum("ij->i", self._xs*xs + self._ys*ys).reshape((-1,1))
        l = np.sqrt(l1*l1 + l2*l2)
        err_rot_only = np.einsum("ij->i", (xs*l2/l - ys*l1/l - self._xs*self.as_)**2 + (ys*l2/l + xs*l1/l - self._ys*self.as_)**2)
        best_err_rot_only = np.sqrt(err_rot_only.min()/len(xys))

        if only_error:
            return min(best_err, best_err_rot_only)

        #construct the 2x2 transormation matrix for the best fit
        if best_err < best_err_rot_only:
            ind = (np.argmin(err),0)
            S = np.array([[m11[ind], m12[ind]], [m21[ind], m22[ind]]])*scale[ind]
        else:
            ind = (np.argmin(err_rot_only),0)
            S = np.array([[l2[ind]/l[ind], -l1[ind]/l[ind]], [l1[ind]/l[ind], l2[ind]/l[ind]]])*self.as_[ind]
            best_err = best_err_rot_only

        ex,ey = self.exs[ind], self.eys[ind]
        e = np.hypot(ey,ex) #eccentricity
        a = self.as_[ind] #semi-major axis
        #true and mean anomaly at t=0
        theta0 = -np.arctan2(ey,ex)
        M0 = np.arctan2(np.sqrt(1 - e**2)*np.sin(theta0), e + np.cos(theta0)) - e*np.sqrt(1 - e**2)*np.sin(theta0)/(1 + e*np.cos(theta0)) - self.ts_mean*np.sqrt(self.mu/a**3)

        #decompose S to get the rotation angles
        U,s,V = np.linalg.svd(S)

        #make U and V rotation matrices (no reflections) but keep the SVD coorect
        if U[0,0]*U[1,1] < 0:
            U[:,1] *= -1
            s[1] *= -1

        if V[0,0]*V[1,1] < 0:
            V[1] *= -1
            s[1] *= -1

        o = np.arctan2(-U[1,0], U[0,0]) - theta0 #argument of periapsis
        O = np.arctan2(-V[1,0], V[0,0]) #ascending node

        if best_err < best_err_rot_only:
            i = np.arccos(s[1]/a) #inclination
        else:
            i = 0

        return best_err, (a, e, i, o, O, M0)

    def _get_error_sensitivites(self, a):
        """
        Numerically estimates the maximum sensitivity of orbit fitting error to (a, ex, ey).
        This is done by numerically differentiating the positions of the outer points in the orbit plane.

        Parameters
        ----------
        a: semi-major axis

        Returns
        -------
        tuple of 3 floats
            the sensitivities of the error to a, ex, ey
        """
        #stencil for simplest forward-difference scheme (axis 2)
        das = 1e-3*np.array([0,self.tol,0,0]).reshape((1,1,4))
        dexs = 1e-3*np.array([0,0,self.tol/a,0]).reshape((1,1,4))
        deys = 1e-3*np.array([0,0,0,self.tol/a]).reshape((1,1,4))

        #eight (ex,ey) points along the outer ring e=e_max (axis 1)
        as_ = np.ones((1,8,1))*a + das
        exs = self.max_e*np.array([1,np.sqrt(0.5),0,-np.sqrt(0.5),-1,-np.sqrt(0.5),0,np.sqrt(0.5)]).reshape((1,8,1)) + dexs
        eys = self.max_e*np.array([0,np.sqrt(0.5),1,np.sqrt(0.5),0,-np.sqrt(0.5),-1,-np.sqrt(0.5)]).reshape((1,8,1)) + deys

        #mean anomalies (times are in axis 0)
        dMs = np.sqrt(self.mu/as_**3)*self.ts.reshape((-1,1,1))

        #get points in orbit plane corresponding to all ex,ey,ts, and stencil above; (len(ts),8,4) total
        xs, ys = _modified_kepler_newton(dMs, exs, eys, self.kepler_iter, 1e-3*self.tol/a)
        xs, ys = as_*xs, as_*ys

        #the first point in the stencil is "unperturbed"
        xs0 = xs[:,:,:1]
        ys0 = ys[:,:,:1]
        
        #find an angle to rotate the rest of the points to minimize average distance squared
        domegas = np.arctan2(np.sum(ys0*xs - xs0*ys, axis=0, keepdims=True), np.sum(xs0*xs + ys0*ys, axis=0, keepdims=True))

        #compute squared errors - distances due to stencil shifts
        errs = (xs*np.cos(domegas) - ys*np.sin(domegas) - xs0)**2 + (ys*np.cos(domegas) + xs*np.sin(domegas) - ys0)**2

        #find the maximum sensitivities of RMS error
        #the meanis across detection times, the maximum is across the 8 (ex,ey) points tested
        sa = np.max(np.sqrt(np.mean(errs[:,:,1], axis=0))/das[:,:,1])
        sex = np.max(np.sqrt(np.mean(errs[:,:,2], axis=0))/dexs[:,:,2])
        sey = np.max(np.sqrt(np.mean(errs[:,:,3], axis=0))/deys[:,:,3])

        return sa, sex, sey

class OrbitFitter:
    """
    A class for managing several grid-search classes with different ranges of sami-major axis.
    The regions are chosen such that at most one "perfect" fit can be found in each.
    Regions in which there could be no "perfect fit" (within tolerance) are ignored.
    """
    def __init__(self, mu, ts, min_a, max_a, max_e, tol):
        """
        Constructor - precomputes points in orbit plane for all combinations of (a,ex,ey) to be searched over

        Parameters
        ----------
        mu:    gravitatoinal parameter
        ts:    times at which detections will be given
        max_e: upper bound on the eccentricity of orbits
        min_a: lower bound on the semi-major axis of orbits (units must be consistent with mu and possible_ts)
        max_a: upper bound on the semi-major axis of orbits (in the same units of min_a)
        tol:   astrometry error tollerance (RMS across all detections in the same units of min_a)
        """
        self.tol = tol
        self.ts = np.array(ts, dtype=np.float64)
        self.mu = mu
        self.max_e = max_e

        #split the semi-major axis into regions based on when points "cross" each other (see get_a_regions)
        self.a_regions = get_a_regions(self.ts, mu, min_a)

        #set upper bound to be max_a (lower bound is already min_a)
        max_reg_i = self.a_regions.searchsorted(max_a)
        if self.a_regions[-1] < max_a:
            self.a_regions = np.concatenate([self.a_regions, [max_a]])
        else:
            self.a_regions = self.a_regions[:self.a_regions.searchsorted(max_a)+1]
            self.a_regions[-1] = max_a

        #initialize grid search for the given regions
        self.grids = [OrbitGridSearch(self.mu, self.ts, self.max_e, self.a_regions[i], self.a_regions[i+1], tol) for i in range(len(self.a_regions)-1)]

    def fit(self, xys, only_error=False):
        """
        Fits an orbit image-plane points xys at times self.ts.
        Iterates over all relevant grids and invokes OrbitGridSearch.fit

        Parameters
        ----------
        xys:        detections in image plane; np.array of shape (?,2)
        only_error: whether to only return fit error

        Returns
        -------
        iterator over:
            float
                fit error

            tuple of 6 floats (optional)
                orbital parameters of best fit (a, e, i, o, O, M0)
        """ 

        #void iterating over all grids by bounding the semi-major axis
        min_a, max_a = estimate_min_max_a(xys, self.ts, self.mu, self.max_e, self.tol)
        min_reg_i = max(0, self.a_regions.searchsorted(min_a) - 1)
        max_reg_i = min(self.a_regions.searchsorted(max_a), len(self.grids))

        #fit points to all grids that might contain a perfect fit
        for reg_i in range(min_reg_i, max_reg_i):
            yield self.grids[reg_i].fit(xys, only_error)

def _modified_kepler_newton(dM, ex, ey, max_iter, tol):
    """
    Solves Kepler's equation in terms of the eccentrcity vector (ex,ey instead of just e).
    Instead of the mean anomaly, the difference of the mean anomaly fromt t=0 is given.
    By definition, at t=0: x>0 and y=0.

    Parameters
    ----------
    dM:       difference in mean anomaly dM = M - M(t=0)
    ex:       x componenet of the eccentricity vector
    ey:       y componenet of the eccentricity vector
    max_iter: maximum number of iterations of Newton's method
    tol:      tolerance of eccentric anomaly (convergence condition)

    Returns
    -------
    np.array of floats
        x corrdinates of points in orbital plane (with a = 1)

    np.array of floats
        y corrdinates of points in orbital plane (with a = 1)
    """
    #definitions of constants
    e2 = ex**2 + ey**2

    c1 = (ex + e2)/(1+ex)
    c3 = np.sqrt(1-e2)
    c2 = ey*c3/(1+ex)

    #Newton's method for solving the modified Kepler's equation
    dE = dM*1.0 #eccentric anomaly difference dE = E-E(t=0)
    for _ in range(max_iter):
        sE = np.sin(dE)
        cE = np.cos(dE)
        dE_update = (dE + c2*cE - c2 - c1*sE - dM)/(1 - c2*sE - c1*cE)
        if np.max(np.abs(dE_update)) < tol: break
        dE = dE - dE_update

    #translating the eccentric anomaly to x and y position (normalized by a)
    xs = (1 + ex - ey**2)/(1 + ex)*cE - c3/(1 + ex)*ey*sE - ex
    ys = ey*cE + c3*sE - ey

    return xs, ys

def get_a_regions(ts, mu, min_a):
    """
    Splits semi-major axes into regions based on the topology of the observations.
    In the same region, the number of revolutions the planet makes between every two detections remains the same.
    #The the transition between semi-major axes is where two or more times in ts correspond to the same position.

    Parameters
    ----------
    ts:    times at which detections will be given
    mu:    gravitational parameter
    min_a: lower bound on semi-major axis

    Returns
    -------
    np.array of floats
        semi-major axes at the transitions between regions
    """
    max_n = np.sqrt(mu/min_a**3) #maximum mean angular motion

    dts = list(set([ts[i] - ts[j] for i in range(1, len(ts)) for j in range(i)])) #time differences
    max_k = int(np.floor(max_n*max(dts)/(2*np.pi))) #maximum number of revolutions

    #compute all time differences that cause two points to be in the same location after one or multiple periods
    expanded_dts = np.array([2*np.pi/max_n] + [dt/k for dt in dts for k in range(1,max_k+1) if 2*np.pi*k < max_n*dt])

    #compute all semi-major axes that correspond to period=one of the time differences
    a_regions = np.sort((mu/4/np.pi**2*expanded_dts**2)**(1.0/3))

    #filter out duplicates and sort semi-major axes at wich transition between regions occurs
    return np.delete(a_regions, np.where(np.abs(a_regions[1:] - a_regions[:-1]) < min_a*1e-3)[0])

def estimate_min_max_a(xys, ts, mu, max_e, tol):
    """
    Find quick bounds on max/min semi-major axes that gives an orbit that fits data within tol.

    Parameters
    ----------
    xys:   detections in image plane; np.array of shape (?,2)
    ts:    times at which detections will be given
    mu:    gravitational parameter (units consistent with xys and ts)
    max_e: upper bound on the eccentricity of orbits
    tol:   astrometry error tollerance (RMS across all detections in the same units of xys)

    Returns
    -------
    float
        lower bound on semi-major axis

    float
        upper bound on semi-major axis
    """
    #minimum semi-major axis must be large enough to reach the further detection
    min_a = np.max((np.hypot(xys[:,0], xys[:,1]) - tol)/(1 + max_e))

    #compute distances and time differences between detections all
    DX2 = (xys[:,0].reshape((-1,1)) - xys[:,0].reshape((1,-1)))**2 + (xys[:,1].reshape((-1,1)) - xys[:,1].reshape((1,-1)))**2
    DX2 = (DX2 - 2*tol)*(DX2 > 2*tol) + np.finfo(float).eps
    DT2 = (ts.reshape((-1,1)) - ts.reshape((1,-1)))**2
    DT2[np.arange(len(xys)),np.arange(len(xys))] = np.finfo(float).max

    #maximum semi-major axis must be small enough so that the planets are quick enough to move between detections
    max_a = (DT2/DX2).min()*mu*(1 + 2*max_e) + tol

    return min_a, max_a
