import numpy as np
import orbit_fitting
import itertools

class OrbitGrouper:
    """
    A class for efficiently grouping detections by orbits using grid search.
    Tries to fit orbits to all possible combinations of observations (across all or some of the observation times).

    """
    def __init__(self, mu, possible_ts, min_a, max_a, max_e, tol, lazy_init=True):
        """
        Constructor

        Parameters
        ----------
        mu:          gravitatoinal parameter
        possible_ts: times at which detections will be given
                     (it saves time and memory if time differences are repeated, 
                      e.g., possible_ts[2:]-possible_ts[1:-1] == possible_ts[1:-1]-possible_ts[:-2])
        min_a:       lower bound on the semi-major axis of orbits (units must be consistent with mu and possible_ts)
        max_a:       upper bound on the semi-major axis of orbits (in the same units of min_a)
        max_e:       upper bound on the eccentricity of orbits
        tol:         astrometry error tollerance (RMS across all detections in the same units of min_a)
        lazy_init:   whether the grids should be created on-the-fly or in the contructor
        """
        self.possible_ts = np.sort(list(set(possible_ts)))
        self.mu = mu
        self.min_a = min_a
        self.max_a = max_a
        self.max_e = max_e
        self.tol = tol
        self.fitters_dict = {} #orbit fitters for grid search

        if lazy_init: return

        #initialize orbit fitters for each possible set of time differences
        for l in range(2,len(possible_ts)+1):
            for sub_ts in itertools.combinations(possible_ts, l):
                #sub_ts is a subgroup of observation times
                #since we only care about time differences, subtract the first time
                key = tuple(np.array(sub_ts) - sub_ts[0])

                #the resulting time differences might have already been initialized
                if key in self.fitters_dict: continue

                #if not, initialize the corresponding lookup able
                self.fitters_dict[key] = orbit_fitting.OrbitFitter(self.mu, sub_ts, self.min_a, self.max_a, self.max_e, self.tol)

    def group_orbits(self, xys, ts):
        """
        Find all subsets of observations (groupings) that have an orbit passing within self.tol of them.

        Parameters
        ----------
        xys: np.array of all detections of shape (?,2)
        ts:  np.array of all detections times

        Returns
        -------
        array of tuples
            indices of detections that have an orbit passing within self.tol of them
        """
        assert(set(ts).issubset(self.possible_ts))

        #start with single detection that can always be fitted with an orbit
        all_valid_groupings = [(i,) for i in range(len(xys))]
        #prepare to check all pairs of detections
        potential_groupings = list(itertools.combinations(range(len(xys)), 2))

        #chekc gorupings of increasing length
        for _ in range(1, len(self.possible_ts)):
            #validate the grouping by fitting orbits to them
            valid_groupings = [g for g in potential_groupings if self.check_grouping(xys, ts, g)]
            all_valid_groupings += valid_groupings

            #get potential grouping with one more detection
            potential_groupings = get_next_level_groups(valid_groupings)

        return all_valid_groupings

    def check_grouping(self, xys, ts, g):
        """
        Check whether one specific grouping g can be fitted with an orbit within specified tolerance

        Parameters
        ----------
        xys: np.array of all detections of shape (?,2)
        ts:  np.array of all detections times
        g:   the grouping - indices of observations to be checked

        Returns
        -------
        boolean
            True if at least one orbit can be fitted within self.tol
        """

        sub_ts = ts[list(g)] #selected observation times
        sorted_indices = np.argsort(sub_ts) #sort times to align with orbit fitters
        key = tuple(sub_ts[sorted_indices] - sub_ts[sorted_indices[0]]) #fitter id

        if len(set(key)) < len(g):
            #here if some observation times are repeated (a planet cannot corresponds to two detections)
            return False

        sub_xys = xys[list(g)][sorted_indices] #also align astrometry with fitters time order

        if not key in self.fitters_dict: #lazy initialization of fitters
            self.fitters_dict[key] = orbit_fitting.OrbitFitter(self.mu, key, self.min_a, self.max_a, self.max_e, self.tol)

        #lookup talbes try many regions; check if within at least one the error is low enough
        for err in self.fitters_dict[key].fit(sub_xys, only_error=True):
            if err < self.tol:
                return True

        return False

def get_next_level_groups(groups):
    """
    Find all groups of k+1 elements such that all of their subgroups with k elements are given.

    Parameters
    ----------
    groups: array of sorted touples - all valid groups of (indicies f) k detections identifies so far

    Returns
    -------
    array of sorted touples
        all potentially valid groups of k+1 detections to be chekced
    """
    if len(groups) == 0:
        return []

    lengths = list(map(len, groups))
    l = max(lengths)
    assert(l == min(lengths)) #all groups must be the same length

    max_i = max(map(max, groups)) #maximum element across all groups

    #extend all groups by adding one element
    #the added elements are larger then the largest element already in each group
    #this endures uniquness of all results groups with l+1 elements/indices
    potential_groups = [g + (j,) for g in groups for j in range(g[-1]+1,max_i+1)]

    #a function for cheking that all sub-groups of the new groups are given
    all_sub_exist = lambda g: all(g[:j] + g[j+1:] in groups for j in range(l+1))

    #return only  new groups withh all sub-groups given
    return list(filter(all_sub_exist, potential_groups))

