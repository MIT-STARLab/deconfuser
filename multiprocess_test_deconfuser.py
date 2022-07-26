import numpy as np
import multiprocessing
import itertools
import argparse
import os

import deconfuser.sample_planets as sample_planets
import deconfuser.orbit_fitting as orbit_fitting
import deconfuser.orbit_grouping as orbit_grouping
import deconfuser.partition_ranking as partition_ranking

mu_sun = 4*np.pi**2 #Sun's gravitational parameter in AU^3/year^2

parser = argparse.ArgumentParser(description="Monte-Carlo testing of the deconfuser")
parser.add_argument("--n_planets", type=int, default=3, help="number of planet per system (default: 3)")
parser.add_argument("--n_epochs", type=int, default=4, help="number of observation epochs (default: 4)")
parser.add_argument("--cadence", type=float, default=0.5, help="observation candence in years (default: 0.5)")
parser.add_argument("--mu", type=float, default=mu_sun, help="gravitational parameter in AU^3/year^2 (default: 4pi^2)")
parser.add_argument("--min_a", type=float, default=0.25, help="minimum semi-major axis in AU (default: 0.25)")
parser.add_argument("--max_a", type=float, default=2.0, help="maximum semi-major axis in AU (default: 2.0)")
parser.add_argument("--sep_a", type=float, default=0.3, help="minimum semi-major difference in AU (default: 0.3)")
parser.add_argument("--min_i", type=float, default=0, help="minimum inclination in radians (default: 0)")
parser.add_argument("--max_i", type=float, default=np.pi/2, help="maximum inclination in radians (default: pi/2)")
parser.add_argument("--max_e", type=float, default=0.3, help="maximum eccentricity (default: 0.3)")
parser.add_argument("--spread_i_O", type=float, default=0.0, help="spread of inclination and LAN in radians (default: 0.0 - coplanar)")
parser.add_argument("--n_processes", type=int, default=4, help="number of concurrent processes (default: 4)")
parser.add_argument("--n_systems", type=int, default=10, help="number of systems per process (default: 10)")
parser.add_argument("-v", "--verbose", action="store_true", help="print planet data")
parser.add_argument("toleranes", type=float, nargs="+", help="orbit fit tollerances")
args = parser.parse_args()

#observation epochs (years)
ts = args.cadence*np.arange(args.n_epochs)

#the correct partition of detection by planets
correct_partition = [tuple(range(i*len(ts),(i+1)*len(ts))) for i in range(args.n_planets)]

#to speed up computation, begin with coarsest tolerance and progress to finest:
#1. full orbit grouping will be performed with the coarsest tolerance (i.e., recursively consider all groupings of observation)
#2. only "full" groups that fit observation within a coarser tolerance will be fitted with a finer tolerance
#Note: "missed" detections are not simulataed here so confusion will only "arise" with full groups (n_epochs observations per planet)
tolerances = sorted(args.toleranes, reverse=True)
orbit_grouper = orbit_grouping.OrbitGrouper(args.mu, ts, args.min_a-tolerances[0], args.max_a+tolerances[0], args.max_e, tolerances[0], lazy_init=False)
orbit_fitters = [orbit_fitting.OrbitFitter(args.mu, ts, args.min_a-tol, args.max_a+tol, args.max_e, tol) for tol in tolerances[1:]]

#multi-process printing
printing_lock = multiprocessing.Lock()
def _print(*v):
    printing_lock.acquire()
    print(os.getpid(), *v)
    os.sys.stdout.flush()
    printing_lock.release()

#main function to be ran from multiple processors (the large lookup tables are read-only and shared between processes)
def generate_and_test_systems():
    np.random.seed(os.getpid())

    for _ in range(args.n_systems):
        #choose random orbit parameters for each planet
        a,e,i,o,O,M0 = sample_planets.random_planet_elements(args.n_planets, args.min_a, args.max_a, args.max_e, args.sep_a, args.min_i, args.max_i, args.spread_i_O)
     
        #get coordinates of planets when observed
        xs,ys = sample_planets.get_observations(a, e, i, o, O, M0, ts, args.mu)
        observations = np.stack([xs,ys], axis=2).reshape((-1,2))

        #add radially bounded astrometry error
        noise_r = tolerances[-1]*np.random.random(len(observations))
        noise_a = 2*np.pi*np.random.random(len(observations))
        observations[:,0] += noise_r*np.cos(noise_a)
        observations[:,1] += noise_r*np.sin(noise_a)

        if args.verbose:
            _print("ts =", list(ts))
            for ip in range(args.n_planets):
                _print("a,e,i,o,O,M0 = ", (a[ip],e[ip],i[ip],o[ip],O[ip],M0[ip]))
                _print("xys =", list(map(list, observations[ip*len(ts):(ip+1)*len(ts)])))

        #all detection times for all obesrvations
        all_ts = np.tile(ts, args.n_planets)
        
        #get all possible (full or patrial) groupings of detection by orbits that fit them with the coarsest tolerance
        groupings = orbit_grouper.group_orbits(observations, all_ts)

        #select only groupings that include all epochs (these will be most highly ranked, so no need to check the rest)
        groupings = [g for g in groupings if len(g) == args.n_epochs]

        #check for spurious orbits and repeat for finer tolerances
        for j in range(len(tolerances)):
            found_correct = sum(cg in groupings for cg in correct_partition)

            _print("Tolerance %f: found %d correct and %d spurious orbits out of %d"%(tolerances[j], found_correct, len(groupings) - found_correct, args.n_planets))
            if args.verbose:
                _print("Tolerance %f:"%(tolerances[j]), groupings)

            #find all partitions of observations to exactly n_planets groups
            #note that since all partial grouping were filtered out, all partitions will have exactly n_planets groups
            top_partitions = list(partition_ranking.get_ranked_partitions(groupings))

            if found_correct < args.n_planets:
                for ip in range(args.n_planets):
                    if not correct_partition[ip] in groupings:
                        _print("Failed to fit a correct orbit for planet %d!"%(ip))
            elif len(top_partitions) == 1:
                _print("Tolerance %f: no confusion"%(tolerances[j]))
            else:
                assert(len(top_partitions) > 1)
                _print("Tolerance %f: found %d spurious \"good\" paritions of detections by planets (confusion)"%(tolerances[j], len(top_partitions) - 1))
                if args.verbose:
                    _print("Tolerance %f:"%(tolerances[j]), top_partitions)

            #move to a finer tolerance
            if j < len(tolerances) - 1:
                #only keep groupings that cna be fitted with an orbit with the finer tolerance
                groupings = [g for g in groupings if any(err < tolerances[j+1] for err in orbit_fitters[j].fit(observations[list(g)], only_error=True))]

if __name__ == '__main__':
    #run testing from multiple processes
    processes = []
    for i in range(args.n_processes):
        p = multiprocessing.Process(target=generate_and_test_systems)
        p.start()
        processes.append(p)

    #wait for all processes to finish
    for p in processes:
        p.join()
