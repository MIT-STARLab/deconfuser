'''
Script to run MC simulations with the deconfuser. Original script w/o phase information: test_deconfuser.py

How to run in command line: 
    $ python test_deconfuser_addphase.py TOLERANCE --ARG ARG_VALUE
    example: $ python test_deconfuser_addphase.py 0.05 --n_planets 3 --min_a 0.2 -v
'''
import numpy as np
import argparse
import os
import sys
import csv

import deconfuser.sample_planets as sample_planets
import deconfuser.orbit_fitting as orbit_fitting
import deconfuser.orbit_grouping as orbit_grouping
import deconfuser.partition_ranking as partition_ranking

import photometry.photometry as phot
import photometry.likelihood as L
import photometry.ranking as ranking
from datetime import datetime 

start = datetime.now()
now = start.strftime("%Y-%m-%d_%H%M%S") # for text file

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
parser.add_argument("--n_systems", type=int, default=10, help="number of systems (default: 10)")
parser.add_argument("-v", "--verbose", action="store_true", help="print planet data")
parser.add_argument("tolerances", type=float, nargs="+", help="orbit fit tollerances")
args = parser.parse_args()

# Create text file and log file for output
output_file = f"test_deconfuser_output_{now}.txt"
try:
    f = open(f"output_files/{output_file}", "a")
    logfile = open(f"output_files/run_log_{now}.log", "a") 
    sys.stdout = logfile  # redirect output to log file
    sys.stderr = logfile  # redirect error output to log file also
except FileNotFoundError: # if directory doesn't exist, create it
    print('output_files directory not found. Creating directory.')
    os.mkdir('output_files')
    f = open(f"output_files/{output_file}", "a")
    logfile = open(f"output_files/run_log_{now}.log", "a")
    sys.stdout = logfile 
    sys.stderr = logfile

# Write headers to text file
writer = csv.writer(f)
headers = ['system', 'n_planets', 'planet', 'n_orbit_options', 'a', 'e', 'i', \
           'o', 'O', 'M0', 'ts', 'xyzs', 'correct_partition', 'top_partitions', \
           'partition', 'group', 'L_group_options', 'L_partition_options', \
           'L_detections', 'noisy_detections', 'detection_photon_rates'] 
run_parameters = f"Run parameters: {args.n_systems} systems, {args.n_planets} planets, \
                    {args.n_epochs} epochs, {args.cadence} cadence (yr), \
                    {args.min_a} min_a (AU), {args.max_a} max_a (AU), {args.sep_a} sep_a (AU), \
                    {args.min_i} min_i (rad), {args.max_i} max_i (rad), {args.max_e} max_e, \
                    {args.tolerances} tolerances"
writer.writerow([run_parameters]) # save run parameters in file
writer.writerow(headers)          # add headers to file

# Set up planet, star, and detector parameters for photometry
star = phot.Star(T=5778, R_star=695700e3, d_system=10, mu=mu_sun) # system distance in parsecs -- values for the Sun
planet = phot.Planet(R_p=6.371e6, Ag=0.3)                         # values for Earth
detector = phot.Detector(qe=0.837, cic=0.016, dark_current=1.3e-4, read_noise=120, gain=1000, 
                    fwc=80000, conversion_gain=1.0, t=3600, D=2.36, throughput=0.38, f_pa=0.039,
                    wavelength=573.8e-9, bandwidth=56.5e-9) # Roman instrument parameters

# Observation epochs (years)
ts = args.cadence*np.arange(args.n_epochs)

# The correct partition of detection by planets
correct_partition = [tuple(range(i*len(ts),(i+1)*len(ts))) for i in range(args.n_planets)]
print(f'correct_partition: {correct_partition}') # TODO: remove -- SH added

# To speed up computation, begin with coarsest tolerance and progress to finest:
# 1. full orbit grouping will be performed with the coarsest tolerance (i.e., recursively consider all groupings of observation)
# 2. only "full" groups that fit observation within a coarser tolerance will be fitted with a finer tolerance
# Note: "missed" detections are not simulataed here so confusion will only "arise" with full groups (n_epochs observations per planet)
tolerances = sorted(args.tolerances, reverse=True)
tol = tolerances[0] # TODO: ADJUST THIS SECTION + ORBIT_FITTERS LATER TO HANDLE MULTIPLE TOLERANCES

orbit_grouper = orbit_grouping.OrbitGrouper(args.mu, ts, args.min_a-tolerances[0], args.max_a+tolerances[0], args.max_e, tolerances[0], lazy_init=False)
orbit_fitters = [orbit_fitting.OrbitFitter(args.mu, ts, args.min_a-tol, args.max_a+tol, args.max_e, tol) for tol in tolerances[1:]]
orbit_fitter = orbit_fitting.OrbitFitter(args.mu, ts, args.min_a-tol, args.max_a+tol, args.max_e, tol) # TODO: remove later -- SH added


for _ in range(args.n_systems):
    # -------------------- Generate simulated systems --------------------
    print(f'\nSystem #{_} \n----------') 
    # Choose random orbit parameters for each planet
    a,e,i,o,O,M0 = sample_planets.random_planet_elements(args.n_planets, args.min_a, args.max_a, args.max_e, args.sep_a, args.min_i, args.max_i, args.spread_i_O)
    # TODO: ADD SECTION --> CHOOSE RANDOM RADII FOR EACH PLANET

    # Get coordinates of planets when observed
    xs,ys,zs = sample_planets.get_observations(a, e, i, o, O, M0, ts, args.mu) 
    observations = np.stack([xs,ys,zs], axis=2).reshape((-1,3))

    # Add radially bounded astrometry error
    # TODO: ADD CHANGES FOR NOISE BASED ON PHOTOMETRY HERE
    noise_r = tolerances[-1]*np.random.random(len(observations)) 
    noise_a = 2*np.pi*np.random.random(len(observations))
    observations[:,0] += noise_r*np.cos(noise_a) # x-direction error 
    observations[:,1] += noise_r*np.sin(noise_a) # y-direction error 

    # Calculate photometry of simulated system (these are your "observations")
    all_coords = [] 
    for ip in range(args.n_planets):
        all_coords.append(list(map(list, observations[ip*len(ts):(ip+1)*len(ts)])))
    all_coords = np.asarray(all_coords)
    # get noisy and not noisy photometric detections for simulated system
    noisy_detections, detections_photon_rates = phot.get_detections_counts(args.n_planets, args.n_epochs, xyzs=all_coords, 
                                                                               Planet=planet, Star=star, Detector=detector)

    if args.verbose:
        print("\nts =", list(ts)) 
        for ip in range(args.n_planets): # for every planet
            print("\nplanet ", ip+1, ": ")
            print("a,e,i,o,O,M0 = ", (a[ip],e[ip],i[ip],o[ip],O[ip],M0[ip])) # true orbital parameters for each planet
            print("xyzs =", list(map(list, observations[ip*len(ts):(ip+1)*len(ts)]))) # true coordinates of detection for each planet
            print("photon_rates = ", detections_photon_rates[ip]) # calculated photon rates for each planet detection (format: [detection1, detection2, ..., detectionN])
            print("noisy_detections = ", noisy_detections[ip]) # noisy planet detections (format: [array([planet1_detection1, ..., planet1_detectionN])], ..., array([planetM_detection1, ..., planetM_detectionN])])

    # output simulated planet info to text file
    for ip in range(args.n_planets):
        planet_params = [_, args.n_planets, ip+1, np.NaN, a[ip], e[ip], i[ip], o[ip], O[ip], M0[ip], ts, list(map(list, \
                        observations[ip*len(ts):(ip+1)*len(ts)])), correct_partition, None, None, None, None, None, noisy_detections[ip], \
                        detections_photon_rates[ip]] # system #, # planets simulated, planet #, a, e, i, o, O, M0, confused?, ts, xyzs, correct_partition, top_partitions, group, 'L_detections', 'L_group_options', 'L_partition_options', 'noisy_detections', 'detection_photon_rates' 
        writer.writerow(planet_params)

    # all detection times for all obesrvations
    all_ts = np.tile(ts, args.n_planets)
    
    # -------------------- Do the orbit fitting --------------------
    # get all possible (full or partial) groupings of detection by orbits that fit them with the coarsest tolerance
    groupings = orbit_grouper.group_orbits(observations, all_ts)

    # select only groupings that include all epochs (these will be most highly ranked, so no need to check the rest)
    groupings = [g for g in groupings if len(g) == args.n_epochs] 
    
    # Check for spurious orbits and repeat for finer tolerances
    for j in range(len(tolerances)):
        found_correct = sum(cg in groupings for cg in correct_partition)

        print('-------------------------------------------------------------')
        print("Tolerance %f: found %d correct and %d spurious orbits out of %d"%(tolerances[j], found_correct, len(groupings) - found_correct, args.n_planets))
        if args.verbose:
            print("Tolerance %f:"%(tolerances[j]), groupings)

        # Find all partitions of observations to exactly n_planets groups
        # Note that since all partial grouping were filtered out, all partitions will have exactly n_planets groups
        top_partitions = list(partition_ranking.get_ranked_partitions(groupings))

        if found_correct < args.n_planets:
            for ip in range(args.n_planets):
                if not correct_partition[ip] in groupings:
                    print("Failed to fit a correct orbit for planet %d!"%(ip))
        elif len(top_partitions) == 1:
            print("Tolerance %f: no confusion"%(tolerances[j]))
        else:
            assert(len(top_partitions) > 1)
            L_system_options = [] # for system likelihoods
            # Get orbital parameters of spurious orbits
            for partition in top_partitions: 
                print('partition: ', partition)
                L_partition_options = [] # group options list
                i = 0                    # for getting correct noisy counts per planet orbit option
                for group in partition:  # which data points lie on the orbit
                    print('\ngroup: ', group)
                    L_group_options = [] # orbit options list
                    group_orbit_parameters = []
                    k = 0                # keep track of how many orbit options per group

                    # Get groups + orbital parameters to rank with photometry
                    for err, parameters in orbit_fitter.fit(observations[group]): 
                        print('\nParameters: ', parameters)
                        print('err: ', err) # Added to return fit errors

                        # Phase information section
                        # Phase info is buried in likelihood function -- add as a return parameter in likelihood.py if you want to back it out
                        # Calculate likelihood of orbit option
                        L_orbit, L_detections = L.get_L_orbit(n_detections=args.n_epochs,
                                                            a=parameters[0], e=parameters[1],
                                                            i=parameters[2],
                                                            o=parameters[3],
                                                            O=parameters[4],
                                                            M0=parameters[5],
                                                            ts=ts,
                                                            noisy_counts=noisy_detections[i],
                                                            Star=star, Planet=planet, Detector=detector)
                        print(f'L_orbit: {L_orbit}')    # Likelihood of entire orbit (alldetections) 
                        L_group_options.append(L_orbit) # save L of each orbit option per group
                        print(f'L_group_options: {L_group_options}') # Likelihood of each orbit option in a group 
                        group_orbit_parameters.append(parameters)
                        k += 1 # track number of orbit options

                    L_partition_options.append(L_group_options) # Likelihood of all orbit options in a partition -- will be empty if i = nan

                    # Write confused orbit options to output file
                    a_s = [orbit[0] for orbit in group_orbit_parameters] # separate orbital parameters for all orbit options in a group
                    e_s = [orbit[1] for orbit in group_orbit_parameters]
                    i_s = [orbit[2] for orbit in group_orbit_parameters]
                    o_s = [orbit[3] for orbit in group_orbit_parameters]
                    O_s = [orbit[4] for orbit in group_orbit_parameters]
                    M0_s = [orbit[5] for orbit in group_orbit_parameters]
                    option_parameters = [_, args.n_planets, i+1, k, a_s, e_s, i_s, o_s, O_s, M0_s, ts, None, correct_partition, top_partitions, partition, \
                                        group, L_group_options, L_partition_options, L_detections, None, None] # parameters for writing to text file
                    
                    writer.writerow(option_parameters)
                    i += 1 # advance to next detected planet in the system for comparison

            # ------------------------------------------------------------------
            print("Tolerance %f: found %d spurious \"good\" paritions of detections by planets (confusion)"%(tolerances[j], len(top_partitions) - 1))
            if args.verbose:
                print("Tolerance %f:"%(tolerances[j]), top_partitions)

        #move to a finer tolerance
        if j < len(tolerances) - 1:
            #only keep groupings that cna be fitted with an orbit with the finer tolerance
            groupings = [g for g in groupings if any(err < tolerances[j+1] for err in orbit_fitters[j].fit(observations[list(g)], only_error=True))]

# Re-rank systems with photometry
ranking_filepath = "output_files/ranking_files/"
try: # create ranking files directory if it doesn't exist
    os.makedirs("output_files/ranking_files", exist_ok=True) 
except OSError as error: 
    print("ranking_files directory cannot be created.")

# Create photometry ranking object -- houses file dataframe
confused_systems = ranking.PhotometryRanking(filepath=f"output_files/{output_file}", n_planets=args.n_planets)
df_confused = confused_systems.get_top_group_options()  # iterate over options with multiple groups
df_ranked = confused_systems.top_ranked_partition()     # Get top ranked partition in each system
df_recombined = confused_systems.combine_and_cleanup(save_file=True, save_path=ranking_filepath + f"systems_ranked_{now}.txt")  # Combine original and ranked dataframes
# If you want to calculate percent difference between simulated and fit orbits:
df_final = confused_systems.orbit_percent_diff()      
df_final_wperc = confused_systems.final_recombined(save_file=True, save_path=ranking_filepath + f"systems_ranked_wPercDiff_{now}.txt")    
print('\nPhotometry ranking complete.')

end  = datetime.now()
runtime = end - start
runtime_string = [f"Run time: {runtime} s"]
writer.writerow(runtime_string) # write run time to end of file

f.close()       
logfile.close() 
sys.stdout = sys.__stdout__ # reset standard output to terminal
sys.stderr = sys.__stderr__ # reset error output to terminal