# Run deconfuser with phase information for specific systems
# S. Hasler
import numpy as np
import sys
import pandas as pd
import csv
from datetime import datetime 

import deconfuser.sample_planets as sample_planets
import deconfuser.orbit_fitting as orbit_fitting
import deconfuser.orbit_grouping as orbit_grouping
import deconfuser.partition_ranking as partition_ranking
import photometry.photometry as phot
import photometry.likelihood as L

# ------------------------------------------------------------
# Specify parameters
mu = 4*np.pi**2 #Sun's gravitational parameter in AU^3/year^2
n_planets = 3
n_epochs = 3
cadence = 0.5
verbose = True
tolerances = [0.05]
n_systems = 11

# Deconfuser parameters
max_a = 6.0 # max a [AU]
min_a = 0.5 # AU
max_e = 0.1 # eccentricity max
sep_a = 0.3
min_i = 0.0 # rad
max_i = 1.5707963267948966 # rad
spread_i_O = 0.0 # spread of inclination and LAN in radians

start = datetime.now()
now = start.strftime("%Y-%m-%d_%H%M%S") # for text file

# File with systems to run
# systems_orb_params_file = "/Users/shasler/Documents/Projects/Deconfusion/publication/ten_systems/orbparams_10confused_systems_lowi.txt" # low inclination systems
# systems_orb_params_file = "/Users/shasler/Documents/Projects/Deconfusion/publication/ten_systems/orbparams_10confused_systems_medi.txt" # med incl. systems
systems_orb_params_file = "/Users/shasler/Documents/Projects/Deconfusion/publication/ten_systems/orbparams_10confused_systems_highi.txt" # high incl. systems

# Output file path
path = "/Users/shasler/Code/deconfuser/output_files/ten_systems_for_paper/"
f = open(path + f"highi_wErr_output_{now}.txt", "a")
logfile = open(path + f"run_log_highi_wErr_10systems_{now}.log", "a") 
sys.stdout = logfile # redirect output to log file
sys.stderr = logfile # redirect error output to log file also
# ------------------------------------------------------------
# Read in systems
df_system_params = pd.read_csv(systems_orb_params_file, header=0, delimiter=', ', skipfooter=1)

# Write headers to text file
writer = csv.writer(f)
headers = ['system', 'n_planets', 'planet', 'n_orbit_options', 'a', 'e', 'i', 'o', 'O', 'M0', 'ts', 'xyzs', 'correct_partition', 'top_partitions',\
           'partition', 'group', 'rms_fit_err', 'L_group_options', 'L_partition_options', 'L_detections', 'noisy_detections', 'detection_photon_rates'] 

run_parameters = f'Run parameters: {n_systems} systems, {n_planets} planets, \
    {n_epochs} epochs, {cadence} cadence (yr), {tolerances} tolerances'
writer.writerow([run_parameters]) # save run parameters in file
writer.writerow(headers) # add headers to file

# Set up planet, star, and detector parameters for photometry
star = phot.Star(T=5778, R_star=695700e3, d_system=10, mu=mu) # system distance in parsecs -- values for the Sun
planet = phot.Planet(R_p=6.371e6, Ag=0.3)              # values for Earth
detector = phot.Detector(qe=0.837, cic=0.016, dark_current=1.3e-4, read_noise=120, gain=1000, 
                    fwc=80000, conversion_gain=1.0, t=3600, D=2.36, throughput=0.38, f_pa=0.039,
                    wavelength=573.8e-9, bandwidth=56.5e-9) # Roman instrument parameters
                    # f_pa & throughput from V. Bailey (core throughput and optical throughput)

#observation epochs (years)
ts = cadence*np.arange(n_epochs)

#the correct partition of detection by planets
correct_partition = [tuple(range(i*len(ts),(i+1)*len(ts))) for i in range(n_planets)]
print(f'correct_partition: {correct_partition}') # TODO: remove -- SH added

#to speed up computation, begin with coarsest tolerance and progress to finest:
#1. full orbit grouping will be performed with the coarsest tolerance (i.e., recursively consider all groupings of observation)
#2. only "full" groups that fit observation within a coarser tolerance will be fitted with a finer tolerance
#Note: "missed" detections are not simulataed here so confusion will only "arise" with full groups (n_epochs observations per planet)
tolerances = sorted(tolerances, reverse=True)
tol = tolerances[0] # TODO: ADJUST THIS SECTION + ORBIT_FITTERS LATER TO HANDLE MULTIPLE TOLERANCES

orbit_grouper = orbit_grouping.OrbitGrouper(mu, ts, min_a-tolerances[0], max_a+tolerances[0], max_e, tolerances[0], lazy_init=False)
orbit_fitters = [orbit_fitting.OrbitFitter(mu, ts, min_a-tol, max_a+tol, max_e, tol) for tol in tolerances[1:]]
orbit_fitter = orbit_fitting.OrbitFitter(mu, ts, min_a-tol, max_a+tol, max_e, tol) # TODO: remove later -- SH added

for _ in range(n_systems):
    #%% -------------------- Generate simulated systems --------------------
    print(f'\nSystem #{_+1} \n----------') # outputs which number system for readability

    # Get orbit parameters
    system_data = df_system_params[df_system_params['system'] == _ + 1] # system numbers are 1-indexed
    a_vals = np.array(system_data['a'].values)
    e_vals = np.array(system_data['e'].values)
    i_vals = np.array(system_data['i'].values)
    o_vals = np.array(system_data['o'].values)
    O_vals = np.array(system_data['O'].values)
    M0_vals = np.array(system_data['M0'].values)

    #get coordinates of planets when observed
    xs,ys,zs = sample_planets.get_observations(a_vals, e_vals, i_vals, o_vals, O_vals, M0_vals, ts, mu) # TODO: remove E and remove from sample_planets
    observations = np.stack([xs,ys,zs], axis=2).reshape((-1,3))

    #add radially bounded astrometry error
    # TODO: ADD CHANGES FOR NOISE BASED ON PHOTOMETRY HERE
    noise_r = tolerances[-1]*np.random.random(len(observations)) # returns array of len(obs) * final tolerance value 
    noise_a = 2*np.pi*np.random.random(len(observations)) # radial error?  
    observations[:,0] += noise_r*np.cos(noise_a) # x-direction error  # TODO: remove 0, added for testing no astro noise
    observations[:,1] += noise_r*np.sin(noise_a) # y-direction error 
        # observations format: array([[group1_x, group1_y, group1_z], [group2_x, ..., ...], [groupN_x, groupN_y, groupN_z]])
        # observations are the x,y,z coordinates for each of the orbit groupings, which potential orbital parameters are drawn from

    # SECTION TO CALCULATE PHOTOMETRY OF SIMULATED SYSTEM
    # first adjust coordinates for use in get_detections_counts function
    all_coords = []
    for ip in range(n_planets):
        all_coords.append(list(map(list, observations[ip*len(ts):(ip+1)*len(ts)])))
    all_coords = np.asarray(all_coords)
    # get noisy and not noisy photometric detections for simulated system -- phase information buried in this function
    noisy_detections, detections_photon_rates = phot.get_detections_counts(n_planets, n_epochs, xyzs=all_coords, 
                                                                           Planet=planet, Star=star, Detector=detector)

    if verbose:
        print("\nts =", list(ts)) # observation epochs
        for ip in range(n_planets): # for every planet
            print("\nplanet ", ip+1, ": ")
            print("a,e,i,o,O,M0 = ", (a_vals[ip],e_vals[ip],i_vals[ip],o_vals[ip],O_vals[ip],M0_vals[ip])) # true orbital parameters for each planet
            print("xyzs =", list(map(list, observations[ip*len(ts):(ip+1)*len(ts)]))) # true coordinates of detection for each planet
            print("photon_rates = ", detections_photon_rates[ip]) # calculated photon rates for each planet detection (format: [detection1, detection2, ..., detectionN])
            print("noisy_detections = ", noisy_detections[ip]) # noisy planet detections (format: [array([planet1_detection1, ..., planet1_detectionN])], ..., array([planetM_detection1, ..., planetM_detectionN])])

    # output simulated planet info to text file
    for ip in range(n_planets):
        planet_params = [_+1, n_planets, ip+1, np.NaN, a_vals[ip], e_vals[ip], i_vals[ip], o_vals[ip], O_vals[ip], M0_vals[ip], ts, list(map(list, \
                        observations[ip*len(ts):(ip+1)*len(ts)])), correct_partition, None, None, None, None, None, None, None, noisy_detections[ip], \
                        detections_photon_rates[ip]] # system #, # planets simulated, planet #, a, e, i, o, O, M0, confused?, ts, xyzs, correct_partition, top_partitions, group, 'L_detections', 'L_group_options', 'L_partition_options', 'noisy_detections', 'detection_photon_rates' 
        writer.writerow(planet_params)

    # All detections times for all observations
    all_ts = np.tile(ts, n_planets)

#%% -------------------- Do the orbit fitting --------------------
    #get all possible (full or partial) groupings of detection by orbits that fit them with the coarsest tolerance
    groupings = orbit_grouper.group_orbits(observations, all_ts)

    #select only groupings that include all epochs (these will be most highly ranked, so no need to check the rest)
    groupings = [g for g in groupings if len(g) == n_epochs] # lists found groupings of detections
    
    #check for spurious orbits and repeat for finer tolerances
    for j in range(len(tolerances)):
        found_correct = sum(cg in groupings for cg in correct_partition)

        print(f'found_correct: {found_correct}') # TODO: remove -- added for testing

        print('-------------------------------------------------------------')
        print("Tolerance %f: found %d correct and %d spurious orbits out of %d"%(tolerances[j], found_correct, len(groupings) - found_correct, n_planets))
        if verbose:
            print("Tolerance %f:"%(tolerances[j]), groupings)

        #find all partitions of observations to exactly n_planets groups
        #note that since all partial grouping were filtered out, all partitions will have exactly n_planets groups
        top_partitions = list(partition_ranking.get_ranked_partitions(groupings))

        if found_correct < n_planets:
            for ip in range(n_planets):
                if not correct_partition[ip] in groupings:
                    print("Failed to fit a correct orbit for planet %d!"%(ip))
        elif len(top_partitions) == 1:
            print("Tolerance %f: no confusion"%(tolerances[j]))
        else:
            assert(len(top_partitions) > 1)
            L_system_options = [] # for system likelihoods
            # Get orbital parameters of spurious orbits
            for partition in top_partitions: # add to print orbit parameters
                print('partition: ', partition)
                L_partition_options = [] # pre-allocate and clear group options list
                l = 0 # for getting correct noisy counts per planet orbit option
                for group in partition: # which data points lie on the orbit
                    p=0 # TODO: remove -- SH added for testing

                    print('\ngroup: ', group)
                    L_group_options = [] # pre-allocate and clear orbit options list
                    group_orbit_parameters = []
                    errs = []
                    k = 0 # keep track of how many orbit options per group

                    # Print groups + orbital parameters
                    for err, parameters in orbit_fitter.fit(observations[group]): 
                        p=1 # TODO: remove -- SH added for testing
                        print('\nParameters: ', parameters)
                        print('err: ', err) 

                        if parameters[2] == np.nan:
                            print('Error! inclination is NaN!')
                        else:
                            # Phase information section
                            # Phase info buried in likelihood function -- add as a return parameter in likelihood.py if you want to back it out
                            # Calculate likelihood of orbit option
                            L_orbit, L_detections = L.get_L_orbit(n_detections=n_epochs,
                                                                  a=parameters[0], e=parameters[1],
                                                                  i=parameters[2],
                                                                  o=parameters[3],
                                                                  O=parameters[4],
                                                                  M0=parameters[5],
                                                                  ts=ts,
                                                                  noisy_counts=noisy_detections[l],
                                                                  Star=star, Planet=planet, 
                                                                  Detector=detector
                                                                  )
                            print(f'L_orbit: {L_orbit}') # Likelihood of entire orbit (alldetections) 
                            L_group_options.append(L_orbit) # save L of each orbit option per group
                            print(f'L_group_options: {L_group_options}') # Likelihood of each orbit option in a group 
                            group_orbit_parameters.append(parameters)
                            errs.append(err) 
                            k += 1 # track number of orbit options

                    L_partition_options.append(L_group_options) # Likelihood of all orbit options in a partition -- will be empty if i = nan

                    # Write confused orbit options to output file
                    a_s = [orbit[0] for orbit in group_orbit_parameters] # separate orbital parameters for all orbit options in a group
                    e_s = [orbit[1] for orbit in group_orbit_parameters]
                    i_s = [orbit[2] for orbit in group_orbit_parameters]
                    o_s = [orbit[3] for orbit in group_orbit_parameters]
                    O_s = [orbit[4] for orbit in group_orbit_parameters]
                    M0_s = [orbit[5] for orbit in group_orbit_parameters]
                    option_parameters = [_+1, n_planets, l+1, k, a_s, e_s, i_s, o_s, O_s, \
                                         M0_s, ts, None, correct_partition, \
                                         top_partitions, partition, group, errs, \
                                         L_group_options, L_partition_options, \
                                         L_detections, None, None] # parameters for writing to text file
                    
                    writer.writerow(option_parameters)
                    l += 1 # advance to next detected planet in the system for comparison
                    
                    if p==0: # TODO: remove -- SH added for testing
                        print('FLAG -- P=0, NO PARAMETERS TO OUTPUT')
                    
            # ------------------------------------------------------------------
            print("Tolerance %f: found %d spurious \"good\" partitions of detections by planets (confusion)"%(tolerances[j], len(top_partitions) - 1))
            if verbose:
                print("Tolerance %f:"%(tolerances[j]), top_partitions)

        # Move to a finer tolerance
        #move to a finer tolerance
        if j < len(tolerances) - 1:
            #only keep groupings that cna be fitted with an orbit with the finer tolerance
            groupings = [g for g in groupings if any(err < tolerances[j+1] for err in orbit_fitters[j].fit(observations[list(g)], only_error=True))]

# record run time
end  = datetime.now()
runtime = end - start
runtime_string = [f"Run time: {runtime} s"]
writer.writerow(runtime_string) # write run time to end of file

f.close() # close text file
logfile.close() # close log file
sys.stdout = sys.__stdout__ # reset standard output to terminal
sys.stderr = sys.__stderr__ # reset error output to terminal