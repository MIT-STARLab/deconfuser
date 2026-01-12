'''
@author: S. N. Hasler

Functions to read text files output by test_deconfuser_wPhotometry.py, pull information into dataframe, and return new 
dataframe with partitions re-ranked and highest ranked partitions marked as True.

How to use:
    1. Read text file into dataframe with pd.read_csv()
        df = pd.read_csv(file_path)
    2. df_confused_only = get_top_group_options()
        Return only highest likelihood groups for groups with multiple orbit options and remove
        simulated planets from dataframe.
    3. Get number of planets simulated from first row of dataframe (the same for all columns when using test_deconfuser_addphase.py)
        n_planets = df['n_planets'][0]
    3. df_top_ranked = top_ranked_partition(df_confused_only, n_planets)
        Gets top ranked partition for all systems and returns dataframe with assigned ranking ids.
    4. df_recombined = combine_and_cleanup(df, df_top_ranked)
        Combines original dataframe and dataframe with ranked partitions. Removes unnecessary/duplicate columns.
'''
import pandas as pd
import numpy as np
from tqdm import tqdm 

def percent_difference(val1, val2):
    '''
    Returns the percent difference of two values.
    '''
    return (np.abs(val1 -  val2) / ((val1 +  val2) / 2) ) * 100

class Planet:
    '''
    Planet class to store orbital parameters
    '''
    def __init__(self, number, a, e, i, omega, Omega, M0):
        self.number = number
        self.a = a
        self.e = e
        self.i = i
        self.o = omega
        self.O = Omega
        self.M0 = M0

    def __str__(self):
        return f"Planet #{self.number}: a = {self.a} au, e = {self.e}, i = {self.i} rad, o = {self.omega}, O = {self.Omega}, M0 = {self.M0}"

class PlanetarySystem:
    '''
    System class to store n_planets and system information
    '''
    def __init__(self, number):
        self.number = number
        self.planets = []
    
    def add_planet(self, planet):
        if isinstance(planet, Planet):
            self.planets.append(planet)
        else:
            print("Invalid planet object. Please provide a valid Planet object.")
    
    def __str__(self):
        system_info = f"System #{self.number}\n"
        planets_info = f"\n".join([str(planet) for planet in self.planets])
        return system_info + planets_info

class PhotometryRanking:
    """
    A class for updating orbit rankings using photometric information
    """
    def __init__(self, filepath, n_planets):
        '''
        Constructor

        Parameters
        ----------
        filepath : string
            Full path to output text file from running simulation with 
            the deconfuser. 
        n_planets : int
            Number of planets simulated per system in MC deconfuser run.
        
        '''
        self.filepath = filepath
        self.n_planets = n_planets

        # Read file into dataframe
        self.df = pd.read_csv(filepath, skiprows=1, skipfooter=1, engine='python')
    
    def get_top_group_options(self):
        '''
        Iterates over dataframe of systems output by the deconfuser with 
        photometry likelihood values to sort groups with multiple orbit options by 
        orbits with the highest likelihoods. 
        Creates new columns to save orbital parameters and likelihoods for
        the highest likelihood options in each group. 
        Returns new dataframe with new columns.

        '''
        df = self.df
        print(f"Sorting group options for each system...")
        for index, row in tqdm(df.iterrows()):
            # if there is more than one orbit option in a group, 
            # need to pick the highest likelihood option for partition
            if row['n_orbit_options'] > 1:
                L_group_options = eval(row['L_group_options'])   # convert L_group_options to list
                n_orbit_options = int(row['n_orbit_options'])    # get number of orbit options in the group

                # separate group_options into tuples with (likelihood, index)
                indices = list(range(n_orbit_options))                     # get indices for tuple
                group_options = list(zip(L_group_options, indices))        # pair group options + indices 
                sorted_group_options = sorted(group_options, reverse=True) # sort group_options by highest likelihood
                top_group = sorted_group_options[0]                        # keep only highest likelihood option
                highest_L_index = top_group[1]

                # save highest likelihood option
                df.loc[index, 'L_orbit'] = top_group[0]                    # add new column for highest likelihood orbit -- keep highest likelihood
                df.loc[index, 'id_top_orbit_in_group'] = highest_L_index   # save index for highest L orbit in group
                df.loc[index, 'top_a'] = eval(df.loc[index, 'a'])[highest_L_index]  # copy parameters for highest L option to new top_'parameter' column
                df.loc[index, 'top_e'] = eval(df.loc[index, 'e'])[highest_L_index]
                df.loc[index, 'top_i'] = eval(df.loc[index, 'i'])[highest_L_index]
                df.loc[index, 'top_o'] = eval(df.loc[index, 'o'])[highest_L_index]
                df.loc[index, 'top_O'] = eval(df.loc[index, 'O'])[highest_L_index]
                df.loc[index, 'top_M0'] = eval(df.loc[index, 'M0'])[highest_L_index]

            elif row['n_orbit_options'] == 1: # for groups with only one orbit option
                df.loc[index, 'L_orbit'] = eval(df.loc[index, 'L_group_options'])[0]
                df.loc[index, 'id_top_orbit_in_group'] = 0
                df.loc[index, 'top_a'] = eval(df.loc[index, 'a'])[0]  # copy a to top_a for groups with only one orbit option
                df.loc[index, 'top_e'] = eval(df.loc[index, 'e'])[0]
                df.loc[index, 'top_i'] = eval(df.loc[index, 'i'])[0]
                df.loc[index, 'top_o'] = eval(df.loc[index, 'o'])[0]
                df.loc[index, 'top_O'] = eval(df.loc[index, 'O'])[0]
                df.loc[index, 'top_M0'] = eval(df.loc[index, 'M0'])[0]
            
        # remove columns that we don't need anymore (original orbital parameter columns)
        df = df.drop(columns=['ts', 'xyzs', 'correct_partition', 'noisy_detections', \
                            'detection_photon_rates', 'L_detections', 'L_partition_options', 'L_group_options'])
        # finally, remove rows for simulated planets and only keep orbit options
        df = df[~df['n_orbit_options'].isnull()]
        
        self.df_confused_options = df
    
        return self.df_confused_options
    

    def top_ranked_partition(self):
        '''
        Function to assign the top ranked partition for each system. 
        Uses the dataframe of only confused system options returned
        by the get_top_group_options function.
        
        '''
        system_numbers = list(set([num for num in self.df_confused_options['system']]))
        partition_list = []

        print("Calculating likelihood of each partition...")
        for index, row in tqdm(self.df_confused_options.iterrows()):    
            if row['planet'] == self.n_planets:                                   
                partition_list.append(row['L_orbit'])  # save L_orbit for group
                L_partition = np.prod(partition_list)
                self.df_confused_options.loc[index, 'L_partition_list'] = str(partition_list) # append partition list to end of df
                self.df_confused_options.loc[index, 'L_partition'] = str(L_partition)
                partition_list = []                     # reset partition_list
            else:
                partition_list.append(row['L_orbit'])  

        print("Sorting partitions by likelihoods...")
        for num in tqdm(system_numbers): # for each system
            system_df = self.df_confused_options[self.df_confused_options['system'] == num]  
            partition_options = np.unique([partition for partition in system_df['partition']]) # get possible partitions for system
            L_partitions = [float(row['L_partition']) for index, row in system_df[~system_df['L_partition'].isnull()].iterrows()] # get L_partition for each partition
            ids = list(range(len(L_partitions)))
            
            # Put likelihood and partitions into tuple
            possible_partitions = list(zip(L_partitions, partition_options))     # create tuples of (partition likelihood, partition)       
            sorted_partitions = sorted(possible_partitions, reverse=True)        # sort tuples from highest -> lowest likelihood
            numbered_sorted_partitions = list(zip(sorted_partitions, ids))       # add ranking ids to sorted partitions
            highest_L_partition = sorted_partitions[0][1]                        # save partition of highest likelihood

            rankings = []
            for partition in numbered_sorted_partitions:
                part_tuple = (partition[0][1], partition[1])                     # create new tuple to keep rankings with partitions
                rankings.append(part_tuple)
            
            for partition in rankings: # add ranking value to df
                for index, row in self.df_confused_options.iterrows():
                    if row['partition'] == partition[0] and row['system'] == num:
                        self.df_confused_options.loc[index, 'ranking'] = partition[1] # add ranking number to df

                        # Mark highest likelihood partition in dataframe as True, the rest as False
                        if self.df_confused_options.loc[index, 'partition'] == highest_L_partition:
                            self.df_confused_options.loc[index, 'top_ranked_partition'] = True
                        else:
                            self.df_confused_options.loc[index, 'top_ranked_partition'] = False

        return self.df_confused_options # All confused options now ranked
    
    def combine_and_cleanup(self, save_file=False, save_path=None):
        '''
        Combine original dataframe and dataframe with ranked partitions.
        Also removes duplicate/unecessary columns.
        Saves file if save_file is True

        Parameters
        ----------
        save_file : bool, optional
            Whether or not you want to save the dataframe to a text file, by default False
        save_path : _type_, optional
            Path to save text file to. Recommend to save where original text files are located., by default None
        
        '''
        df_recombined = self.df.join(self.df_confused_options, how='outer', lsuffix='_original')
        df_recombined = df_recombined.drop(columns=['system', 'n_planets', 'planet', 'n_orbit_options', 'top_partitions_original', \
                           'partition_original', 'group_original', 'L_partition_options'])
    
        if save_file == True:
            df_recombined.to_csv(save_path, header=True, index=None)

        self.df_recombined = df_recombined

        return df_recombined
    
    def orbit_percent_diff(self):
        '''
        Calculate the percent difference between the simulated orbital
        parameters and the confused partitions.

        Parameters
        ----------
        confused_system_numbers : numpy.ndarray 
            Array of integers corresponding to the confused system numbers.
        '''
        confused_system_numbers = self.df['system'].unique()

        # Separate out simulated systems
        df_confused_sim = self.df_recombined[(self.df_recombined['system_original'].isin(confused_system_numbers)) & (self.df_recombined['n_orbit_options_original'].isna())]
        # Separate out the confused partitions for each system
        df_confused_partitions = self.df_recombined[(self.df_recombined['system_original'].isin(confused_system_numbers)) & (~self.df_recombined['n_orbit_options_original'].isna())]

        # For each confused system, get the number of confused partitions
        for system_num in confused_system_numbers:
            simulated_system = PlanetarySystem(system_num) # create system object

            df_simulated = df_confused_sim[df_confused_sim['system_original'] == system_num]
            df_confused = df_confused_partitions[df_confused_partitions['system_original'] == system_num]

            # For the simulated system, grab the orbital parameters
            for index, row in df_simulated.iterrows():
                simulated_planet = Planet(row['planet_original'], row['a_original'], row['e_original'], row['i_original'], 
                                    row['o_original'], row['O_original'], row['M0_original'])
                simulated_system.add_planet(simulated_planet) # add planets to system

            # Check to see how many confused partitions there are
            num_partitions = len(df_confused) / self.n_planets

            # For each confused option, compare the orbital parameters to the planet with the
            # corresponding number in the simulated system
            for index, row in df_confused.iterrows():
                planet_number = row['planet_original']

                # Compare each parameter
                sim_a = float(simulated_system.planets[planet_number-1].a)
                sim_e = float(simulated_system.planets[planet_number-1].e)
                sim_i = float(simulated_system.planets[planet_number-1].i)
                sim_o = float(simulated_system.planets[planet_number-1].o)
                sim_O = float(simulated_system.planets[planet_number-1].O)
                sim_M0 = float(simulated_system.planets[planet_number-1].M0)
                        
                pdiff_a = percent_difference(sim_a, float(row['top_a']))
                pdiff_e = percent_difference(sim_e, float(row['top_e']))
                pdiff_i = percent_difference(sim_i, float(row['top_i']))
                pdiff_o = percent_difference(sim_o, float(row['top_o']))
                pdiff_O = percent_difference(sim_O, float(row['top_O']))
                pdiff_M0 = percent_difference(sim_M0, float(row['top_M0']))

                # Add percent difference to columns
                df_confused_partitions.loc[index, 'a_%diff'] = pdiff_a
                df_confused_partitions.loc[index, 'e_%diff'] = pdiff_e
                df_confused_partitions.loc[index, 'i_%diff'] = pdiff_i
                df_confused_partitions.loc[index, 'o_%diff'] = pdiff_o
                df_confused_partitions.loc[index, 'O_%diff'] = pdiff_O
                df_confused_partitions.loc[index, 'M0_%diff'] = pdiff_M0

        self.df_confused_final = df_confused_partitions

        return df_confused_partitions
    
    def final_recombined(self, save_file=False, save_path=None):
        '''
        Final recombination after percent differences have been calculated

        Parameters
        ----------
        save_file : bool, optional
            Flag to save the file or not., by default False
        save_path : _type_, optional
            Path to save file to. Must be defined if save_file = True;
            By default None
        '''
        colnames = [col for col in self.df_confused_final.columns]
        cols_to_remove = colnames[:len(colnames)-6] # duplicate cols to remove
        
        df_final = self.df_recombined.join(self.df_confused_final, how='outer', lsuffix='1')
        df_final = df_final.drop(columns=cols_to_remove)

        if save_file == True:
            df_final.to_csv(save_path, header=True, index=None)

        self.df_final = df_final

        return df_final
    
    