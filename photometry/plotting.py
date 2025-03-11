"""
Author: Sammy Hasler

Class for plotting with the deconfusion + photometry work.

"""
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../') # TODO: remove later -- place plotting.py in deconfuser code structure
import deconfuser.sample_planets as sample_planets
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update({'font.size':14})

class Plotting:
    """
    Class to generate plots from deconfuser+photometry ranking scheme. Requires dataframes generated 
    from reading text file output from deconfuser and supporting_functions.
    """
    
    def __init__(self, df_all, system_number, mu=4*np.pi**2, colors=['#72D7E1', '#912F56', '#E99D98']):
        '''
        Constructor 

        Note: Will not be able to plot more than 18 epochs for a system. Extend markers list if you need to plot more.

        Parameters
        ----------
        df_all : pandas.DataFrame
            Dataframe of all output from the deconfuser (from test_deconfuser_output_*.txt file)
        system_number : int
            Number of the system of interest
        mu : float, optional
            Solar gravitational parameter, by default 4*np.pi**2 (Sun)
        colors : list, optional
            Colors for plotting, by default ['#72D7E1', '#912F56', '#E99D98']

        '''
        self.df_all = df_all
        self.system_number = system_number
        self.mu = mu
        self.markers = ['o', 's', 'v', 'P', '<', '>', '1', '2', '3', '4', '8', ',', 'p', 'x', 'h', 'H', '+', '.']
        self.colors = colors 

    def system_detections(self, plot_tracks=True, save_plot=False, save_path=None, xlim=None, ylim=None, plot_title=True, title_with_i=None,
                          legend_bbox_anchor=None, legend_ncol=1, plot_legend=True):
        '''
        Function to plot system detections for system number of interest.
        Plots detections for system number of interest. 

        Parameters
        ----------
        plot_tracks : bool, optional
            If True, plots the orbit tracks of the planets of interest, by default False
            
        '''
        
        i = 0 # for labeling
        
        coords_df = self.df_all.loc[(self.df_all.system == self.system_number) & (self.df_all.n_orbit_options.isnull()), ['xyzs']] # gets xyz coordinates for true system orbits
        params_df = self.df_all.loc[(self.df_all.system == self.system_number) & (self.df_all.n_orbit_options.isnull()), ['a', 'e', 'i', 'o', 'O', 'M0']] # get orbital parameters for true system orbits
        
        fig, ax = plt.subplots()
        
        for index, row in coords_df.iterrows():
            xyzs = eval(row['xyzs']) # convert string to list
            for detection in range(len(xyzs)):
                plt.scatter(xyzs[detection][0], xyzs[detection][1], marker=self.markers[detection],
                        color='k', s=100, label=f'epoch #{detection+1}' if i == 0 else "") # plots each detection with different shape
            i += 1
            
        if plot_tracks: # if we want orbit tracks, calculate those orbits tracks given the orbit parameters
            m = 0
            for index, row in params_df.iterrows(): # for each planet in the system
                # Calculate orbit track
                xs_more, ys_more, zs_more = sample_planets.get_observations(eval(row['a']), eval(row['e']), 
                                                                            eval(row['i']), eval(row['o']), 
                                                                            eval(row['O']), eval(row['M0']),
                                                                            2*np.pi*np.sqrt(eval(row['a'])**3/self.mu)*np.arange(0,1.1,0.01), 
                                                                            self.mu)
                # Plot orbit track
                plt.plot(xs_more[0], ys_more[0], linewidth=2, color='k')
        # Get inclination value
        inc = np.rad2deg(float(params_df['i'].values[0]))
                
        plt.scatter(0, 0, marker='*', color='gold', s=150)
        if plot_legend:
            plt.legend(bbox_to_anchor=legend_bbox_anchor, ncols=legend_ncol)#loc='upper left', framealpha=0.4)#bbox_to_anchor=(1.0,1.0))
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.xlim(xlim)
        plt.ylim(ylim)
        if plot_title:
            plt.title(f'System detections (System #{self.system_number})')
        if title_with_i:
            plt.title(f"{title_with_i}, i = {inc:.2f}°")
        if save_plot:
            plt.savefig(save_path, dpi='figure', bbox_inches='tight')
        plt.show()


    def system_options(self, confused_options_df, save_plot=False, save_path=None, xlim=None, ylim=None, darkmode_axes=False):
        '''
        Function to plot confused system orbit options on top of detections for system number of interest.
        Accepts dataframe of only confused options (output of get_top_group_options) from the deconfuser.
        
        Parameters
        ----------
        confused_options_df : pandas.DataFrame
            Dataframe of the confused system options returned by get_top_group_options function.

        '''    

        # get dfs for system number of interest
        system_df = self.df_all[(self.df_all.system == self.system_number) & (self.df_all.n_orbit_options.isnull())]
        confused_system_df = confused_options_df[(confused_options_df.system == self.system_number) & 
                                                (confused_options_df.n_orbit_options != 0)] # check n_orbit_options != 0 -- why are there zero options being output?
        
        # get unique partitions
        confused_partitions = confused_system_df['partition'].unique()
        
        # iterate over top partitions
        for partition in confused_partitions: 
            i = 0 # for labeling detections      
            if darkmode_axes:
                facecolor = 'black'
            else:
                facecolor = 'white'  
            fig, ax = plt.subplots(facecolor=facecolor) # reset figure
            partition_df = confused_system_df[(confused_system_df.partition == partition)] # gets df of only partition being iterated over
            
            # plot system detections
            for index, row in system_df.iterrows():
                xyzs = eval(row['xyzs']) # convert str to list
                for detection in range(len(xyzs)):
                    plt.scatter(xyzs[detection][0], xyzs[detection][1], marker=self.markers[detection],
                            label=f'detection #{detection}' if i == 0 else "", color='k', s=100) 
                i += 1
                
            # plot partition orbit options
            for index, row in partition_df.iterrows(): # iterate over each planet in the partition
                # calculate orbit track
                xs_more, ys_more, zs_more = sample_planets.get_observations(float(row['top_a']), float(row['top_e']),
                                                                        float(row['top_i']), float(row['top_o']),
                                                                        float(row['top_O']), float(row['top_M0']),
                                                                        2*np.pi*np.sqrt(float(row['top_a'])**3/self.mu)*np.arange(0,1,0.01), 
                                                                        self.mu)
                plt.plot(xs_more[0], ys_more[0], linewidth=3, alpha=0.5)

                
            plt.scatter(0, 0, marker='*', color='gold', s=150)
            plt.legend(bbox_to_anchor=(1.0,1.0))
            plt.xlabel('x (AU)')
            plt.ylabel('y (AU)')
            plt.xlim(xlim)
            plt.ylim(ylim)
            if darkmode_axes:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(which='major', length=12, colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
            plt.title(f'System {self.system_number} Partition options\n(Partition: {partition})')
            if save_plot:
                plt.savefig(save_path, dpi='figure', bbox_inches='tight')
            plt.show()

    def top_partition(self, top_ranked_df):
        '''
        Function to plot top ranked orbit options on top of detections for system number of interest.
        Accepts dataframe of top ranked options (output of top_ranked_partition) from the deconfuser.
        
        Parameters
        ----------
        top_ranked_df : pandas.DataFrame
            Dataframe of output with partitions ranked (from top_ranked_partition function)
        
        '''
    
        # get dfs for system number of interest
        system_df = self.df_all[(self.df_all.system == self.system_number) & (self.df_all.n_orbit_options.isnull())]
        top_partition_df = top_ranked_df[(top_ranked_df.system == self.system_number) & 
                                                (top_ranked_df.top_ranked_partition == True)]
        L_partition = np.prod(top_partition_df['L_orbit'].values)

        i = 0 # for labeling plot
        plt.figure()
        for index, row in system_df.iterrows():
            xyzs = eval(row['xyzs'])
            for detection in range(len(xyzs)):
                plt.scatter(xyzs[detection][0], xyzs[detection][1], marker=self.markers[detection],
                        label=f'detection #{detection}' if i == 0 else "", color='k', s=100)
            i += 1
        j = 0
        for index, row in top_partition_df.iterrows():
            xs_more, ys_more, zs_more = sample_planets.get_observations(float(row['top_a']), float(row['top_e']),
                                                                        float(row['top_i']), float(row['top_o']),
                                                                        float(row['top_O']), float(row['top_M0']),
                                                                        2*np.pi*np.sqrt(float(row['top_a'])**3/self.mu)*np.arange(0,1,0.01), 
                                                                        self.mu)
            plt.plot(xs_more[0], ys_more[0], linewidth=3, alpha=0.5, c=self.colors[j])
            j += 1

        plt.scatter(0, 0, marker='*', color='gold', s=150)
        plt.legend(bbox_to_anchor=(1.0,1.0))
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.title(f'Top ranked partition (System #{self.system_number})\nL={L_partition:.3e}')
        plt.show()

    def compare_top_to_true(self, top_ranked_df, save_plot=False, save_path=None, xlim=None, ylim=None, plot_legend=True, plot_title=True):
        '''
        Function to plot top ranked orbit options on top of detections and true orbits for system number of interest.
        Accepts dataframe of top ranked options (output of top_ranked_partition) from the deconfuser.
        
        Parameters
        ----------
        top_ranked_df : pandas.DataFrame
            Dataframe of output with partitions ranked (from top_ranked_partition function)

        '''
        # get dfs for system number of interest
        system_df = self.df_all[(self.df_all.system == self.system_number) & (self.df_all.n_orbit_options.isnull())]
        top_partition_df = top_ranked_df[(top_ranked_df.system == self.system_number) & 
                                                (top_ranked_df.top_ranked_partition == True)]
        i = 0 # for labeling plot
        j = 0
        plt.figure()
        for index, row in system_df.iterrows():
            xyzs = eval(row['xyzs'])
            true_xs_more, true_ys_more, true_zs_more = sample_planets.get_observations(eval(row['a']), eval(row['e']), 
                                                                            eval(row['i']), eval(row['o']), 
                                                                            eval(row['O']), eval(row['M0']),
                                                                            2*np.pi*np.sqrt(eval(row['a'])**3/self.mu)*np.arange(0,1,0.01), 
                                                                            self.mu)
            for detection in range(len(xyzs)):
                plt.scatter(xyzs[detection][0], xyzs[detection][1], marker=self.markers[detection], color='k', s=100)
            plt.plot(true_xs_more[0], true_ys_more[0], linewidth=2, alpha=0.7, color='k', 
                     label=f"$a_{i+1}$={float(row['a']):.2f} AU\n$e_{i+1}$={float(row['e']):.2f}\n$i_{i+1}$={np.rad2deg(float(row['i'])):.2f}°")

            i += 1
        for index, row in top_partition_df.iterrows():
            xs_more, ys_more, zs_more = sample_planets.get_observations(float(row['top_a']), float(row['top_e']),
                                                                        float(row['top_i']), float(row['top_o']),
                                                                        float(row['top_O']), float(row['top_M0']),
                                                                        2*np.pi*np.sqrt(float(row['top_a'])**3/self.mu)*np.arange(0,1,0.01), 
                                                                        self.mu)

            plt.plot(xs_more[0], ys_more[0], linewidth=3, alpha=0.9, linestyle='--', color=self.colors[j], 
                     label=f"$a_{j+1}$={row['top_a']:.2f} AU\n$e_{j+1}$={row['top_e']:.2f}\n$i_{j+1}$={np.rad2deg(float(row['top_i'])):.2f}°\nL={row['L_orbit']:.4f}")  
            
            j += 1 
            
                
        plt.scatter(0, 0, marker='*', color='gold', s=150)
        if plot_legend:
            plt.legend(bbox_to_anchor=(1.05,-0.15), ncols=3)
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.xlim(xlim)
        plt.ylim(ylim)
        if plot_title:
            plt.title(f"True system vs. top ranked partition \n(System #{row['system']})")
        if save_plot:
            plt.savefig(save_path, dpi='figure', bbox_inches='tight')
        plt.show()

    def multiple_group_options_per_system(self, confused_df, system_num):
        '''
        Function to plot confused system orbit options for systems that have multiple
        orbit options per planet ("group")

        Parameters
        ----------
        confused_df : pandas.DataFrame
            Dataframe of confused system options/

        '''
        simulated_df = self.df_all[(self.df_all.system == system_num) & (self.df_all.n_orbit_options.isnull())]
        confused_df = confused_df[(confused_df.system == system_num)]

        # get unique partitions
        confused_partitions = confused_df['partition'].unique()
        
        # iterate over top partitions
        for partition in confused_partitions: 
            k = 0 # for labeling detections        
            plt.figure() # reset figure
            partition_df = confused_df[(confused_df.partition == partition)] # gets df of only partition being iterated over

            # plot system detections
            for index, row in simulated_df.iterrows():
                xyzs = eval(row['xyzs']) # convert str to list
                for detection in range(len(xyzs)):
                    plt.scatter(xyzs[detection][0], xyzs[detection][1], marker=self.markers[detection],
                            label=f'detection #{detection}' if k == 0 else "", color='k', s=100) 
                k += 1
            # plot partition orbit options
            for index, row in partition_df.iterrows():
                # check for multiple options
                if len(eval(row['a'])) > 1: # if more than 1 option per planet
                    for j in range(len(eval(row['a']))):
                        a = eval(row['a'])[j]
                        e = eval(row['e'])[j]
                        i = eval(row['i'])[j]
                        o = eval(row['o'])[j]
                        O = eval(row['O'])[j]
                        M0 = eval(row['M0'])[j]

                        # calculate orbit track
                        xs_more, ys_more, zs_more = sample_planets.get_observations(a, e, i, o, O, M0,
                                                                                    2*np.pi*np.sqrt(a**3/self.mu)*np.arange(0,1,0.01), 
                                                                                    self.mu)
                        plt.plot(xs_more[0], ys_more[0], linewidth=2, alpha=0.5)

                else:
                # calculate orbit track
                    a = eval(row['a'])[0]
                    e = eval(row['e'])[0]
                    i = eval(row['i'])[0]
                    o = eval(row['o'])[0]
                    O = eval(row['O'])[0]
                    M0 = eval(row['M0'])[0]
                    # calculate orbit track
                    xs_more, ys_more, zs_more = sample_planets.get_observations(a, e, i, o, O, M0,
                                                                                2*np.pi*np.sqrt(a**3/self.mu)*np.arange(0,1,0.01), 
                                                                                self.mu)
                    plt.plot(xs_more[0], ys_more[0], linewidth=3, alpha=0.75, linestyle='dashed')
                    
            plt.scatter(0, 0, marker='*', color='gold', s=150)
            plt.legend(bbox_to_anchor=(1.0,1.0))
            plt.xlabel('x (AU)')
            plt.ylabel('y (AU)')
            plt.title(f'Partition: {partition}')
            plt.show()
        
    def post_individual_group_ranking(self, confused_df, system_num, save_plot=False, save_path=None, xlim=None, ylim=None,
                                      compare_to_sim=False):
        '''
        Plot top ranked groups (orbits) together after running rank_partitions_from_test_deconfuser_output.rank_by_individual_groups()

        Parameters
        ----------
        confused_df : pandas.dataframe
            Dataframe of onyl confused orbit options
        system_num : int
            Number of system from dataframe
        '''

        df_system = self.df_all[(self.df_all.system == system_num) & (self.df_all.n_orbit_options.isnull())]
        df_top_groups = confused_df[(confused_df.system == system_num) & (confused_df.highest_Lingroup == True)]

        i = 0 # for labeling
        plt.figure()
        # First plot detections
        for index, row in df_system.iterrows():
            xyzs = eval(row['xyzs'])
            if compare_to_sim:
                true_xs_more, true_ys_more, true_zs_more = sample_planets.get_observations(eval(row['a']), eval(row['e']), 
                                                                            eval(row['i']), eval(row['o']), 
                                                                            eval(row['O']), eval(row['M0']),
                                                                            2*np.pi*np.sqrt(eval(row['a'])**3/self.mu)*np.arange(0,1,0.01), 
                                                                            self.mu)
                plt.plot(true_xs_more[0], true_ys_more[0], linewidth=2, alpha=0.7, color='k', 
                     label=f"$a_{i+1}$={float(row['a']):.2f} AU\n$e_{i+1}$={float(row['e']):.2f}\n$i_{i+1}$={np.rad2deg(float(row['i'])):.2f}°")

            for detection in range(len(xyzs)):
                plt.scatter(xyzs[detection][0], xyzs[detection][1], marker=self.markers[detection], color='k', s=100)
            i += 1
        # Get matched orbits to plot
        j = 0
        for index, row in df_top_groups.iterrows():
            xs_more, ys_more, zs_more = sample_planets.get_observations(float(row['top_a']), float(row['top_e']),
                                                                        float(row['top_i']), float(row['top_o']),
                                                                        float(row['top_O']), float(row['top_M0']),
                                                                        2*np.pi*np.sqrt(float(row['top_a'])**3/self.mu)*np.arange(0,1,0.01), 
                                                                        self.mu)
            inclination = np.rad2deg(row['top_i']) # get inclination in degrees for labeling
            if self.colors != None:
                plt.plot(xs_more[0], ys_more[0], linewidth=3, linestyle='dashed', alpha=0.75, color=self.colors[j], 
                         label=f"$a_{j+1}$={row['top_a']:.2f} AU\n$e_{j+1}$={row['top_e']:.2f}\n$i_{j+1}$={inclination:.2f}°\n$L_{j+1}$={row['L_orbit']:.4f}")
                j += 1
            else:
                plt.plot(xs_more[0], ys_more[0], linewidth=3, alpha=0.5)

        plt.scatter(0, 0, marker='*', color='gold', s=150)
        plt.legend(bbox_to_anchor=(1.05,-0.15), ncols=3) 
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.title(f'Top ranked groups (System #{system_num})')
        plt.xlim(xlim)
        plt.ylim(ylim)

        if save_plot:
            plt.savefig(save_path, dpi='figure', bbox_inches='tight')

        plt.show()


    def iterate_over_all_partitions(self, df_confused, save_plot=False, save_path=None, xlims=None, ylims=None,
                                plot_title=True, L_in_title=False, inc_group_for_title=None):
        '''
        Iterate over all confused orbit options in a dataframe and plot all orbit tracks.

        Parameters
        ----------
        df_confused : pandas.DataFrame
            Dataframe of confused options
        save_plot : bool, optional
            Whether or not to save plot, by default False
        save_path : str, optional
            Path to save plot to if save_plot is True, by default None
        xlims : list of lists, optional
            List of xlims that is the same length as the number of confused options to be plotted, by default None
        ylims : list of lists, optional
            List of ylims that is the same length as the number of confused options to be plotted, by default None
        plot_title : bool, optional
            Whether or not to add a title to the plot, by default True
        L_in_title : bool, optional
            Whether or not to list the system likelihood in the title, by default False
        inc_group_for_title : str, optional
            Extra string to add to title (), by default None
        '''
        
        system_numbers = self.df_all['system'].unique() # get systems in dataframe
        # Iterate over systems
        for system in system_numbers:
            xlim = xlims[system-1]
            ylim = ylims[system-1]
            print('System number: ', system)
            system_df = self.df_all[(self.df_all.system == system) & (self.df_all.n_orbit_options.isnull())]
            confused_system_df = df_confused[(df_confused.system == system) & (df_confused.n_orbit_options != 0)]
            
            # Get unique partitions
            confused_partitions = confused_system_df['partition'].unique()
            
            # iterate over top partitions
            n_partition = 1
            i_title = 0
            for partition in confused_partitions:
                i = 0 # for labeling detections
                fig, ax = plt.subplots() # reset figure
                partition_df = confused_system_df[(confused_system_df.partition == partition)] # gets df of only partition being iterated over
                print('L_orbit: ', partition_df['L_orbit'].values)
                L_partition = np.prod(partition_df['L_orbit'].values)

                # Plot system detections
                for index, row in system_df.iterrows():
                    xyzs = eval(row['xyzs']) # convert str to list
                    for detection in range(len(xyzs)):
                        plt.scatter(xyzs[detection][0], xyzs[detection][1], marker=self.markers[detection],
                                color='k', s=100)
                    i += 1
                    
                # Plot partition orbit options
                j = 0
                for index, row in partition_df.iterrows(): # iterate over each planet in the partition
                    # calculate orbit track
                    xs_more, ys_more, zs_more = sample_planets.get_observations(float(row['top_a']), float(row['top_e']),
                                                                        float(row['top_i']), float(row['top_o']),
                                                                        float(row['top_O']), float(row['top_M0']),
                                                                        2*np.pi*np.sqrt(float(row['top_a'])**3/self.mu)*np.arange(0,1.1,0.01), 
                                                                        self.mu)
                    # Grab inclination for each row, put in +/- 90° frame
                    inclination = np.rad2deg(row['top_i'])
                    print('inclination: ', inclination)
                        
                    plt.plot(xs_more[0], ys_more[0], linewidth=3, color=self.colors[j], alpha=0.75,
                            label=f"$a_{j+1}$={row['top_a']:.2f} AU\n$e_{j+1}$={row['top_e']:.2f}\n$i_{j+1}$={inclination:.2f}°")#\n$L_{j+1}$={row['L_orbit']:.4f}")
                    print(f"a={row['top_a']:.3f}, e={row['top_e']:.3f}, i={inclination:.3f}")
                    j += 1
                
                plt.scatter(0, 0, marker='*', color='gold', s=150)
                plt.legend(bbox_to_anchor=(1.05,-0.15), ncols=3)
                if plot_title:
                    if L_in_title:
                        plt.title(f'Possible Orbits #{i_title+1} (L={L_partition:.3e})')
                    else:
                        system_title = system
                        if system == 11:
                            system_title = 10
                        plt.title(f'System {system_title}, {inc_group_for_title} Inc.\nPossible Orbits #{i_title+1}')
                    i_title += 1

                plt.xlabel('x (AU)')
                plt.ylabel('y (AU)')
                plt.ylim(ylim)
                plt.xlim(xlim)
                if save_plot:
                    plt.savefig(save_path+f'_system{system}_partition{n_partition}.png', 
                                dpi='figure', bbox_inches='tight')
                plt.show()
                n_partition += 1
                    